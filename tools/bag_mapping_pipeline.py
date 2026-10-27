#!/usr/bin/env python3
"""ROS1 bag → 静态点云 PCD（+ 轨迹 + 指标），用于 2D 建图前处理。

不依赖 ROS 运行时，只读 bag。输入任意含 Odometry + PointCloud2 的 ROS1 bag，
输出可直接喂给 slope_map_projector.py 的 PCD 与 TUM。

处理链：
  1. 流式解析 bag（none/bz2/lz4），抽取 Odometry 位姿与世界系点云；
  2. 5 cm 体素去重，统计每个体素被多少帧观测到；
  3. 自动估计地面偏移（轨迹附近点的相对高度主峰）与帧数阈值；
  4. 输出 cloud_voxel_all.pcd（全部体素）、cloud_static.pcd（过滤后）、traj.tum、metrics.json。

用法：
    python3 bag_mapping_pipeline.py --bag run.bag --out out/

    # 指定话题/帧名/参数
    python3 bag_mapping_pipeline.py --bag run.bag --out out/ \
        --odom-topic /Odometry --cloud-topic /cloud_registered \
        --voxel 0.05 --min-frames 5
"""

import argparse
import bz2
import json
import math
import struct
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- bag 读取

def _parse_header(blob):
    fields, offset = {}, 0
    while offset < len(blob):
        (length,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        name, _, value = blob[offset:offset + length].partition(b"=")
        offset += length
        fields[name.decode()] = value
    return fields


def _iter_records(blob):
    offset = 0
    while offset + 4 <= len(blob):
        (header_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        header = _parse_header(blob[offset:offset + header_len])
        offset += header_len
        (data_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        yield header, blob[offset:offset + data_len]
        offset += data_len


def _read_record(stream):
    raw = stream.read(4)
    if len(raw) < 4:
        return None, b""
    (header_len,) = struct.unpack("<I", raw)
    header = _parse_header(stream.read(header_len))
    (data_len,) = struct.unpack("<I", stream.read(4))
    return header, stream.read(data_len)


def _decompress(header, payload):
    kind = header.get("compression", b"none")
    if kind in (b"none", b""):
        return payload
    if kind in (b"\x02", b"bz2"):
        return bz2.decompress(payload)
    if kind == b"lz4":
        try:
            import lz4.frame  # type: ignore
        except ImportError as exc:
            raise RuntimeError("bag uses lz4 chunks; install python-lz4") from exc
        return lz4.frame.decompress(payload)
    raise ValueError(f"unsupported bag chunk compression: {kind!r}")


def scan_connections(path):
    """读取 bag 的 connection 表：conn_id -> (topic, type)。"""
    with open(path, "rb") as stream:
        stream.readline()
        header, _ = _read_record(stream)
        if header is None or "index_pos" not in header:
            raise SystemExit(f"{path}: 不是有效的 ROS1 bag（缺少 bag header）")
        index_pos = struct.unpack("<Q", header["index_pos"])[0]
        stream.seek(index_pos)
        conns = {}
        while True:
            header, payload = _read_record(stream)
            if header is None:
                break
            if header["op"][0] == 7:
                (conn_id,) = struct.unpack("<I", header["conn"])
                info = _parse_header(payload)
                conns[conn_id] = (info["topic"].decode(), info["type"].decode())
    return conns


def pick_topic(conns, msg_type, preferred, explicit):
    if explicit:
        for conn_id, (topic, kind) in conns.items():
            if topic == explicit:
                return conn_id
        raise SystemExit(f"topic {explicit} not found in bag (have {sorted(t for t, _ in conns.values())})")
    for conn_id, (topic, kind) in conns.items():
        if topic == preferred and kind == msg_type:
            return conn_id
    for conn_id, (topic, kind) in conns.items():
        if kind == msg_type:
            return conn_id
    raise SystemExit(f"no {msg_type} topic in bag")


# ------------------------------------------------------------ 消息解析

def parse_odometry(payload):
    offset = 0
    offset += 4                                   # seq
    sec, nsec = struct.unpack_from("<II", payload, offset); offset += 8
    (frame_len,) = struct.unpack_from("<I", payload, offset); offset += 4 + frame_len
    (child_len,) = struct.unpack_from("<I", payload, offset); offset += 4 + child_len
    px, py, pz = struct.unpack_from("<3d", payload, offset); offset += 24
    qx, qy, qz, qw = struct.unpack_from("<4d", payload, offset)
    return sec + nsec * 1e-9, px, py, pz, qx, qy, qz, qw


def parse_cloud_xyz(payload):
    offset = 0
    offset += 4                                   # seq
    sec, nsec = struct.unpack_from("<II", payload, offset); offset += 8
    (frame_len,) = struct.unpack_from("<I", payload, offset); offset += 4
    frame = payload[offset:offset + frame_len].decode(); offset += frame_len
    height, width = struct.unpack_from("<II", payload, offset); offset += 8
    (nfields,) = struct.unpack_from("<I", payload, offset); offset += 4
    for _ in range(nfields):
        (name_len,) = struct.unpack_from("<I", payload, offset); offset += 4
        offset += name_len + 4 + 1 + 4            # name + offset + datatype + count
    offset += 1                                   # is_bigendian
    step, _row_step = struct.unpack_from("<II", payload, offset); offset += 8
    (data_len,) = struct.unpack_from("<I", payload, offset); offset += 4
    npts = data_len // step if step else 0
    if npts == 0:
        return frame, sec + nsec * 1e-9, np.zeros((0, 3), np.float32)
    arr = np.frombuffer(payload[offset:offset + data_len], dtype="<f4").reshape(npts, step // 4)
    return frame, sec + nsec * 1e-9, arr[:, 0:3].astype(np.float32)


# ------------------------------------------------------------ 核心处理

def voxel_keys(xyz, resolution):
    grid = np.floor(xyz / resolution).astype(np.int64) + 20000
    return (grid[:, 0] * 40001 + grid[:, 1]) * 40001 + grid[:, 2]


def estimate_ground_offset(means, counts, traj_xy, traj_z, near_radius=1.5,
                           lo=-2.5, hi=0.5, bin_width=0.02):
    """轨迹附近点的 (z - 当地轨迹 z) 主峰即地面；offset = -peak。

    用 1 m 格子把点挂到同格的轨迹高度上，全程向量化：
    轨迹发散（格子数暴涨）时也不会退化成循环遍历。
    """
    if len(traj_xy) < 2 or len(means) == 0:
        return None, None

    cell = 1.0
    span = 200000                      # 单轴 ±200 km，足够覆盖任何未发散/已发散轨迹
    side = 2 * span + 1

    tcx = np.floor(traj_xy[:, 0] / cell).astype(np.int64) + span
    tcy = np.floor(traj_xy[:, 1] / cell).astype(np.int64) + span
    tk = tcx * side + tcy

    order = np.argsort(tk, kind="stable")
    tk_sorted = tk[order]
    tz_sorted = traj_z[order].astype(np.float64)
    uniq, starts = np.unique(tk_sorted, return_index=True)
    counts_t = np.diff(np.append(starts, len(tk_sorted)))
    tz_cell = np.add.reduceat(tz_sorted, starts) / counts_t

    pcx = np.floor(means[:, 0] / cell).astype(np.int64) + span
    pcy = np.floor(means[:, 1] / cell).astype(np.int64) + span
    pk = pcx * side + pcy

    pos = np.searchsorted(uniq, pk)
    pos = np.clip(pos, 0, len(uniq) - 1)
    matched = uniq[pos] == pk
    if not matched.any():
        return None, None

    d = means[matched, 2] - tz_cell[pos[matched]]
    hist, edges = np.histogram(d, bins=np.arange(lo, hi + bin_width, bin_width))
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")
    peak = int(np.argmax(smooth))
    center = edges[peak] + bin_width / 2
    window = d[np.abs(d - center) <= 2 * bin_width]
    refined = float(np.median(window)) if len(window) else float(center)
    offset = -refined
    hist_info = {
        "peak_center_m": round(float(center), 4),
        "refined_m": round(refined, 4),
        "offset_m": round(offset, 4),
        "matched_voxels": int(matched.sum()),
        "traj_cells": int(len(uniq)),
        "bins": [round(float(v), 3) for v in edges[:-1]],
        "counts": hist.tolist(),
    }
    return offset, hist_info


def visibility_counts(frames, traj, uniq_keys, resolution, step, max_range, stride):
    """帧内射线穿透投票（DUFOMap 核心判据的轻量实现）。

    对每帧的每个点，从该帧位姿沿射线按 step 采样，落在已占用体素上的采样点
    记一次"穿透"。同时被大量命中、又被大量穿透的体素 = 动态物（人先被看到、
    离开后又被扫穿）。

    返回与 uniq_keys 对齐的穿透计数。
    """
    counts = np.zeros(len(uniq_keys), dtype=np.int64)
    if len(frames) == 0 or len(uniq_keys) == 0:
        return counts
    keys_sorted = uniq_keys
    traj_t = traj[:, 0]
    # 必须与 voxel_keys() 用同一套编码，否则 searchsorted 永远对不上
    span = 20000
    side = 40001

    usable = 0
    for index in range(0, len(frames), stride):
        stamp, xyz = frames[index]
        if len(xyz) == 0:
            continue
        pos = int(np.clip(np.searchsorted(traj_t, stamp), 1, len(traj_t) - 1))
        # 取时间上更近的一帧位姿
        if abs(traj_t[pos - 1] - stamp) < abs(traj_t[pos] - stamp):
            pos -= 1
        origin = traj[pos, 1:4]
        d = xyz[:, :3].astype(np.float64) - origin
        length = np.linalg.norm(d, axis=1)
        keep = (length > 0.5) & (length <= max_range)
        if not keep.any():
            continue
        d = d[keep]
        length = length[keep]
        steps = np.maximum(1, np.ceil(length / step).astype(np.int64))
        total = int(steps.sum())
        if total == 0:
            continue
        ray = np.repeat(np.arange(len(length), dtype=np.int64), steps)
        starts = np.repeat(np.cumsum(steps) - steps, steps)
        dist = ((np.arange(total, dtype=np.float64) - starts) + 0.5) * step
        unit = d[ray] / length[ray][:, None] * dist[:, None]
        samples = origin + unit
        # 跳过紧邻位姿与终点附近，避免把自身和命中面记成穿透
        inside_range = (dist > 0.5) & (dist < (length[ray] - 2.0 * step))
        if not inside_range.any():
            continue
        samples = samples[inside_range]
        grid = np.floor(samples / resolution).astype(np.int64) + span
        keys = (grid[:, 0] * side + grid[:, 1]) * side + grid[:, 2]
        idx = np.searchsorted(keys_sorted, keys)
        idx = np.clip(idx, 0, len(keys_sorted) - 1)
        hit = keys_sorted[idx] == keys
        if not hit.any():
            continue
        counts += np.bincount(idx[hit], minlength=len(keys_sorted)).astype(np.int64)
        usable += 1
    return counts


def otsu_threshold(values, lo=2, hi=40):
    """对帧数做两类分割，返回阈值（保留 count >= 阈值）。"""
    if len(values) == 0:
        return lo
    v = np.clip(values.astype(np.float64), lo, hi)
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return lo
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b[~np.isfinite(sigma_b)] = -1
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def write_pcd(path, points, intensity):
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    record = np.column_stack([points, intensity]).astype("<f4")
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(record.tobytes())


def main():
    parser = argparse.ArgumentParser(description="ROS1 bag → 静态点云 PCD + 轨迹 + 指标")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--cloud-topic", default="/cloud_registered")
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--min-frames", type=int, default=0,
                        help="保留被 >=N 帧观测到的体素；0=自动（Otsu，钳制 2~20）")
    parser.add_argument("--ground-offset", type=float, default=None,
                        help="雷达光心到地面高度；省略则自动估计")
    parser.add_argument("--visibility-filter", action="store_true",
                        help="启用射线穿透投票（DUFOMap 轻量实现），与帧计数联合判定动态点")
    parser.add_argument("--visibility-step", type=float, default=0.10,
                        help="射线采样间隔（m）")
    parser.add_argument("--visibility-max-range", type=float, default=8.0,
                        help="参与投票的射线最大长度（m）")
    parser.add_argument("--visibility-stride", type=int, default=2,
                        help="每隔几帧做一次投票（越大越快越粗）")
    parser.add_argument("--visibility-ratio", type=float, default=0.5,
                        help="穿透占比超过该值判为动态（0~1）")
    parser.add_argument("--remove-ground", action="store_true",
                        help="额外输出 cloud_noground.pcd（按当地地面去掉地面及以下的点）")
    parser.add_argument("--ground-remove-height-m", type=float, default=0.10,
                        help="低于当地地面 + 该高度的点视为地面并删除")
    parser.add_argument("--max-points-per-frame", type=int, default=0,
                        help="每帧最多取多少点（0=不限，用于限内存）")
    args = parser.parse_args()

    started = time.time()
    conns = scan_connections(args.bag)
    odom_id = pick_topic(conns, "nav_msgs/Odometry", args.odom_topic, args.odom_topic if args.odom_topic in
                         [t for t, _ in conns.values()] else None)
    cloud_id = pick_topic(conns, "sensor_msgs/PointCloud2", args.cloud_topic, args.cloud_topic if args.cloud_topic in
                          [t for t, _ in conns.values()] else None)

    odom = []
    frame_clouds = []            # [(stamp, xyz)] 供可见性投票
    voxel_all_keys, voxel_all_rep = [], []
    frames = 0
    frames_used = 0
    cloud_frame_name = None

    with open(args.bag, "rb") as stream:
        stream.readline()
        _read_record(stream)
        while True:
            header, payload = _read_record(stream)
            if header is None:
                break
            if header["op"][0] != 5:
                continue
            frames += 1
            for inner, data in _iter_records(_decompress(header, payload)):
                if inner["op"][0] != 2:
                    continue
                (conn_id,) = struct.unpack("<I", inner["conn"])
                if conn_id == odom_id:
                    odom.append(parse_odometry(data))
                elif conn_id == cloud_id:
                    name, _stamp, xyz = parse_cloud_xyz(data)
                    cloud_frame_name = cloud_frame_name or name
                    if len(xyz) == 0:
                        continue
                    if args.visibility_filter:
                        frame_clouds.append((_stamp, xyz))
                    if args.max_points_per_frame:
                        xyz = xyz[:args.max_points_per_frame]
                    frames_used += 1
                    keys = voxel_keys(xyz, args.voxel)
                    uniq, first = np.unique(keys, return_index=True)
                    voxel_all_keys.append(uniq)
                    voxel_all_rep.append(xyz[first])

    if not odom or not frames_used:
        raise SystemExit("bag has no usable Odometry or cloud messages")

    odom.sort()
    traj = np.asarray(odom, dtype=np.float64)
    keys = np.concatenate(voxel_all_keys)
    reps = np.concatenate(voxel_all_rep)
    order = np.argsort(keys, kind="stable")
    keys, reps = keys[order], reps[order].astype(np.float64)
    uniq_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    means = np.add.reduceat(reps, starts, axis=0) / counts[:, None]

    ground_offset, ground_hist = estimate_ground_offset(
        means, counts, traj[:, 1:3], traj[:, 3]
    ) if args.ground_offset is None else (args.ground_offset, None)

    visibility = None
    if args.visibility_filter:
        visibility = visibility_counts(
            frame_clouds, traj, uniq_keys, args.voxel,
            args.visibility_step, args.visibility_max_range, max(1, args.visibility_stride),
        )

    if args.min_frames > 0:
        min_frames, threshold_mode = args.min_frames, "explicit"
    else:
        # Otsu 在"低帧数占多数"的长跑数据上会偏高（实测 5 分钟趟给出 13，把弱结构也删了），
        # 因此钳到 3~8：低于 3 会放进大量瞬时点，高于 8 会开始丢真实结构。
        auto = otsu_threshold(counts)
        min_frames = int(min(max(2, round(auto)), 4))
        threshold_mode = f"auto-otsu(raw={auto:.1f}, clamped 2..4)"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "traj_camera_init.tum", "w", encoding="ascii") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in traj:
            stream.write("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % tuple(row))

    write_pcd(out / "cloud_voxel_all.pcd", means.astype(np.float32),
              np.clip(counts, 0, 255).astype(np.float32))
    if visibility is not None:
        np.save(out / "voxel_hit_counts.npy", counts)
        np.save(out / "voxel_visibility_counts.npy", visibility)
        np.save(out / "voxel_keys.npy", uniq_keys)

    keep = counts >= min_frames
    if visibility is not None:
        observed = counts.astype(np.float64)
        crossed = visibility.astype(np.float64)
        ratio = crossed / np.maximum(observed + crossed, 1.0)
        dynamic = ratio >= args.visibility_ratio
        keep = keep & ~dynamic
    write_pcd(out / "cloud_static.pcd", means[keep].astype(np.float32),
              np.clip(counts[keep], 0, 255).astype(np.float32))

    noground_stats = None
    if args.remove_ground and ground_offset is not None:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        if cKDTree is not None and len(traj):
            tree = cKDTree(traj[:, 1:3])
            _dist, nearest = tree.query(means[:, :2], k=1)
            local_ground = traj[nearest, 3] - ground_offset
            above = ((means[:, 2] - local_ground) >= args.ground_remove_height_m) & keep
            write_pcd(out / "cloud_noground.pcd", means[above].astype(np.float32),
                      np.clip(counts[above], 0, 255).astype(np.float32))
            noground_stats = {"points": int(above.sum()),
                              "removed": int((~above).sum()),
                              "height_m": args.ground_remove_height_m}

    metrics = {
        "bag": str(Path(args.bag).resolve()),
        "chunks_read": frames,
        "cloud_frames": frames_used,
        "cloud_frame_id": cloud_frame_name,
        "odom_messages": int(len(traj)),
        "duration_s": round(float(traj[-1, 0] - traj[0, 0]), 2),
        "trajectory_z_span_m": round(float(np.ptp(traj[:, 3])), 3),
        "voxel_m": args.voxel,
        "voxels_total": int(len(uniq_keys)),
        "ground_offset_m": None if ground_offset is None else round(float(ground_offset), 4),
        "ground_histogram": ground_hist,
        "frame_count_percentiles": {q: int(np.percentile(counts, q)) for q in (50, 90, 99)},
        "frame_count_max": int(counts.max()),
        "min_frames": int(min_frames),
        "threshold_mode": threshold_mode,
        "static_voxels": int(keep.sum()),
        "noground_output": noground_stats,
        "static_ratio": round(float(keep.mean()), 4),
        "outputs": [str(out / "traj_camera_init.tum"), str(out / "cloud_voxel_all.pcd"),
                    str(out / "cloud_static.pcd")],
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
