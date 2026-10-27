#!/usr/bin/env python3
"""把世界系 FAST-LIO PCD 快速投成 2D 占据栅格（PGM/YAML），用于参数对比预览。

这是**诊断/预览**工具，不是 slope_map_projector 的替代：
  - 不做轨迹分段、不做离群过滤、不做自由空间的射线裁剪
  - 地面基准只有两种模式：全局偏移，或"每格最低点"近似
  - 输出只有占据格，没有 free/unknown 区分，不能直接当导航交付图

典型用途：同一份 PCD 扫不同 ground offset，看地面是否灌进 2D 图。

示例
----
全局地面偏移（地面在 -0.85，雷达光心到地面 0.85 m）::

    python3 quick_2d_preview.py --pcd scans.pcd --ground-offset 0.85 \
        --out-prefix /tmp/prev_085

逐格局部地面（地面起伏/坡道场景，会把地面上的矮障碍一并抹掉）::

    python3 quick_2d_preview.py --pcd scans.pcd --local-ground \
        --out-prefix /tmp/prev_local

带轨迹距离门限（裁掉远场离群点）::

    python3 quick_2d_preview.py --pcd scans.pcd --local-ground \
        --trajectory trajectory.tum --max-traj-dist 8.0 --out-prefix /tmp/prev_gated
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- PCD 读取

_PCD_TYPES = {
    ("F", 2): "<f2",
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def read_pcd_xyz(path):
    """读 PCD 的 xyz（仅支持 DATA binary 未压缩）。返回 (N,3) float64。"""
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: PCD header ended before DATA")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                offset = stream.tell()
                break

    if header["DATA"] != ["binary"]:
        raise ValueError(f"{path}: only uncompressed DATA binary is supported")

    names = header.get("FIELDS") or header.get("FIELD")
    if not names:
        raise ValueError(f"{path}: PCD header has no FIELDS")
    sizes = [int(v) for v in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(names))]
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"{path}: FIELDS/SIZE/TYPE/COUNT lengths differ")
    if not {"x", "y", "z"}.issubset(set(names)):
        raise ValueError(f"{path}: PCD must contain x, y and z")

    dtype = np.dtype(
        [(name, _PCD_TYPES[(kind, size)], count) if count > 1 else (name, _PCD_TYPES[(kind, size)])
         for name, kind, size, count in zip(names, types, sizes, counts)]
    )
    points = np.fromfile(path, dtype=dtype, offset=offset)
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def read_tum_xy(path):
    """读 TUM 轨迹（timestamp tx ty tz [qx qy qz qw]），返回 (N,2)。"""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((float(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"{path}: no TUM poses parsed")
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------- 投 影

def local_ground_height(xyz, cell):
    """每个点所属 XY 格的最低 Z（近似当地地面）。"""
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ix = np.floor(xs / cell).astype(np.int64)
    iy = np.floor(ys / cell).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    width = int(ix.max()) + 1
    height = int(iy.max()) + 1
    zmin = np.full(height * width, np.inf)
    np.minimum.at(zmin, iy * width + ix, zs)
    return zmin[iy * width + ix]


def despeckle_mask(xyz, cell, min_points):
    """去掉稀疏格子里的点（粗格内点数 < min_points），用于裁掉远场孤立散点。"""
    if min_points <= 1:
        return np.ones(len(xyz), dtype=bool)
    ix = np.floor(xyz[:, 0] / cell).astype(np.int64)
    iy = np.floor(xyz[:, 1] / cell).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    width = int(ix.max()) + 1
    counts = np.bincount(iy * width + ix, minlength=width * (int(iy.max()) + 1))
    return counts[iy * width + ix] >= min_points


def trajectory_gate_mask(xyz, traj_xy, max_dist, cell=0.5):
    """XY 距离轨迹 <= max_dist 的点为 True（栅格化 + 方形膨胀，足够粗筛）。"""
    from numpy.lib.stride_tricks import sliding_window_view

    if max_dist <= 0:
        return np.ones(len(xyz), dtype=bool)
    tx = np.floor(traj_xy[:, 0] / cell).astype(np.int64)
    ty = np.floor(traj_xy[:, 1] / cell).astype(np.int64)
    ox, oy = tx.min(), ty.min()
    tx -= ox
    ty -= oy
    grid = np.zeros((int(ty.max()) + 1, int(tx.max()) + 1), dtype=bool)
    grid[ty, tx] = True

    radius = max(1, int(math.ceil(max_dist / cell)))
    padded = np.pad(grid, radius, constant_values=False)
    window = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    dilated = window.any(axis=(2, 3))

    ix = np.floor(xyz[:, 0] / cell).astype(np.int64) - ox
    iy = np.floor(xyz[:, 1] / cell).astype(np.int64) - oy
    inside = (ix >= 0) & (ix < dilated.shape[1]) & (iy >= 0) & (iy < dilated.shape[0])
    mask = np.zeros(len(xyz), dtype=bool)
    mask[inside] = dilated[iy[inside], ix[inside]]
    return mask


def project_to_grid(xyz, resolution, height_min, height_max, ground):
    """把点投成占据栅格。返回 (occ, x0, y0)，occ[row, col]，row 0 对应最大 Y。"""
    xs, ys = xyz[:, 0], xyz[:, 1]
    x0 = float(xs.min())
    y0 = float(ys.min())
    cols = int(math.floor((xs.max() - x0) / resolution)) + 1
    rows = int(math.floor((ys.max() - y0) / resolution)) + 1
    occ = np.zeros((rows, cols), dtype=bool)

    dz = xyz[:, 2] - ground
    hit = (dz >= height_min) & (dz <= height_max)
    if not hit.any():
        return occ, x0, y0

    cx = np.floor((xs[hit] - x0) / resolution).astype(np.int64)
    cy = np.floor((ys[hit] - y0) / resolution).astype(np.int64)
    occ[cy, cx] = True
    return occ, x0, y0


def write_pgm(occ, path, occupied=0, free=205):
    """写 P5 PGM：row 0 = 最大 Y（map_server 契约，图像上下翻转）。"""
    image = np.where(occ[::-1], occupied, free).astype(np.uint8)
    header = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    Path(path).write_bytes(header + image.tobytes())


def write_yaml(path, image_name, resolution, origin):
    Path(path).write_text(
        "\n".join(
            [
                f'image: "{image_name}"',
                f"resolution: {resolution}",
                "origin: [{:.6f}, {:.6f}, 0.0]".format(origin[0], origin[1]),
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "mode: trinary",
                "",
            ]
        )
    )


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="把世界系 PCD 快速投成 2D 占据栅格（PGM/YAML），用于参数对比预览",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pcd", required=True, help="输入 PCD（DATA binary）")
    ground = parser.add_mutually_exclusive_group(required=True)
    ground.add_argument("--ground-offset", type=float,
                        help="全局地面偏移：地面 Z = -offset（如雷达光心离地 0.85 m 则传 0.85）")
    ground.add_argument("--local-ground", action="store_true",
                        help="用每格最低点作为当地地面（会让地面上的矮障碍一并消失）")
    parser.add_argument("--local-ground-cell", type=float, default=0.10,
                        help="局部地面取最低点的 XY 格边长（m）")
    parser.add_argument("--height-min", type=float, default=0.15, help="障碍窗下沿（相对地面，m）")
    parser.add_argument("--height-max", type=float, default=1.80, help="障碍窗上沿（相对地面，m）")
    parser.add_argument("--resolution", type=float, default=0.05, help="栅格分辨率（m）")
    parser.add_argument("--max-abs-z", type=float, default=0.0,
                        help="原始 Z 绝对值超过该值的点直接丢弃（0=不过滤，用于剔帧级发散点）")
    parser.add_argument("--trajectory", help="可选 TUM 轨迹，用于距离门限")
    parser.add_argument("--max-traj-dist", type=float, default=8.0,
                        help="点距轨迹超过该值则丢弃（m）")
    parser.add_argument("--despeckle-cell", type=float, default=0.5,
                        help="去散点的粗格边长（m）")
    parser.add_argument("--despeckle-min-points", type=int, default=3,
                        help="粗格内点数低于该值的点丢弃（0/1=关闭），用于裁远场孤立点")
    parser.add_argument("--out-prefix", required=True, help="输出前缀，生成 <prefix>.pgm 与 <prefix>.yaml")
    args = parser.parse_args()

    xyz = read_pcd_xyz(args.pcd)
    total = len(xyz)
    report = {"pcd": str(args.pcd), "points_in": total}

    if args.max_abs_z > 0:
        keep = np.abs(xyz[:, 2]) <= args.max_abs_z
        report["dropped_abs_z"] = int((~keep).sum())
        xyz = xyz[keep]

    if args.trajectory:
        keep = trajectory_gate_mask(xyz, read_tum_xy(args.trajectory), args.max_traj_dist)
        report["dropped_traj_gate"] = int((~keep).sum())
        xyz = xyz[keep]

    if args.despeckle_min_points > 1:
        keep = despeckle_mask(xyz, args.despeckle_cell, args.despeckle_min_points)
        report["dropped_despeckle"] = int((~keep).sum())
        xyz = xyz[keep]

    if len(xyz) == 0:
        raise SystemExit("no points left after filtering")

    ground = -args.ground_offset if not args.local_ground else local_ground_height(xyz, args.local_ground_cell)

    occ, x0, y0 = project_to_grid(xyz, args.resolution, args.height_min, args.height_max, ground)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_pgm(occ, prefix.with_suffix(".pgm"))
    write_yaml(prefix.with_suffix(".yaml"), prefix.name + ".pgm", args.resolution, (x0, y0))

    occupied = int(occ.sum())
    report.update(
        {
            "points_used": int(len(xyz)),
            "grid": {"width": int(occ.shape[1]), "height": int(occ.shape[0]), "resolution": args.resolution},
            "origin": [round(x0, 4), round(y0, 4)],
            "occupied_cells": occupied,
            "occupied_area_m2": round(occupied * args.resolution ** 2, 2),
            "occupied_ratio": round(occupied / occ.size, 4),
            "ground_mode": "local" if args.local_ground else f"global(-{args.ground_offset})",
            "height_window": [args.height_min, args.height_max],
            "outputs": [str(prefix.with_suffix(".pgm")), str(prefix.with_suffix(".yaml"))],
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
