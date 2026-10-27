#!/usr/bin/env python3
"""Project a world-frame FAST-LIO PCD to a 2-D map using trajectory-local Z.

The PCD and TUM trajectory must describe the same mapping run and the same
world frame. ``--ground-offset-m`` is the vertical distance from the trajectory
pose origin (normally FAST-LIO's body/IMU origin) down to the local floor.

The trajectory is smoothed and split into short contiguous segments. Each
segment contributes a local floor-height reference; nearby XY trajectory
segments are considered for each PCD point so ramps follow the route instead
of being classified against one global Z band. Segment hits share one global
XY grid and are unioned into the existing PGM/YAML occupancy convention.
Points also pass an absolute trajectory-distance gate, so far-field PCD
outliers cannot expand the map or become spurious obstacles. Accepted voxel
centers are rasterized with their configured physical footprint so walls stay
continuous at finer map resolution.

This is an offline projection utility. It does not alter FAST-LIO, navigation,
or the 3-D source PCD. A single 2-D projection cannot distinguish two
overlapping traversable surfaces; those must be separated by the upstream map
segmentation stage.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


PCD_TYPE_MAP = {
    ("F", 2): np.dtype("<f2"),
    ("F", 4): np.dtype("<f4"),
    ("F", 8): np.dtype("<f8"),
    ("I", 1): np.dtype("i1"),
    ("I", 2): np.dtype("<i2"),
    ("I", 4): np.dtype("<i4"),
    ("I", 8): np.dtype("<i8"),
    ("U", 1): np.dtype("u1"),
    ("U", 2): np.dtype("<u2"),
    ("U", 4): np.dtype("<u4"),
    ("U", 8): np.dtype("<u8"),
}


def read_binary_pcd(path):
    """Return a structured memmap and field names for an uncompressed PCD."""
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header ended before DATA")
            decoded = line.decode("ascii", errors="strict").strip()
            if not decoded or decoded.startswith("#"):
                continue
            pieces = decoded.split()
            header[pieces[0].upper()] = pieces[1:]
            if pieces[0].upper() == "DATA":
                data_offset = stream.tell()
                break

    if header["DATA"] != ["binary"]:
        raise ValueError("only uncompressed PCD DATA binary is supported")
    names = header.get("FIELDS") or header.get("FIELD")
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(names))]
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths differ")
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PCD must contain x, y, and z fields")

    formats = []
    offsets = []
    offset = 0
    for name, size, type_code, count in zip(names, sizes, types, counts):
        base_dtype = PCD_TYPE_MAP.get((type_code.upper(), size))
        if base_dtype is None:
            raise ValueError("unsupported PCD field type: %s %s %s" % (name, type_code, size))
        if count < 1:
            raise ValueError("PCD field COUNT must be positive")
        formats.append(base_dtype if count == 1 else (base_dtype, (count,)))
        offsets.append(offset)
        offset += size * count

    if "POINTS" in header:
        point_count = int(header["POINTS"][0])
    else:
        point_count = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    expected_bytes = point_count * offset
    available_bytes = path.stat().st_size - data_offset
    if available_bytes < expected_bytes:
        raise ValueError("PCD data is truncated: expected %d bytes, found %d" % (expected_bytes, available_bytes))

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": offset})
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(point_count,)), point_count, offset


def read_tum_trajectory(path):
    """Read a TUM trajectory: timestamp x y z qx qy qz qw."""
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 8 or values.shape[0] < 2:
        raise ValueError("trajectory must contain at least two TUM rows with 8 columns")
    if not np.isfinite(values[:, :8]).all():
        raise ValueError("trajectory contains NaN or infinite values")
    if np.any(np.diff(values[:, 0]) < 0.0):
        raise ValueError("trajectory timestamps must be nondecreasing")
    positions = values[:, 1:4].astype(np.float64, copy=True)
    return values[:, 0], positions


def smooth_trajectory_z(positions, window_m):
    """Median-smooth Z over a distance window, not a fixed sample count."""
    positions = np.asarray(positions, dtype=np.float64)
    if window_m <= 0.0:
        raise ValueError("smooth window must be positive")
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    half_window = float(window_m) * 0.5
    result = np.empty(len(positions), dtype=np.float64)
    for index, center in enumerate(distance):
        left = int(np.searchsorted(distance, center - half_window, side="left"))
        right = int(np.searchsorted(distance, center + half_window, side="right"))
        result[index] = np.median(positions[left:max(left + 1, right), 2])
    return distance, result


def segment_trajectory(distance, smoothed_z, max_length_m, max_z_span_m):
    """Split the route when either local path length or Z span is exceeded."""
    if max_length_m <= 0.0 or max_z_span_m <= 0.0:
        raise ValueError("segment length and Z span must be positive")
    ids = np.zeros(len(distance), dtype=np.int32)
    start = 0
    segment_id = 0
    for index in range(1, len(distance)):
        z_span = float(np.ptp(smoothed_z[start:index + 1]))
        if distance[index] - distance[start] >= max_length_m or z_span >= max_z_span_m:
            segment_id += 1
            start = index
        ids[index] = segment_id
    return ids


def make_segment_references(positions, smoothed_z, segment_ids):
    """Reduce dense trajectory samples to stable XY/Z representatives."""
    centers = []
    heights = []
    lengths = []
    for segment_id in range(int(segment_ids[-1]) + 1):
        indices = np.flatnonzero(segment_ids == segment_id)
        if not len(indices):
            continue
        centers.append(np.median(positions[indices, :2], axis=0))
        heights.append(float(np.median(smoothed_z[indices])))
        lengths.append(int(len(indices)))
    if not centers:
        raise ValueError("trajectory segmentation produced no segments")
    return np.asarray(centers), np.asarray(heights), np.asarray(lengths)


def classify_points(points, segment_xy, segment_z, tree, ground_offset_m,
                    height_low_m, height_high_m, neighbor_count,
                    candidate_distance_tolerance_m=0.10,
                    max_route_distance_m=None):
    """Return occupied candidates and their selected local trajectory segment."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
    k = min(int(neighbor_count), len(segment_z))
    distances, indices = tree.query(points[:, :2], k=k, workers=1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    relative_height = points[:, 2, None] - (segment_z[indices] - ground_offset_m)
    # Only compare a point against trajectory segments that are spatially
    # local to its nearest route reference. Without this gate, a distant
    # lower ramp segment can make the uphill floor look like a raised object.
    local = distances <= (distances[:, :1] + candidate_distance_tolerance_m)
    valid = local & (relative_height >= height_low_m) & (relative_height <= height_high_m)
    if max_route_distance_m is not None:
        valid &= distances <= float(max_route_distance_m)
    ranked_distance = np.where(valid, distances, np.inf)
    selected_column = np.argmin(ranked_distance, axis=1)
    keep = np.isfinite(ranked_distance[np.arange(len(points)), selected_column])
    selected_segment = indices[np.arange(len(points)), selected_column].astype(np.int32)
    return keep, selected_segment


def pcd_xyz_chunk(cloud, start, end):
    return np.column_stack((cloud["x"][start:end], cloud["y"][start:end], cloud["z"][start:end])).astype(np.float64, copy=False)


def compute_bounds(cloud, point_count, chunk_points, resolution, margin_m):
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    finite_points = 0
    for start in range(0, point_count, chunk_points):
        xyz = pcd_xyz_chunk(cloud, start, min(start + chunk_points, point_count))
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.any():
            continue
        block = xyz[finite]
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        finite_points += len(block)
    if finite_points == 0:
        raise ValueError("PCD has no finite XYZ points")
    xmin = math.floor((minimum[0] - margin_m) / resolution) * resolution
    ymin = math.floor((minimum[1] - margin_m) / resolution) * resolution
    xmax = math.ceil((maximum[0] + margin_m) / resolution) * resolution
    ymax = math.ceil((maximum[1] + margin_m) / resolution) * resolution
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1
    return xmin, ymin, width, height, finite_points, minimum.tolist(), maximum.tolist()


def compute_hit_bounds(points, resolution, margin_m):
    """Build the map extent from accepted projection points, not raw outliers."""
    if len(points) == 0:
        raise ValueError("height-filtered projection produced no points")
    minimum = np.min(points[:, :2], axis=0)
    maximum = np.max(points[:, :2], axis=0)
    xmin = math.floor((minimum[0] - margin_m) / resolution) * resolution
    ymin = math.floor((minimum[1] - margin_m) / resolution) * resolution
    xmax = math.ceil((maximum[0] + margin_m) / resolution) * resolution
    ymax = math.ceil((maximum[1] + margin_m) / resolution) * resolution
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1
    return xmin, ymin, width, height


def rasterize_hits(points, resolution, xmin, ymin, width, height,
                   occupancy_radius_m):
    """Rasterize accepted voxel centers and cover their physical footprint."""
    gx = np.floor((points[:, 0] - xmin) / resolution).astype(np.int64)
    gy = np.floor((points[:, 1] - ymin) / resolution).astype(np.int64)
    inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    occupied = np.zeros((height, width), dtype=bool)
    occupied[gy[inside], gx[inside]] = True
    radius_cells = int(math.ceil(max(0.0, occupancy_radius_m) / resolution))
    if radius_cells:
        structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        occupied = ndimage.binary_dilation(occupied, structure=structure)
        occupied = ndimage.binary_closing(occupied, structure=structure)
    return occupied, int(np.count_nonzero(inside))


def raycast_free_cells(points, segment_ids, segment_xy, segment_z, ground_offset_m,
                       xmin, ymin, width, height, resolution, occupied,
                       skip_near_m=0.20, max_range_m=None, block=20000):
    """从观测点对应的轨迹位姿向该点连线，连线上经过的格子标为可通行。

    近似点：pcd 没有逐点时间戳，这里用"该点被哪个轨迹分段观测到"的段位姿
    作为射线起点（classify_points 已经算好对应段）。机器人在连续移动时，
    段中心与实际帧位姿最多差半段（默认 0.5 m），足够做自由空间填充。

    射线遇到已占用格就停止（标准 ray casting 语义），避免"点位与段位姿跨墙"
    时长线穿墙、在地图外画出白色条纹。
    """
    free = np.zeros((height, width), dtype=bool)
    if len(points) == 0:
        return free
    flat = free.reshape(-1)
    occ_flat = occupied.reshape(-1)
    for start in range(0, len(points), block):
        stop = min(start + block, len(points))
        pts = points[start:stop]
        seg = segment_ids[start:stop]
        ox = segment_xy[seg, 0]
        oy = segment_xy[seg, 1]
        dx = pts[:, 0] - ox
        dy = pts[:, 1] - oy
        planar = np.hypot(dx, dy)
        usable = planar > skip_near_m
        if max_range_m is not None:
            usable &= planar <= float(max_range_m)
        if not usable.any():
            continue
        ox, oy, dx, dy, planar = ox[usable], oy[usable], dx[usable], dy[usable], planar[usable]
        steps = np.maximum(1, np.ceil(planar / resolution).astype(np.int64))
        total = int(steps.sum())
        ray_id = np.repeat(np.arange(len(planar), dtype=np.int64), steps)
        starts = np.repeat(np.cumsum(steps) - steps, steps)
        frac = (np.arange(total, dtype=np.float64) - starts + 0.5) / steps[ray_id]
        sx = (ox[ray_id] + frac * dx[ray_id]).astype(np.float32)
        sy = (oy[ray_id] + frac * dy[ray_id]).astype(np.float32)
        gx = np.floor((sx - xmin) / resolution).astype(np.int32)
        gy = np.floor((sy - ymin) / resolution).astype(np.int32)
        valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        valid &= frac * planar[ray_id] > skip_near_m
        if not valid.any():
            continue
        rid = ray_id[valid]
        lin = gy[valid].astype(np.int64) * width + gx[valid]
        kept = np.ones(len(lin), dtype=bool)
        hits = occ_flat[lin]
        if hits.any():
            hit_positions = np.flatnonzero(hits)
            first_ray, first_occurrence = np.unique(rid[hit_positions], return_index=True)
            first_stop_index = hit_positions[first_occurrence]
            limit = np.full(int(rid.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
            limit[first_ray] = first_stop_index
            kept = np.arange(len(lin), dtype=np.int64) < limit[rid]
        flat[np.unique(lin[kept])] = True
    return free


def write_pgm_yaml(occupied, pgm_path, resolution, origin_x, origin_y, free=None):
    pgm_path = Path(pgm_path)
    yaml_path = pgm_path.with_suffix(".yaml")
    image = np.full(occupied.shape, 205, dtype=np.uint8)
    if free is not None:
        image[free] = 254
    image[occupied] = 0
    image = np.flipud(image)
    with pgm_path.open("wb") as stream:
        stream.write(("P5\n%d %d\n255\n" % (image.shape[1], image.shape[0])).encode("ascii"))
        stream.write(image.tobytes())
    with yaml_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "image: %s\nresolution: %.6f\norigin: [%.6f, %.6f, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
            % (pgm_path.name, resolution, origin_x, origin_y)
        )
    return yaml_path


def process(args):
    started = time.monotonic()
    prefix = Path(args.out_prefix)
    pgm_path = Path(str(prefix) + ".pgm")
    yaml_path = Path(str(prefix) + ".yaml")
    metrics_path = Path(str(prefix) + "_metrics.json")
    outputs = (pgm_path, yaml_path, metrics_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("refusing to overwrite existing output(s): " + ", ".join(existing) + " (pass --overwrite to replace)")

    cloud, point_count, point_step = read_binary_pcd(args.pcd)
    trajectory_times, positions = read_tum_trajectory(args.trajectory)
    # Auto ground offset: sensor origin closest to the floor. Without this
    # the height band is shifted and everything lands wrong.
    _, smoothed_z_for_offset = smooth_trajectory_z(positions, args.smooth_window_m)
    auto_offset = float(np.min(smoothed_z_for_offset))
    if args.auto_ground_offset or args.ground_offset_m is None:
        args.ground_offset_m = max(0.0, auto_offset)
        print("ground-offset-m = %.3f (auto from trajectory min smoothed z)"
              % args.ground_offset_m)
    distance, smoothed_z = smooth_trajectory_z(positions, args.smooth_window_m)
    segment_ids = segment_trajectory(
        distance, smoothed_z, args.segment_length_m, args.segment_z_span_m
    )
    segment_xy, segment_z, segment_sample_counts = make_segment_references(
        positions, smoothed_z, segment_ids
    )
    if len(segment_xy) < 1:
        raise ValueError("trajectory has no usable segment references")
    tree = cKDTree(segment_xy)
    finite_points = 0
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    hit_blocks = []
    hit_seg_blocks = []
    ground_blocks = []
    ground_seg_blocks = []
    segment_hits = np.zeros(len(segment_xy), dtype=np.int64)
    floor_lo = -0.30  # below local floor: noise margin for ground band
    for start in range(0, point_count, args.chunk_points):
        end = min(start + args.chunk_points, point_count)
        xyz = pcd_xyz_chunk(cloud, start, end)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.any():
            continue
        xyz = xyz[finite]
        minimum = np.minimum(minimum, np.min(xyz, axis=0))
        maximum = np.maximum(maximum, np.max(xyz, axis=0))
        finite_points += len(xyz)
        keep, selected_segment = classify_points(
            xyz,
            segment_xy,
            segment_z,
            tree,
            args.ground_offset_m,
            args.height_low_m,
            args.height_high_m,
            args.neighbor_count,
            args.candidate_distance_tolerance_m,
            args.max_route_distance_m,
        )
        # Ground-band points (below height_low, above floor margin) near
        # the route are walkable floor -> mark FREE so the output map
        # distinguishes traversable area from unknown.
        k = min(int(args.neighbor_count), len(segment_z))
        g_dist, g_idx = tree.query(xyz[:, :2], k=k, workers=1)
        if k == 1:
            g_dist = g_dist[:, None]; g_idx = g_idx[:, None]
        g_rel = xyz[:, 2, None] - (segment_z[g_idx] - args.ground_offset_m)
        g_local = g_dist <= (g_dist[:, :1] + args.candidate_distance_tolerance_m)
        g_band = g_local & (g_rel >= floor_lo) & (g_rel < args.height_low_m)
        g_any = g_band.any(axis=1)
        ground_blocks.append(xyz[g_any])
        ground_seg_blocks.append(g_idx[g_any, 0])
        if not keep.any():
            continue
        hit_blocks.append(xyz[keep])
        hit_seg_blocks.append(selected_segment[keep])
        segment_hits += np.bincount(selected_segment[keep], minlength=len(segment_hits))
        if end == point_count or end % (args.chunk_points * 10) == 0:
            print("classified %d/%d points; occupied=%d ground=%d" %
                  (end, point_count,
                   sum(len(b) for b in hit_blocks),
                   sum(len(b) for b in ground_blocks)), flush=True)

    if not hit_blocks:
        raise ValueError("height and route-distance filters accepted no points")
    hits = np.concatenate(hit_blocks, axis=0)
    # Free space: union of ground-band points and hit points (for bounds)
    ground_pts = (np.concatenate(ground_blocks, axis=0)
                  if ground_blocks else np.zeros((0, 3)))
    ground_segs = (np.concatenate(ground_seg_blocks)
                   if ground_seg_blocks else np.zeros(0, dtype=np.int32))
    hit_segs = (np.concatenate(hit_seg_blocks)
                if hit_seg_blocks else np.zeros(0, dtype=np.int32))
    # Bounds must cover both hits and ground observations
    all_proj = np.vstack([hits, ground_pts]) if len(ground_pts) else hits
    xmin, ymin, width, height = compute_hit_bounds(
        all_proj, args.resolution, args.margin_m
    )
    if width * height > args.max_grid_cells:
        raise ValueError("grid would contain %d cells (limit %d)" % (width * height, args.max_grid_cells))
    occupied, points_used = rasterize_hits(
        hits, args.resolution, xmin, ymin, width, height,
        args.occupancy_radius_m,
    )
    # Free: ground-band cells + (可选) 射线穿透格子
    ground_free = np.zeros((height, width), dtype=bool)
    if len(ground_pts):
        gx = np.floor((ground_pts[:, 0] - xmin) / args.resolution).astype(np.int64)
        gy = np.floor((ground_pts[:, 1] - ymin) / args.resolution).astype(np.int64)
        inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        ground_free[gy[inside], gx[inside]] = True

    ray_free = np.zeros((height, width), dtype=bool)
    if args.raycast_free:
        if args.raycast_source == "ground" and len(ground_pts):
            ray_pts, ray_segs = ground_pts, ground_segs
        else:
            ray_pts = np.vstack([hits, ground_pts]) if len(ground_pts) else hits
            ray_segs = np.concatenate([hit_segs, ground_segs]) if len(ground_segs) else hit_segs
        # 孤立点（周围没有足够邻居）不参与射线：否则每个离群点都会从位姿拉出一条
        # 穿过整张图的白色细线（现场看到的"扇形拖尾"）。
        if len(ray_pts) >= 2 and args.raycast_min_support > 0:
            k = min(len(ray_pts), args.raycast_min_support + 1)
            support_tree = cKDTree(ray_pts[:, :2])
            dists, _ = support_tree.query(ray_pts[:, :2], k=k, workers=1)
            if dists.ndim == 1:
                dists = dists[:, None]
            near = (dists[:, 1:] <= args.raycast_support_radius_m).sum(axis=1)
            supported = near >= args.raycast_min_support
            ray_pts = ray_pts[supported]
            ray_segs = ray_segs[supported]
        ray_free = raycast_free_cells(
            ray_pts, ray_segs, segment_xy, segment_z, args.ground_offset_m,
            xmin, ymin, width, height, args.resolution, occupied,
            skip_near_m=args.raycast_skip_near_m,
            max_range_m=args.max_route_distance_m,
        )

    free = (ground_free | ray_free) & ~occupied
    if args.raycast_free:
        # 射线是逐条 1 格宽的线，相邻射线之间会留下零星空洞；
        # 闭运算补洞，再用膨胀后的障碍把墙面重新切开，避免白色穿墙。
        structure = np.ones((3, 3), dtype=bool)
        free = ndimage.binary_closing(free, structure=structure)
        free &= ~ndimage.binary_dilation(occupied, structure=structure)
    min_xyz = minimum.tolist()
    max_xyz = maximum.tolist()

    pgm_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path = write_pgm_yaml(occupied, pgm_path, args.resolution, xmin, ymin, free=free)
    segment_records = []
    for index, (z, samples, hits) in enumerate(zip(segment_z, segment_sample_counts, segment_hits)):
        segment_records.append({
            "segment_id": index,
            "representative_z_m": float(z),
            "trajectory_samples": int(samples),
            "projected_points": int(hits),
        })
    metrics = {
        "pcd": str(Path(args.pcd).resolve()),
        "trajectory": str(Path(args.trajectory).resolve()),
        "trajectory_frame": "must match PCD world frame",
        "trajectory_rows": int(len(trajectory_times)),
        "pcd_points": int(point_count),
        "pcd_point_step_bytes": int(point_step),
        "finite_points": int(finite_points),
        "projected_points": int(points_used),
        "resolution_m": float(args.resolution),
        "height_window_m": [float(args.height_low_m), float(args.height_high_m)],
        "ground_offset_m": float(args.ground_offset_m),
        "trajectory_smoothing_window_m": float(args.smooth_window_m),
        "segment_max_length_m": float(args.segment_length_m),
        "segment_max_z_span_m": float(args.segment_z_span_m),
        "candidate_distance_tolerance_m": float(args.candidate_distance_tolerance_m),
        "max_route_distance_m": float(args.max_route_distance_m),
        "occupancy_radius_m": float(args.occupancy_radius_m),
        "trajectory_segments": segment_records,
        "grid_width": int(width),
        "grid_height": int(height),
        "origin": [float(xmin), float(ymin), 0.0],
        "point_bounds_min_xyz": min_xyz,
        "point_bounds_max_xyz": max_xyz,
        "occupied_cells": int(occupied.sum()),
        "free_cells": int(free.sum()),
        "free_cells_ground": int((ground_free & ~occupied).sum()),
        "free_cells_raycast": int((ray_free & ~occupied).sum()),
        "unknown_cells": int((~occupied & ~free).sum()),
        "raycast_free_enabled": bool(args.raycast_free),
        "pgm": str(pgm_path.resolve()),
        "yaml": str(yaml_path.resolve()),
        "elapsed_s": float(time.monotonic() - started),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", required=True, help="uncompressed world-frame binary PCD")
    parser.add_argument("--trajectory", required=True, help="matching TUM trajectory file")
    parser.add_argument("--out-prefix", required=True, help="output path prefix; creates .pgm/.yaml/_metrics.json")
    parser.add_argument("--ground-offset-m", type=float, default=None,
                        help="pose-origin height above local floor; omit or pass --auto-ground-offset to auto-estimate")
    parser.add_argument("--auto-ground-offset", action="store_true",
                        help="auto-estimate ground offset from trajectory min z")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--height-low-m", type=float, default=0.30)
    parser.add_argument("--height-high-m", type=float, default=1.00)
    parser.add_argument("--smooth-window-m", type=float, default=0.50)
    parser.add_argument("--segment-length-m", type=float, default=1.00)
    parser.add_argument("--segment-z-span-m", type=float, default=0.10)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--candidate-distance-tolerance-m", type=float, default=0.10,
                        help="consider near-tied route segments only within this XY distance of the closest segment")
    parser.add_argument("--max-route-distance-m", type=float, default=8.0,
                        help="reject points farther than this from the nearest trajectory segment")
    parser.add_argument("--occupancy-radius-m", type=float, default=0.05,
                        help="cover each accepted voxel by this radius when rasterizing")
    parser.add_argument("--margin-m", type=float, default=1.0)
    parser.add_argument("--raycast-free", dest="raycast_free", action="store_true", default=True,
                        help="从轨迹位姿向每个观测点连线，把连线经过的格子标为可通行（默认开）")
    parser.add_argument("--no-raycast-free", dest="raycast_free", action="store_false",
                        help="关闭射线自由空间填充，只把地面带标为可通行")
    parser.add_argument("--raycast-skip-near-m", type=float, default=0.20,
                        help="射线起点附近这段距离不标记（避开机器人自身）")
    parser.add_argument("--raycast-source", choices=("ground", "all"), default="ground",
                        help="射线来源：ground=只用地带点（干净），all=障碍点也投射（覆盖广但有拖尾）")
    parser.add_argument("--raycast-min-support", type=int, default=3,
                        help="参与射线的点至少要有这么多邻居（0=关闭支撑过滤）")
    parser.add_argument("--raycast-support-radius-m", type=float, default=0.30,
                        help="支撑过滤的邻居半径")
    parser.add_argument("--chunk-points", type=int, default=500_000)
    parser.add_argument("--max-grid-cells", type=int, default=100_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    finite_parameters = (
        args.resolution, args.height_low_m, args.height_high_m,
        args.smooth_window_m, args.segment_length_m, args.segment_z_span_m,
        args.candidate_distance_tolerance_m, args.max_route_distance_m,
        args.occupancy_radius_m, args.margin_m, args.raycast_skip_near_m,
        args.raycast_support_radius_m,
    )
    if not all(math.isfinite(value) for value in finite_parameters):
        raise SystemExit("all distance, height, and resolution parameters must be finite")
    if args.resolution <= 0.0 or args.height_low_m < 0.0 or args.height_high_m <= args.height_low_m:
        raise SystemExit("require resolution>0 and 0<=height-low<height-high")
    if args.ground_offset_m is not None and args.ground_offset_m < 0.0:
        raise SystemExit("ground-offset-m must be nonnegative")
    if (args.neighbor_count < 1 or args.chunk_points < 1 or args.margin_m < 0.0
            or args.candidate_distance_tolerance_m < 0.0
            or args.max_route_distance_m <= 0.0 or args.occupancy_radius_m < 0.0):
        raise SystemExit("neighbor-count/chunk-points/max-route-distance must be positive; margin/tolerance/radius must be nonnegative")
    if args.smooth_window_m <= 0.0 or args.segment_length_m <= 0.0 or args.segment_z_span_m <= 0.0:
        raise SystemExit("smooth-window, segment-length, and segment-z-span must be positive")
    if args.max_grid_cells < 1:
        raise SystemExit("max-grid-cells must be positive")
    process(args)


if __name__ == "__main__":
    main()
