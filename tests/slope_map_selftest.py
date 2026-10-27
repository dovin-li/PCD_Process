#!/usr/bin/env python3
"""Deterministic synthetic checks for slope_map_projector's core logic."""

import contextlib
import io
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from rosbag_odometry_to_tum import extract_rows

from slope_map_projector import (
    make_segment_references,
    process,
    read_binary_pcd,
    segment_trajectory,
    smooth_trajectory_z,
    classify_points,
    write_pgm_yaml,
)


def references(positions, smooth_window=0.5, segment_length=1.0, z_span=0.10):
    distance, smooth_z = smooth_trajectory_z(positions, smooth_window)
    segment_ids = segment_trajectory(distance, smooth_z, segment_length, z_span)
    segment_xy, segment_z, counts = make_segment_references(positions, smooth_z, segment_ids)
    return segment_ids, segment_xy, segment_z, counts


def assert_mask(name, points, trajectory, expected, **kwargs):
    _, segment_xy, segment_z, _ = references(trajectory)
    keep, _ = classify_points(
        points,
        segment_xy,
        segment_z,
        cKDTree(segment_xy),
        ground_offset_m=kwargs.get("ground_offset_m", 0.5),
        height_low_m=kwargs.get("height_low_m", 0.15),
        height_high_m=kwargs.get("height_high_m", 1.80),
        neighbor_count=kwargs.get("neighbor_count", 16),
        candidate_distance_tolerance_m=kwargs.get("candidate_distance_tolerance_m", 0.10),
    )
    assert keep.tolist() == expected, "%s: expected %s, got %s" % (name, expected, keep.tolist())


def run():
    # Exercise ROS bag extraction without requiring a ROS installation on the
    # offline workstation; the fake messages mirror nav_msgs/Odometry fields.
    messages = []
    for stamp, x_pose, quaternion in (
        (1.0, 0.0, (3.0, 0.0, 0.0, 4.0)),
        (1.05, 0.1, (0.0, 0.0, 0.0, 0.5)),
    ):
        position = SimpleNamespace(x=x_pose, y=2.0, z=0.5)
        orientation = SimpleNamespace(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        pose = SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation))
        header = SimpleNamespace(frame_id="camera_init", stamp=SimpleNamespace(to_sec=lambda value=stamp: value))
        messages.append(("/Odometry", SimpleNamespace(header=header, child_frame_id="body", pose=pose), None))

    class FakeBag:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read_messages(self, topics):
            return (item for item in messages if item[0] in topics)

    fake_rosbag = SimpleNamespace(Bag=FakeBag)
    with patch.dict(sys.modules, {"rosbag": fake_rosbag}):
        rows, frame_ids, child_frame_ids = extract_rows("fixture.bag", "/Odometry", "camera_init", "body")
    assert np.allclose(rows[0], (1.0, 0.0, 2.0, 0.5, 0.6, 0.0, 0.0, 0.8))
    assert np.allclose(rows[1], (1.05, 0.1, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    assert np.isclose(np.linalg.norm(rows[0][4:8]), 1.0)
    assert np.isclose(np.linalg.norm(rows[1][4:8]), 1.0)
    assert frame_ids == ["camera_init"] and child_frame_ids == ["body"]
    print("PASS ROS1 Odometry-to-TUM field mapping")

    # Binary PCD parser must honor field sizes and offsets, including fields
    # beyond XYZ that FAST-LIO writes into its accumulated cloud.
    with tempfile.TemporaryDirectory(prefix="slope-map-pcd-selftest-") as temp_dir:
        pcd = Path(temp_dir) / "tiny.pcd"
        header = (
            "VERSION .7\nFIELDS x y z intensity curvature\nSIZE 4 4 4 4 4\n"
            "TYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\n"
            "DATA binary\n"
        ).encode("ascii")
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "<f4"), ("curvature", "<f4"),
        ])
        records = np.asarray([(1.0, 2.0, 3.0, 4.0, 5.0), (-1.0, -2.0, -3.0, 6.0, 7.0)], dtype=dtype)
        pcd.write_bytes(header + records.tobytes())
        cloud, count, point_step = read_binary_pcd(pcd)
        assert count == 2 and point_step == dtype.itemsize
        assert np.allclose(np.column_stack((cloud["x"], cloud["y"], cloud["z"])),
                           [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    print("PASS binary PCD fields and point stride")

    # Exercise the complete PCD + TUM -> PGM/YAML/metrics path with a tiny
    # world-frame fixture: floor and overhead are removed; wall and obstacle
    # remain occupied.
    with tempfile.TemporaryDirectory(prefix="slope-map-e2e-selftest-") as temp_dir:
        root = Path(temp_dir)
        pcd = root / "cloud.pcd"
        tum = root / "trajectory.tum"
        out_prefix = root / "map"
        header = (
            "VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
            "COUNT 1 1 1 1\nWIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n"
        ).encode("ascii")
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        cloud = np.asarray([
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 0.5, 0.8, 1.0),
            (1.0, 1.0, 0.2, 1.0),
            (1.0, 1.5, 2.4, 1.0),
        ], dtype=dtype)
        pcd.write_bytes(header + cloud.tobytes())
        with tum.open("w", encoding="ascii") as stream:
            for index, x_pose in enumerate(np.linspace(0.0, 2.0, 41)):
                stream.write("%.3f %.6f 0.0 0.5 0 0 0 1\n" % (index * 0.05, x_pose))
        args = Namespace(
            pcd=str(pcd), trajectory=str(tum), out_prefix=str(out_prefix),
            ground_offset_m=0.5, resolution=0.05, height_low_m=0.15,
            height_high_m=1.80, smooth_window_m=0.5, segment_length_m=0.5,
            segment_z_span_m=0.1, neighbor_count=16,
            candidate_distance_tolerance_m=0.10, max_route_distance_m=8.0,
            occupancy_radius_m=0.0, margin_m=0.1,
            chunk_points=2, max_grid_cells=10000, overwrite=False,
            auto_ground_offset=False,
            raycast_free=True, raycast_skip_near_m=0.20,
            raycast_min_support=3, raycast_support_radius_m=0.30,
            raycast_source="ground",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = process(args)
        assert metrics["projected_points"] == 2 and metrics["occupied_cells"] == 2
        assert (root / "map.pgm").exists() and (root / "map.yaml").exists()
        written_metrics = json.loads((root / "map_metrics.json").read_text(encoding="utf-8"))
        assert written_metrics["projected_points"] == 2
    print("PASS end-to-end binary PCD + TUM projection")

    # Flat floor is dropped; wall and a low obstacle remain; overhead is out.
    x = np.linspace(0.0, 10.0, 201)
    flat = np.column_stack((x, np.zeros_like(x), np.full_like(x, 0.70)))
    flat_points = np.asarray([
        [2.0, 0.0, 0.20],  # local floor
        [2.0, 1.0, 0.80],  # wall, +0.60 m
        [3.0, 1.0, 0.40],  # obstacle, +0.20 m above local floor
        [4.0, 1.0, 2.10],  # overhead, +1.90 m
    ])
    assert_mask("flat floor/wall/height window", flat_points, flat, [False, True, True, False])
    print("PASS flat floor/wall/height window")

    # A 12-degree ramp must follow the local trajectory Z instead of becoming
    # an occupied strip. Noise in the path is removed by distance-based median.
    ramp_x = np.linspace(0.0, 12.0, 481)
    slope = np.tan(np.deg2rad(12.0))
    ramp_ground = 0.20 + slope * ramp_x
    rng = np.random.default_rng(42)
    ramp = np.column_stack((ramp_x, np.zeros_like(ramp_x), ramp_ground + 0.50 + rng.normal(0.0, 0.015, len(ramp_x))))
    ramp_points = np.column_stack((ramp_x[::12], np.zeros_like(ramp_x[::12]), ramp_ground[::12]))
    segment_ids, segment_xy, segment_z, _ = references(ramp)
    ground_keep, _ = classify_points(
        ramp_points, segment_xy, segment_z, cKDTree(segment_xy), 0.50, 0.15, 1.80, 16, 0.10
    )
    assert not ground_keep.any(), "ramp floor must not be projected as a wall"
    ramp_wall = np.column_stack((ramp_x[::12], np.full_like(ramp_x[::12], 0.8), ramp_ground[::12] + 0.60))
    wall_keep, _ = classify_points(
        ramp_wall, segment_xy, segment_z, cKDTree(segment_xy), 0.50, 0.15, 1.80, 16, 0.10
    )
    assert wall_keep.all(), "walls beside the ramp must remain occupied"
    assert int(segment_ids[-1]) >= 5, "rising trajectory should split into local-Z segments"
    print("PASS ramp follows trajectory; adjacent wall remains")

    steep_angle = 22.0
    steep_x = np.linspace(0.0, 8.0, 321)
    steep_ground = 0.10 + np.tan(np.deg2rad(steep_angle)) * steep_x
    steep_path = np.column_stack((steep_x, np.zeros_like(steep_x), steep_ground + 0.50))
    _, steep_xy, steep_z, _ = references(steep_path)
    steep_floor = np.column_stack((steep_x[::8], np.zeros_like(steep_x[::8]), steep_ground[::8]))
    steep_keep, _ = classify_points(
        steep_floor, steep_xy, steep_z, cKDTree(steep_xy), 0.50, 0.15, 1.80, 16, 0.10
    )
    assert not steep_keep.any(), "22-degree ramp floor must not become an occupied strip"
    print("PASS steep-ramp ground rejection")

    # Two traversals with overlapping XY but different elevations must choose
    # the height-compatible local segment rather than the first/lowest route.
    lower_x = np.linspace(0.0, 4.0, 81)
    upper_x = lower_x[::-1]
    overlap = np.vstack((
        np.column_stack((lower_x, np.zeros_like(lower_x), np.full_like(lower_x, 0.70))),
        np.column_stack((upper_x, np.zeros_like(upper_x), np.full_like(upper_x, 3.70))),
    ))
    overlap_points = np.asarray([
        [2.0, 0.0, 0.20],  # lower floor
        [2.0, 0.5, 0.80],  # lower wall
        [2.0, 0.0, 3.20],  # upper floor
        [2.0, 0.5, 3.80],  # upper wall
    ])
    assert_mask("overlapping levels select local height", overlap_points, overlap,
                [False, True, False, True], segment_length=1.0)
    print("PASS overlapping levels select local height")

    # PGM must preserve the existing image orientation and value convention:
    # world lower-left occupied -> bottom-left black pixel, free background 205.
    occupancy = np.asarray([[True, False], [False, True]], dtype=bool)
    with tempfile.TemporaryDirectory(prefix="slope-map-selftest-") as temp_dir:
        pgm = Path(temp_dir) / "map.pgm"
        yaml_path = write_pgm_yaml(occupancy, pgm, 0.05, -1.0, -2.0)
        data = pgm.read_bytes()
        header, pixels = data.split(b"255\n", 1)
        assert header.startswith(b"P5\n2 2\n")
        assert np.frombuffer(pixels, dtype=np.uint8).tolist() == [205, 0, 0, 205]
        text = yaml_path.read_text(encoding="utf-8")
        assert "resolution: 0.050000" in text and "origin: [-1.000000, -2.000000, 0.0]" in text
    print("PASS PGM/YAML origin, orientation, and gray-value contract")
    print("ALL 8 synthetic checks passed")


if __name__ == "__main__":
    run()
