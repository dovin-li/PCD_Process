#!/usr/bin/env python3
"""Export FAST-LIO nav_msgs/Odometry messages from a ROS1 bag as TUM poses.

The output keeps the Odometry header timestamp and position unchanged and
normalizes its orientation quaternion. By default, the expected frame contract
is camera_init -> body, matching the TRON2 FAST-LIO mapping setup. This script
only reads the bag; it does not start ROS nodes.
"""

import argparse
import math
from pathlib import Path


def extract_rows(bag_path, topic, expected_frame, expected_child_frame):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS1 rosbag Python modules are required to read this bag") from exc

    rows = []
    frame_ids = set()
    child_frame_ids = set()
    previous_stamp = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, message, _ in bag.read_messages(topics=[topic]):
            frame_id = message.header.frame_id
            child_frame_id = message.child_frame_id
            frame_ids.add(frame_id)
            child_frame_ids.add(child_frame_id)
            if expected_frame and frame_id != expected_frame:
                raise ValueError("unexpected header.frame_id %r (expected %r)" % (frame_id, expected_frame))
            if expected_child_frame and child_frame_id != expected_child_frame:
                raise ValueError("unexpected child_frame_id %r (expected %r)" % (child_frame_id, expected_child_frame))

            stamp = float(message.header.stamp.to_sec())
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            values = (
                stamp,
                float(position.x), float(position.y), float(position.z),
                float(orientation.x), float(orientation.y),
                float(orientation.z), float(orientation.w),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Odometry contains NaN or infinite pose values")
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError("Odometry timestamps must be strictly increasing")
            quaternion_norm = math.sqrt(sum(value * value for value in values[4:8]))
            if quaternion_norm < 1e-6:
                raise ValueError("Odometry contains a zero-length orientation quaternion")
            values = values[:4] + tuple(value / quaternion_norm for value in values[4:8])
            previous_stamp = stamp
            rows.append(values)

    if len(rows) < 2:
        raise ValueError("bag topic %s has fewer than two Odometry messages" % topic)
    return rows, sorted(frame_ids), sorted(child_frame_ids)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="ROS1 bag recorded during the same mapping run")
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument("--out", required=True, help="output TUM trajectory path")
    parser.add_argument("--expected-frame", default="camera_init")
    parser.add_argument("--expected-child-frame", default="body")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise SystemExit("refusing to overwrite %s (pass --overwrite to replace)" % output)

    rows, frame_ids, child_frame_ids = extract_rows(
        args.bag, args.topic, args.expected_frame, args.expected_child_frame
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw; frame_id=%s child_frame_id=%s\n" % (
            ",".join(frame_ids), ",".join(child_frame_ids)
        ))
        for row in rows:
            stream.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % row)
    print("wrote %d poses to %s (frames %s -> %s)" % (
        len(rows), output, ",".join(frame_ids), ",".join(child_frame_ids)
    ))


if __name__ == "__main__":
    main()
