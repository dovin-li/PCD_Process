#!/usr/bin/env python3
"""一条命令：ROS1 bag → 静态点云 PCD + 2D 占据栅格地图（PGM/YAML）。

参数已固化（2026-10-27 现场验收版），不接受逐次调参；需要修改请改本文件常量。

消费链：
    bag ──► bag_mapping_pipeline.py ──► cloud_voxel_all.pcd / cloud_static.pcd / traj.tum
          ──► 轨迹体检（发散即拒绝出图）
          ──► slope_map_projector.py ──► map.pgm / map.yaml

固化参数（与 scan_0 验收图一致）：
    体素 0.05 m；帧数阈值 自动（Otsu，钳制 3~8）；地面偏移 自动估计；
    障碍窗 0.10~1.00 m（相对当地地面）；栅格 0.05 m；轨迹分段 1.0 m / 高差 0.10 m；
    轨迹距离门限 8 m；占据栅格化半径 0.05 m。

用法：
    python3 tools/bag_to_map.py --bag run.bag --out out/run1
    python3 tools/bag_to_map.py --bag run.bag --out out/run1 --force   # 跳过体检

输出目录：
    cloud_voxel_all.pcd   全部 5 cm 体素（intensity = 观测帧数）
    cloud_static.pcd      过滤后点云（送进投影的就是它）
    traj_camera_init.tum  轨迹
    map.pgm / map.yaml    2D 地图（0=障碍 205=未知 254=可通行）
    metrics.json          自动估计值 + 判据
    report.md             本次运行的体检结论
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---- 固化参数（改动需同步 README_bag_to_map.md 并登记 BASELINE） ----
VOXEL_M = 0.05
HEIGHT_LOW_M = 0.10
HEIGHT_HIGH_M = 1.00
GRID_RESOLUTION_M = 0.05
MIN_FRAMES = 3          # 固定 3 帧：2 帧杂点太多、4 帧丢细节，2026-10-27 现场拍板
MAX_ROUTE_DISTANCE_M = 8.0

# ---- 轨迹体检阈值（超过即拒绝出图） ----
MAX_STEP_M = 1.0        # 相邻位姿位移上限；健康值 < 0.1
MAX_Z_SPAN_M = 5.0      # 全程高差上限；办公场景健康值 < 1.0
MIN_DURATION_S = 10.0

TOOLS = Path(__file__).resolve().parent


def run(cmd, echo=True):
    if echo:
        print("+ " + " ".join(str(part) for part in cmd), flush=True)
    result = subprocess.run([str(part) for part in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed: {cmd[0]}")
    return result.stdout


def sanity_check(traj_path):
    """读 TUM 做体检，返回 (ok, report)。"""
    traj = np.loadtxt(traj_path)
    if traj.ndim == 1:
        traj = traj[None, :]
    steps = np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2]))
    step_max = float(steps.max()) if len(steps) else 0.0
    z_span = float(np.ptp(traj[:, 3]))
    duration = float(traj[-1, 0] - traj[0, 0])
    problems = []
    if step_max > MAX_STEP_M:
        problems.append(f"相邻位姿最大位移 {step_max:.2f} m > {MAX_STEP_M} m（里程计疑似发散）")
    if z_span > MAX_Z_SPAN_M:
        problems.append(f"全程高差 {z_span:.2f} m > {MAX_Z_SPAN_M} m（里程计疑似发散）")
    if duration < MIN_DURATION_S:
        problems.append(f"时长 {duration:.1f} s < {MIN_DURATION_S} s（数据太短）")
    report = {
        "poses": int(len(traj)),
        "duration_s": round(duration, 2),
        "max_step_m": round(step_max, 4),
        "z_span_m": round(z_span, 4),
        "ok": not problems,
        "problems": problems,
    }
    return not problems, report


def main():
    parser = argparse.ArgumentParser(description="bag → PCD + 2D 地图（参数固化版）")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true", help="跳过轨迹体检（发散数据也出图）")
    args = parser.parse_args()

    bag = Path(args.bag).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not bag.is_file():
        raise SystemExit(f"bag not found: {bag}")
    out.mkdir(parents=True, exist_ok=True)

    print("== 1/3 抽点云与轨迹 ==", flush=True)
    run([sys.executable, TOOLS / "bag_mapping_pipeline.py",
         "--bag", bag, "--out", out,
         "--voxel", VOXEL_M, "--min-frames", MIN_FRAMES,
         "--remove-ground"])

    raw_metrics = json.loads((out / "metrics.json").read_text())
    ground_offset = raw_metrics.get("ground_offset_m")
    if ground_offset is None:
        raise SystemExit("地面偏移未能自动估计；请检查点云是否贴近轨迹，或改用 --force 前先手工给值")

    print("== 2/3 轨迹体检 ==", flush=True)
    ok, health = sanity_check(out / "traj_camera_init.tum")
    print(json.dumps(health, ensure_ascii=False, indent=2), flush=True)
    if not ok and not args.force:
        (out / "report.md").write_text(
            "# 拒绝出图\n\n" + "\n".join(f"- {p}" for p in health["problems"]) + "\n",
            encoding="utf-8")
        raise SystemExit("轨迹体检未通过，已停止（加 --force 可强制出图）：\n  " +
                         "\n  ".join(health["problems"]))

    print("== 3/3 投影 2D 地图 ==", flush=True)
    run([sys.executable, TOOLS / "slope_map_projector.py",
         "--pcd", out / "cloud_static.pcd",
         "--trajectory", out / "traj_camera_init.tum",
         "--ground-offset-m", f"{ground_offset:.4f}",
         "--height-low-m", HEIGHT_LOW_M,
         "--height-high-m", HEIGHT_HIGH_M,
         "--resolution", GRID_RESOLUTION_M,
         "--max-route-distance-m", MAX_ROUTE_DISTANCE_M,
         "--overwrite",
         "--out-prefix", out / "map"])

    map_metrics_path = out / "map_metrics.json"
    map_metrics = json.loads(map_metrics_path.read_text()) if map_metrics_path.is_file() else {}

    summary = {
        "bag": str(bag),
        "out": str(out),
        "fixed_params": {
            "voxel_m": VOXEL_M,
            "height_window_m": [HEIGHT_LOW_M, HEIGHT_HIGH_M],
            "grid_resolution_m": GRID_RESOLUTION_M,
            "min_frames": MIN_FRAMES if MIN_FRAMES else "auto(otsu,2..4)",
            "max_route_distance_m": MAX_ROUTE_DISTANCE_M,
        },
        "auto_estimated": {
            "ground_offset_m": ground_offset,
            "min_frames": raw_metrics.get("min_frames"),
            "threshold_mode": raw_metrics.get("threshold_mode"),
        },
        "health": health,
        "cloud": {
            "frames": raw_metrics.get("cloud_frames"),
            "voxels_total": raw_metrics.get("voxels_total"),
            "static_voxels": raw_metrics.get("static_voxels"),
        },
        "map": {
            "occupied_cells": map_metrics.get("occupied_cells"),
            "grid": [map_metrics.get("grid_width"), map_metrics.get("grid_height")],
            "pgm": str(out / "map.pgm"),
            "yaml": str(out / "map.yaml"),
        },
        "outputs": {
            "pcd_all": str(out / "cloud_voxel_all.pcd"),
            "pcd_static": str(out / "cloud_static.pcd"),
            "pcd_noground": str(out / "cloud_noground.pcd"),
            "traj": str(out / "traj_camera_init.tum"),
            "map_pgm": str(out / "map.pgm"),
            "map_yaml": str(out / "map.yaml"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(
        "# bag → PCD + 2D 地图\n\n"
        f"- bag：`{bag.name}`\n"
        f"- 帧数：{summary['cloud']['frames']}，时长 {health['duration_s']} s\n"
        f"- 自动地面偏移：**{ground_offset:.3f} m**\n"
        f"- 自动帧数阈值：{summary['auto_estimated']['min_frames']}（{summary['auto_estimated']['threshold_mode']}）\n"
        f"- 体素：全部 {summary['cloud']['voxels_total']}，保留 {summary['cloud']['static_voxels']}\n"
        f"- 2D 障碍格：{summary['map']['occupied_cells']}\n"
        f"- 体检：{'通过' if health['ok'] else '未通过，已强制出图'}"
        f"（最大帧间位移 {health['max_step_m']} m，高差 {health['z_span_m']} m）\n",
        encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
