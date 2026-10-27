# pointcloud-tools

把一份 FAST-LIO 的 ROS1 bag 处理成 **3D 点云（PCD）** 和 **2D 占据栅格地图（PGM/YAML）**。
本机离线运行，**不需要 ROS 环境**。

- 版本：2026-10-27 v1.0（现场验收：scan_0 通过 / 发散 bag 自动拒绝）
- 依赖：Python 3.8+、numpy、scipy（`/usr/bin/python3` 已满足）

## 快速开始

```bash
cd pointcloud-tools
/usr/bin/python3 tools/bag_to_map.py --bag /path/to/scan.bag --out out/跑次名
```

约 2 GB 的 bag 在这台 Mac 上 5~10 秒跑完。产物在 `out/跑次名/`。

## Web 页面

```bash
双击 launch/启动点云工作台.command
# 或：/usr/bin/python3 tools/bag_studio.py
```

打开 <http://127.0.0.1:8770/>：选 bag → 调参数 → 看进度和日志 → 直接预览 2D 地图、下载点云。

> 由 Finder 双击启动的服务可以写桌面等任意目录；用命令行在受限环境里启动时，
> 程序会自动把结果落到本仓库 `out/` 并提示原因。

## 输出

| 文件 | 内容 |
|---|---|
| `cloud_static.pcd` | 静态点云（帧计数过滤后） |
| `cloud_noground.pcd` | 静态点云 + 去地面（推荐看这个） |
| `cloud_voxel_all.pcd` | 全部 5 cm 体素，intensity = 被观测帧数 |
| `traj_camera_init.tum` | 轨迹（相机初始帧为原点） |
| `map.pgm` / `map.yaml` | 2D 地图：0=障碍，205=未知，254=可通行 |
| `metrics.json` / `map_metrics.json` | 自动估计值、帧数分布、判据 |

## 参数（默认值已固化，来自 2026-10-27 现场对比）

| 参数 | 默认 | 说明 |
|---|---|---|
| 帧数阈值 | **3** | 同一 5 cm 体素至少被几帧看到才保留。2=细节最多但杂，4=干净但丢矮结构 |
| 体素 | 0.05 m | 点云去重粒度 |
| 障碍窗下沿 | 0.10 m | 相对当地地面，低于此不算障碍（地面噪声余量） |
| 障碍窗上沿 | 1.00 m | 相对当地地面，高于此不算障碍（桌面/人上半身/横梁） |
| 地图分辨率 | 0.05 m | 与 `map_server` 契约一致 |
| 轨迹距离门限 | 8 m | 离轨迹更远的点丢弃，防远场飞点撑大地图 |
| 地面偏移 | 自动 | 雷达光心到地面高度，本项目实测约 **1.174 m** |
| 点云去地面 | 开 | 额外输出 `cloud_noground.pcd` |
| 2D 图射线填充 | 开 | 从位姿向观测点连线，把经过的格子标可通行 |

改参数请同时更新本表与 `docs/参数与流程.md`。

## 目录结构

```
pointcloud-tools/
├── tools/
│   ├── bag_to_map.py            一条命令入口（参数固化）
│   ├── bag_mapping_pipeline.py  bag 解析 → 体素 → 帧计数 → 自动地面偏移
│   ├── slope_map_projector.py   逐段地面基准投影 + 射线自由空间填充
│   ├── quick_2d_preview.py      无轨迹时的快速预览
│   ├── bag_studio.py            本机 Web 工作台
│   └── rosbag_odometry_to_tum.py 轨迹导出（需 ROS 环境）
├── launch/启动点云工作台.command
├── tests/slope_map_selftest.py  投影核心的合成自检
└── docs/参数与流程.md            参数取值理由与验收标准
```

## 已知限制

1. 只处理 ROS1 bag（v2.0，none/bz2 压缩）；lz4 需装 `python-lz4`。
2. 必须有 `/Odometry` 和世界系 `/cloud_registered`；缺轨迹只能退化成一刀切/预览。
3. 会做轨迹体检：相邻位姿位移 > 1 m 或高差 > 5 m 直接拒绝出图（`--force` 可强出）。
4. 动态点用「帧数投票」近似，长时间静止的人会留下；需要更严格需射线穿透投票或多趟比对。
5. 2D 图表达不了重叠楼层；跨层数据要按楼层分段建图。
6. `/cloud_registered` 若被 `dense_publish_en=false` 降采样，墙面会偏细、有断点——源头问题。

## 测试

```bash
PYTHONPATH=tools /usr/bin/python3 tests/slope_map_selftest.py
```

## 来源

从 `tron2_ws_backup/tools/` 中当天验证过的脚本整理而来，行为与命令行参数保持一致。

## 基线声明（唯一）

本仓库是点云处理的**唯一基线**，2026-10-27 定稿。另一套并行实现（`tools/mapgen/`）已下线，
代码保留在 `tron2_ws_backup` 的 git 历史里，不再维护、不再作为对照。

后续所有改动都在本仓库主线进行，改参数/换算法都要同步更新本 README 与 `docs/参数与流程.md`。

## 路线图：DUFOMap 接入点

当前动态点去除用的是**帧计数投票**（同一 5 cm 体素被 ≥3 帧看到才保留），快、但不处理
"长时间站立不动的人"。后续在主线上增加 DUFOMap 精修，接入位置固定在**第 2 步之后**：

```
bag ──► 1. 体素去重(5cm) ──► 2. 帧计数过滤(≥3帧) ──► [3. DUFOMap 精修] ──► 4. 去地面 ──► 5. 投影 + 射线填充
                                                          ↑
                                                    预留扩展点（默认关闭）
```

接入约束（写死，避免以后又出现两套）：

1. **默认关闭**，用 `--dufomap` 显式开启；不开时行为与现在完全一致。
2. 只在**有 bag 输入**时可用——DUFOMap 需要逐帧序列，只有 PCD 的场景退化为不启用。
3. 输入输出契约：读原始 per-frame 点云（`/cloud_registered`）+ 轨迹，输出 **dynamic 掩码**；
   掩码作用在 `cloud_voxel_all.pcd` 上，生成 `cloud_static_dufomap.pcd`，下游投影不变。
4. 耗时显著上升（逐帧射线推理），必须在页面上有进度提示，且不允许静默降级。
5. 开启后产物文件名带 `_dufomap` 后缀，便于和基线结果对比。
