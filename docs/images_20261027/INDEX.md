# 点云处理分析图（2026-10-27）

当天调参与验收过程中生成的全部对比图，按主题归档。所有图都是从真实 bag 跑出来的，
不是示意图。

数据来源两趟：
- **scan_0**：`20261027_patchwork/scan_0.bag`（307 s，6,018 帧，走廊+房间）
- **rerun**：`fastlio_rerun_scan1_cov10_20261027/output_0.bag`（295 s，5,786 帧，跨 2.4 m 坡道）

## 01_地面基准（9 张）

地面高度怎么定，直接决定 2D 图是否被地面灌满。

| 图 | 说明 |
|---|---|
| `2d_ground_offset_cmp.png` | 三组对比：offset 1.2（地面灌满）/ 0.85（正常）/ 局部地面（最干净） |
| `cmp_offset120_low010_vs_low000.png` | 同一 offset 下，下沿 0.10 正常 vs 0.00 地面全黑 |
| `floor_mask_136m2.png` | 地面层覆盖范围：136 m²，占已覆盖面积 79% |
| `map_off085.png` / `map_off120.png` / `map_local.png` | 三种地面基准的单独出图 |
| `onecut_global085_030_080.png` | "一刀切"方案的实际效果（作为反例） |
| `final_local.png` / `map_local_ground.png` | 局部地面投影结果 |

结论：地面基准必须来自 odometry（轨迹 z − 1.174 m），窗口相对当地地面，不能一刀切。

## 02_高度窗（8 张）

障碍窗下沿的取值影响。

| 图 | 说明 |
|---|---|
| `cmp_heightlow_010_030.png` | 下沿 0.10 vs 0.30：矮结构保留度 |
| `cmp_heightlow_000_010.png` | 下沿 0.00 vs 0.10：可通行区开始被吃掉 |
| `cmp_heightlow_neg010_000.png` | 下沿 −0.10 vs 0.00（负值无意义，仅作边界验证） |
| `cmp_low.png` / `cmp_low0.png` / `cmp_neg.png` | 同上三组的早期版本 |
| `cmp_flood.png` | 地面灌入 2D 图的典型失败样例 |
| `final_cmp.png` | 汇总对比 |

结论：下沿 **0.10 m**；降到 0 会把整片地面判成障碍。

## 03_帧数阈值（9 张）

"同一个 5 cm 体素至少被几帧看到才保留"。

| 图 | 说明 |
|---|---|
| `f234.png` | ≥4 / ≥3 / ≥2 帧三档并排（rerun 数据） |
| `f2_vs_f4.png` | 2 帧 vs 4 帧细节对比 |
| `scan0_thresholds.png` | 3 / 5 / 13 帧三档（scan_0 数据） |
| `scan0_f5_vs_f8.png` | 5 帧 vs 8 帧 |
| `f5f8.png` / `detail_cmp.png` / `detail_cmp2.png` / `fix_cmp.png` | 早期阈值对比与修复前后 |
| `cmp_static.png` | 静态过滤前后对比 |

结论：**3 帧**是平衡点。8 帧删掉约 90% 细节（花坛边缘这类矮结构消失）。

## 04_射线填充（4 张）

从轨迹位姿向观测点连线，把经过的格子标为可通行。

| 图 | 说明 |
|---|---|
| `bag_traj_vs_static.png` | 同一数据：全量体素 vs ≥5 帧过滤，人形重影消失 |
| `final_scan0_raycast.png` / `final_run2_raycast.png` | 射线填充后的最终地图 |
| `vis_overlay.png` | 射线穿透投票判为动态的点（红）叠加在保留点（灰）上 |

## 05_最终地图（7 张）

| 图 | 说明 |
|---|---|
| `final_scan0_map.png` | scan_0 最终 2D 地图（3 帧 + 射线填充） |
| `fastlio_rerun_cov10_map.png` | rerun 数据最终地图（70.7 × 42.7 m） |
| `run2_scan0_map.png` | run2 数据地图（覆盖较小） |
| `scan0_map.png` / `f3_vs_f8.png` | 早期版本与 3/8 帧对比 |
| `pcd_static_top.png` / `pcd_preview.png` | 点云俯视（按高度上色） |

## 06_早期诊断（15 张）

最初定位问题时的原始分析，保留作追溯。

| 图 | 说明 |
|---|---|
| `density.png` | 点云密度图，暴露 91.7% 重复率 |
| `zmin.png` / `zmin_fixed.png` / `zmax.png` | 逐格最低/最高点，判断地面是否存在 |
| `crop_zoom.png` / `crop_full.png` | 局部放大 |
| `farfield.png` | 远场离群点分布 |
| `floor_height.png` / `floor_mask.png` | 地面层高度着色与掩膜 |
| `pcd_top.png` / `preview_local_ground.png` | 点云俯视与局部地面投影 |
| `tool_g085.png` / `tool_local.png` / `tool_local_z3.png` / `onecut.png` | 预览工具的早期输出 |

## 重新生成

任何一张都可以用 `tools/bag_to_map.py` 或 `tools/bag_studio.py` 换参数重跑，
参数与对应关系见 `docs/参数与流程.md`。
