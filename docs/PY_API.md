# Yosegment Python 模块 / API 解析

本文档用于“读代码的人”：按模块（.py 文件）梳理主要职责、关键数据结构与核心函数，方便快速定位改动点。

> 约定：文中模块路径均以仓库根为起点，例如 `app/mapping/octomap.py`。

---

## 1. 顶层包结构（`app/`）

- `app/inference/`：分割推理入口（标准落盘 / ONNX realtime）
- `app/mapping/`：mask → 栅格地图 / 2.5D columns / 导出与可视化
- `app/planning/`：路径规划（D* Lite 2D/3D）与渲染
- `app/tooling/`：独立小工具脚本

---

## 2. `app/config.py`

职责：项目默认配置集中点。

常见用途：
- 统一改默认输入/输出目录
- 统一改默认设备、置信度阈值、grid_scale 等

关键对象：
- `DEFAULT_CONFIG`：默认配置实例（被多个 CLI 入口读取）

---

## 3. `app/paths.py`

职责：路径解析与 repo 根目录定位。

关键函数：
- `resolve_path(user_value, default_value)`：
  - 若用户传入参数为空，则回落到默认值
  - 支持相对/绝对路径

---

## 4. 推理层（`app/inference/`）

### 4.1 `app/inference/segmentation.py`

职责：标准分割入口（通常通过 subprocess 调用 YOLOv5 分割脚本），并把结果落到 `runs/segment/exp*/`。

你最常改的点：
- 文件顶部的 `MANUAL_*` 默认值（便于本地快速调参）

典型输出：
- `runs/segment/exp*/masks/*.png`

### 4.2 `app/inference/onnx_realtime.py`

职责：ONNX realtime 分割入口：
- 不依赖磁盘中转
- 将单帧推理结果转成规划层能消费的 `mask_entries`

常见扩展方向：摄像头/视频流/服务化封装。

---

## 5. 建图层（`app/mapping/`）

### 5.1 `app/mapping/grid_map.py`

职责：**mask → 栅格** 的核心逻辑（阻塞/可通行障碍物、目标点、代价、障碍高度等）。

关键概念：
- `GridMapHandler(grid_w, grid_h, grid_scale)`：
  - 持有各种中间结果：
    - `blocked_obstacles: set[(x,y)]`
    - `traversable_obstacles: set[(x,y)]`
    - `terrain_penalties: dict[(x,y)->float]`
    - `obstacle_heights: dict[(x,y)->int]`
    - `obstacle_class_ids: dict[(x,y)->int]`
  - 以及 `target_point`、`mask_instances` 等

关键函数：
- `load_mask_entries(mask_dir, grid_handler)`：读取磁盘 mask PNG 并构建 `mask_entries` 列表
- `GridMapHandler.batch_masks_to_obs(mask_list, ...)`：
  - **当前语义的“权威实现”**
  - 输入：mask entries（来自磁盘或 realtime）
  - 输出：障碍物集合与 target_point，并写入 handler 的各类字段

### 5.2 `app/mapping/octomap.py`

定位：本仓库里的 “OctoMap” 并非全局 3D octree；当前更贴近 **局部 2.5D columns（柱体）地图**。

关键常量：
- `TREE_CANOPY_DOWNWARD_EXPANSION = 1.0`：树冠危险带向下膨胀 1m
- `CANOPY_CLASSES = {4, 6}`：tree/forest 类
- `HOUSE_CLASS = 9`

关键数据结构：
- `ColumnState`：每个 cell 一根柱体
  - `height`：柱体高度（用于显示和 top_z）
  - `display_base_z`：显示基底（当前默认 0）
  - `collision_base_z`：碰撞危险带的底部（canopy 会是 `top_z - 1m`）
  - `top_z`：`display_base_z + height`
  - `terrain_penalty`、`class_id`、`mode` 等

关键方法（Phase2 v0 相关）：
- `OctoMap.build_occ2d(occupied_value=255, use_columns=False) -> np.ndarray`：
  - 导出 dense 2D 占据图 `uint8[H,W]`
  - `use_columns=False`：用 `blocked_obstacles` 标占据（更保守）
  - `use_columns=True`：用 `columns.keys()` 标占据（更“全”，用于对齐 columns 语义）
- `OctoMap.build_z_band2d() -> np.ndarray`：
  - 导出 dense 2D 危险高度带 `float32[H,W,2]`
  - `[y,x,0]=collision_base_z`，`[y,x,1]=top_z`

内部流程要点：
- `grid_handler.batch_masks_to_obs(...)` 先产出 obstacle_* 字段
- `_sync_from_grid_handler()` 把 handler 的结果同步进 OctoMap
- `build_octomap(...)` 用同步后的高度/类别信息构建 columns，并刷新 blocked 等视图

### 5.3 `app/mapping/octomap_export.py`

定位：Phase2 v0 导出工具：把 masks 构建为 OctoMap，然后导出：
- `occ2d`（uint8[H,W]）
- `z_band2d`（float32[H,W,2]）

关键函数：
- `parse_args()`：定义 CLI 参数（mask-dir、grid-scale、out-npz、preview、use-columns 等）
- `main()`：执行导出并可选 matplotlib 预览

输出 npz 的字段：
- `occ2d`
- `z_band2d`
- `grid_w/grid_h/grid_scale/revision`

### 5.4 `app/mapping/mask_reduce.py`

定位：**mask→grid 归约后端抽象**（Phase2 GPU 调试 / 未来 NPU 接口预留）。

关键类型：
- `ReduceResult`：与 GridMapHandler 的语义对齐的输出集合
- `ReduceBackend.reduce(...)`：后端接口

实现：
- `NumpyBackend`：CPU 参考实现
- `TorchBackend`：可选 torch 后端（当前用于 debug，对齐 Numpy 语义；后续可继续矢量化）

### 5.5 `app/mapping/ros2_publish_occ_zband.py`

定位：在 ROS2 Python 环境里发布 Phase2 v0 导出结果（不强依赖 ROS2，运行时 import）。

核心能力：
- 发布 `/yoseg/occ2d_grid`（`nav_msgs/OccupancyGrid`）
- 发布 `/yoseg/z_band_markers`（`visualization_msgs/MarkerArray`，用 `LINE_LIST` 画每个 cell 的竖直危险段）

关键函数：
- `_to_occupancygrid_data(occ2d, threshold)`：occ2d→OccupancyGrid data（并做 y flip）
- `_build_z_band_markers(z_band2d, ...)`：构造 MarkerArray（并做 y flip 对齐 RViz）

---

## 6. 规划层（`app/planning/`）

### 6.1 `app/planning/dstar_lite.py`

职责：2D D* Lite 路径规划器。

你通常关注：
- 邻域扩张、代价/惩罚策略
- 障碍物膨胀与安全边界

### 6.2 `app/planning/dstar_lite_3d.py`

职责：3D D* Lite（26 邻域）。

注意事项（当前实现特性）：
- open list 仍是 dict + `min(U)` 扫描，适合小图验证，不适合上板高频。

### 6.3 `app/planning/path_planner.py`

职责：交互式规划入口（matplotlib），支持鼠标改点与保存结果。

### 6.4 `app/planning/pathplan_batch.py`

职责：批量规划入口 + 公共渲染层。

关键函数：
- `build_plan_result(...)`
- `render_plan_view(...)`

### 6.5 `app/planning/realtime_pathplan.py`

职责：realtime 入口：打开输入源、调用 ONNX 分割、直接走内存链路做规划与渲染。

---

## 7. 测试（`tests/`）

- `tests/test_octomap_occ_zband.py`：覆盖 `build_occ2d/build_z_band2d` 与 canopy 规则
- `tests/test_mask_reduce_backends.py`：覆盖 NumpyBackend 与 TorchBackend（若无 torch 自动 skip）
- 其他测试：覆盖 grid_map、planner、realtime 等
