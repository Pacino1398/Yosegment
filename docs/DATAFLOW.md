# Yosegment 数据流图（地图 ↔ 规划 ↔ 导出/ROS2）

本文档把你现在工程里 **分割结果 → 地图（grid）→ 路径规划**，以及 **grid → 2.5D columns → occ2d/z_band2d → ROS2** 两条链路，整理成一张可对照代码的“数据流图”。

---

## 1. 总览数据流图（Mermaid）

> 你可以在 GitHub / 支持 Mermaid 的 Markdown 查看器里直接渲染。

```mermaid
flowchart TD
  A[分割输出
masks/*.png 或内存 mask_entries] --> B[加载/组织 mask_entries]

  B --> C[GridMapHandler.batch_masks_to_obs
(构建 2D 栅格语义)]

  C -->|blocked_obstacles(set)| D[DStarLite
2D 路径规划]
  C -->|traversable_obstacles(set)| D
  C -->|terrain_penalties(dict)| D
  C -->|target_point| D

  C --> E[OctoMap
(2.5D columns)]

  E --> F[octomap_export.py
导出 npz]
  F --> G[occ2d(uint8[H,W])]
  F --> H[z_band2d(float32[H,W,2])]

  G --> I[ROS2 OccupancyGrid 发布]
  H --> J[ROS2 MarkerArray 发布
(竖直危险段)]
```

---

## 2. 关键“数据对象”说明（你在代码里会看到它们频繁出现）

### 2.1 mask_entries（分割结果的统一中间格式）
- 形态：`list[list]`（每个元素代表一个 instance mask）
- 主要来源：
  - 磁盘：`load_mask_entries(...)` / `load_grouped_mask_entries(...)`
  - 内存：`app/inference/onnx_realtime.py` 产生并直接传给规划

### 2.2 GridMapHandler 的输出（地图语义）
由 `GridMapHandler.batch_masks_to_obs(mask_entries)` 生成/填充：
- `blocked_obstacles: set[(x,y)]`：不可通行
- `traversable_obstacles: set[(x,y)]`：可通行但带惩罚（tree/forest）
- `terrain_penalties: dict[(x,y)->float]`：惩罚值
- `obstacle_heights: dict[(x,y)->int]`：高度
- `obstacle_class_ids: dict[(x,y)->int]`：类别
- `target_point: (x,y) | None`

### 2.3 DStarLite 消费地图的方式
在 `DStarLite.__init__`：
- `obs`（传入的 blocked）会被处理为：`self.obs = set(obs) - passable_obs`
  - 即 traversable 会从绝对障碍里剔除，允许通行
- `cost(a,b)` 会把 `terrain_penalties[b]` 加进总代价

### 2.4 OctoMap/columns（2.5D 表达）
`OctoMap` 复用 `GridMapHandler` 的结果，构建每个 cell 的 `ColumnState`：
- `height/top_z`
- `collision_base_z`（canopy 类向下膨胀 1m）

然后可导出：
- `occ2d: uint8[H,W]`
- `z_band2d: float32[H,W,2]`（[z_lo,z_hi]）

---

## 3. 对应 .py 文件清单（按数据流从上游到下游）

### A) mask_entries 的来源
- `app/mapping/grid_map.py`
  - `load_mask_entries(mask_dir, grid_handler)`：读取磁盘 masks 成 mask_entries
  - `load_grouped_mask_entries(mask_dir)`：按 image_stem 分组（给 batch/video 用）
- `app/inference/onnx_realtime.py`
  - 产生内存版 mask_entries（realtime 链路）

### B) 地图构建（权威语义）
- `app/mapping/grid_map.py`
  - `GridMapHandler.batch_masks_to_obs(...)`

### C) 2D 路径规划（D* Lite）
- `app/planning/dstar_lite.py`
  - `DStarLite`
- 入口/调用处：
  - `app/planning/pathplan.py`（交互式 GUI）
  - `app/planning/pathplanbatch.py`（批量 + 公共函数 `build_plan_result`）
  - `app/planning/realtime_pathplan.py`（realtime 入口，内部复用 batch 公共逻辑）

### D) 2.5D columns + 导出（Phase2 v0）
- `app/mapping/octomap.py`
  - `OctoMap`、`ColumnState`、`build_occ2d/build_z_band2d`
- `app/mapping/octomap_export.py`
  - 导出 `occ2d + z_band2d` 到 `.npz`

### E) ROS2 发布（Phase2 v0）
- `app/mapping/ros2_publish_occ_zband.py`
  - 发布 `/yoseg/occ2d_grid`（OccupancyGrid）
  - 发布 `/yoseg/z_band_markers`（MarkerArray）

---

## 4. 最常见的两条“可运行链路”（对应上图两条分支）

### 4.1 2D 规划主链路（离线 masks）
1) `runs/segment/exp*/masks/*.png`
2) `app/planning/pathplan.py` / `app/planning/pathplanbatch.py`
3) `GridMapHandler.batch_masks_to_obs` → `DStarLite`

### 4.2 Phase2 v0 导出 + ROS2 可视化链路
1) `runs/segment/exp*/masks/*.png`
2) `app/mapping/octomap_export.py`
   - 内部：`GridMapHandler.batch_masks_to_obs` → `OctoMap` → `occ2d/z_band2d`
3) `app/mapping/ros2_publish_occ_zband.py` 发布到 RViz
