# OctoMap/occ2d/z-band 通过 ROS2 话题发布（Yosegment）

本文说明如何把 `app/mapping/octomap.py` 生成的地图结果，通过 ROS2 话题发布出去，便于 RViz2 可视化/下游规划节点订阅。

> 说明：当前仓库主要以 Python 管线为主，ROS2 部分采取“可选依赖、延迟 import”的方式，避免在未安装 ROS2 时导入失败。

---

## 1. 已实现的发布能力

### 1.1 从 `.npz` 文件定频发布（离线/回放）

文件：`app/ros2/occ_zband_publisher.py`

发布话题（默认）：

- `/yoseg/occ2d_grid`：`nav_msgs/OccupancyGrid`
- `/yoseg/z_band_markers`：`visualization_msgs/MarkerArray`

运行示例：

```bash
python -m app.ros2.occ_zband_publisher \
  --npz runs/debug/map_v0.npz \
  --frame-id map \
  --resolution 0.1 \
  --rate 2
```

其中 `.npz` 需要包含：

- `occ2d`: `uint8[H,W]`
- `z_band2d`: `float32/float64[H,W,2]`（low/high）

### 1.2 从 JSON snapshot 实时订阅并发布（在线）

文件：`app/ros2/realtime_occ_zband_node.py`

订阅：

- `/yoseg/octomap_snapshot_json`：`std_msgs/String`（JSON）

发布：

- `/yoseg/occ2d_grid`：`nav_msgs/OccupancyGrid`
- `/yoseg/z_band_markers`：`visualization_msgs/MarkerArray`

运行示例：

```bash
python -m app.ros2.realtime_occ_zband_node \
  --sub-topic /yoseg/octomap_snapshot_json \
  --frame-id map \
  --resolution 0.1
```

---

## 2. Snapshot JSON 格式（与 OctoMap.export_planner_snapshot() 对齐）

节点 `realtime_occ_zband_node.py` 采用“best-effort”解析，至少需要字段：

```json
{
  "bounds": [80, 60, 3.0],
  "grid_scale": 8,
  "blocked_obstacles": [[10, 10], [11, 10]]
}
```

如果提供 `columns`（更丰富，可恢复 z_band）：

```json
{
  "bounds": [80, 60, 3.0],
  "grid_scale": 8,
  "columns": {
    "(10,10)": {"height": 1.2, "class_id": 0},
    "(11,10)": {"height": 0.5, "class_id": 1}
  }
}
```

---

## 3. RViz2 可视化建议

### 3.1 OccupancyGrid

- Fixed Frame：`map`（与 `--frame-id` 保持一致）
- 添加 Display：`Map`，订阅 `/yoseg/occ2d_grid`

### 3.2 z-band MarkerArray

- 添加 Display：`MarkerArray`，订阅 `/yoseg/z_band_markers`

> 当前 marker 使用 `LINE_LIST`，每个 cell 画一条竖线表示 `[z_low, z_high]`。

---

## 4. 与现有实时管线的推荐集成方式（更低延迟）

如果你的实时分割/建图与 ROS2 节点在同一 Python 进程，推荐不要走 JSON/订阅模式：

- 直接在你的循环里创建一个 `rclpy.node.Node`
- 实例化 `OccZBandPublisher`
- 每帧调用 `publish(occ2d, z_band2d)`

这样可避免序列化/反序列化和 ROS subscription 的额外开销。

---

## 5. 依赖说明

需要 ROS2 环境（例如 Humble）并安装：

- `rclpy`
- `nav_msgs`
- `visualization_msgs`
- `geometry_msgs`
- `std_msgs`
- `builtin_interfaces`

Windows 上一般不建议跑 ROS2；更常见是 Ubuntu 端运行 ROS2 节点。