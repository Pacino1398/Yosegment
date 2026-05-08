# Yosegment

一个把 **YOLOv5 分割结果** 转成 **栅格障碍物地图**，再做 **D* Lite 路径规划** 的工程。

当前仓库包含两条主链路：
- **离线落盘链路**：分割结果输出到 `runs/segment/exp*/masks`，再读取做建图/规划
- **ONNX realtime 内存链路**：分割结果直接保存在内存里，不走 mask 落盘中转

---

## 文档导航

- 代码/架构/函数解析（按 .py 模块）：[`docs/PY_API.md`](docs/PY_API.md)
- 运行方式/参数/示例（按入口脚本）：[`docs/USAGE.md`](docs/USAGE.md)
- 地图↔规划↔导出/ROS2 的数据流图：[`docs/DATAFLOW.md`](docs/DATAFLOW.md)

---

## 快速开始

### 1) 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2) 运行测试

```bash
python -m pytest -q
```

### 3) 常用入口（更完整请看 `docs/USAGE.md`）

- 标准分割推理（落盘）：

```bash
python -m app.inference.segmentation --load-masks
```

- 交互式路径规划：

```bash
python -m app.planning.path_planner
```

- Phase2 v0：导出 `occ2d + z_band2d`（用于本地预览或后续 ROS2 发布）：

```bash
python -m app.mapping.octomap_export --mask-dir runs/segment/exp2/masks --out-npz runs/debug/map_v0.npz --preview
```

- Phase2 v0：在 ROS2 环境发布 `OccupancyGrid + z_band markers`：

```bash
python -m app.mapping.ros2_publish_occ_zband --npz runs/debug/map_v0.npz --frame-id map --resolution 1.0 --rate 2
```

---

## ROS2 话题发布格式（工程约定）

本仓库的 ROS2 代码采用“**可选依赖、延迟 import**”策略：不在默认环境硬依赖 `rclpy`。
更完整说明见：[`docs/ROS2_OCTOMAP_PUBLISH.md`](docs/ROS2_OCTOMAP_PUBLISH.md)

### 1) 已实现的 Topics

#### A. `/yoseg/occ2d_grid`（`nav_msgs/OccupancyGrid`）

- **用途**：2D 占据栅格（从 `occ2d: uint8[H,W]` 转换）
- **关键字段约定**：
  - `header.frame_id`：由参数 `--frame-id` / `OccZBandPublishConfig.frame_id` 指定（默认 `map`）
  - `header.stamp`：节点当前时间 `node.get_clock().now().to_msg()`
  - `info.resolution`：`--resolution`（单位：m/格）
  - `info.width/info.height`：对应 `occ2d.shape = [H,W]`
  - `info.origin`：固定为 `(0,0,0)`，四元数 `w=1.0`
  - `data`：int8 数组（`len = H*W`），本工程仅输出 `0/100`（不输出 `-1 unknown`）

**坐标/翻转说明**：工程内部数组坐标是 **x 向右、y 向下（图像坐标）**；为适配 RViz 显示，发布前会对 `occ2d` 做 `np.flipud()`（Y 轴翻转），使原点更接近“左下角”。

#### B. `/yoseg/z_band_markers`（`visualization_msgs/MarkerArray`）

- **用途**：可视化每个 cell 的高度危险段 `[z_low, z_high]`
- **Marker 形式**：单个 `Marker`，`type=LINE_LIST`，`ns="z_band"`，`id=0`
- **关键字段约定**：
  - `markers[0].header.frame_id`：同 `frame_id`
  - `markers[0].header.stamp`：与 `occ2d_grid` 同一时间戳
  - `scale.x`：线宽，由 `--marker-scale`/`marker_scale` 指定
  - `color`：RGBA，由 `--marker-alpha`/`marker_alpha` 控制透明度
  - `points`：每个 cell 对应两点（线段端点）
    - `x = (cell_x + 0.5) * resolution`
    - `y = (H - 1 - cell_y + 0.5) * resolution`（同样做 Y 翻转以对齐 RViz）
    - `z = z_low / z_high`（单位：m）

### 2) 输入数据格式（离线/回放：`.npz`）

用于发布的 `.npz` 文件需要至少包含：

- `occ2d`: `uint8[H,W]`
- `z_band2d`: `float32/float64[H,W,2]`（low/high）

对应发布入口：

- `python -m app.ros2.occ_zband_publisher ...`（推荐，封装更清晰）
- `python -m app.mapping.ros2_publish_occ_zband ...`（旧入口，功能类似）

### 3) 实时输入格式（在线：snapshot JSON）

实时节点：`python -m app.ros2.realtime_occ_zband_node --sub-topic /yoseg/octomap_snapshot_json ...`

订阅：`/yoseg/octomap_snapshot_json`（`std_msgs/String`），`msg.data` 为 JSON。

至少需要字段（best-effort 解析）：

```json
{
  "bounds": [80, 60, 3.0],
  "grid_scale": 8,
  "blocked_obstacles": [[10, 10], [11, 10]]
}
```

如果提供 `columns`（更丰富，可恢复 `z_band2d`）：

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

### 4) QoS/队列与频率（当前实现）

- 发布端 `queue_size`：默认 `1`（`create_publisher(..., queue_size)`）
- 离线回放发布频率：`--rate`（Hz）
- 实时节点：按消息到达触发发布，同时提供 `--rate` 作为配置项（上限语义）

> 注：当前代码未显式设置 ROS2 QoSProfile（Reliability/Durability 等），使用 rclpy 默认 QoS；如要与 Nav2/RViz 特定 QoS 对齐，可在 `app/ros2/occ_zband_publisher.py` 里扩展为显式 QoSProfile。

---

## 目录概览

- `app/`：本地业务逻辑（建议优先在这里改）
  - `app/inference/`：分割推理入口
  - `app/mapping/`：建图与导出（mask→grid、2.5D columns、occ2d/z_band2d）
  - `app/planning/`：路径规划与渲染
  - `app/tooling/`：独立工具脚本
- `tests/`：pytest 测试
- `yolo/`：vendored YOLOv5 代码（除非必要，不建议大范围改动）

