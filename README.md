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

## 目录概览

- `app/`：本地业务逻辑（建议优先在这里改）
  - `app/inference/`：分割推理入口
  - `app/mapping/`：建图与导出（mask→grid、2.5D columns、occ2d/z_band2d）
  - `app/planning/`：路径规划与渲染
  - `app/tooling/`：独立工具脚本
- `tests/`：pytest 测试
- `yolo/`：vendored YOLOv5 代码（除非必要，不建议大范围改动）

