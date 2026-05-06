# Yosegment 使用说明（运行方式 / 参数 / 示例）

本文档用于“跑起来的人”：集中记录各个 Python 入口脚本/模块的调用方式、常用参数、示例与注意事项。

---

## 0. 环境与安装

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
python -m pytest -q
```

---

## 1. 标准分割推理（落盘链路）

入口：`app/inference/segmentation.py`

最常用：

```bash
python -m app.inference.segmentation --load-masks
```

手动指定输入与输出命名：

```bash
python -m app.inference.segmentation --source test_input --name exp2 --load-masks
```

说明：
- 会调用 `yolo/segment/predict.py`
- 输出目录：`runs/segment/exp*/masks/*.png`

---

## 2. 交互式路径规划（matplotlib GUI）

入口：`app/planning/path_planner.py`

读取最新一次分割结果：

```bash
python -m app.planning.path_planner
```

手动指定 mask 目录：

```bash
python -m app.planning.path_planner --mask-dir runs/segment/exp2/masks
```

适用场景：
- 调试已有分割结果
- 鼠标交互改起点/终点/障碍物
- 保存规划结果到 `runs/pathplan/exp*`

---

## 3. 批量路径规划

入口：`app/planning/pathplan_batch.py`

对图片/视频/文件夹先分割再规划：

```bash
python -m app.planning.pathplan_batch --source test_input
```

输出目录：
- `runs/segment/exp*`（分割产物）
- `runs/pathplan/exp*`（规划产物）

---

## 4. ONNX realtime 路径规划（内存链路）

入口：`app/planning/realtime_pathplan.py`

单图/视频/目录：

```bash
python -m app.planning.realtime_pathplan --source test_input --weights weights/0414_qy++.onnx
```

实时显示（不等于保存）：

```bash
python -m app.planning.realtime_pathplan --source test_input/demo.mp4 --weights weights/0414_qy++.onnx --view
```

摄像头：

```bash
python -m app.planning.realtime_pathplan --source 0 --weights weights/0414_qy++.onnx --view
```

注意：
- `onnx_realtime` 要求输入权重是 `.onnx`；误传 `.pt` 会报错

---

## 5. Phase2 v0：导出整图 occ2d + z_band2d（危险高度带）

入口：`app/mapping/octomap_export.py`

导出 npz：

```bash
python -m app.mapping.octomap_export --mask-dir runs/segment/exp2/masks --out-npz runs/debug/map_v0.npz
```

带 matplotlib 预览：

```bash
python -m app.mapping.octomap_export --mask-dir runs/segment/exp2/masks --out-npz runs/debug/map_v0.npz --preview
```

可选参数：
- `--grid-scale`：覆盖默认 pixel→grid 缩放
- `--use-columns`：occ2d 使用 columns 语义（更“全”）；默认使用 blocked_obstacles（更保守）

npz 内容：
- `occ2d: uint8[H,W]`，0=free，255=occupied
- `z_band2d: float32[H,W,2]`，[...,0]=collision_base_z，[...,1]=top_z
- 以及 `grid_w/grid_h/grid_scale/revision`

---

## 6. Phase2 v0：ROS2 发布（OccupancyGrid + z_band 可视化）

入口：`app/mapping/ros2_publish_occ_zband.py`

### 6.1 前置条件

仅在**有 ROS2 Python 环境**的机器上运行（必须能 import：`rclpy/nav_msgs/visualization_msgs` 等）。

本仓库本身不强依赖 ROS2：该脚本采用“运行时 import ROS2”策略，所以你在非 ROS2 环境下 import `app` 不会炸；但运行该脚本会失败。

### 6.2 发布命令

```bash
python -m app.mapping.ros2_publish_occ_zband --npz runs/debug/map_v0.npz --frame-id map --resolution 1.0 --rate 2
```

发布 topics：
- `/yoseg/occ2d_grid`：`nav_msgs/OccupancyGrid`
- `/yoseg/z_band_markers`：`visualization_msgs/MarkerArray`

RViz2 建议添加：
- Map: topic=`/yoseg/occ2d_grid`
- MarkerArray: topic=`/yoseg/z_band_markers`

可视化参数：
- `--occupied-threshold`：`occ2d > threshold` 判为 occupied
- `--marker-step`：对 marker 下采样（大图建议用 2/4/8）
- `--marker-alpha`：透明度
- `--marker-scale`：线宽

坐标约定：
- 导出的数组坐标是 (x 右, y 下)
- 为 RViz 更直观，脚本对 y 做了 flip（`np.flipud`）

---

## 7. Phase2：GPU 调试 / 未来 NPU 接口预留（mask→grid 归约后端）

模块：`app/mapping/mask_reduce.py`

用途：把现有 `GridMapHandler.batch_masks_to_obs` 的语义，用“后端接口”的方式抽出来：
- `NumpyBackend`：CPU 参考实现
- `TorchBackend`：可选 torch 后端（用于桌面 GPU/CPU debug）

相关测试：
- `tests/test_mask_reduce_backends.py`

---

## 8. 其他：导出 ONNX

如果你手里只有 `.pt` 权重：

```bash
python yolo/export.py --weights weights/0414_qy++.pt --include onnx --imgsz 640 640
```

得到 `weights/xxx.onnx` 后，交给 realtime 入口使用。
