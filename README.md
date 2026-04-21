# Yosegment

一个把 **YOLOv5 分割结果** 转成 **栅格障碍物地图**，再做 **D* Lite 路径规划** 的工程。

现在项目里的本地业务逻辑已经基本收拢到 `app/` 下：
- `app/inference/` 负责分割推理入口
- `app/mapping/` 负责把 mask 转成栅格地图
- `app/planning/` 负责路径规划和结果渲染
- `app/tooling/` 放独立小工具
- `yolo/` 保持为 vendored YOLOv5 代码

---

## 1. 这个项目现在有两条主链路

### 1.1 标准落盘链路

适合离线处理、批量处理、调试已有分割结果。

```text
source
  -> app.inference.segmentation
  -> runs/segment/exp*/masks
  -> app.mapping.grid_map
  -> app.planning.path_planner / app.planning.pathplan_batch
  -> runs/pathplan/exp*
```

这条链路里，mask 会先落到磁盘，再被规划模块读取。

### 1.2 ONNX 实时内存链路

适合单帧、视频、摄像头、流式回传，不再走“先写 mask 再读 mask”。

```text
frame / video / stream
  -> app.inference.onnx_realtime
  -> 内存版 mask_entries
  -> build_plan_result(...)
  -> render_plan_view(...)
  -> app.planning.realtime_pathplan
```

这条链路里：
- 分割结果直接保存在内存里
- 规划直接复用现有 `pathplan_batch.py` 的公共函数
- `runs/pathplan/` 只保存规划结果，不再额外保存 `masks`

---

## 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

当前 `requirements.txt` 已包含：
- 基础运行：`numpy`、`opencv-python`、`matplotlib`、`PyYAML`
- 推理相关：`torch`、`torchvision`
- ONNX 相关：`onnxruntime`、`onnx`
- 开发相关：`pytest`、`black`、`ruff`

---

## 3. 常用命令

## 3.1 标准分割推理

```bash
python -m app.inference.segmentation --load-masks
```

手动指定输入：

```bash
python -m app.inference.segmentation --source test_input --name exp2 --load-masks
```

默认会调用：
- `yolo/segment/predict.py`
- 输出到 `runs/segment/exp*`

### 直接在 Python 文件顶部改默认值

`app/inference/segmentation.py` 顶部支持直接改这些常用默认项：

```python
MANUAL_SOURCE
MANUAL_WEIGHTS
MANUAL_DATA_YAML
MANUAL_PROJECT
MANUAL_NAME
```

例如：

```python
MANUAL_SOURCE = "test_input"
MANUAL_WEIGHTS = "weights/0414_qy++.pt"
MANUAL_DATA_YAML = "data/my.yaml"
MANUAL_PROJECT = "runs/segment"
MANUAL_NAME = "exp"
```

优先级规则：
1. 命令行参数
2. `segmentation.py` 顶部手动默认值
3. `app/config.py` 里的项目默认值

---

## 3.2 交互式路径规划

直接读取 `runs/segment/` 下最新一轮分割结果：

```bash
python -m app.planning.path_planner
```

手动指定 mask 目录：

```bash
python -m app.planning.path_planner --mask-dir runs/segment/exp/masks
```

这个入口适合：
- 查看当前障碍物分布
- 鼠标交互改起点/终点/障碍物
- 保存当前规划图到 `runs/pathplan/exp*`

---

## 3.3 批量路径规划

对图片、视频、或混合文件夹先分割再规划：

```bash
python -m app.planning.pathplan_batch --source test_input
```

这个入口会：
1. 先调用标准分割流程
2. 读取 `runs/segment/exp*/masks`
3. 对图片或视频批量生成规划结果
4. 输出到 `runs/pathplan/exp*`

---

## 3.4 ONNX 实时路径规划

单图 / 视频 / 文件夹：

```bash
python -m app.planning.realtime_pathplan --source test_input --weights weights/0414_qy++.onnx
```

实时显示：

```bash
python -m app.planning.realtime_pathplan --source test_input/demo.mp4 --weights weights/0414_qy++.onnx --view
```

摄像头：

```bash
python -m app.planning.realtime_pathplan --source 0 --weights weights/0414_qy++.onnx --view
```

流地址：

```bash
python -m app.planning.realtime_pathplan --source rtsp://your-stream-url --weights weights/0414_qy++.onnx --view
```

只显示不保存：

```bash
python -m app.planning.realtime_pathplan --source 0 --weights weights/0414_qy++.onnx --view --nosave
```

这个入口的特点：
- 输入可以是图片、视频、目录、摄像头索引、流地址
- 直接走 `frame -> mask_entries -> planning -> render`
- 不依赖 `runs/segment/.../masks` 中转

---

## 3.5 导出 ONNX

如果你手里只有 `.pt` 权重，先导出 `.onnx`：

```bash
python yolo/export.py --weights weights/0414_qy++.pt --include onnx --imgsz 640 640
```

导出后会得到类似：

```text
weights/0414_qy++.onnx
```

然后再交给 realtime 入口：

```bash
python -m app.planning.realtime_pathplan --source test_input --weights weights/0414_qy++.onnx
```

注意：
- `app.inference.onnx_realtime` 明确要求输入是 `.onnx`
- 如果误传 `.pt`，代码会直接报错提醒

---

## 3.6 测试、格式化、检查

运行全部测试：

```bash
python -m pytest
```

只跑 realtime 相关测试：

```bash
python -m pytest tests/test_realtime_pathplan.py -q
```

格式化：

```bash
python -m black app tests utils
```

静态检查：

```bash
python -m ruff check app tests utils
```

---

## 4. 输出目录规则

## 4.1 分割输出

标准分割统一放在：

```text
runs/segment/exp
runs/segment/exp1
runs/segment/exp2
...
```

这里通常包含：
- `masks/`
- `labels/`
- YOLO 推理输出的其他文件

## 4.2 路径规划输出

路径规划统一放在：

```text
runs/pathplan/exp
runs/pathplan/exp1
runs/pathplan/exp2
...
```

这里现在只保存规划结果，例如：
- `*_planned.png`
- `*_planned.mp4`
- `obstacles.png`
- `planned.png`

不再额外复制 `masks`。

## 4.3 为什么会有 `exp`、`exp1`、`exp2`

当前命名规则是：
- 第一次输出：`exp`
- 后续依次：`exp1`、`exp2`、`exp3`

所以不是少了 `exp1`，而是第一轮本来就叫 `exp`。

---

## 5. 路径规划里的关键约定

- 目标类别：`0`
- 可通行但有代价的类别：`4`、`6`（tree、forest）
- 主要障碍物高度定义在：`app/mapping/grid_map.py`
- 路径规划算法在：`app/planning/dstar_lite.py`

当前含义是：
- `tree` / `forest` 会画出来
- 但不是绝对不可通行
- 规划器会优先绕开，实在绕不开时允许通过，并叠加代价

---

## 6. `app/` 目录总览

这一部分是现在最值得看的内容。后续如果要改业务逻辑，基本都从这里进。

### 6.1 `app/config.py`

项目默认配置集中点。

主要负责：
- 仓库根目录
- `runs/` 目录
- 默认输入目录
- 默认权重路径
- 默认设备
- 默认置信度阈值
- 默认栅格缩放

如果你想统一改默认行为，先看这里。

### 6.2 `app/paths.py`

路径解析工具。

主要负责：
- repo 根目录定位
- `data/`、`runs/`、`test_input/`、`weights/` 定位
- 环境变量覆盖默认路径
- 常见路径解析

如果你想改“默认从哪里读、往哪里写”，通常这里和 `config.py` 一起看。

---

## 7. `app/inference/` 总结

### 7.1 `app/inference/segmentation.py`

标准分割入口。

它的职责很明确：
- 拼装 YOLOv5 分割命令
- 调 `yolo/segment/predict.py`
- 把结果写到 `runs/segment/exp*`
- 需要时再把生成的 mask 读回来

适合：
- 离线图片/视频分割
- 保留原始 YOLO 输出目录
- 给 `path_planner.py` / `pathplan_batch.py` 提供磁盘版 mask

### 7.2 `app/inference/onnx_realtime.py`

ONNX 实时分割入口。

它不再调 subprocess，也不依赖磁盘中转，而是直接：
- 加载 `.onnx` 权重
- 对单帧 `numpy.ndarray` 做前处理
- 执行推理 + NMS + mask 重建
- 把结果转成规划层能直接消费的 `mask_entries`

这个文件是 realtime 链路的核心。

如果以后你要接：
- 摄像头
- 视频回传
- Web 服务
- 流式推理

主要就是沿着这个文件继续扩展。

---

## 8. `app/mapping/` 总结

### 8.1 `app/mapping/grid_map.py`

这是“分割结果变规划输入”的核心文件。

主要负责：
- 读取 mask 或内存版 `mask_entries`
- 把像素掩码压到栅格地图上
- 区分阻塞障碍物和可通行障碍物
- 计算目标点中心
- 记录障碍物高度、类别、代价

可以直接把它理解成：

```text
mask -> obstacle cells + target point + terrain penalty
```

无论是标准落盘链路还是 realtime 内存链路，最后都会对齐到这里的输入契约。

### 8.2 `app/mapping/octomap.py`

一个比较薄的包装层，目前只是复用 `GridMapHandler` 做掩码到障碍物的转换。

当前工程里真正的主逻辑还是在 `grid_map.py`。

---

## 9. `app/planning/` 总结

### 9.1 `app/planning/dstar_lite.py`

自定义 D* Lite 规划器。

主要负责：
- 栅格图上的路径搜索
- 障碍物膨胀
- 安全代价
- 可通行区域附加代价
- 起点/终点更新后的重规划

### 9.2 `app/planning/path_planner.py`

交互式规划入口。

主要特点：
- 默认读取 `runs/segment/` 下最新一轮 `exp*` 的 `masks/`
- 自动构建栅格图
- 自动找目标点
- Matplotlib 交互显示
- 支持鼠标改起点、终点、障碍物
- 同时保存当前规划图到 `runs/pathplan/exp*`

适合：
- 调试已有分割结果
- 看规划效果
- 手工补障碍物验证路径变化

### 9.3 `app/planning/pathplan_batch.py`

批量规划入口 + 公共渲染层。

它一方面负责批量处理图片/视频，另一方面也沉淀了 realtime 复用的公共函数。

这里最关键的是两个函数：
- `build_plan_result(...)`
- `render_plan_view(...)`

它们已经成了现在规划层的公共接口：
- 标准批处理可以复用
- realtime 入口也可以复用

### 9.4 `app/planning/realtime_pathplan.py`

新的实时路径规划入口。

主要负责：
- 打开图片、视频、目录、摄像头、流地址
- 调 `OnnxRealtimeSegmenter` 做分割
- 直接把内存版结果喂给 `build_plan_result(...)`
- 调 `render_plan_view(...)` 出最终规划图
- 按需显示、按需保存

可以把它理解成：

```text
输入编排层 + realtime 业务入口
```

如果你后面要做 ONNX 封装、接口化、实时视频回传，这个文件就是主入口。

---

## 10. `app/tooling/` 总结

### 10.1 `app/tooling/json_group_fix.py`

按 `label` 自动修正标注 JSON 里的 `group_id`。

命令：

```bash
python -m app.tooling.json_group_fix path/to/json_dir
```

### 10.2 `app/tooling/rename_images.py`

批量重命名图片。

默认先预览，不直接执行。

命令：

```bash
python -m app.tooling.rename_images path/to/images --prefix group6 --start-num 1 --padding 6
```

实际执行时加：

```bash
--execute
```

---

## 11. 其他目录说明

### `data/`
主要放数据配置 YAML，例如 `data/my.yaml`。

### `weights/`
主要放 `.pt` 或 `.onnx` 权重。

### `test_input/`
放本地测试图片、视频、文件夹。

### `tests/`
放自动化测试。当前已经有 realtime 相关测试，用来验证：
- `.onnx` 权重后缀检查
- 检测结果到 `mask_entries` 的转换
- 内存版 mask 是否能继续喂给规划与渲染

### `utils/`
目前只剩少量遗留辅助代码；当前业务主入口已经基本迁到 `app/`。

### `yolo/`
vendored YOLOv5 代码。

通常建议：
- 优先改 `app/`
- 不要在 `yolo/` 里做大范围重构
- 只有确实要改上游推理逻辑时再动它

---

## 12. 如果你现在要改代码，优先看哪里

### 想改标准分割入口
看：
- `app/inference/segmentation.py`
- `app/config.py`
- `app/paths.py`

### 想改 realtime / ONNX / 视频流
看：
- `app/inference/onnx_realtime.py`
- `app/planning/realtime_pathplan.py`
- `app/planning/pathplan_batch.py`

### 想改 mask 到栅格的规则
看：
- `app/mapping/grid_map.py`

### 想改路径规划策略
看：
- `app/planning/dstar_lite.py`

### 想改展示图样式
看：
- `app/planning/pathplan_batch.py`
- `app/planning/path_planner.py`

---

## 13. 一句话总结

如果只记住一句：

- **离线链路**：`segmentation.py -> runs/segment -> path_planner.py / pathplan_batch.py`
- **实时链路**：`onnx_realtime.py -> realtime_pathplan.py -> build_plan_result(...) -> render_plan_view(...)`

而现在整个项目真正需要长期维护的本地业务逻辑，重点都已经放在 `app/` 下面了。
