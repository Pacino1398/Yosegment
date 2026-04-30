# Yosegmentne

补齐了 **全 3D 路径规划（Phase 1）** 与 **3D 可视化 demo**，并保留原有 **局部 2.5D 建图与展示**。

### A) 局部 2.5D 建图与展示（OctoMap columns）

集中在 `app/mapping/octomap.py`：

- `octomap.py` 现在不再按“全局八叉树地图”来理解，当前更适合作为 **局部 2.5D 地图构建与可视化入口**。
- 默认会优先读取 `runs/segment/` 下最新一轮分割结果的 `masks/`，也可以手动传 `--mask-dir` 覆盖。
- 输入仍然复用现有分割结果 `masks`，底层继续走 `app.mapping.grid_map` 的 mask → 栅格投影逻辑。
- 建图规则按类别区分：
  - `tree / forest`：按树冠 footprint 建图，内部碰撞语义按 **向下膨胀 1 米** 处理。
  - `house`：按观测 footprint **直接拉到底部**。
  - 其他已支持障碍类：暂时按普通 grounded 柱体处理。
- 可视化效果：
  - 障碍物统一表现为 **从地面向上生长**；
  - 无人机显示在局部地图中心上空，默认高度 **15m**；
  - 展示重点是局部 2.5D 柱状图，不再走之前那种逐 voxel 的慢速显示方式。

命令行入口：

```bash
python -m app.mapping.octomap --mask-dir runs/segment/exp2/masks
```

可选：对每个 mask instance 在**像素域**做固定尺寸 tile 栅格化，并在导出的 snapshot 中附带稀疏矩形集合（xywh + coverage）：

```bash
python -m app.mapping.octomap --mask-dir runs/segment/exp2/masks --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3
```

- `tile-w-px / tile-h-px`：tile 宽高（单位：像素）
- `min-coverage`：tile 内前景像素占比阈值（0~1），低于该值的 tile 会被丢弃
- 导出字段名：`mask_instance_tiles`
  - 类型：`list[dict]`
  - 每个元素包含：`class_id / confidence / mask_index / image_stem / filename / tiles`
- `tiles` 为 `tuple[{x,y,w,h,coverage}]`（`x,y` 为 tile 左上角像素坐标）

#### A.1 pixel tiles 可视化（调试用）

如果你想直接查看某一张 mask 在像素域被切成了哪些 tile（绿色框为保留的 tile，黄色数字为 coverage），可以用：

```bash
python -m app.mapping.pixel_tiles_viz --mask-path runs/segment/exp2/masks/4_xxx.png --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3
```

也可以保存为图片：

```bash
python -m app.mapping.pixel_tiles_viz --mask-path runs/segment/exp2/masks/4_xxx.png --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3 --save runs/debug/tile_viz.png
```

如果你是在 `app/mapping/` 目录下直接执行脚本，也可以这样运行：

```bash
python octomap.py --mask-dir D:/qingyu/Yosegment/runs/segment/exp2/masks
```

### B) 全 3D 路径规划（D* Lite 3D + 26 邻域）

这一部分是当前工程从“局部 2.5D 柱状障碍物表达”平滑迈向“全 3D 体素规划”的关键。

新增文件/模块：

- `app/planning/dstar_lite_3d.py`
  - 3D D* Lite（state=(x,y,z)）
  - 26 邻域（dx,dy,dz ∈ {-1,0,1} 且不全为 0）
  - 3D Euclidean heuristic（Phase 1 直接用离散格单位）
  - 3D step cost（26 邻域仅 1/√2/√3 三类步长）
  - `update_vertex` 已升维到 (x,y,z)
  - **注意：目前 open list `U` 仍用 dict + `min(U)` 扫描**，适合小图验证，不适合 RK3588 长期运行

- `app/mapping/octomap_voxel_adapter.py`
  - 将 `app/mapping/octomap.py::OctoMap.columns`（2.5D 柱体：top_z/collision_base_z/terrain_penalty）适配为 `VoxelOccupancy`
  - 核心：预计算每个 (x,y) 的占用 z 区间 `ColumnVoxelRange[z_lo, z_hi)`，使得 `is_occupied((x,y,z))` 为 O(1)
  - penalty 目前按 (x,y) 投影复用 terrain_penalty（Phase 1 简化）

关键约定：

- 体素 z 区间语义为 **[z_lo, z_hi)**（半开区间），避免 `top_z` 为整数时出现边界层误判占用。
- 3D demo 起点高度由 `ManualHeightProvider(default_z=args.z0)` 给出；目标点默认落地（goal_z=0）。

#### B.1 工程现状分析（优缺点）

优点：

1) **迁移路径清晰、侵入性低**：
- `OctoMap` 继续维持 2.5D columns（易从语义分割的“高度表”构建）；
- 通过 `OctoMapVoxelAdapter` 提供统一的 `VoxelOccupancy.is_occupied()` 查询，把 3D 规划与建图解耦。

2) **3D 查询开销可控**：
- occupancy 查询通过区间 `[z_lo,z_hi)` 判定，避免了全体素存储与 3D 数组的内存开销。

3) **D* Lite 3D 已具备正确的升维骨架**：
- `neighbors/cost/heuristic/update_vertex` 均为 3D；
- `g/rhs` 采用惰性 dict（避免初始化 W*H*Z）。

不足/风险：

1) **核心性能瓶颈（RK3588 风险最大）**：
- `compute_path()` 里 `s = min(self.U, key=self.U.get)` 是 O(|U|) 扫描；3D 下 |U| 会显著变大（邻域从 8->26），这会成为首要瓶颈。

2) **启发式尺度未引入真实分辨率**：
- `heuristic()` 直接用离散格的欧氏距离；若未来 xy_resolution 与 z_resolution 不同（真实世界往往不同），需要改为“尺度化欧氏距离”。

3) **起点/终点高度策略仍偏 demo**：
- 当前 `DStarLite3D.__post_init__` 里 goal 强制 z=0；start 使用 height_source 给的 z，但接口使用上目前存在不一致（详见后续迁移建议）。

4) **3D 可视化 demo 仍存在 brute-force 扫描**：
- `pathplan_3d.py` / `pathplan_3d_dynamic.py` 的 occupied voxels 抽样是扫全空间（O(W*H*Z)），仅适合 PC 调试。

#### B.2 后续“全 3D 路径规划”演进思路（从 Phase 1 到可上板）

下面给出一个尽量“平滑迁移”的路线图：先保持现有 2.5D columns 建图不变，再逐步提升为真正的 3D 占用体（可接点云/深度）。

**Phase 2：把 3D D* Lite 做到可在 RK3588 上稳定跑（必须做）**

1) Open list 优化：
- 将 `U: dict[state->key]` + `min(U)` 扫描替换为 `heapq`（优先队列），并用“惰性删除/版本号”处理重复 key。
- 这是 3D 下最关键的性能改造点。

2) 邻域与代价进一步工程化：
- 26 邻域已实现，但建议把 motion 与 step_cost 预生成（目前已做 `_step_cost` 常量化）；
- 将 `neighbors()` 改为生成器/预分配 list，减少 Python 层对象分配。

3) 启发式与代价尺度化：
- 将 `heuristic()` 改为：
  - dx = (ax-bx)*xy_resolution
  - dy = (ay-by)*xy_resolution
  - dz = (az-bz)*z_resolution
  - h = sqrt(dx^2+dy^2+dz^2)
- 同理 `cost()` 的 base step cost 也应使用分辨率加权（特别是 z_resolution != xy_resolution 时）。

4) 约束/不可行性剪枝：
- 增加“飞行高度上下界”（比如 z_min/z_max 或动态层范围），避免无意义地搜索过高或过低层。
合理，初次规划路径之后，后续仅需要更新无人机所在高度附件几层即可。

**Phase 3：数据层真正升维（从 2.5D columns -> 真 3D occupancy）**

1) 从 columns 扩展为“多段占用区间”或“稀疏体素集合”
- 当前每个 (x,y) 只有一个区间（grounded 或 canopy 的单段）。真实 3D（点云/稠密障碍）可能出现多段占用。
- 可将 `_ranges[(x,y)]` 升级为 `list[range]` 或更紧凑的结构（例如两个区间：ground + canopy）。

2) 融合动态/增量更新
- `OctoMap` 有 `revision` 与 `update_columns()`，可作为增量更新入口；
- 后续可将“列变化”映射为“受影响体素”的增删，并在 D* Lite 中触发 `update_vertex`（更贴近 D* Lite 本来用于动态环境的优势）。

**Phase 4：上板性能路线（RK3588）**

可能的瓶颈点（按优先级）：

- P0：D* Lite open list 扫描（必须换 heap）
- P1：3D 26 邻域带来的 neighbors 扩张（每步 26 次 cost/occupied 检查）
- P1：Python dict 频繁 get/set（g/rhs/U）
- P2：occupied voxels 可视化扫描 O(W*H*Z)（上板应完全禁用或改为直接从 columns/ranges 生成点集）

建议优化顺序：
1) heapq + lazy delete
2) 将关键循环局部变量绑定（如把 `is_occupied`、`_g` 本地化），减少 attribute lookup
3) 若仍不够：考虑用 `numba`/Cython/pybind11 把 inner loop（neighbors+cost+update_vertex）下沉

> 提示：若你希望我后续继续把 Phase 2 的 heapq 版本落地，我建议先在 PC 上用 `tests/test_planner_3d.py` 跑通，再做性能 profiling（cProfile）。

### C) 3D 可视化 demo（matplotlib mplot3d）
先写在pathplan_3d 和pathplan_3d_dynamic就好

验证命令：

```bash
python -m pytest -q
python -m pytest -q tests/test_planner_3d.py
```

注意：当前 demo 的 3D 可视化为了“最小可用”，会对 voxel 空间做 brute-force 扫描采样，因此提供 `viz-*` 参数控制绘制体素数量；后续如要上板（RK3588）长期运行，建议改为直接从 column 生成点集（避免 O(W*H*Z) 扫描）。

---

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
python -m app.planning.path_planner --mask-dir runs/segment/exp2/masks
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
