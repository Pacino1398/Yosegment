from __future__ import annotations
"""
3D D* Lite 动态规划演示（交互式） - 性能满血优化版

- 立体 3D 展示：occupied voxels（黑点） + 3D path（绿线） + start（绿点） + goal（红点）
- 交互方式：
  - 中键点击：设置 Start 的 (x, y)，z 由高度策略给出，并自动将点 snap 到附近 free voxel
  - 右键点击：设置 Goal 的 (x, y)，强制在 z=0 层进行 xy 扩散寻找 free voxel
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 允许 VSCode 直接运行本文件（不要求 python -m）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.mapping.grid_map import GridMapHandler, load_mask_entries
from app.mapping.octomap import OctoMap
from app.paths import resolve_path
from app.planning.dstar_lite_3d import DStarLite3D
from app.mapping.octomap_voxel_adapter import OctoMapVoxelAdapter
from app.planning.pathplan_batch import get_latest_segmentation_run_dir
# 提前引入，避免在交互事件中反复触发模块检索
from app.planning.space3d import ManualHeightProvider


def get_default_mask_dir() -> Path:
    latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
    return (latest_run_dir / "masks").resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive 3D D* Lite demo (3D view, middle=start, right=goal).")
    p.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files.")
    p.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="Pixel-to-grid scale.")
    p.add_argument("--z0", type=int, default=12, help="Default cruise height layer for start.")
    p.add_argument("--z-max-cap", type=int, default=0, help="If >0, cap z_max to this value to control search size.")
    p.add_argument("--max-steps", type=int, default=8000, help="Maximum path extraction steps.")
    p.add_argument("--viz-max-voxels", type=int, default=60000, help="Hard cap of drawn occupied voxels.")
    p.add_argument("--viz-skip", type=int, default=2, help="Draw every N-th voxel (1=all).")
    p.add_argument("--viz-alpha", type=float, default=0.10, help="Voxel point alpha (transparency).")
    return p.parse_args()


def _snap_to_free_cell_xyz(
    occupancy: OctoMapVoxelAdapter,
    preferred_xyz: tuple[int, int, int],
    r_xy_max: int = 10,
    z_search: int = 2,
) -> tuple[int, int, int]:
    """稳如老狗的空闲点寻找算法"""
    grid = occupancy.grid
    x0 = min(max(0, int(preferred_xyz[0])), grid.x_max - 1)
    y0 = min(max(0, int(preferred_xyz[1])), grid.y_max - 1)
    z0 = min(max(0, int(preferred_xyz[2])), grid.z_max - 1)

    if not occupancy.is_occupied((x0, y0, z0)):
        return (x0, y0, z0)

    # 1. 优先只在 Z 轴上下找 (不改变 XY)
    if z_search > 0:
        for dz in range(1, z_search + 1):
            for sign in (-1, 1):
                z_test = z0 + sign * dz
                if 0 <= z_test < grid.z_max and not occupancy.is_occupied((x0, y0, z_test)):
                    return (x0, y0, z_test)

    # 2. 生成安全的 Z 候选层
    z_candidates = [z0]
    if z_search > 0:
        for dz in range(1, z_search + 1):
            if 0 <= z0 - dz < grid.z_max: z_candidates.append(z0 - dz)
            if 0 <= z0 + dz < grid.z_max: z_candidates.append(z0 + dz)

    # 3. 经典的同心方环 XY 扩散 (放弃复杂语法，追求 100% 稳定)
    for r in range(1, r_xy_max + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                # 只查边缘一圈，内部已经查过
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < grid.x_max and 0 <= ny < grid.y_max:
                    for z in z_candidates:
                        if not occupancy.is_occupied((nx, ny, z)):
                            return (nx, ny, z)

    return (x0, y0, z0)


def _sample_occupied_voxels(
    occupancy: OctoMapVoxelAdapter,
    max_voxels: int,
    skip: int,
) -> tuple[list[int], list[int], list[int]]:
    """使用 numpy 和 itertools 大幅提升大地图体素抽取速度"""
    grid = occupancy.grid
    
    # 巧妙利用 Python itertools 避免多重函数调用开销
    coords = []
    sampled = 0
    
    # 生成基础坐标流，直接计算 skip
    coord_stream = itertools.product(range(grid.x_max), range(grid.y_max), range(grid.z_max))
    
    for x, y, z in coord_stream:
        if skip > 1 and ((x + y + z) % skip != 0):
            continue
        if occupancy.is_occupied((x, y, z)):
            coords.append((x, y, z))
            sampled += 1
            if sampled >= max_voxels:
                break
                
    if not coords:
        return [], [], []
        
    # 利用 Numpy 高速切片
    arr = np.array(coords, dtype=np.int32)
    return arr[:, 0].tolist(), arr[:, 1].tolist(), arr[:, 2].tolist()


class PathPlanner3DDynamic:
    def __init__(self, mask_dir: Path, grid_scale: int, z0: int, z_max_cap: int, 
                 max_steps: int, viz_max_voxels: int, viz_skip: int, viz_alpha: float):
        self.mask_dir = Path(mask_dir)
        self.grid_scale = int(grid_scale)
        self.z0 = int(z0)
        self.z_max_cap = int(z_max_cap)
        self.max_steps = int(max_steps)
        self.viz_max_voxels = int(viz_max_voxels)
        self.viz_skip = max(1, int(viz_skip))
        self.viz_alpha = float(viz_alpha)

        self.grid_w, self.grid_h = OctoMap.infer_grid_size(self.mask_dir, self.grid_scale)

        self.grid_handler = GridMapHandler(grid_w=self.grid_w, grid_h=self.grid_h, grid_scale=self.grid_scale)
        mask_entries = load_mask_entries(self.mask_dir, self.grid_handler)
        obs, target_point = self.grid_handler.batch_masks_to_obs(mask_entries)

        self.octomap = OctoMap(grid_w=self.grid_w, grid_h=self.grid_h, grid_scale=self.grid_scale)
        self.octomap.grid_handler = self.grid_handler
        self.octomap._sync_from_grid_handler()
        self.octomap.build_octomap(obs)

        cap = None if self.z_max_cap <= 0 else int(self.z_max_cap)
        self.occupancy = OctoMapVoxelAdapter(self.octomap, z_resolution=1.0, z_max_cap=cap)

        self.start_xy = (self.grid_w // 2, self.grid_h // 2)
        self.goal_xy = (int(target_point[0]), int(target_point[1])) if target_point is not None else (self.grid_w - 5, self.grid_h - 5)

        self.planner: DStarLite3D | None = None
        self.path3d: list[tuple[int, int, int]] = []

        self.fig, self.ax = None, None
        self.path_artist, self.start_artist, self.goal_artist = None, None, None

        self._init_planner()
        self._init_plot()
        self.replan()
        self.update_plot()

    def _init_planner(self) -> None:
        height_source = ManualHeightProvider(default_z=self.z0)
        
        # Start 处理：允许 Z 轴小幅微调以确保起始合法
        safe_xyz = _snap_to_free_cell_xyz(self.occupancy, (self.start_xy[0], self.start_xy[1], self.z0), r_xy_max=12, z_search=2)
        self.start_xy = (safe_xyz[0], safe_xyz[1])

        self.planner = DStarLite3D(
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            occupancy=self.occupancy,
            height_source=height_source,
        )

        # 二次校验
        s = self.planner.start
        if self.occupancy.is_occupied(s):
            safe_s = _snap_to_free_cell_xyz(self.occupancy, s, r_xy_max=12, z_search=2)
            if safe_s != s:
                self.start_xy = (safe_s[0], safe_s[1])
                self.planner.update_start(self.start_xy)
                self.planner.start = safe_s

    def replan(self) -> None:
        if self.planner is None:
            self.path3d = []
            return
        try:
            self.path3d = self.planner.plan(max_steps=self.max_steps)
        except Exception as e:
            print(f"[Warning] Planning failed: {e}")
            self.path3d = []

    def _init_plot(self) -> None:
        if self.planner is None:
            return

        plt.rcParams["figure.facecolor"] = "white"
        xs, ys, zs = _sample_occupied_voxels(self.occupancy, max_voxels=self.viz_max_voxels, skip=self.viz_skip)

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("3D D* Lite Dynamic Demo\nMid: Set Start (Fly) | Right: Set Goal (Ground)")

        if xs:
            self.ax.scatter(xs, ys, zs, s=2, c="k", alpha=float(self.viz_alpha), depthshade=False, label="Occupied")

        (self.path_artist,) = self.ax.plot([], [], [], c="#06E706", linewidth=2.5, label="Path")
        
        s, g = self.planner.start, self.planner.goal
        self.start_artist = self.ax.scatter([s[0]], [s[1]], [s[2]], c="#079107", s=60, label="Start")
        self.goal_artist = self.ax.scatter([g[0]], [g[1]], [g[2]], c="red", s=60, label="Goal")

        grid = self.occupancy.grid
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(0, grid.x_max)
        self.ax.set_ylim(0, grid.y_max)
        self.ax.set_zlim(0, grid.z_max)
        self.ax.legend(loc="upper right")
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        plt.tight_layout()

    def update_plot(self) -> None:
        if not all([self.fig, self.ax, self.planner, self.path_artist]):
            return

        if self.path3d:
            p = np.asarray(self.path3d, dtype=np.int32)
            self.path_artist.set_data(p[:, 0], p[:, 1])
            self.path_artist.set_3d_properties(p[:, 2])
        else:
            self.path_artist.set_data([], [])
            self.path_artist.set_3d_properties([])

        s, g = self.planner.start, self.planner.goal
        if self.start_artist:
            self.start_artist._offsets3d = ([s[0]], [s[1]], [s[2]])
        if self.goal_artist:
            self.goal_artist._offsets3d = ([g[0]], [g[1]], [g[2]])

        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        # 1. 基础防护
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        # --- 调试代码：如果还是 (0,0)，请看终端打印的 raw 值是多少 ---
        # print(f"DEBUG Raw Click: x={event.xdata:.2f}, y={event.ydata:.2f}")

        # 2. 转换坐标 (使用 round 减轻 int 强制截断导致的 0 偏置)
        x, y = int(round(event.xdata)), int(round(event.ydata))
        
        # 3. 边界约束，防止点到坐标轴外面去
        x = max(0, min(x, self.grid_w - 1))
        y = max(0, min(y, self.grid_h - 1))

        # ==========================================
        # 中键 (Button 2): 设置起点
        # ==========================================
        if event.button == 2:
            # 保持当前的 Z 轴高度进行平移
            current_z = self.planner.start[2] if self.planner else self.z0
            preferred = (x, y, current_z)
            safe = _snap_to_free_cell_xyz(self.occupancy, preferred, r_xy_max=12, z_search=2)
            
            self.start_xy = (safe[0], safe[1])
            if self.planner:
                self.planner.update_start(self.start_xy)
                self.planner.start = safe
            print(f"[Start] Set to {safe}")

        # ==========================================
        # 右键 (Button 3): 设置终点
        # ==========================================
        elif event.button == 3:
            # 强制探测地面 (Z=0) 的空位
            safe_g = _snap_to_free_cell_xyz(self.occupancy, (x, y, 0), r_xy_max=15, z_search=0)
            self.goal_xy = (safe_g[0], safe_g[1])

            # Phase 2：goal 变化仍采用简单重建（路径实时更新优先）
            hp = ManualHeightProvider(default_z=self.z0)
            self.planner = DStarLite3D(self.start_xy, self.goal_xy, self.occupancy, hp)
            self.planner.goal = safe_g
            print(f"[Goal] Set to {safe_g}")
        else:
            return

        # 4. 关键：计算并【强制】通知画布更新
        self.replan()
        self.update_plot()
        
        # 如果是交互模式，必须调用这个才能看到变化
        if self.fig:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()  
    
    
    def run(self) -> None:
        print(f"\n3D 动态规划演示\nMask 来源: {self.mask_dir}")
        plt.show(block=True)


def main() -> None:
    args = parse_args()
    mask_dir = resolve_path(args.mask_dir, get_default_mask_dir())
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    demo = PathPlanner3DDynamic(
        mask_dir=mask_dir,
        grid_scale=args.grid_scale,
        z0=args.z0,
        z_max_cap=args.z_max_cap,
        max_steps=args.max_steps,
        viz_max_voxels=args.viz_max_voxels,
        viz_skip=args.viz_skip,
        viz_alpha=args.viz_alpha,
    )
    demo.run()


if __name__ == "__main__":
    main()
# python -m app.planning.pathplan_3d_dynamic