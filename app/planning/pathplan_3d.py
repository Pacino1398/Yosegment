from __future__ import annotations

"""
3D 路径规划最小示例（Phase 1）

目标：
- 不破坏现有 2D 交互/批处理入口
- 提供一个可以直接运行的 3D 规划 demo：OctoMap(columns) -> VoxelOccupancy -> DStarLite3D -> 3D path

说明：
- z 语义：离散高度层（z_resolution=1.0）
- 起终点 z：由 ManualHeightProvider(default_z=15) 提供（可通过参数调整）
- 当前 demo 使用 GridMapHandler 的 masks->obs 同步状态来构建 OctoMap columns（与现有工程数据一致）
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from app.planning.space3d import ManualHeightProvider
from app.planning.pathplanbatch import get_latest_segmentation_run_dir
from app.planning.space3d import GridSpec3D


def get_default_mask_dir() -> Path:
    latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
    return (latest_run_dir / "masks").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D D* Lite demo: build OctoMap and plan in full 3D voxels.")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files.")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="Pixel-to-grid scale.")
    parser.add_argument("--z0", type=int, default=12, help="Default cruise height layer (start_z=goal_z=z0).")
    parser.add_argument("--z-max-cap", type=int, default=0, help="If >0, cap z_max to this value to control search size.")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum path extraction steps.")

    parser.add_argument(
        "--viz",
        action="store_true",
        help="Show 3D visualization (matplotlib mplot3d): occupied voxels + planned path.",
    )
    parser.add_argument(
        "--viz-max-voxels",
        type=int,
        default=60000,
        help="Hard cap of drawn occupied voxels (avoid rendering too slow).",
    )
    parser.add_argument(
        "--viz-skip",
        type=int,
        default=1,
        help="Draw every N-th voxel (1=all). Increase to speed up rendering.",
    )
    parser.add_argument(
        "--viz-alpha",
        type=float,
        default=0.10,
        help="Voxel point alpha (transparency).",
    )
    return parser.parse_args()


def _visualize_3d(
    occupancy: OctoMapVoxelAdapter,
    path3d: list[tuple[int, int, int]],
    start: tuple[int, int, int] | None,
    goal: tuple[int, int, int] | None,
    max_voxels: int,
    skip: int,
    alpha: float,
) -> None:
    grid = occupancy.grid
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []

    # brute-force scan (Phase 1 demo)：对大空间会慢，因此提供 max_voxels/skip 限制
    sampled = 0
    for x in range(grid.x_max):
        for y in range(grid.y_max):
            for z in range(grid.z_max):
                if skip > 1 and ((x + y + z) % skip != 0):
                    continue
                if occupancy.is_occupied((x, y, z)):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    sampled += 1
                    if sampled >= max_voxels:
                        break
            if sampled >= max_voxels:
                break
        if sampled >= max_voxels:
            break

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D D* Lite Demo (occupied voxels + path)")

    if xs:
        ax.scatter(xs, ys, zs, s=2, c="k", alpha=float(alpha), depthshade=False, label="occupied")

    if path3d:
        p = np.asarray(path3d, dtype=np.int32)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], c="#06E706", linewidth=2.5, label="path")

    if start is not None:
        ax.scatter([start[0]], [start[1]], [start[2]], c="#015E01", s=60, label="start")
    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], [goal[2]], c="red", s=60, label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, grid.x_max)
    ax.set_ylim(0, grid.y_max)
    ax.set_zlim(0, grid.z_max)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _snap_to_free_cell(
    occupancy: OctoMapVoxelAdapter,
    preferred_xy: tuple[int, int],
    z: int,
    r_max: int = 8,
) -> tuple[int, int, int]:
    """
    在固定 z 层上，从 preferred_xy 开始向外搜索最近的 free voxel。
    目的：避免 start/goal 落在占用体素里导致 3D 路径直接为空。
    """
    grid: GridSpec3D = occupancy.grid
    x0, y0 = preferred_xy
    x0 = min(max(0, int(x0)), grid.x_max - 1)
    y0 = min(max(0, int(y0)), grid.y_max - 1)
    z = min(max(0, int(z)), grid.z_max - 1)

    if not occupancy.is_occupied((x0, y0, z)):
        return (x0, y0, z)

    for r in range(1, int(r_max) + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                x = x0 + dx
                y = y0 + dy
                if x < 0 or x >= grid.x_max or y < 0 or y >= grid.y_max:
                    continue
                if not occupancy.is_occupied((x, y, z)):
                    return (x, y, z)

    return (x0, y0, z)


def main() -> None:
    args = parse_args()
    mask_dir = resolve_path(args.mask_dir, get_default_mask_dir())
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    # 1) Build 2D grid handler state from masks (same as existing pipeline)
    # 2) Sync into OctoMap so we have columns/top_z/collision_base_z for voxelization
    first_mask = None
    for mask_file in sorted(Path(mask_dir).glob("*.png")):
        first_mask = mask_file
        break
    if first_mask is None:
        raise FileNotFoundError(f"no mask png found in: {mask_dir}")

    # Infer grid size via GridMapHandler inputs
    # NOTE: We reuse OctoMap.infer_grid_size to keep consistent with existing code paths.
    grid_w, grid_h = OctoMap.infer_grid_size(mask_dir, args.grid_scale)
    grid_handler = GridMapHandler(grid_w=grid_w, grid_h=grid_h, grid_scale=args.grid_scale)
    mask_entries = load_mask_entries(mask_dir, grid_handler)
    _obs, target_point = grid_handler.batch_masks_to_obs(mask_entries)

    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=args.grid_scale)
    octomap.grid_handler = grid_handler
    octomap._sync_from_grid_handler()  # populate obstacle_heights/class_ids/penalties
    octomap.build_octomap(_obs)  # build columns/max_z

    # 3) Voxel occupancy
    z_max_cap = None if args.z_max_cap <= 0 else int(args.z_max_cap)
    occupancy = OctoMapVoxelAdapter(octomap, z_resolution=1.0, z_max_cap=z_max_cap)

# 4) 3D D* Lite
    raw_start_x, raw_start_y = grid_w // 2, grid_h // 2
    if target_point is not None:
        raw_goal_x, raw_goal_y = target_point[0], target_point[1]
    else:
        raw_goal_x, raw_goal_y = grid_w - 5, grid_h - 4

    start_xy = (
        min(max(0, int(raw_start_x)), grid_w - 1),
        min(max(0, int(raw_start_y)), grid_h - 1)
    )
    goal_xy = (
        min(max(0, int(raw_goal_x)), grid_w - 1),
        min(max(0, int(raw_goal_y)), grid_h - 1)
    )

    height_source = ManualHeightProvider(default_z=int(args.z0))

    # 4.1) 先让规划器按“高度策略”生成 start_z（goal_z 在规划器内部强制为 0）
    planner = DStarLite3D(
        start_xy=start_xy,
        goal_xy=goal_xy,
        occupancy=occupancy,
        height_source=height_source,
    )

    # 4.2) 若 start 在占用体素中，自动 snap 到同一 z 层最近 free voxel（确保能出 3D 路径）
    snapped_start = _snap_to_free_cell(occupancy, preferred_xy=start_xy, z=planner.start[2], r_max=10)
    if snapped_start != planner.start:
        planner.update_start((snapped_start[0], snapped_start[1]))
        # 强制写回 z（update_start 会走 height_source，这里保持同一 z 层）
        planner.start = snapped_start

    print("-" * 50)
    print(f"[DEBUG] Grid: {grid_w}x{grid_h} | Z0 Layer: {height_source.get_z(start_xy)}")
    print(f"[DEBUG] Final Start: {planner.start} | Occupied: {occupancy.is_occupied(planner.start)}")
    print(f"[DEBUG] Final Goal:  {planner.goal} | Occupied: {occupancy.is_occupied(planner.goal)}")
    print("-" * 50)

    path3d = planner.plan(max_steps=int(args.max_steps))
    print(f"grid: {grid_w}x{grid_h} z_max={occupancy.grid.z_max}")
    print(f"start: {planner.start} goal: {planner.goal}")
    print(f"path length: {len(path3d)}")
    if path3d:
        print("path head:", path3d[: min(5, len(path3d))])
        print("path tail:", path3d[max(0, len(path3d) - 5) :])

    if bool(args.viz):
        _visualize_3d(
            occupancy,
            path3d,
            start=planner.start,
            goal=planner.goal,
            max_voxels=int(args.viz_max_voxels),
            skip=max(1, int(args.viz_skip)),
            alpha=float(args.viz_alpha),
        )


if __name__ == "__main__":
    main()

# python -m app.planning.pathplan_3d --viz