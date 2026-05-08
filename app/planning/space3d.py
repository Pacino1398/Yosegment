from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple

Cell2D = tuple[int, int]
Cell3D = tuple[int, int, int]

__all__ = [
    "Cell2D",
    "Cell3D",
    "GridSpec3D",
    "VoxelOccupancy",
    "HeightSource",
    "GlobalSafeHeightProvider",
    "ManualHeightProvider",
    "compute_z_max_from_height",
    "main",
]


@dataclass(frozen=True, slots=True)
class GridSpec3D:

    x_max: int
    y_max: int
    z_max: int
    xy_resolution: float = 1.0
    z_resolution: float = 1.0

    def in_bounds(self, cell: Cell3D) -> bool:
        x, y, z = cell
        return 0 <= x < self.x_max and 0 <= y < self.y_max and 0 <= z < self.z_max

    def clamp(self, cell: Cell3D) -> Cell3D:
        x, y, z = cell
        x = min(max(x, 0), self.x_max - 1)
        y = min(max(y, 0), self.y_max - 1)
        z = min(max(z, 0), self.z_max - 1)
        return x, y, z

    def to_metric(self, cell: Cell3D) -> Tuple[float, float, float]:
        """将离散 cell 转为“尺度化坐标”（单位由 resolution 决定）。"""
        x, y, z = cell
        return x * self.xy_resolution, y * self.xy_resolution, z * self.z_resolution


class VoxelOccupancy(Protocol):
    """
    体素查询接口（仅定义协议）。

    说明：
    - D* Lite 3D 的 cost/neighbor 检查将依赖该接口判断 voxel 是否可通行
    - 实现来自 OctoMap columns 的 adapter（app/planning/octomap_voxel_adapter.py）
    """

    grid: GridSpec3D

    def is_occupied(self, cell: Cell3D) -> bool:
        """cell 为离散体素坐标。True 表示不可通行。"""
        ...

    def penalty(self, cell: Cell3D) -> float:
        """可选代价项（地形/安全/风险等）。默认可返回 0.0。"""
        ...


class HeightSource(Protocol):
    """
    起点/终点高度（z）来源协议。

    用途：
    - 将“默认巡航高度 z=15”这类策略从规划算法中解耦出来，便于后续接入真实高度/传感器/任务策略。
    - 规划器在生成 start/goal 的 (x, y, z) 时，通过该接口获取 z。
    """

    def get_z(self, cell_xy: Cell2D, occupancy: VoxelOccupancy) -> int:
        ...


@dataclass(frozen=True, slots=True)
class GlobalSafeHeightProvider:
    """
    全局安全高度 provider：飞在整张地图所有障碍物顶端之上（max_z + margin）。

    说明：
- 该策略与 (x,y) 无关，输出一个“全局安全高度”，适合快速 demo/保守避障
- 若后续要做更激进的贴地/穿行策略，可替换为局部高度策略
    """

    margin_layers: int = 1
    min_z: int = 0

    def get_z(self, cell_xy: Cell2D, occupancy: VoxelOccupancy) -> int:
        # OctoMapVoxelAdapter 上通常会挂载 .octomap
        max_terrain_z = 0
        if hasattr(occupancy, "octomap") and getattr(occupancy, "octomap") is not None:
            max_terrain_z = int(getattr(occupancy.octomap, "max_z", 0))
        safe_z = max(self.min_z, max_terrain_z + int(self.margin_layers))
        return min(int(safe_z), int(occupancy.grid.z_max) - 1)


@dataclass(frozen=True, slots=True)
class ManualHeightProvider:
    """
    固定高度 provider（与 occupancy 无关）。

    注意：为兼容 DStarLite3D（会调用 get_z(cell_xy, occupancy)），这里接受 occupancy 参数但忽略它。
    """

    default_z: int = 10

    def get_z(self, cell_xy: Cell2D, occupancy: VoxelOccupancy | None = None) -> int:
        return int(self.default_z)


def compute_z_max_from_height(max_z: float, z_resolution: float = 1.0, min_layers: int = 1) -> int:
    """
    将 OctoMap 的 max_z（连续高度）转换为离散 z_max（层数）。

    - max_z: 来自 OctoMap.max_z（例如 ColumnState.top_z 的最大值）
    - z_resolution: 每层高度
    - 返回的 z_max 至少为 min_layers（避免 0 层空间）
    """
    if z_resolution <= 0:
        raise ValueError("z_resolution must be > 0")

    layers = int((max_z / z_resolution) + 0.999999)  # ceil without importing math
    return max(min_layers, layers)


def _sample_occupied_voxels(
    occupancy: VoxelOccupancy,
    max_voxels: int,
    skip: int,
    surface_only: bool,
) -> tuple[list[int], list[int], list[int]]:
    """
    采样占用体素点云。

    surface_only=True 时只采样“外表面”体素，避免把障碍物内部全部画出来导致过于密集：
    - 若当前 occupied voxel 的 6-邻域（面邻域）里存在 free/out-of-bounds，则认为它在表面上
    """
    grid = occupancy.grid
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    sampled = 0

    neighbors6 = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    for x in range(grid.x_max):
        for y in range(grid.y_max):
            for z in range(grid.z_max):
                if skip > 1 and ((x + y + z) % skip != 0):
                    continue
                if not occupancy.is_occupied((x, y, z)):
                    continue

                if surface_only:
                    is_surface = False
                    for dx, dy, dz in neighbors6:
                        nb = (x + dx, y + dy, z + dz)
                        if (not grid.in_bounds(nb)) or (not occupancy.is_occupied(nb)):
                            is_surface = True
                            break
                    if not is_surface:
                        continue

                xs.append(x)
                ys.append(y)
                zs.append(z)
                sampled += 1
                if sampled >= max_voxels:
                    return xs, ys, zs

    return xs, ys, zs


def _visualize_3d(
    occupancy: VoxelOccupancy,
    max_voxels: int,
    skip: int,
    alpha: float,
    point_size: float,
    surface_only: bool,
) -> None:
    import matplotlib.pyplot as plt

    xs, ys, zs = _sample_occupied_voxels(
        occupancy,
        max_voxels=int(max_voxels),
        skip=max(1, int(skip)),
        surface_only=bool(surface_only),
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Map Convert (occupied voxels)")

    if xs:
        ax.scatter(
            xs,
            ys,
            zs,
            s=float(point_size),
            c="k",
            alpha=float(alpha),
            depthshade=False,
            label="occupied",
        )

    grid = occupancy.grid
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, grid.x_max)
    ax.set_ylim(0, grid.y_max)
    ax.set_zlim(0, grid.z_max)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _get_default_mask_dir():
    from app.config import DEFAULT_CONFIG
    from app.planning.pathplanbatch import get_latest_segmentation_run_dir

    latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
    return (latest_run_dir / "masks").resolve()


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Convert masks -> OctoMap -> 3D voxel occupancy visualization (like pathplan_3d)."
    )
    p.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files.")
    p.add_argument("--grid-scale", type=int, default=None, help="Pixel-to-grid scale (default from config).")
    p.add_argument("--z-resolution", type=float, default=1.0, help="Z resolution (layer height).")
    p.add_argument("--z-max-cap", type=int, default=0, help="If >0, cap z_max to this value to control scan size.")

    p.add_argument(
        "--viz",
        dest="viz",
        action="store_true",
        default=True,
        help="Show 3D visualization (default: on). Use --no-viz to disable.",
    )
    p.add_argument(
        "--no-viz",
        dest="viz",
        action="store_false",
        help="Disable visualization.",
    )

    # 体素块形态
    p.add_argument("--viz-max-voxels", type=int, default=1000000, help="Hard cap of drawn occupied voxels.")
    p.add_argument("--viz-skip", type=int, default=1, help="Draw every N-th voxel.")
    p.add_argument("--viz-alpha", type=float, default=1.0, help="Voxel point alpha.")
    p.add_argument("--viz-point-size", type=float, default=2.0, help="Voxel point size for scatter.")
    p.add_argument(
        "--viz-surface-only",
        action="store_true",
        default=True,
        help="Draw only surface voxels."
    )

    p.add_argument(
        "--viz-solid",
        dest="viz_surface_only",
        action="store_false",
        help="Draw solid voxels."
    )
    return p.parse_args()


def main() -> None:
    """
    入口：把 masks 转成 OctoMap，再转成 3D voxel occupancy，并可视化。

    用法：
      python -m app.planning.space3d --viz
    """
    from pathlib import Path

    from app.config import DEFAULT_CONFIG
    from app.mapping.grid_map import GridMapHandler, load_mask_entries
    from app.mapping.octomap import OctoMap
    from app.paths import resolve_path
    from app.mapping.octomap_voxel_adapter import OctoMapVoxelAdapter

    args = _parse_args()
    mask_dir = resolve_path(args.mask_dir, _get_default_mask_dir())
    if not Path(mask_dir).exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    grid_scale = DEFAULT_CONFIG.default_grid_scale if args.grid_scale is None else int(args.grid_scale)

    grid_w, grid_h = OctoMap.infer_grid_size(Path(mask_dir), grid_scale)
    grid_handler = GridMapHandler(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)
    mask_entries = load_mask_entries(Path(mask_dir), grid_handler)
    obs, _target_point = grid_handler.batch_masks_to_obs(mask_entries)

    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)
    octomap.grid_handler = grid_handler
    octomap._sync_from_grid_handler()
    octomap.build_octomap(obs)

    z_max_cap = None if int(args.z_max_cap) <= 0 else int(args.z_max_cap)
    occupancy = OctoMapVoxelAdapter(octomap, z_resolution=float(args.z_resolution), z_max_cap=z_max_cap)

    print(f"[space3d] grid: {grid_w}x{grid_h} z_max={occupancy.grid.z_max} max_z={getattr(octomap, 'max_z', None)}")

    if bool(args.viz):
        _visualize_3d(
            occupancy,
            max_voxels=int(args.viz_max_voxels),
            skip=int(args.viz_skip),
            alpha=float(args.viz_alpha),
            point_size=float(args.viz_point_size),
            surface_only=bool(args.viz_surface_only),
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    main()

# python -m app.planning.space3d --viz