from __future__ import annotations

"""
OctoMap(2.5D columns) -> 3D voxel occupancy adapter (Phase 1)

目的：
- 将 app/mapping/octomap.py::OctoMap.columns 的“柱体”表达，转换为 3D 体素查询接口 VoxelOccupancy
- z 语义：离散高度层（z_resolution=1.0），与 ColumnState.top_z / collision_base_z 对齐

实现要点：
- 预计算每个 (x, y) 的占用 z 区间（grounded/canopy）
- 查询 is_occupied((x, y, z)) 时只做 O(1) 的区间比较
- penalty((x, y, z)) 先按 (x, y) 投影复用 terrain_penalty（Phase 1 简化策略）
"""

from dataclasses import dataclass
from typing import Mapping

from app.mapping.octomap import ColumnState, OctoMap
from app.planning.space3d import Cell2D, Cell3D, GridSpec3D, VoxelOccupancy, compute_z_max_from_height


def _ceil_to_int(value: float) -> int:
    # 不引入 math 的 ceil：对正数可用 int(x + (1 - eps))
    if value <= 0:
        return 0
    return int(value + 0.999999)


def _floor_to_int(value: float) -> int:
    if value <= 0:
        return 0
    return int(value)


@dataclass(frozen=True, slots=True)
class ColumnVoxelRange:
    """
    将 ColumnState 转为离散体素占用区间 [z_lo, z_hi)。
    """

    z_lo: int
    z_hi: int
    terrain_penalty: float = 0.0

    def contains(self, z: int) -> bool:
        return self.z_lo <= z < self.z_hi


class OctoMapVoxelAdapter(VoxelOccupancy):
    """
    OctoMap -> VoxelOccupancy 适配器。

    约定：
    - x/y 范围使用 OctoMap.grid_w/grid_h
    - z_max 默认由 OctoMap.max_z 离散化得到（z_resolution=1.0），可通过 z_max_cap 限制空间规模
    """

    def __init__(
        self,
        octomap: OctoMap,
        *,
        z_resolution: float = 1.0,
        z_max_cap: int | None = None,
        min_layers: int = 1,
        xy_resolution: float = 1.0,
    ):
        self._octomap = octomap
        z_max = compute_z_max_from_height(octomap.max_z, z_resolution=z_resolution, min_layers=min_layers)
        if z_max_cap is not None:
            z_max = min(z_max, int(z_max_cap))
        # z_max 表示“层数”，有效 z 为 [0, z_max)。为确保 top_z 恰好为整数时，
        # 区间 [0, top_z) 中的最高 z=top_z-1 仍然可表达，需要至少 top_z+1 层。
        z_max = max(int(z_max), _ceil_to_int(octomap.max_z) + 1)

        self.grid = GridSpec3D(
            x_max=int(octomap.grid_w),
            y_max=int(octomap.grid_h),
            z_max=int(z_max),
            xy_resolution=float(xy_resolution),
            z_resolution=float(z_resolution),
        )

        self._ranges: dict[Cell2D, ColumnVoxelRange] = self._build_ranges(octomap.columns)

    def _build_ranges(self, columns: Mapping[Cell2D, ColumnState]) -> dict[Cell2D, ColumnVoxelRange]:
        ranges: dict[Cell2D, ColumnVoxelRange] = {}
        for cell_xy, col in columns.items():
            # 将连续高度转离散层
            top = _ceil_to_int(col.top_z)
            if top <= 0:
                continue

            base = 0
            if getattr(col, "mode", "grounded") == "canopy":
                base = _floor_to_int(col.collision_base_z)
            else:
                base = _floor_to_int(getattr(col, "collision_base_z", 0.0))

            z_lo = max(0, min(base, self.grid.z_max - 1))
            z_hi = max(0, min(top, self.grid.z_max))
            if z_hi <= z_lo:
                continue

            ranges[cell_xy] = ColumnVoxelRange(
                z_lo=z_lo,
                z_hi=z_hi,
                terrain_penalty=float(getattr(col, "terrain_penalty", 0.0) or 0.0),
            )
        return ranges

    def is_occupied(self, cell: Cell3D) -> bool:
        x, y, z = cell
        if not (0 <= x < self.grid.x_max and 0 <= y < self.grid.y_max and 0 <= z < self.grid.z_max):
            return True  # 越界视为不可通行
        r = self._ranges.get((x, y))
        if r is None:
            return False
        return r.contains(z)

    def penalty(self, cell: Cell3D) -> float:
        x, y, z = cell
        if not (0 <= x < self.grid.x_max and 0 <= y < self.grid.y_max and 0 <= z < self.grid.z_max):
            return 0.0
        r = self._ranges.get((x, y))
        if r is None:
            return 0.0
        # Phase 1：按 (x,y) 投影复用 terrain_penalty，不随 z 变化
        return float(r.terrain_penalty)