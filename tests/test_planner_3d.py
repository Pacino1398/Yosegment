from __future__ import annotations

from dataclasses import dataclass

from app.mapping.octomap import ColumnState
from app.planning.dstar_lite_3d import DStarLite3D
from app.planning.octomap_voxel_adapter import OctoMapVoxelAdapter
from app.planning.space3d import GridSpec3D, ManualHeightProvider, VoxelOccupancy


@dataclass(slots=True)
class _DummyOctoMap:
    grid_w: int
    grid_h: int
    grid_scale: int
    columns: dict[tuple[int, int], ColumnState]
    max_z: float


def test_octomap_voxel_adapter_grounded_range():
    col = ColumnState(cell=(1, 1), height=3.0, mode="grounded", collision_base_z=0.0, terrain_penalty=2.5)
    # top_z = 3.0
    octomap = _DummyOctoMap(grid_w=4, grid_h=4, grid_scale=1, columns={(1, 1): col}, max_z=col.top_z)
    occ = OctoMapVoxelAdapter(octomap)  # type: ignore[arg-type]

    assert occ.is_occupied((1, 1, 0)) is True
    assert occ.is_occupied((1, 1, 2)) is True
    assert occ.is_occupied((1, 1, 3)) is False
    assert occ.penalty((1, 1, 0)) == 2.5
    assert occ.penalty((0, 0, 0)) == 0.0


def test_octomap_voxel_adapter_canopy_range():
    col = ColumnState(
        cell=(2, 2),
        height=6.0,
        mode="canopy",
        display_base_z=0.0,
        collision_base_z=4.0,
        terrain_penalty=0.0,
    )
    # occupied z in [4, 6)
    octomap = _DummyOctoMap(grid_w=5, grid_h=5, grid_scale=1, columns={(2, 2): col}, max_z=col.top_z)
    occ = OctoMapVoxelAdapter(octomap)  # type: ignore[arg-type]

    assert occ.is_occupied((2, 2, 3)) is False
    assert occ.is_occupied((2, 2, 4)) is True
    assert occ.is_occupied((2, 2, 5)) is True
    assert occ.is_occupied((2, 2, 6)) is False


class _EmptyOccupancy(VoxelOccupancy):
    def __init__(self, grid: GridSpec3D):
        self.grid = grid

    def is_occupied(self, cell: tuple[int, int, int]) -> bool:
        x, y, z = cell
        return not self.grid.in_bounds((x, y, z))

    def penalty(self, cell: tuple[int, int, int]) -> float:
        return 0.0


def test_dstar_lite_3d_neighbors_26():
    grid = GridSpec3D(x_max=3, y_max=3, z_max=3)
    occ = _EmptyOccupancy(grid)
    planner = DStarLite3D((1, 1), (2, 2), occ, height_source=ManualHeightProvider(1))
    n = planner.neighbors((1, 1, 1))
    assert len(n) == 26
    assert (0, 1, 1) in n
    assert (2, 2, 2) in n


def test_dstar_lite_3d_simple_plan_reaches_goal():
    grid = GridSpec3D(x_max=6, y_max=6, z_max=6)
    occ = _EmptyOccupancy(grid)
    planner = DStarLite3D((0, 0), (5, 5), occ, height_source=ManualHeightProvider(2))
    path = planner.plan(max_steps=500)
    assert path
    assert path[0] == (0, 0, 2)

    # 目标点约定落地：goal_z=0
    assert path[-1] == (5, 5, 0)
