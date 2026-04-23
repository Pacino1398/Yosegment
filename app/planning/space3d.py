from __future__ import annotations

"""
3D 空间规范（Phase 0）

目标：
- 为后续 3D D* Lite / 体素查询提供统一的坐标、尺度与边界定义
- 本阶段只做“规范与类型定义”，不改动现有 2D/2.5D 规划逻辑
"""

from dataclasses import dataclass
from typing import Protocol, Tuple

Cell2D = tuple[int, int]
Cell3D = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class GridSpec3D:
    """
    3D 栅格空间定义（离散空间）。

    坐标系约定：
    - cell = (x, y, z) 均为整数索引
    - x: [0, x_max)
    - y: [0, y_max)
    - z: [0, z_max)

    尺度约定：
    - xy_resolution: 单个栅格在平面方向代表的“真实长度单位”（可以是米，也可以是任意单位）
    - z_resolution: 单个 z 层代表的“高度单位”
    - 若你的高度来自语义类别（如 app/mapping/grid_map.py 的 CLASS_HEIGHTS），通常是“离散层数/相对高度”，
      可令 z_resolution=1.0 作为单位层；后续接入真实测距/点云时再替换为米。
    """

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
    - Phase 0 只定义接口，不提供实现；实现将来自 OctoMap columns 的 adapter（Phase 1/2）
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
    - D* Lite 3D 在生成 start/goal 的 (x, y, z) 时，通过该接口获取 z。
    """

    def get_z(self, cell_xy: Cell2D) -> int:
        """返回离散高度层 z（整数层索引）。"""
        ...


@dataclass(frozen=True, slots=True)
class ManualHeightProvider:
    """
    手动/固定高度 provider。

    默认值 z=15（与你当前 OctoMap 可视化 DEFAULT_UAV_ALTITUDE=15.0 保持一致的“层”概念）。
    """

    default_z: int = 15

    def get_z(self, cell_xy: Cell2D) -> int:
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