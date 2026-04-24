from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.mapping.grid_map import GridMapHandler, load_mask_entries
from app.paths import resolve_path
from app.planning.pathplan_batch import get_latest_segmentation_run_dir

Cell2D = tuple[int, int]
DEFAULT_UAV_ALTITUDE = 12.0
TREE_CANOPY_DOWNWARD_EXPANSION = 1.0
CANOPY_CLASSES = {4, 6}
GROUNDED_CLASSES = {1, 3, 5, 9}
HOUSE_CLASS = 9


@dataclass(slots=True)
class ColumnState:
    cell: Cell2D
    height: float
    class_id: int | None = None
    observed: bool = True
    display_base_z: float = 0.0
    collision_base_z: float = 0.0
    terrain_penalty: float = 0.0
    mode: str = "grounded"

    @property
    def top_z(self) -> float:
        return self.display_base_z + self.height


class OctoMap:
    def __init__(self, grid_w: int, grid_h: int, grid_scale: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid_scale = grid_scale
        self.grid_handler = GridMapHandler(grid_w, grid_h, grid_scale)
        self._reset_state()

    @staticmethod
    def infer_grid_size(mask_dir: str | Path, grid_scale: int) -> tuple[int, int]:
        mask_path = Path(mask_dir)
        first_mask = None
        if mask_path.exists():
            for mask_file in sorted(mask_path.glob("*.png")):
                first_mask = plt.imread(mask_file)
                if first_mask is not None:
                    break

        if first_mask is None:
            return 64, 64

        if getattr(first_mask, "ndim", 0) == 3:
            mask_h, mask_w = first_mask.shape[:2]
        else:
            mask_h, mask_w = first_mask.shape
        return max(1, mask_w // grid_scale), max(1, mask_h // grid_scale)

    def _reset_state(self):
        self.obstacles: set[Cell2D] = set()
        self.blocked_obstacles: set[Cell2D] = set()
        self.traversable_obstacles: set[Cell2D] = set()
        self.terrain_penalties: dict[Cell2D, float] = {}
        self.obstacle_heights: dict[Cell2D, int] = {}
        self.obstacle_class_ids: dict[Cell2D, int] = {}
        self.mask_instances: list[dict[str, object]] = []
        self.target_point: Cell2D | None = None
        self.columns: dict[Cell2D, ColumnState] = {}
        self.max_z = 0.0
        self.revision = 0

    def _sync_from_grid_handler(self):
        self.obstacles = set(self.grid_handler.obstacles)
        self.blocked_obstacles = set(self.grid_handler.blocked_obstacles)
        self.traversable_obstacles = set(self.grid_handler.traversable_obstacles)
        self.terrain_penalties = dict(self.grid_handler.terrain_penalties)
        self.obstacle_heights = dict(self.grid_handler.obstacle_heights)
        self.obstacle_class_ids = dict(self.grid_handler.obstacle_class_ids)
        self.mask_instances = list(self.grid_handler.mask_instances)
        self.target_point = self.grid_handler.target_point

    def _normalize_column_state(self, cell: Cell2D, value: ColumnState | Mapping[str, Any] | int | float | None) -> ColumnState | None:
        if value is None:
            return None

        if isinstance(value, ColumnState):
            column = ColumnState(
                cell=cell,
                height=max(0.0, float(value.height)),
                class_id=value.class_id,
                observed=bool(value.observed),
                display_base_z=float(value.display_base_z),
                collision_base_z=float(value.collision_base_z),
                terrain_penalty=float(value.terrain_penalty),
                mode=str(value.mode),
            )
        elif isinstance(value, Mapping):
            class_id = value.get("class_id")
            column = ColumnState(
                cell=cell,
                height=max(0.0, float(value.get("height", 1.0))),
                class_id=None if class_id is None else int(class_id),
                observed=bool(value.get("observed", True)),
                display_base_z=float(value.get("display_base_z", 0.0)),
                collision_base_z=float(value.get("collision_base_z", 0.0)),
                terrain_penalty=float(value.get("terrain_penalty", 0.0)),
                mode=str(value.get("mode", "grounded")),
            )
        else:
            column = ColumnState(cell=cell, height=max(0.0, float(value)))

        if column.height <= 0.0:
            return None
        column.display_base_z = max(0.0, column.display_base_z)
        column.collision_base_z = max(0.0, min(column.collision_base_z, column.top_z))
        return column

    def _build_column_for_cell(self, cell: Cell2D, class_id: int | None, height: int | float) -> ColumnState:
        top_z = max(1.0, float(height))
        terrain_penalty = float(self.terrain_penalties.get(cell, 0.0))

        if class_id in CANOPY_CLASSES:
            collision_base_z = max(0.0, top_z - TREE_CANOPY_DOWNWARD_EXPANSION)
            mode = "canopy"
        else:
            collision_base_z = 0.0
            mode = "grounded"

        if class_id == HOUSE_CLASS:
            collision_base_z = 0.0
            mode = "grounded"

        return ColumnState(
            cell=cell,
            height=top_z,
            class_id=class_id,
            observed=True,
            display_base_z=0.0,
            collision_base_z=collision_base_z,
            terrain_penalty=terrain_penalty,
            mode=mode,
        )

    def _build_columns_from_synced_state(self):
        self.columns = {}
        cells = set(self.obstacles) | set(self.obstacle_heights)
        for cell in cells:
            class_id = self.obstacle_class_ids.get(cell)
            height = self.obstacle_heights.get(cell, 1)
            self.columns[cell] = self._build_column_for_cell(cell, class_id, height)

    def _build_columns_from_obstacle_set(self, obstacle_set):
        self.columns = {}
        for cell in set(obstacle_set or set()):
            self.columns[cell] = ColumnState(cell=cell, height=1.0, observed=True)

    def _rebuild_local_views(self):
        self.obstacles = set()
        self.blocked_obstacles = set()
        self.traversable_obstacles = set()
        self.terrain_penalties = {}
        self.obstacle_heights = {}
        self.obstacle_class_ids = {}
        self.max_z = 0.0

        for cell, column in self.columns.items():
            if column.height <= 0.0:
                continue
            self.obstacles.add(cell)
            self.blocked_obstacles.add(cell)
            self.obstacle_heights[cell] = int(round(column.height))
            if column.class_id is not None:
                self.obstacle_class_ids[cell] = column.class_id
            if column.terrain_penalty > 0.0:
                self.terrain_penalties[cell] = column.terrain_penalty
            if column.top_z > self.max_z:
                self.max_z = column.top_z

    def masks_to_obstacle(self, mask_list):
        obstacle_set, target_point = self.grid_handler.batch_masks_to_obs(mask_list or [])
        self._sync_from_grid_handler()
        self.build_octomap(obstacle_set)
        if not mask_list:
            print("地图构建：未检测到掩码，障碍物集合为空")
            return set(), None

        print(f"地图构建：生成 {len(obstacle_set)} 个局部 2.5D 障碍栅格")
        return set(self.blocked_obstacles), target_point

    def build_octomap(self, obstacle_set):
        print("开始构建局部 2.5D 地图...")
        has_synced_state = bool(self.obstacles or self.obstacle_heights or self.obstacle_class_ids)
        if has_synced_state:
            self._build_columns_from_synced_state()
        else:
            self._build_columns_from_obstacle_set(obstacle_set)
            self.mask_instances = []
            self.target_point = None

        self.revision += 1
        self._rebuild_local_views()
        print("局部 2.5D 地图构建完成！")
        return set(self.blocked_obstacles)

    def get_bounds(self) -> tuple[int, int, float]:
        return self.grid_w, self.grid_h, self.max_z

    def get_blocked_obstacles_2d(self) -> set[Cell2D]:
        return set(self.blocked_obstacles)

    def get_obstacle_heights_2d(self) -> dict[Cell2D, int]:
        return dict(self.obstacle_heights)

    def export_planner_snapshot(self) -> dict[str, object]:
        return {
            "bounds": self.get_bounds(),
            "grid_scale": self.grid_scale,
            "target_point": self.target_point,
            "blocked_obstacles": self.get_blocked_obstacles_2d(),
            "obstacle_heights": self.get_obstacle_heights_2d(),
            "obstacle_class_ids": dict(self.obstacle_class_ids),
            "columns": dict(self.columns),
            "mask_instances": list(self.mask_instances),
            "revision": self.revision,
        }

    def show_local_map_3d(
        self,
        elev: float = 28.0,
        azim: float = -58.0,
        uav_altitude: float = DEFAULT_UAV_ALTITUDE,
    ):
        figure = plt.figure(figsize=(10, 8))
        axis = figure.add_subplot(111, projection="3d")

        grounded_columns = sorted(
            (column for column in self.columns.values() if column.mode == "grounded"),
            key=lambda column: column.cell,
        )
        canopy_columns = sorted(
            (column for column in self.columns.values() if column.mode == "canopy"),
            key=lambda column: column.cell,
        )

        if grounded_columns:
            xs = [column.cell[0] for column in grounded_columns]
            ys = [column.cell[1] for column in grounded_columns]
            zs = [column.display_base_z for column in grounded_columns]
            dx = [1.0] * len(grounded_columns)
            dy = [1.0] * len(grounded_columns)
            dz = [column.height for column in grounded_columns]
            axis.bar3d(xs, ys, zs, dx, dy, dz, color="#4c4c4c", alpha=0.88, shade=True)

        if canopy_columns:
            xs = [column.cell[0] for column in canopy_columns]
            ys = [column.cell[1] for column in canopy_columns]
            zs = [column.collision_base_z for column in canopy_columns]
            dx = [1.0] * len(canopy_columns)
            dy = [1.0] * len(canopy_columns)
            dz = [max(column.top_z - column.collision_base_z, 0.1) for column in canopy_columns]
            axis.bar3d(xs, ys, zs, dx, dy, dz, color="#2ca02c", alpha=0.55, shade=True)

        if self.target_point is not None:
            tx, ty = self.target_point
            axis.scatter([tx + 0.5], [ty + 0.5], [0.5], color="red", s=70, depthshade=False)

        uav_x = self.grid_w / 2.0
        uav_y = self.grid_h / 2.0
        uav_z = max(0.0, float(uav_altitude))
        axis.scatter([uav_x], [uav_y], [uav_z], color="#1f77b4", s=90, depthshade=False)

        axis.set_xlim(0, max(self.grid_w, 1))
        axis.set_ylim(0, max(self.grid_h, 1))
        axis.set_zlim(0, max(float(self.max_z), uav_z + 1.0, 1.0))
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_title(f"Local 2.5D Map | UAV z={uav_z:.1f}")
        axis.view_init(elev=elev, azim=azim)

        legend_items = [
            Line2D([0], [0], marker="s", color="w", label="grounded obstacle", markerfacecolor="#4c4c4c", markersize=10),
            Line2D([0], [0], marker="s", color="w", label="canopy obstacle", markerfacecolor="#2ca02c", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="uav", markerfacecolor="#1f77b4", markersize=8),
        ]
        if self.target_point is not None:
            legend_items.append(
                Line2D([0], [0], marker="o", color="w", label="target", markerfacecolor="red", markersize=8)
            )
        axis.legend(handles=legend_items, loc="upper right")
        plt.tight_layout()
        plt.show()

    def update_columns(self, added_or_changed: Mapping[Cell2D, ColumnState | Mapping[str, Any] | int | float | None], removed: set[Cell2D] | None = None):
        for cell in removed or set():
            self.columns.pop(cell, None)

        for cell, value in added_or_changed.items():
            column = self._normalize_column_state(cell, value)
            if column is None:
                self.columns.pop(cell, None)
            else:
                self.columns[cell] = column

        self.revision += 1
        self._rebuild_local_views()

    def clear_column(self, cell: Cell2D):
        self.update_columns({}, removed={cell})


def get_default_mask_dir() -> Path:
    latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
    return (latest_run_dir / "masks").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and display a local 2.5D map view from segmentation masks.")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files.")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="Pixel-to-grid scale.")
    parser.add_argument("--uav-altitude", type=float, default=DEFAULT_UAV_ALTITUDE, help="Default UAV altitude in the local 2.5D view.")
    parser.add_argument("--elev", type=float, default=28.0, help="3D camera elevation angle.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth angle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = resolve_path(args.mask_dir, get_default_mask_dir())
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    grid_w, grid_h = OctoMap.infer_grid_size(mask_dir, args.grid_scale)
    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=args.grid_scale)
    mask_list = load_mask_entries(mask_dir, octomap.grid_handler)
    octomap.masks_to_obstacle(mask_list)
    octomap.show_local_map_3d(elev=args.elev, azim=args.azim, uav_altitude=args.uav_altitude)


if __name__ == "__main__":
    main()

# python -m app.mapping.octomap --mask-dir runs/segment/exp2/masks