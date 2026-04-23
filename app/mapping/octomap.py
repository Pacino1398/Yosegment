from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from app.mapping.grid_map import GridMapHandler

Cell2D = tuple[int, int]
Voxel3D = tuple[int, int, int]
OctreeNodeKey = tuple[int, int, int, int]


@dataclass(slots=True)
class ColumnState:
    cell: Cell2D
    height: int
    class_id: int | None = None
    blocked: bool = True
    traversable: bool = False
    terrain_penalty: float = 0.0


@dataclass(slots=True)
class VoxelRecord:
    voxel: Voxel3D
    hard_occupied: bool
    soft_occupied: bool
    class_id: int | None
    terrain_penalty: float
    column: Cell2D
    revision: int


@dataclass(slots=True)
class OctreeNode:
    key: OctreeNodeKey
    occupied_count: int = 0
    soft_count: int = 0
    max_penalty: float = 0.0
    max_height: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def dominant_class_id(self) -> int | None:
        if not self.class_counts:
            return None
        return max(self.class_counts.items(), key=lambda item: (item[1], -item[0]))[0]


@dataclass(slots=True)
class OctoMapDelta:
    added_voxels: set[Voxel3D] = field(default_factory=set)
    removed_voxels: set[Voxel3D] = field(default_factory=set)
    changed_columns: set[Cell2D] = field(default_factory=set)
    changed_nodes: set[OctreeNodeKey] = field(default_factory=set)
    revision: int = 0


class OctoMap:
    def __init__(self, grid_w: int, grid_h: int, grid_scale: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid_scale = grid_scale
        self.grid_handler = GridMapHandler(grid_w, grid_h, grid_scale)
        self._reset_state()

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
        self.voxels: dict[Voxel3D, VoxelRecord] = {}
        self.octree_nodes: dict[OctreeNodeKey, OctreeNode] = {}
        self.max_z = 0
        self.root_size = 1
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

    def _next_power_of_two(self, value: int) -> int:
        size = 1
        target = max(1, int(value))
        while size < target:
            size *= 2
        return size

    def _node_signature(self, node: OctreeNode) -> tuple[object, ...]:
        return (
            node.occupied_count,
            node.soft_count,
            node.max_penalty,
            node.max_height,
            tuple(sorted(node.class_counts.items())),
        )

    def _snapshot_nodes(self) -> dict[OctreeNodeKey, tuple[object, ...]]:
        return {key: self._node_signature(node) for key, node in self.octree_nodes.items()}

    def _changed_node_keys(self, old_nodes: dict[OctreeNodeKey, tuple[object, ...]]) -> set[OctreeNodeKey]:
        new_nodes = self._snapshot_nodes()
        changed_keys = set(old_nodes) | set(new_nodes)
        return {key for key in changed_keys if old_nodes.get(key) != new_nodes.get(key)}

    def _normalize_column_state(self, cell: Cell2D, value: ColumnState | Mapping[str, Any] | int | None) -> ColumnState | None:
        if value is None:
            return None

        if isinstance(value, ColumnState):
            normalized = ColumnState(
                cell=cell,
                height=max(0, int(value.height)),
                class_id=value.class_id,
                blocked=bool(value.blocked),
                traversable=bool(value.traversable),
                terrain_penalty=float(value.terrain_penalty),
            )
        elif isinstance(value, Mapping):
            height = max(0, int(value.get("height", 1)))
            blocked = bool(value.get("blocked", True))
            traversable = bool(value.get("traversable", False))
            class_id = value.get("class_id")
            class_id = None if class_id is None else int(class_id)
            terrain_penalty = float(value.get("terrain_penalty", 0.0))
            normalized = ColumnState(
                cell=cell,
                height=height,
                class_id=class_id,
                blocked=blocked,
                traversable=traversable,
                terrain_penalty=terrain_penalty,
            )
        else:
            normalized = ColumnState(cell=cell, height=max(0, int(value)), blocked=True)

        if normalized.height <= 0:
            return None
        if normalized.blocked:
            normalized.traversable = False
            normalized.terrain_penalty = 0.0
        elif normalized.traversable:
            normalized.terrain_penalty = max(0.0, float(normalized.terrain_penalty))
        else:
            return None
        return normalized

    def _build_columns_from_synced_state(self):
        self.columns = {}
        cells = set(self.blocked_obstacles) | set(self.traversable_obstacles) | set(self.obstacle_heights)
        for cell in cells:
            height = max(1, int(self.obstacle_heights.get(cell, 1)))
            blocked = cell in self.blocked_obstacles
            traversable = cell in self.traversable_obstacles and not blocked
            column = ColumnState(
                cell=cell,
                height=height,
                class_id=self.obstacle_class_ids.get(cell),
                blocked=blocked,
                traversable=traversable,
                terrain_penalty=self.terrain_penalties.get(cell, 0.0),
            )
            self.columns[cell] = column

    def _build_columns_from_obstacle_set(self, obstacle_set):
        self.columns = {}
        for cell in set(obstacle_set or set()):
            self.columns[cell] = ColumnState(cell=cell, height=1, blocked=True)

    def _rebuild_2d_views_from_columns(self):
        self.obstacles = set()
        self.blocked_obstacles = set()
        self.traversable_obstacles = set()
        self.terrain_penalties = {}
        self.obstacle_heights = {}
        self.obstacle_class_ids = {}

        for cell, column in self.columns.items():
            if column.height <= 0:
                continue
            self.obstacles.add(cell)
            self.obstacle_heights[cell] = column.height
            if column.class_id is not None:
                self.obstacle_class_ids[cell] = column.class_id
            if column.blocked:
                self.blocked_obstacles.add(cell)
            elif column.traversable:
                self.traversable_obstacles.add(cell)
                if column.terrain_penalty > 0.0:
                    self.terrain_penalties[cell] = column.terrain_penalty

    def _column_to_voxels(self, column: ColumnState) -> dict[Voxel3D, VoxelRecord]:
        voxels: dict[Voxel3D, VoxelRecord] = {}
        if column.height <= 0:
            return voxels
        if not column.blocked and not column.traversable:
            return voxels

        x, y = column.cell
        for z in range(column.height):
            voxel = (x, y, z)
            voxels[voxel] = VoxelRecord(
                voxel=voxel,
                hard_occupied=column.blocked,
                soft_occupied=column.traversable and not column.blocked,
                class_id=column.class_id,
                terrain_penalty=column.terrain_penalty if column.traversable and not column.blocked else 0.0,
                column=column.cell,
                revision=self.revision,
            )
        return voxels

    def _node_keys_for_voxel(self, voxel: Voxel3D) -> list[OctreeNodeKey]:
        x, y, z = voxel
        keys: list[OctreeNodeKey] = []
        cell_span = self.root_size
        depth = 0

        while True:
            keys.append((depth, x // cell_span, y // cell_span, z // cell_span))
            if cell_span == 1:
                break
            cell_span //= 2
            depth += 1

        return keys

    def _rebuild_voxels_from_columns(self):
        self.voxels = {}
        self.max_z = 0
        for column in self.columns.values():
            if column.height > self.max_z:
                self.max_z = column.height
            self.voxels.update(self._column_to_voxels(column))
        self.root_size = self._next_power_of_two(max(self.grid_w, self.grid_h, self.max_z or 1))

    def _rebuild_octree_nodes(self):
        self.octree_nodes = {}
        for record in self.voxels.values():
            column_height = self.columns.get(record.column).height if record.column in self.columns else record.voxel[2] + 1
            for key in self._node_keys_for_voxel(record.voxel):
                node = self.octree_nodes.get(key)
                if node is None:
                    node = OctreeNode(key=key)
                    self.octree_nodes[key] = node
                if record.hard_occupied:
                    node.occupied_count += 1
                if record.soft_occupied:
                    node.soft_count += 1
                if record.terrain_penalty > node.max_penalty:
                    node.max_penalty = record.terrain_penalty
                if column_height > node.max_height:
                    node.max_height = column_height
                if record.class_id is not None:
                    node.class_counts[record.class_id] = node.class_counts.get(record.class_id, 0) + 1

    def _records_for_cell(self, cell: Cell2D) -> list[VoxelRecord]:
        return [record for record in self.voxels.values() if record.column == cell]

    def _reconcile_columns_from_voxels(self, cells: set[Cell2D]):
        for cell in cells:
            records = self._records_for_cell(cell)
            if not records:
                self.columns.pop(cell, None)
                continue

            height = max(record.voxel[2] for record in records) + 1
            blocked = any(record.hard_occupied for record in records)
            traversable = any(record.soft_occupied for record in records) and not blocked
            terrain_penalty = max((record.terrain_penalty for record in records if record.soft_occupied), default=0.0)
            class_id = None
            classified_records = [record for record in records if record.class_id is not None]
            if classified_records:
                class_id = max(classified_records, key=lambda record: (record.voxel[2], record.class_id or -1)).class_id

            self.columns[cell] = ColumnState(
                cell=cell,
                height=height,
                class_id=class_id,
                blocked=blocked,
                traversable=traversable,
                terrain_penalty=terrain_penalty,
            )

    def masks_to_obstacle(self, mask_list):
        obstacle_set, target_point = self.grid_handler.batch_masks_to_obs(mask_list or [])
        self._sync_from_grid_handler()
        self.build_octomap(obstacle_set)
        if not mask_list:
            print("地图构建：未检测到掩码，障碍物集合为空")
            return set(), None

        print(f"地图构建：生成 {len(obstacle_set)} 个栅格障碍物")
        return set(self.blocked_obstacles), target_point

    def build_octomap(self, obstacle_set):
        print("开始构建八叉树地图...")
        has_synced_state = bool(self.blocked_obstacles or self.traversable_obstacles or self.obstacle_heights)
        if has_synced_state:
            self._build_columns_from_synced_state()
        else:
            self._build_columns_from_obstacle_set(obstacle_set)
            self.mask_instances = []
            self.target_point = None

        self._rebuild_2d_views_from_columns()
        self.revision += 1
        self._rebuild_voxels_from_columns()
        self._rebuild_octree_nodes()
        print("八叉树地图构建完成！")
        return set(self.blocked_obstacles)

    def get_bounds(self) -> tuple[int, int, int]:
        return self.grid_w, self.grid_h, self.max_z

    def get_blocked_obstacles_2d(self) -> set[Cell2D]:
        return set(self.blocked_obstacles)

    def get_traversable_obstacles_2d(self) -> set[Cell2D]:
        return set(self.traversable_obstacles)

    def get_terrain_penalties_2d(self) -> dict[Cell2D, float]:
        return dict(self.terrain_penalties)

    def get_obstacle_heights_2d(self) -> dict[Cell2D, int]:
        return dict(self.obstacle_heights)

    def get_occupied_voxels(self, hard_only: bool = True) -> set[Voxel3D]:
        if hard_only:
            return {voxel for voxel, record in self.voxels.items() if record.hard_occupied}
        return set(self.voxels)

    def get_soft_voxels(self) -> set[Voxel3D]:
        return {voxel for voxel, record in self.voxels.items() if record.soft_occupied}

    def get_octree_nodes(self, depth: int | None = None) -> dict[OctreeNodeKey, OctreeNode]:
        if depth is None:
            return dict(self.octree_nodes)
        return {key: node for key, node in self.octree_nodes.items() if key[0] == depth}

    def is_occupied(self, voxel: Voxel3D, hard_only: bool = False) -> bool:
        record = self.voxels.get(voxel)
        if record is None:
            return False
        if hard_only:
            return record.hard_occupied
        return record.hard_occupied or record.soft_occupied

    def get_penalty(self, voxel: Voxel3D) -> float:
        record = self.voxels.get(voxel)
        if record is None:
            return 0.0
        return record.terrain_penalty

    def export_planner_snapshot(self) -> dict[str, object]:
        return {
            "bounds": self.get_bounds(),
            "grid_scale": self.grid_scale,
            "target_point": self.target_point,
            "blocked_obstacles": self.get_blocked_obstacles_2d(),
            "traversable_obstacles": self.get_traversable_obstacles_2d(),
            "terrain_penalties": self.get_terrain_penalties_2d(),
            "obstacle_heights": self.get_obstacle_heights_2d(),
            "obstacle_class_ids": dict(self.obstacle_class_ids),
            "mask_instances": list(self.mask_instances),
            "occupied_voxels": self.get_occupied_voxels(hard_only=True),
            "soft_voxels": self.get_soft_voxels(),
            "root_size": self.root_size,
            "revision": self.revision,
        }

    def update_columns(
        self,
        added_or_changed: Mapping[Cell2D, ColumnState | Mapping[str, Any] | int | None],
        removed: set[Cell2D] | None = None,
    ) -> OctoMapDelta:
        old_voxels = set(self.voxels)
        old_nodes = self._snapshot_nodes()
        changed_columns = set(removed or set()) | set(added_or_changed)

        for cell in removed or set():
            self.columns.pop(cell, None)

        for cell, value in added_or_changed.items():
            column = self._normalize_column_state(cell, value)
            if column is None:
                self.columns.pop(cell, None)
            else:
                self.columns[cell] = column

        self._rebuild_2d_views_from_columns()
        self.revision += 1
        self._rebuild_voxels_from_columns()
        self._rebuild_octree_nodes()

        return OctoMapDelta(
            added_voxels=set(self.voxels) - old_voxels,
            removed_voxels=old_voxels - set(self.voxels),
            changed_columns=changed_columns,
            changed_nodes=self._changed_node_keys(old_nodes),
            revision=self.revision,
        )

    def clear_column(self, cell: Cell2D) -> OctoMapDelta:
        return self.update_columns({}, removed={cell})

    def update_voxels(
        self,
        occupied: set[Voxel3D] | None = None,
        freed: set[Voxel3D] | None = None,
        soft: set[Voxel3D] | None = None,
    ) -> OctoMapDelta:
        old_voxels = set(self.voxels)
        old_nodes = self._snapshot_nodes()
        next_revision = self.revision + 1
        changed_columns: set[Cell2D] = set()

        for voxel in freed or set():
            record = self.voxels.pop(voxel, None)
            if record is not None:
                changed_columns.add(record.column)
            else:
                changed_columns.add((voxel[0], voxel[1]))

        for voxel in occupied or set():
            cell = (voxel[0], voxel[1])
            previous = self.voxels.get(voxel)
            class_id = previous.class_id if previous is not None else self.obstacle_class_ids.get(cell)
            self.voxels[voxel] = VoxelRecord(
                voxel=voxel,
                hard_occupied=True,
                soft_occupied=False,
                class_id=class_id,
                terrain_penalty=0.0,
                column=cell,
                revision=next_revision,
            )
            changed_columns.add(cell)

        for voxel in soft or set():
            cell = (voxel[0], voxel[1])
            previous = self.voxels.get(voxel)
            if previous is not None and previous.hard_occupied:
                changed_columns.add(cell)
                continue
            class_id = previous.class_id if previous is not None else self.obstacle_class_ids.get(cell)
            penalty = previous.terrain_penalty if previous is not None else self.terrain_penalties.get(cell, 0.0)
            self.voxels[voxel] = VoxelRecord(
                voxel=voxel,
                hard_occupied=False,
                soft_occupied=True,
                class_id=class_id,
                terrain_penalty=penalty,
                column=cell,
                revision=next_revision,
            )
            changed_columns.add(cell)

        self._reconcile_columns_from_voxels(changed_columns)
        self._rebuild_2d_views_from_columns()
        self.revision = next_revision
        self.max_z = max((record.voxel[2] for record in self.voxels.values()), default=-1) + 1
        self.root_size = self._next_power_of_two(max(self.grid_w, self.grid_h, self.max_z or 1))
        self._rebuild_octree_nodes()

        return OctoMapDelta(
            added_voxels=set(self.voxels) - old_voxels,
            removed_voxels=old_voxels - set(self.voxels),
            changed_columns=changed_columns,
            changed_nodes=self._changed_node_keys(old_nodes),
            revision=self.revision,
        )
