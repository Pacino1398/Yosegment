from __future__ import annotations

"""
D* Lite 3D (Phase 1)

说明：
- 基于现有 app/planning/dstar_lite.py 的最小改动升维版本
- state 从 (x, y) 扩展为 (x, y, z)
- neighbors 使用 3D 26 邻域
- heuristic/cost 使用 3D 欧氏距离（离散栅格尺度由 GridSpec3D 决定）
- g/rhs/U 使用惰性存储（避免 3D 全量初始化导致的内存/时间开销）

性能注意：
- Phase 1 仍沿用 dict 扫描 min(U) 的方式，适合小规模验证
- Phase 2 将替换为 heapq 优先队列
"""

import heapq
import math
from dataclasses import dataclass

from app.planning.space3d import Cell2D, Cell3D, HeightSource, ManualHeightProvider, VoxelOccupancy


def _step_cost(dx: int, dy: int, dz: int) -> float:
    # 26 邻域只有 3 种距离：1, sqrt(2), sqrt(3)
    s = dx * dx + dy * dy + dz * dz
    if s == 1:
        return 1.0
    if s == 2:
        return 1.4142135623730951
    if s == 3:
        return 1.7320508075688772
    return math.sqrt(float(s))


@dataclass(slots=True)
class DStarLite3D:
    start_xy: Cell2D
    goal_xy: Cell2D
    occupancy: VoxelOccupancy
    height_source: HeightSource = ManualHeightProvider()

    inflation_radius: int = 0
    penalty_weight: float = 0.0

    start: Cell3D | None = None
    goal: Cell3D | None = None
    g: dict[Cell3D, float] | None = None
    rhs: dict[Cell3D, float] | None = None

    # open list: heapq + lazy deletion
    _open_heap: list[tuple[float, float, int, Cell3D]] | None = None
    _open_entries: dict[Cell3D, tuple[float, float, int]] | None = None
    _open_seq: int = 0

    km: float = 0.0
    motions: tuple[tuple[int, int, int], ...] | None = None

    def _open_insert_or_update(self, s: Cell3D) -> None:
        """插入/更新 open list 中 s 的 key（惰性删除：旧条目留在 heap 里）。"""
        if self._open_heap is None or self._open_entries is None:
            raise RuntimeError("open list not initialized")
        k1, k2 = self.calc_key(s)
        self._open_seq += 1
        seq = int(self._open_seq)
        self._open_entries[s] = (k1, k2, seq)
        heapq.heappush(self._open_heap, (k1, k2, seq, s))

    def _open_remove(self, s: Cell3D) -> None:
        if self._open_entries is None:
            raise RuntimeError("open list not initialized")
        self._open_entries.pop(s, None)

    def _open_top(self) -> tuple[tuple[float, float], Cell3D] | None:
        """返回当前最小 key 的 (key, state)，会清理 heap 顶部的过期条目。"""
        if self._open_heap is None or self._open_entries is None:
            raise RuntimeError("open list not initialized")

        while self._open_heap:
            k1, k2, seq, s = self._open_heap[0]
            cur = self._open_entries.get(s)
            if cur is None:
                heapq.heappop(self._open_heap)
                continue
            if cur != (k1, k2, seq):
                heapq.heappop(self._open_heap)
                continue
            return (k1, k2), s
        return None

    def __post_init__(self):
        # start 的 z：由 HeightSource/HeightProvider 决定（比如巡航高度、离地高度等策略）
        self.start = (self.start_xy[0], self.start_xy[1], int(self.height_source.get_z(self.start_xy)))

        # goal 的 z：当前工程约定“投递点在地面”，因此强制 goal_z=0
        # 若未来需要“空中目标点”，可再引入 GoalHeightSource 或参数开关。
        self.goal = (self.goal_xy[0], self.goal_xy[1], 0)

        # 核心状态（惰性初始化：未出现节点默认 inf）
        self.g = {}
        self.rhs = {}

        # open list (heapq)
        self._open_heap = []
        self._open_entries = {}
        self._open_seq = 0

        self.km = 0.0

        # 初始化 rhs(goal)=0，并插入 open list
        self.rhs[self.goal] = 0.0
        self._open_insert_or_update(self.goal)

        motions = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    motions.append((dx, dy, dz))
        self.motions = tuple(motions)

    # ---------- helpers ----------
    def _g(self, s: Cell3D) -> float:
        return float(self.g.get(s, float("inf")))

    def _rhs(self, s: Cell3D) -> float:
        return float(self.rhs.get(s, float("inf")))

    def heuristic(self, a: Cell3D, b: Cell3D) -> float:
        # 欧氏距离启发式（尺度化：xy_resolution / z_resolution）
        ax, ay, az = self.occupancy.grid.to_metric(a)
        bx, by, bz = self.occupancy.grid.to_metric(b)
        dx = ax - bx
        dy = ay - by
        dz = az - bz
        return math.sqrt(float(dx * dx + dy * dy + dz * dz))

    def calc_key(self, s: Cell3D) -> tuple[float, float]:
        g = self._g(s)
        rhs = self._rhs(s)
        k1 = min(g, rhs) + self.heuristic(self.start, s) + float(self.km)
        k2 = min(g, rhs)
        return (k1, k2)

    def neighbors(self, s: Cell3D) -> list[Cell3D]:
        x, y, z = s
        nlist: list[Cell3D] = []
        grid = self.occupancy.grid
        for dx, dy, dz in self.motions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < grid.x_max and 0 <= ny < grid.y_max and 0 <= nz < grid.z_max:
                nlist.append((nx, ny, nz))
        return nlist

    def cost(self, a: Cell3D, b: Cell3D) -> float:
        if self.occupancy.is_occupied(a) or self.occupancy.is_occupied(b):
            return float("inf")

        ax, ay, az = self.occupancy.grid.to_metric(a)
        bx, by, bz = self.occupancy.grid.to_metric(b)
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        base = math.sqrt(float(dx * dx + dy * dy + dz * dz))

        # penalty 仍由 occupancy 提供（Phase 1 通常按 (x,y) 投影；后续可升级为随 z 变化）
        return base + float(self.occupancy.penalty(b))

    # ---------- D* Lite core ----------
    def update_vertex(self, u: Cell3D) -> None:
        if u != self.goal:
            best = float("inf")
            for v in self.neighbors(u):
                c = self.cost(u, v)
                if c == float("inf"):
                    continue
                val = c + self._g(v)
                if val < best:
                    best = val
            self.rhs[u] = best

        # 等价于 2D 版的：若 u 在 open list，先移除，再视需要插入
        self._open_remove(u)

        if self._g(u) != self._rhs(u):
            self._open_insert_or_update(u)

    def compute_path(self) -> None:
        while True:
            top = self._open_top()
            if top is None:
                break
            k_old, s = top
            k_new = self.calc_key(s)

            # 终止条件：k_old >= key(start) 且 start 一致
            if k_old >= self.calc_key(self.start) and self._rhs(self.start) == self._g(self.start):
                break

            # pop current valid top
            self._open_remove(s)
            heapq.heappop(self._open_heap)

            if k_old < k_new:
                self._open_insert_or_update(s)
                continue

            if self._g(s) > self._rhs(s):
                self.g[s] = self._rhs(s)
                for neighbor in self.neighbors(s):
                    self.update_vertex(neighbor)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                for neighbor in self.neighbors(s):
                    self.update_vertex(neighbor)

    def plan(self, max_steps: int = 5000) -> list[Cell3D]:
        self.compute_path()
        path: list[Cell3D] = []
        cur = self.start
        for _ in range(int(max_steps)):
            path.append(cur)
            if cur == self.goal:
                break

            min_cost = float("inf")
            best: Cell3D | None = None
            for neighbor in self.neighbors(cur):
                c = self.cost(cur, neighbor)
                if c == float("inf"):
                    continue
                val = c + self._g(neighbor)
                if val < min_cost:
                    min_cost = val
                    best = neighbor

            if best is None:
                break
            cur = best
        return path

    def update_start(self, new_start_xy: Cell2D) -> None:
        new_start = (new_start_xy[0], new_start_xy[1], int(self.height_source.get_z(new_start_xy)))
        # D* Lite：起点移动时只更新 km 与 start，不需要清空 g/rhs/open list
        self.km += self.heuristic(self.start, new_start)
        self.start = new_start

    def update_occupancy_changed(self, changed_cells: set[Cell3D]) -> None:
        """实时更新入口：当 occupancy（障碍/代价）发生变化时，增量更新相关顶点。

        约定：
        - changed_cells 是占用状态或 penalty 发生变化的体素集合
        - D* Lite 标准做法：对每条受影响的边 (u,v)，调用 update_vertex(u)
          这里简化为：对 changed cell 及其 26 邻域都 update_vertex。
        """
        if not changed_cells:
            return
        to_update: set[Cell3D] = set()
        for c in changed_cells:
            to_update.add(c)
            for nb in self.neighbors(c):
                to_update.add(nb)

        for u in to_update:
            self.update_vertex(u)

    def update_goal(self, new_goal_xy: Cell2D) -> None:
        # Phase 1：简单重建
        # 目标点强制落地：goal_z=0
        new_goal = (new_goal_xy[0], new_goal_xy[1], 0)
        self.goal = new_goal
        self.g.clear()
        self.rhs.clear()
        self._open_heap.clear()
        self._open_entries.clear()
        self._open_seq = 0
        self.km = 0.0
        self.rhs[self.goal] = 0.0
        self._open_insert_or_update(self.goal)
