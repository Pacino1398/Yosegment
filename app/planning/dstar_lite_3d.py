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
    U: dict[Cell3D, tuple[float, float]] | None = None
    km: float = 0.0
    motions: tuple[tuple[int, int, int], ...] | None = None

    def __post_init__(self):
        self.start = (self.start_xy[0], self.start_xy[1], int(self.height_source.get_z(self.start_xy)))
        self.goal = (self.goal_xy[0], self.goal_xy[1], int(self.height_source.get_z(self.goal_xy)))

        # 核心状态（惰性初始化：未出现节点默认 inf）
        self.g = {}
        self.rhs = {}
        self.U = {}
        self.km = 0.0

        # 初始化 rhs(goal)=0，并插入 U
        self.rhs[self.goal] = 0.0
        self.U[self.goal] = self.calc_key(self.goal)

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
        # 欧氏距离启发式（离散层单位）
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
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

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        base = _step_cost(dx, dy, dz)

        # Phase 1：penalty 由 occupancy 适配器提供（通常按 (x,y) 投影）
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

        if u in self.U:
            del self.U[u]

        if self._g(u) != self._rhs(u):
            self.U[u] = self.calc_key(u)

    def compute_path(self) -> None:
        while True:
            if not self.U:
                break

            s = min(self.U, key=self.U.get)
            k_old = self.U[s]
            k_new = self.calc_key(s)

            # 终止条件与 2D 版本保持一致
            if k_old >= k_new and self._rhs(s) == self._g(s):
                break

            del self.U[s]

            if k_old < k_new:
                self.U[s] = k_new
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
        self.km += self.heuristic(self.start, new_start)
        self.start = new_start

    def update_goal(self, new_goal_xy: Cell2D) -> None:
        # Phase 1：简单重建
        new_goal = (new_goal_xy[0], new_goal_xy[1], int(self.height_source.get_z(new_goal_xy)))
        self.goal = new_goal
        self.g.clear()
        self.rhs.clear()
        self.U.clear()
        self.km = 0.0
        self.rhs[self.goal] = 0.0
        self.U[self.goal] = self.calc_key(self.goal)