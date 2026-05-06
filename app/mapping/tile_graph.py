from __future__ import annotations

"""Tile-graph utilities (pixel-domain).

This module is intentionally *additive* and does not affect the existing
(grid_scale -> (gx,gy) -> D* Lite) planning chain.

It provides a lightweight graph view over `GridMapHandler.mask_instance_tiles`
(which is a list of dicts, each containing a `tiles` list with xywh+coverage in
pixel coordinates).

Typical use:
- Extract tiles in `GridMapHandler.batch_masks_to_obs(..., tile_w_px=..., tile_h_px=...)`
- Build graph for visualization or future tile-graph planners.

Notes
-----
- This graph is built on pixel-domain tiles (fixed w/h).
- Nodes are tiles; edges connect 4-neighbors or 8-neighbors in *tile-grid* space.
- This is mainly for visualization/debug; cost model is intentionally minimal.


通过pathplan.py可视化：
python -m app.planning.pathplan --mask-dir runs/segment/exp2/masks --grid-scale 10 --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3 --show-tile-graph --tile-graph-adj 4n


"""

from dataclasses import dataclass
from typing import Iterable, Literal


AdjMode = Literal["4n", "8n"]


@dataclass(frozen=True, slots=True)
class TileNode:
    """A single tile node."""

    id: int
    x: int
    y: int
    w: int
    h: int
    coverage: float
    class_id: int | None = None
    instance_index: int | None = None

    @property
    def cx(self) -> float:
        return float(self.x) + float(self.w) / 2.0

    @property
    def cy(self) -> float:
        return float(self.y) + float(self.h) / 2.0


@dataclass(frozen=True, slots=True)
class TileEdge:
    u: int
    v: int


def _iter_tiles(mask_instance_tiles: list[dict[str, object]]) -> Iterable[tuple[int, int | None, dict[str, object]]]:
    """Yield (instance_index, class_id, tile_dict)."""

    for inst_idx, inst in enumerate(mask_instance_tiles or []):
        class_id = inst.get("class_id")
        try:
            class_id_int = None if class_id is None else int(class_id)
        except Exception:
            class_id_int = None

        tiles = inst.get("tiles")
        if not tiles:
            continue
        for t in list(tiles):
            if isinstance(t, dict):
                yield inst_idx, class_id_int, t


def build_tile_graph(
    mask_instance_tiles: list[dict[str, object]],
    *,
    adj: AdjMode = "4n",
) -> tuple[list[TileNode], list[TileEdge]]:
    """Build a simple adjacency graph from xywh tiles.

    Args:
        mask_instance_tiles: `GridMapHandler.mask_instance_tiles`.
        adj: neighborhood type in tile-grid space.

    Returns:
        nodes, edges
    """

    nodes: list[TileNode] = []
    # Keyed by tile-grid coordinate (tx,ty) in units of (tile_w, tile_h).
    # This assumes all tiles share same w/h; if not, we fall back to exact xywh adjacency.
    index_by_grid: dict[tuple[int, int], int] = {}
    index_by_xy: dict[tuple[int, int, int, int], int] = {}

    tile_w: int | None = None
    tile_h: int | None = None

    for inst_idx, class_id, t in _iter_tiles(mask_instance_tiles):
        try:
            x = int(t.get("x", 0))
            y = int(t.get("y", 0))
            w = int(t.get("w", 0))
            h = int(t.get("h", 0))
            cov = float(t.get("coverage", 0.0))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue

        if tile_w is None:
            tile_w = w
        if tile_h is None:
            tile_h = h

        node_id = len(nodes)
        nodes.append(
            TileNode(
                id=node_id,
                x=x,
                y=y,
                w=w,
                h=h,
                coverage=cov,
                class_id=class_id,
                instance_index=inst_idx,
            )
        )
        index_by_xy[(x, y, w, h)] = node_id
        if tile_w == w and tile_h == h and tile_w > 0 and tile_h > 0:
            index_by_grid[(x // tile_w, y // tile_h)] = node_id

    # Build edges.
    edges_set: set[tuple[int, int]] = set()
    if not nodes:
        return nodes, []

    if tile_w and tile_h and index_by_grid:
        if adj == "8n":
            nbs = (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            )
        else:
            nbs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for (tx, ty), u in index_by_grid.items():
            for dx, dy in nbs:
                v = index_by_grid.get((tx + dx, ty + dy))
                if v is None:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edges_set.add((a, b))
    else:
        # Fallback: exact xywh adjacency (sharing an edge).
        lookup = { (n.x, n.y, n.w, n.h): n.id for n in nodes }
        for n in nodes:
            candidates = [
                (n.x + n.w, n.y, n.w, n.h),
                (n.x - n.w, n.y, n.w, n.h),
                (n.x, n.y + n.h, n.w, n.h),
                (n.x, n.y - n.h, n.w, n.h),
            ]
            for key in candidates:
                v = lookup.get(key)
                if v is None:
                    continue
                a, b = (n.id, v) if n.id < v else (v, n.id)
                edges_set.add((a, b))

    edges = [TileEdge(u=a, v=b) for a, b in sorted(edges_set)]
    return nodes, edges
