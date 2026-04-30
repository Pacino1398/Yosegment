from __future__ import annotations

"""Pixel-domain tiling utilities.

目标：
- 将二值 mask(像素域) 以固定 tile 尺寸 (tile_w_px, tile_h_px) 栅格化
- 输出稀疏矩形集合：List[(x, y, w, h, coverage)]，其中 (x,y) 是 tile 左上角像素坐标

说明：
- coverage = tile 内前景像素占比，用于过滤噪声
- 不依赖 OpenCV，直接用 numpy
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TileRect:
    x: int
    y: int
    w: int
    h: int
    coverage: float


def mask_to_tiles_xywh(
    mask: np.ndarray,
    *,
    tile_w_px: int,
    tile_h_px: int,
    min_coverage: float = 0.3,
) -> list[TileRect]:
    if mask is None or not isinstance(mask, np.ndarray):
        return []
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D array, got shape={getattr(mask, 'shape', None)}")

    h, w = mask.shape
    tw = max(1, int(tile_w_px))
    th = max(1, int(tile_h_px))
    min_cov = float(min_coverage)

    tiles: list[TileRect] = []
    # iterate tiles in image bounds; ignore partial tiles on right/bottom for determinism
    max_x0 = (w // tw) * tw
    max_y0 = (h // th) * th

    for y0 in range(0, max_y0, th):
        for x0 in range(0, max_x0, tw):
            roi = mask[y0 : y0 + th, x0 : x0 + tw]
            # treat non-zero as foreground
            fg = int(np.count_nonzero(roi))
            if fg == 0:
                continue
            cov = fg / float(tw * th)
            if cov < min_cov:
                continue
            tiles.append(TileRect(x=x0, y=y0, w=tw, h=th, coverage=float(cov)))

    return tiles
