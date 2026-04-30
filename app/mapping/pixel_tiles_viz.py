from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Visualization helpers for pixel-domain tiles.

This module is optional and only used for debugging / inspection.

Usage (example):
    python -m app.mapping.pixel_tiles_viz --mask-path path/to/mask.png --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from app.mapping.pixel_tiles import mask_to_tiles_xywh


def draw_tiles_on_mask(
    mask: np.ndarray,
    *,
    tile_w_px: int,
    tile_h_px: int,
    min_coverage: float = 0.3,
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """Draw sparse tiles as rectangles on top of a mask.

    Returns:
        (ax, tiles)
    """

    tiles = mask_to_tiles_xywh(mask, tile_w_px=tile_w_px, tile_h_px=tile_h_px, min_coverage=min_coverage)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # show mask as grayscale
    ax.imshow(mask, cmap="gray", vmin=0, vmax=255)

    for t in tiles:
        rect = plt.Rectangle(
            (t.x, t.y),
            t.w,
            t.h,
            fill=False,
            edgecolor="lime",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            t.x + 1,
            t.y + 10,
            f"{t.coverage:.2f}",
            color="yellow",
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"),
        )

    if title:
        ax.set_title(title)
    ax.set_axis_off()

    return ax, tiles


def _load_mask(mask_path: Path) -> np.ndarray:
    if cv2 is not None:
        m = cv2.imread(str(mask_path), 0)
        if m is not None:
            return m
    # fallback: matplotlib imread (may return float 0..1)
    m = plt.imread(mask_path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = (m * 255.0).astype(np.uint8)
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize pixel-domain tiles (xywh) on top of a mask image")
    p.add_argument("--mask-path", type=str, required=True, help="Path to a single mask PNG")
    p.add_argument("--tile-w-px", type=int, required=True)
    p.add_argument("--tile-h-px", type=int, required=True)
    # p.add_argument("--mask-path", default="runs/segment/exp2/masks/1_000456_2.png")
    # p.add_argument("--tile-w-px", type=int, default=16)
    # p.add_argument("--tile-h-px", type=int, default=16)
    # p.add_argument("--min-coverage", type=float, default=0.3)
    p.add_argument("--save", type=str, default=None, help="Optional output path to save the visualization PNG")
    return p.parse_args()


def main(
    mask_path: str = "runs/segment/exp2/masks/1_000456_2.png",
    tile_w_px: int = 16,
    tile_h_px: int = 16,
    min_coverage: float = 0.3,
    save: str | None = None
) -> None:
    mask_path = Path(mask_path)
    mask = _load_mask(mask_path)

    ax, tiles = draw_tiles_on_mask(
        mask,
        tile_w_px=tile_w_px,
        tile_h_px=tile_h_px,
        min_coverage=min_coverage,
        title=None,
    )
    ax.set_title(f"tiles={len(tiles)} | {mask_path.name}")

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved: {out}")
    else:
        plt.show()

    print([asdict(t) for t in tiles])

if __name__ == "__main__":

    main()

# python -m app.mapping.pixel_tiles_viz --mask-path runs/segment/exp2/masks/1_000456_2.png --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3