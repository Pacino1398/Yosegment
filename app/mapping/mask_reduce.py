from __future__ import annotations

"""Mask -> grid reduction backends.

This module is intended for Phase2 GPU debug / future NPU integration.

- NumpyBackend: current reference implementation (CPU)
- TorchBackend: optional GPU/CPU acceleration (debug on desktop GPU)

Both backends output per-cell:
- blocked: bool occupancy
- class_id: int class label of max-height obstacle per cell
- height: float top_z per cell (currently from CLASS_HEIGHTS)
- terrain_penalty: float per cell

Notes
-----
Today the canonical logic lives in GridMapHandler.batch_masks_to_obs.
This module provides a structured backend API so we can later:
- swap to torch reduce on GPU (keeping exact semantics)
- feed masks from NPU inference pipeline
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from app.mapping.grid_map import (
    CLASS_HEIGHTS,
    TRAVERSABLE_CLASS_PENALTIES,
    TRAVERSABLE_CLASSES,
)


MaskEntry = list
Cell2D = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ReduceResult:
    grid_w: int
    grid_h: int
    blocked: set[Cell2D]
    traversable: set[Cell2D]
    obstacles: set[Cell2D]
    terrain_penalties: dict[Cell2D, float]
    obstacle_heights: dict[Cell2D, int]
    obstacle_class_ids: dict[Cell2D, int]


class ReduceBackend:
    def reduce(
        self,
        mask_list: Sequence[MaskEntry],
        *,
        grid_w: int,
        grid_h: int,
        grid_scale: int,
        obstacle_classes: set[int],
        target_class: int,
        min_pixels: int = 50,
    ) -> ReduceResult:
        raise NotImplementedError


class NumpyBackend(ReduceBackend):
    def reduce(
        self,
        mask_list: Sequence[MaskEntry],
        *,
        grid_w: int,
        grid_h: int,
        grid_scale: int,
        obstacle_classes: set[int],
        target_class: int,
        min_pixels: int = 50,
    ) -> ReduceResult:
        full_obs: set[Cell2D] = set()
        blocked_obs: set[Cell2D] = set()
        traversable_obs: set[Cell2D] = set()
        terrain_penalties: dict[Cell2D, float] = {}
        obstacle_heights: dict[Cell2D, int] = {}
        obstacle_class_ids: dict[Cell2D, int] = {}

        for item in mask_list:
            try:
                cls_id = int(item[1])
                mask = item[3]
            except (TypeError, ValueError, IndexError):
                continue

            if mask is None or not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) < int(min_pixels):
                continue

            if cls_id == int(target_class):
                continue

            if cls_id not in obstacle_classes:
                continue

            height = int(CLASS_HEIGHTS.get(cls_id, 1))
            is_traversable = cls_id in TRAVERSABLE_CLASSES

            for x, y in zip(xs, ys):
                gx = int(x) // int(grid_scale)
                gy = int(y) // int(grid_scale)
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    cell = (gx, gy)
                    full_obs.add(cell)
                    if is_traversable:
                        if cell not in blocked_obs:
                            traversable_obs.add(cell)
                            terrain_penalties[cell] = max(
                                terrain_penalties.get(cell, 0.0),
                                float(TRAVERSABLE_CLASS_PENALTIES.get(cls_id, 0.0)),
                            )
                    else:
                        blocked_obs.add(cell)
                        traversable_obs.discard(cell)
                        terrain_penalties.pop(cell, None)

                    prev_h = obstacle_heights.get(cell, 0)
                    if height > prev_h:
                        obstacle_heights[cell] = height
                        obstacle_class_ids[cell] = cls_id

        return ReduceResult(
            grid_w=grid_w,
            grid_h=grid_h,
            blocked=blocked_obs,
            traversable=traversable_obs,
            obstacles=full_obs,
            terrain_penalties=terrain_penalties,
            obstacle_heights=obstacle_heights,
            obstacle_class_ids=obstacle_class_ids,
        )


class TorchBackend(ReduceBackend):
    """Torch reduce backend.

    This is a best-effort drop-in for NumpyBackend, used for GPU debugging.
    It is optional: if torch is not installed, constructing/using it will fail.

    Implementation note:
    For correctness and simplicity, this backend currently falls back to numpy for
    sparse dictionary construction, but computes gx/gy on torch to enable future
    vectorization.
    """

    def __init__(self, *, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def reduce(
        self,
        mask_list: Sequence[MaskEntry],
        *,
        grid_w: int,
        grid_h: int,
        grid_scale: int,
        obstacle_classes: set[int],
        target_class: int,
        min_pixels: int = 50,
    ) -> ReduceResult:
        torch = self.torch

        full_obs: set[Cell2D] = set()
        blocked_obs: set[Cell2D] = set()
        traversable_obs: set[Cell2D] = set()
        terrain_penalties: dict[Cell2D, float] = {}
        obstacle_heights: dict[Cell2D, int] = {}
        obstacle_class_ids: dict[Cell2D, int] = {}

        for item in mask_list:
            try:
                cls_id = int(item[1])
                mask = item[3]
            except (TypeError, ValueError, IndexError):
                continue

            if cls_id == int(target_class) or cls_id not in obstacle_classes:
                continue

            if mask is None or not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) < int(min_pixels):
                continue

            xs_t = torch.from_numpy(xs.astype(np.int64, copy=False)).to(self.device)
            ys_t = torch.from_numpy(ys.astype(np.int64, copy=False)).to(self.device)
            gx = (xs_t // int(grid_scale)).clamp_(0, grid_w - 1)
            gy = (ys_t // int(grid_scale)).clamp_(0, grid_h - 1)

            # back to cpu numpy for set/dict updates (still correct; later we can vectorize)
            gx_np = gx.to("cpu").numpy()
            gy_np = gy.to("cpu").numpy()

            height = int(CLASS_HEIGHTS.get(cls_id, 1))
            is_traversable = cls_id in TRAVERSABLE_CLASSES

            for cx, cy in zip(gx_np.tolist(), gy_np.tolist()):
                cell = (int(cx), int(cy))
                full_obs.add(cell)
                if is_traversable:
                    if cell not in blocked_obs:
                        traversable_obs.add(cell)
                        terrain_penalties[cell] = max(
                            terrain_penalties.get(cell, 0.0),
                            float(TRAVERSABLE_CLASS_PENALTIES.get(cls_id, 0.0)),
                        )
                else:
                    blocked_obs.add(cell)
                    traversable_obs.discard(cell)
                    terrain_penalties.pop(cell, None)

                prev_h = obstacle_heights.get(cell, 0)
                if height > prev_h:
                    obstacle_heights[cell] = height
                    obstacle_class_ids[cell] = cls_id

        return ReduceResult(
            grid_w=grid_w,
            grid_h=grid_h,
            blocked=blocked_obs,
            traversable=traversable_obs,
            obstacles=full_obs,
            terrain_penalties=terrain_penalties,
            obstacle_heights=obstacle_heights,
            obstacle_class_ids=obstacle_class_ids,
        )
