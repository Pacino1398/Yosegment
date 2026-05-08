import numpy as np
import pytest

from app.mapping.mask_reduce import NumpyBackend, TorchBackend
from app.mapping.grid_map import GridMapHandler


def _make_mask_list(cls_id: int, h: int = 8, w: int = 8):
    mask = np.ones((h, w), dtype=np.uint8) * 255  # 64px => pass >=50 filter
    return [[None, cls_id, 1.0, mask, {"filename": f"{cls_id}_img_0.png", "image_stem": "img", "mask_index": 0}]]


def test_numpy_backend_matches_gridmaphandler_semantics_for_blocked_and_height():
    mask_list = _make_mask_list(1)

    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=1)
    handler.batch_masks_to_obs(mask_list)

    be = NumpyBackend()
    res = be.reduce(
        mask_list,
        grid_w=8,
        grid_h=8,
        grid_scale=1,
        obstacle_classes=handler.OBSTACLE_CLASSES,
        target_class=handler.TARGET_CLASS,
    )

    assert res.blocked == handler.blocked_obstacles
    assert res.traversable == handler.traversable_obstacles
    assert res.obstacle_heights == handler.obstacle_heights
    assert res.obstacle_class_ids == handler.obstacle_class_ids


def test_torch_backend_matches_numpy_backend_when_torch_available():
    mask_list = _make_mask_list(4)

    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=1)

    np_be = NumpyBackend()
    np_res = np_be.reduce(
        mask_list,
        grid_w=8,
        grid_h=8,
        grid_scale=1,
        obstacle_classes=handler.OBSTACLE_CLASSES,
        target_class=handler.TARGET_CLASS,
    )

    try:
        torch_be = TorchBackend(device="cpu")
    except Exception:
        pytest.skip("torch not installed")

    torch_res = torch_be.reduce(
        mask_list,
        grid_w=8,
        grid_h=8,
        grid_scale=1,
        obstacle_classes=handler.OBSTACLE_CLASSES,
        target_class=handler.TARGET_CLASS,
    )

    assert torch_res.blocked == np_res.blocked
    assert torch_res.traversable == np_res.traversable
    assert torch_res.obstacles == np_res.obstacles
    assert torch_res.obstacle_heights == np_res.obstacle_heights
    assert torch_res.obstacle_class_ids == np_res.obstacle_class_ids
    assert torch_res.terrain_penalties == np_res.terrain_penalties
