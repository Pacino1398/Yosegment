import numpy as np

from app.mapping.octomap import OctoMap


def test_octomap_syncs_mask_instance_tiles_from_grid_handler():
    # Build a tiny synthetic mask: class 1 (car) occupies the full 8x8 region.
    # Note: GridMapHandler.batch_masks_to_obs has a built-in filter that skips instances with <50 foreground pixels.
    # Using full-foreground 8x8 (64 pixels) ensures the instance is kept.
    mask = np.ones((8, 8), dtype=np.uint8) * 255

    octomap = OctoMap(grid_w=8, grid_h=8, grid_scale=1)

    # MaskEntry layout: [None, cls_id, confidence, mask, metadata]
    mask_list = [[None, 1, 1.0, mask, {"filename": "1_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap.grid_handler.batch_masks_to_obs(mask_list, tile_w_px=4, tile_h_px=4, min_coverage=0.3)
    octomap._sync_from_grid_handler()

    assert isinstance(octomap.mask_instance_tiles, list)
    assert len(octomap.mask_instance_tiles) == 1

    entry = octomap.mask_instance_tiles[0]
    assert entry["class_id"] == 1
    tiles = entry["tiles"]
    # Full 8x8 mask with 4x4 tiles => 4 tiles.
    assert len(tiles) == 4
    # Spot-check one tile.
    t0 = tiles[0]
    assert (t0["w"], t0["h"]) == (4, 4)
    assert abs(t0["coverage"] - 1.0) < 1e-6


def test_octomap_snapshot_contains_mask_instance_tiles():
    # Full-foreground mask to ensure it passes the >=50 foreground pixel threshold.
    mask = np.ones((8, 8), dtype=np.uint8) * 255

    octomap = OctoMap(grid_w=8, grid_h=8, grid_scale=1)

    mask_list = [[None, 1, 1.0, mask, {"filename": "1_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap.grid_handler.batch_masks_to_obs(mask_list, tile_w_px=4, tile_h_px=4, min_coverage=0.3)
    octomap._sync_from_grid_handler()

    snap = octomap.export_planner_snapshot()
    assert "mask_instance_tiles" in snap
    assert len(snap["mask_instance_tiles"]) == 1
