import numpy as np

from app.mapping.pixel_tiles import mask_to_tiles_xywh


def test_mask_to_tiles_xywh_basic():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 255

    tiles = mask_to_tiles_xywh(mask, tile_w_px=4, tile_h_px=4, min_coverage=0.3)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.x, t.y, t.w, t.h) == (0, 0, 4, 4)
    assert abs(t.coverage - 1.0) < 1e-6


def test_mask_to_tiles_xywh_min_coverage():
    mask = np.zeros((8, 8), dtype=np.uint8)
    # only 1 pixel set inside first tile
    mask[0, 0] = 1
    tiles = mask_to_tiles_xywh(mask, tile_w_px=4, tile_h_px=4, min_coverage=0.2)
    assert tiles == []
