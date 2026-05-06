import numpy as np

from app.mapping.octomap import OctoMap, TREE_CANOPY_DOWNWARD_EXPANSION


def _build_octomap_from_masks(mask_list, *, grid_w=8, grid_h=8, grid_scale=1) -> OctoMap:
    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)
    octomap.grid_handler.batch_masks_to_obs(mask_list)
    octomap._sync_from_grid_handler()
    octomap.build_octomap(octomap.grid_handler.blocked_obstacles)
    return octomap


def test_build_occ2d_marks_blocked_cells_by_default():
    # class 1 is non-traversable => blocked_obstacles
    mask = np.ones((8, 8), dtype=np.uint8) * 255  # 64px, passes >=50 px filter
    mask_list = [[None, 1, 1.0, mask, {"filename": "1_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap = _build_octomap_from_masks(mask_list, grid_w=8, grid_h=8, grid_scale=1)

    occ2d = octomap.build_occ2d()
    assert occ2d.dtype == np.uint8
    assert occ2d.shape == (8, 8)
    assert int(occ2d.max()) == 255
    assert int(occ2d.min()) == 255  # full mask => full grid occupied


def test_build_occ2d_use_columns_can_include_traversable_columns():
    # Note: current GridMapHandler marks traversable classes (4/6) as traversable_obstacles,
    # but they are still included in OctoMap.blocked_obstacles when columns are rebuilt.
    # Therefore we only assert that `use_columns=True` is (at least) inclusive.
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    mask_list = [[None, 4, 1.0, mask, {"filename": "4_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap = _build_octomap_from_masks(mask_list, grid_w=8, grid_h=8, grid_scale=1)

    occ_blocked = octomap.build_occ2d(use_columns=False)
    occ_cols = octomap.build_occ2d(use_columns=True)

    # use_columns should never be less occupied than blocked_obstacles mode.
    assert int(occ_cols.max()) >= int(occ_blocked.max())
    assert int(occ_cols.max()) == 255


def test_build_z_band2d_has_expected_shape_and_ordering():
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    mask_list = [[None, 1, 1.0, mask, {"filename": "1_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap = _build_octomap_from_masks(mask_list, grid_w=8, grid_h=8, grid_scale=1)

    z_band2d = octomap.build_z_band2d()
    assert z_band2d.dtype == np.float32
    assert z_band2d.shape == (8, 8, 2)

    z_lo = z_band2d[..., 0]
    z_hi = z_band2d[..., 1]

    # All occupied => should have non-zero top_z everywhere.
    assert float(z_hi.min()) > 0.0
    assert np.all(z_lo <= z_hi)


def test_build_z_band2d_canopy_uses_collision_base_z_top_z_minus_expansion():
    # class 4 is canopy in OctoMap (CANOPY_CLASSES={4,6})
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    mask_list = [[None, 4, 1.0, mask, {"filename": "4_img_0.png", "image_stem": "img", "mask_index": 0}]]

    octomap = _build_octomap_from_masks(mask_list, grid_w=8, grid_h=8, grid_scale=1)

    z_band2d = octomap.build_z_band2d()
    z_lo = z_band2d[..., 0]
    z_hi = z_band2d[..., 1]

    # For canopy: collision_base_z = top_z - TREE_CANOPY_DOWNWARD_EXPANSION
    # GridMapHandler height for class 4 is 7.
    expected_top = 7.0
    expected_lo = max(0.0, expected_top - float(TREE_CANOPY_DOWNWARD_EXPANSION))

    assert np.allclose(z_hi, expected_top)
    assert np.allclose(z_lo, expected_lo)
