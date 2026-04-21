import numpy as np

from app.mapping.grid_map import (
    CLASS_HEIGHTS,
    GridMapHandler,
    TRAVERSABLE_CLASS_PENALTIES,
    group_mask_entries_by_stem,
    load_label_confidences,
    parse_mask_filename,
)
from app.planning.pathplan_batch import select_display_instances


def test_batch_masks_to_obs_extracts_obstacles_and_target():
    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=10)

    obstacle_mask = np.zeros((80, 80), dtype=np.uint8)
    obstacle_mask[10:40, 20:50] = 255

    target_mask = np.zeros((80, 80), dtype=np.uint8)
    target_mask[50:70, 60:80] = 255

    obs, target = handler.batch_masks_to_obs(
        [
            [None, 1, 1.0, obstacle_mask],
            [None, 0, 1.0, target_mask],
        ]
    )

    assert target == (6, 5)
    assert (2, 1) in obs
    assert (4, 3) in obs
    assert (6, 5) not in obs
    assert handler.obstacle_heights[(2, 1)] == CLASS_HEIGHTS[1]
    assert handler.obstacle_class_ids[(2, 1)] == 1


def test_batch_masks_to_obs_ignores_small_masks():
    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=10)
    small_mask = np.zeros((80, 80), dtype=np.uint8)
    small_mask[0:5, 0:5] = 255

    obs, target = handler.batch_masks_to_obs([[None, 1, 1.0, small_mask]])

    assert obs == set()
    assert target is None
    assert handler.obstacle_heights == {}
    assert handler.obstacle_class_ids == {}


def test_batch_masks_to_obs_marks_tree_and_forest_as_traversable():
    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=10)

    tree_mask = np.zeros((80, 80), dtype=np.uint8)
    tree_mask[10:40, 20:50] = 255

    obs, target = handler.batch_masks_to_obs([[None, 4, 1.0, tree_mask]])

    assert target is None
    assert obs == set()
    assert (2, 1) in handler.obstacles
    assert (2, 1) in handler.traversable_obstacles
    assert (2, 1) not in handler.blocked_obstacles
    assert handler.terrain_penalties[(2, 1)] == TRAVERSABLE_CLASS_PENALTIES[4]


def test_batch_masks_to_obs_keeps_higher_height_on_overlap():
    handler = GridMapHandler(grid_w=8, grid_h=8, grid_scale=10)

    low_mask = np.zeros((80, 80), dtype=np.uint8)
    low_mask[10:40, 20:50] = 255

    high_mask = np.zeros((80, 80), dtype=np.uint8)
    high_mask[10:40, 20:50] = 255

    obs, target = handler.batch_masks_to_obs(
        [
            [None, 1, 1.0, low_mask],
            [None, 9, 1.0, high_mask],
        ]
    )

    assert target is None
    assert (2, 1) in obs
    assert handler.obstacle_heights[(2, 1)] == max(CLASS_HEIGHTS[1], CLASS_HEIGHTS[9])
    assert handler.obstacle_class_ids[(2, 1)] == 9


def test_parse_mask_filename_extracts_video_frame_stem_and_index():
    parsed = parse_mask_filename("4_demo_frame000123_7.png")

    assert parsed["image_stem"] == "demo_frame000123"
    assert parsed["mask_index"] == 7
    assert parsed["filename"] == "4_demo_frame000123_7.png"


def test_group_mask_entries_by_stem_groups_video_frames_separately():
    mask = np.zeros((10, 10), dtype=np.uint8)
    grouped = group_mask_entries_by_stem(
        [
            [None, 4, 1.0, mask, parse_mask_filename("4_demo_frame000001_0.png")],
            [None, 6, 1.0, mask, parse_mask_filename("6_demo_frame000001_1.png")],
            [None, 1, 1.0, mask, parse_mask_filename("1_demo_frame000002_0.png")],
        ]
    )

    assert list(grouped) == ["demo_frame000001", "demo_frame000002"]
    assert len(grouped["demo_frame000001"]) == 2
    assert len(grouped["demo_frame000002"]) == 1


def test_load_label_confidences_reads_mask_index_confidence_pairs(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "demo_frame000001.txt").write_text(
        "4 0.1 0.2 0.3 0.4 0.85\n6 0.5 0.6 0.7 0.8 0.42\n",
        encoding="utf-8",
    )

    confidences = load_label_confidences(tmp_path)

    assert confidences == {
        "demo_frame000001:0": 0.85,
        "demo_frame000001:1": 0.42,
    }


def test_select_display_instances_keeps_highest_confidence_per_overlapping_obstacle():
    selected = select_display_instances(
        [
            {
                "class_id": 1,
                "confidence": 0.61,
                "cells": ((1, 1), (1, 2)),
                "center_cell": (1, 1),
                "mask_index": 0,
            },
            {
                "class_id": 9,
                "confidence": 0.93,
                "cells": ((1, 2), (2, 2)),
                "center_cell": (2, 2),
                "mask_index": 1,
            },
            {
                "class_id": 4,
                "confidence": 0.75,
                "cells": ((7, 7),),
                "center_cell": (7, 7),
                "mask_index": 2,
            },
        ]
    )

    assert len(selected) == 2
    assert any(item["class_id"] == 9 and item["mask_index"] == 1 for item in selected)
    assert any(item["class_id"] == 4 and item["mask_index"] == 2 for item in selected)
