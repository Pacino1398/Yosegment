import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.inference.onnx_realtime import detections_to_mask_entries, ensure_onnx_weights_path
from app.planning.pathplan_batch import build_plan_result, render_plan_view


def test_ensure_onnx_weights_path_rejects_non_onnx():
    try:
        ensure_onnx_weights_path("weights/model.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-.onnx weights")



def test_detections_to_mask_entries_builds_grid_map_contract():
    detections = np.array(
        [
            [10, 10, 40, 40, 0.91, 4],
            [50, 45, 75, 70, 0.77, 0],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((80, 80, 2), dtype=np.uint8)
    masks[10:40, 20:50, 0] = 1
    masks[50:70, 60:80, 1] = 255

    entries = detections_to_mask_entries(detections, masks, "stream_frame000123")

    assert len(entries) == 2
    assert entries[0][1] == 4
    assert abs(entries[0][2] - 0.91) < 1e-6
    assert entries[0][4] == {
        "filename": "4_stream_frame000123_0.png",
        "image_stem": "stream_frame000123",
        "mask_index": 0,
    }
    assert set(np.unique(entries[0][3])) == {0, 255}
    assert entries[1][1] == 0
    assert entries[1][4]["mask_index"] == 1



def test_render_plan_view_accepts_in_memory_mask_entries():
    obstacle_mask = np.zeros((80, 80), dtype=np.uint8)
    obstacle_mask[10:40, 20:50] = 255

    target_mask = np.zeros((80, 80), dtype=np.uint8)
    target_mask[50:70, 60:80] = 255

    mask_entries = [
        [None, 4, 0.91, obstacle_mask, {"filename": "4_demo_0.png", "image_stem": "demo", "mask_index": 0}],
        [None, 0, 0.77, target_mask, {"filename": "0_demo_1.png", "image_stem": "demo", "mask_index": 1}],
    ]

    plan_result = build_plan_result((80, 80), mask_entries, grid_scale=10)
    image = render_plan_view(
        plan_result["canvas_shape"],
        plan_result["grid_handler"],
        plan_result["path"],
        plan_result["start"],
        plan_result["goal"],
        10,
        class_names={-1: "manual", 0: "target", 4: "tree"},
        show_labels=True,
    )

    assert plan_result["goal"] == (6, 5)
    assert image.shape == (80, 80, 3)
    assert image.dtype == np.uint8
