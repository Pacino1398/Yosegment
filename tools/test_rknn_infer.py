"""Minimal RKNN inference sanity check for YOSEGMENT.

Usage on RK3588 board:
  python tools/test_rknn_infer.py --rknn weights/0414_qy++.rknn --image test_input/xxx.jpg

It prints output tensor shapes and tries to parse them as (prediction, proto).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--input-format", type=str, default="nchw_fp32", choices=["nchw_fp32", "nhwc_u8"])
    args = ap.parse_args()

    from app.inference.rknn_realtime import RknnRealtimeSegmenter

    seg = RknnRealtimeSegmenter(weights=args.rknn, imgsz=args.imgsz, input_format=args.input_format)
    entries = seg.predict_image(args.image)
    print(f"mask_entries={len(entries)}")
    if entries:
        # entry: [None, class_id, confidence, binary_mask, metadata]
        m = entries[0][3]
        print(f"first_mask_shape={m.shape} nonzero={int(np.count_nonzero(m))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
