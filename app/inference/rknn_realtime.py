from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
# NOTE: We reuse most of the postprocess utilities from onnx_realtime.py to avoid
# duplicating NMS/mask logic. Import explicitly.
from app.inference.onnx_realtime import (  # noqa: E402
    _clip_boxes,
    _crop_mask,
    _nms,
    _normalize_prediction,
    _normalize_proto,
    _resolve_imgsz,
    _resize_masks,
    _sigmoid,
    _xywh_to_xyxy,
    detections_to_mask_entries,
)

MANUAL_RKNN_WEIGHTS: str | Path | None = None


def _resolve_manual_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DEFAULT_CONFIG.repo_root / path
    return path.resolve()


def get_default_rknn_weights() -> Path:
    """Default to weights/0414_qy++.rknn (user-provided)."""
    # Preferred: repo_root/weights/0414_qy++.rknn
    preferred = DEFAULT_CONFIG.repo_root / "weights" / "0414_qy++.rknn"
    if MANUAL_RKNN_WEIGHTS is not None:
        return _resolve_manual_path(MANUAL_RKNN_WEIGHTS, preferred)
    return preferred.resolve()


def ensure_rknn_weights_path(weights: str | Path) -> Path:
    weights_path = Path(weights)
    if weights_path.suffix.lower() != ".rknn":
        raise ValueError(f"RKNN realtime 入口要求 .rknn 权重，当前收到: {weights_path}")
    return weights_path


class RknnRealtimeSegmenter:
    """RKNN (NPU) realtime segmenter for RK3588.

    This class mirrors OnnxRealtimeSegmenter but replaces ONNXRuntime with rknn_runtime.

    Assumptions:
    - The RKNN model outputs two tensors: prediction (3D) and proto (4D), same semantics
      as YOLOv5-seg export.
    - Input preprocess should match what was used during RKNN build.

    If your RKNN was built with NHWC uint8 input, set input_format='nhwc_u8'.
    If your RKNN was built with NCHW fp32 input, keep default 'nchw_fp32'.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        imgsz: int | tuple[int, int] = 640,
        conf_thres: float | None = None,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
        input_format: str = "nchw_fp32",
    ):
        try:
            # Board-side inference (recommended): rknn-toolkit-lite2
            # pip install rknn-toolkit-lite2
            from rknnlite.api import RKNNLite as RKNN  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RKNN 板端推理需要安装 rknn-toolkit-lite2（提供 rknnlite.api.RKNNLite）。"
                "请在板端执行：pip install rknn-toolkit-lite2，然后验证："
                "python3 -c \"from rknnlite.api import RKNNLite; print('RKNNLite import OK')\""
            ) from exc

        weights_path = get_default_rknn_weights() if weights is None else Path(weights)
        weights_path = ensure_rknn_weights_path(weights_path)
        if not weights_path.is_absolute():
            weights_path = (DEFAULT_CONFIG.repo_root / weights_path).resolve()

        self.weights = weights_path
        self.imgsz = _resolve_imgsz(imgsz)
        self.conf_thres = conf_thres if conf_thres is not None else DEFAULT_CONFIG.default_conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = set(classes) if classes is not None else None
        self.agnostic_nms = agnostic_nms
        self.input_format = input_format.lower().strip()

        self._RKNN = RKNN
        self.rknn = RKNN(verbose=False)
        ret = self.rknn.load_rknn(str(self.weights))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}, weights={self.weights}")

        ret = self.rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame 必须是 HxWxC 的 numpy.ndarray")

        input_h, input_w = self.imgsz
        if frame.shape[0] != input_h or frame.shape[1] != input_w:
            frame = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        if self.input_format == "nhwc_u8":
            # Common RKNN setting: NHWC uint8 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(image, dtype=np.uint8)

        # Default: NCHW float32 RGB normalized to [0,1]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)).astype(np.float32, copy=False)
        image /= 255.0
        return np.expand_dims(np.ascontiguousarray(image), axis=0)

    def _run_inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        # RKNN inference expects a list of inputs.
        outputs = self.rknn.inference(inputs=[input_tensor])
        if outputs is None:
            raise RuntimeError("rknn.inference returned None")
        return [np.asarray(o) for o in outputs]

    def _extract_model_outputs(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Same heuristic as ONNX version: find one 3D and one 4D output.
        prediction = next((output for output in outputs if output.ndim == 3), None)
        proto = next((output for output in outputs if output.ndim == 4), None)
        if prediction is None or proto is None:
            shapes = [tuple(output.shape) for output in outputs]
            raise ValueError(f"无法从 RKNN 输出中识别 prediction/proto，当前 outputs={shapes}")
        return prediction, proto

    def _postprocess_prediction(
        self,
        prediction: np.ndarray,
        proto: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        # Copy from onnx_realtime.OnnxRealtimeSegmenter._postprocess_prediction with minimal changes.
        proto = _normalize_proto(proto)
        mask_dim = int(proto.shape[0])
        prediction = _normalize_prediction(prediction, mask_dim)

        num_classes = prediction.shape[1] - 5 - mask_dim
        if num_classes <= 0:
            raise ValueError(f"无法从预测输出中解析类别数: shape={prediction.shape}, mask_dim={mask_dim}")

        boxes_xywh = prediction[:, :4]
        objectness = prediction[:, 4]
        class_scores = prediction[:, 5 : 5 + num_classes]
        mask_coeffs = prediction[:, 5 + num_classes :]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
        scores = objectness * class_conf

        keep = scores > self.conf_thres
        if self.classes is not None:
            keep &= np.isin(class_ids, list(self.classes))
        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep].astype(np.float32, copy=False)
        class_ids = class_ids[keep].astype(np.float32, copy=False)
        mask_coeffs = mask_coeffs[keep]

        boxes = _xywh_to_xyxy(boxes_xywh)
        boxes = _clip_boxes(boxes, self.imgsz[1], self.imgsz[0])

        if self.agnostic_nms:
            keep_indices = _nms(boxes, scores, self.iou_thres, self.max_det)
        else:
            kept_parts: list[np.ndarray] = []
            for class_id in np.unique(class_ids.astype(np.int32)):
                class_mask = class_ids == float(class_id)
                class_indices = np.where(class_mask)[0]
                class_keep = _nms(boxes[class_mask], scores[class_mask], self.iou_thres, self.max_det)
                if class_keep.size:
                    kept_parts.append(class_indices[class_keep])
            if kept_parts:
                keep_indices = np.concatenate(kept_parts)
                keep_indices = keep_indices[np.argsort(scores[keep_indices])[::-1][: self.max_det]]
            else:
                keep_indices = np.empty((0,), dtype=np.int64)

        if keep_indices.size == 0:
            return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        mask_coeffs = mask_coeffs[keep_indices]

        proto_h, proto_w = proto.shape[1:]
        mask_logits = mask_coeffs @ proto.reshape(mask_dim, -1)
        mask_logits = mask_logits.reshape((-1, proto_h, proto_w))
        masks = _sigmoid(mask_logits)

        scaled_boxes_for_proto = boxes.copy()
        scaled_boxes_for_proto[:, [0, 2]] *= proto_w / float(self.imgsz[1])
        scaled_boxes_for_proto[:, [1, 3]] *= proto_h / float(self.imgsz[0])

        cropped_masks = []
        for mask, box in zip(masks, scaled_boxes_for_proto):
            cropped_masks.append(_crop_mask(mask, box))

        if not cropped_masks:
            return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

        cropped_masks_array = np.stack(cropped_masks, axis=0)
        resized_masks = _resize_masks(np.moveaxis(cropped_masks_array, 0, -1), frame_shape)
        binary_masks = (resized_masks > 0.5).astype(np.uint8)

        frame_h, frame_w = frame_shape
        scaled_boxes = boxes.copy()
        scaled_boxes[:, [0, 2]] *= frame_w / float(self.imgsz[1])
        scaled_boxes[:, [1, 3]] *= frame_h / float(self.imgsz[0])
        scaled_boxes = _clip_boxes(scaled_boxes, frame_w, frame_h)

        detections = np.column_stack((scaled_boxes, scores, class_ids)).astype(np.float32, copy=False)
        return detections, binary_masks

    def predict_frame(self, frame: np.ndarray, frame_stem: str) -> list[list]:
        input_tensor = self.preprocess_frame(frame)
        outputs = self._run_inference(input_tensor)
        prediction, proto = self._extract_model_outputs(outputs)
        detections, masks = self._postprocess_prediction(prediction, proto, frame.shape[:2])
        return detections_to_mask_entries(detections, masks, frame_stem)

    def predict_image(self, image_path: str | Path) -> list[list]:
        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return self.predict_frame(frame, image_path.stem)
