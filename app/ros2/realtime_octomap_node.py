from __future__ import annotations
"""Realtime OctoMap publisher with multi-backend support (ONNX/RKNN)."""
import argparse
import sys
import threading
from http import server
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.inference.onnx_realtime import (
    OnnxRealtimeSegmenter,
    get_default_data_yaml,
    get_default_realtime_weights,
)
from app.inference.rknn_realtime import RknnRealtimeSegmenter
from app.inference.segmentation import get_default_source_dir
from app.paths import resolve_path
from app.mapping.octomap import OctoMap
from app.planning.pathplan_batch import (
    build_plan_result,
    create_pathplan_run_dir,
    get_default_pathplan_project_dir,
    load_class_names,
    render_plan_on_frame,
)
from app.ros2.occ_zband_publisher import OccZBandPublishConfig, OccZBandPublisher

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")

def is_stream_source(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(STREAM_SCHEMES)

def resolve_backend(backend: str, weights) -> str:
    if backend != "auto":
        return backend
    if weights is None:
        default_weights = get_default_realtime_weights()
        return "rknn" if default_weights.suffix.lower() == ".rknn" else "onnx"
    suffix = Path(weights).suffix.lower()
    if suffix == ".rknn":
        return "rknn"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError(f"无法自动判断后端：{weights}")

def create_segmenter(backend, weights, data_yaml, device, imgsz, conf_thres, iou_thres, dnn, half):
    if backend == "rknn":
        return RknnRealtimeSegmenter(
            weights=weights, data_yaml=data_yaml, device=device,
            imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, dnn=dnn, half=half,
        )
    return OnnxRealtimeSegmenter(
        weights=weights, data_yaml=data_yaml, device=device,
        imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, dnn=dnn, half=half,
    )

class MjpegFrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.payload = None
        self.sequence = 0

    def update(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        payload = encoded.tobytes()
        with self.condition:
            self.payload = payload
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 1.0):
        with self.condition:
            if self.sequence == last_sequence:
                self.condition.wait(timeout=timeout)
            return self.sequence, self.payload

class ThreadingMjpegServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_cls, frame_store, stream_path: str):
        super().__init__(server_address, handler_cls)
        self.frame_store = frame_store
        self.stream_path = stream_path

class MjpegRequestHandler(server.BaseHTTPRequestHandler):
    server: ThreadingMjpegServer

    def do_GET(self) -> None:
        if self.path not in {self.server.stream_path, "/"}:
            self.send_error(404)
            return
        if self.path == "/":
            body = f'<html><body><img src="{self.server.stream_path}" style="max-width:100%;height:auto;"></body></html>'.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = 0
        try:
            while True:
                sequence, payload = self.server.frame_store.wait_for_frame(sequence)
                if payload is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        return

class MjpegStreamServer:
    def __init__(self, host: str, port: int, stream_path: str = "/stream.mjpg"):
        self.frame_store = MjpegFrameStore()
        self.server = ThreadingMjpegServer((host, port), MjpegRequestHandler, self.frame_store, stream_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.host = host
        self.port = port
        self.stream_path = stream_path

    def start(self) -> None:
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        self.frame_store.update(frame)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)

class RealtimeOctoMapPublisher:
    def __init__(self, node, source=None, source_topic=None, weights=None, data_yaml=None,
                 backend="auto", device=None, conf_thres=None, iou_thres=0.45,
                 imgsz=640, grid_scale=8, frame_id="map", resolution=1.0,
                 publish_octomap=True, publish_occupancy_grid=False, publish_markers=False,
                 mjpeg_host="0.0.0.0", mjpeg_port=8080, mjpeg_enabled=False,
                 save=False, project=None):
        self._node = node
        self._source = source
        self._source_topic = source_topic
        self._weights = weights
        self._data_yaml = data_yaml
        self._backend = backend
        self._device = device
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._imgsz = imgsz
        self._grid_scale = grid_scale
        self._frame_id = frame_id
        self._resolution = resolution
        self._save = save
        self._project = project
        self._mjpeg_enabled = mjpeg_enabled

        selected_backend = resolve_backend(backend, weights)
        selected_weights = weights if weights is not None else get_default_realtime_weights()
        self._segmenter = create_segmenter(
            selected_backend, selected_weights, data_yaml, device,
            imgsz, conf_thres, iou_thres, dnn=False, half=False,
        )
        node.get_logger().info(f"推理后端：{selected_backend} | 权重：{selected_weights}")

        yaml_path = data_yaml if data_yaml else get_default_data_yaml()
        self._class_names = load_class_names(resolve_path(yaml_path, get_default_data_yaml()))

        cfg = OccZBandPublishConfig(
            frame_id=frame_id, resolution=resolution,
            publish_octomap=publish_octomap,
            publish_occupancy_grid=publish_occupancy_grid,
            publish_markers=publish_markers,
        )
        self._octomap_publisher = OccZBandPublisher(node, cfg)

        self._mjpeg_server = None
        if mjpeg_enabled:
            self._mjpeg_server = MjpegStreamServer(mjpeg_host, mjpeg_port)
            self._mjpeg_server.start()
            node.get_logger().info(f"MJPEG: http://{mjpeg_host if mjpeg_host != '0.0.0.0' else '127.0.0.1'}:{mjpeg_port}/stream.mjpg")

        self._image_sub = None
        if source_topic:
            from functools import partial
            from sensor_msgs.msg import Image as ImageMsg
            self._image_sub = node.create_subscription(ImageMsg, source_topic, partial(self._on_image_msg), 10)
            node.get_logger().info(f"订阅图像话题：{source_topic}")

        self._run_dir = None
        if save:
            project_path = project or get_default_pathplan_project_dir()
            self._run_dir = create_pathplan_run_dir(project_path)
            node.get_logger().info(f"保存目录：{self._run_dir}")

    def _on_image_msg(self, msg):
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_frame(frame)
        except Exception as e:
            self._node.get_logger().warn(f"处理图像失败：{e}")

    def process_frame(self, frame: np.ndarray):
        try:
            mask_entries = self._segmenter.predict_frame(frame, "frame")
            plan_result = build_plan_result(frame.shape[:2], mask_entries, self._grid_scale)
            grid_handler = plan_result["grid_handler"]
            path = plan_result["path"]
            start = plan_result["start"]
            goal = plan_result["goal"]

            octomap = OctoMap(grid_w=grid_handler.grid_w, grid_h=grid_handler.grid_h, grid_scale=self._grid_scale)
            octomap.update_columns(grid_handler.blocked_obstacles, grid_handler.traversable_obstacles)
            occ2d = octomap.build_occ2d(use_columns=True)
            z_band2d = octomap.build_z_band2d()
            self._octomap_publisher.publish(occ2d, z_band2d)

            if self._mjpeg_server is not None:
                rendered = render_plan_on_frame(frame, grid_handler, path, start, goal, self._grid_scale, class_names=self._class_names, show_labels=True)
                self._mjpeg_server.update_frame(rendered)
        except Exception as e:
            self._node.get_logger().error(f"处理帧失败：{e}")

    def close(self):
        if self._mjpeg_server:
            self._mjpeg_server.close()
        close = getattr(self._segmenter, "close", None)
        if callable(close):
            close()

def parse_args():
    parser = argparse.ArgumentParser(description="实时 OctoMap 发布器（支持 ONNX/RKNN 后端）")
    parser.add_argument("--source", default=None, help="输入源：摄像头索引、视频文件、RTSP 流地址")
    parser.add_argument("--source-topic", default=None, help="ROS2 图像话题（与--source 互斥）")
    parser.add_argument("--weights", default=None, help="权重文件路径（.onnx 或.rknn）")
    parser.add_argument("--data", default=None, help="数据配置 YAML")
    parser.add_argument("--backend", choices=("auto", "onnx", "rknn"), default="auto", help="推理后端")
    parser.add_argument("--device", default=DEFAULT_CONFIG.default_device, help="推理设备")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG.default_conf_thres, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸")
    parser.add_argument("--grid-scale", type=int, default=8, help="栅格缩放")
    parser.add_argument("--frame-id", default="map", help="TF 坐标系 ID")
    parser.add_argument("--resolution", type=float, default=1.0, help="地图分辨率（米/格）")
    parser.add_argument("--publish-octomap", action="store_true", default=True, help="发布 Octomap")
    parser.add_argument("--publish-occupancy-grid", action="store_true", default=False, help="发布 OccupancyGrid")
    parser.add_argument("--publish-markers", action="store_true", default=False, help="发布 MarkerArray")
    parser.add_argument("--mjpeg-host", default="0.0.0.0", help="MJPEG 服务器地址")
    parser.add_argument("--mjpeg-port", type=int, default=8080, help="MJPEG 服务器端口")
    parser.add_argument("--mjpeg", action="store_true", help="启用 MJPEG 流")
    parser.add_argument("--save", action="store_true", help="保存结果")
    parser.add_argument("--project", type=Path, default=None, help="项目目录")
    return parser.parse_args()

def main():
    import rclpy
    from rclpy.node import Node
    args = parse_args()
    rclpy.init()

    class RealtimeNode(Node):
        def __init__(self):
            super().__init__("realtime_octomap_node")
            if args.source is None and args.source_topic is None:
                self.get_logger().error("必须指定 --source 或 --source-topic")
                raise ValueError("必须指定输入源")
            if args.source and args.source_topic:
                self.get_logger().error("--source 和 --source-topic 不能同时指定")
                raise ValueError("输入源冲突")

            self._publisher = RealtimeOctoMapPublisher(
                self, source=args.source, source_topic=args.source_topic,
                weights=args.weights, data_yaml=args.data, backend=args.backend,
                device=args.device, conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                imgsz=args.imgsz[0] if len(args.imgsz) == 1 else tuple(args.imgsz),
                grid_scale=args.grid_scale, frame_id=args.frame_id, resolution=args.resolution,
                publish_octomap=args.publish_octomap, publish_occupancy_grid=args.publish_occupancy_grid,
                publish_markers=args.publish_markers, mjpeg_host=args.mjpeg_host,
                mjpeg_port=args.mjpeg_port, mjpeg_enabled=args.mjpeg,
                save=args.save, project=args.project,
            )
            if not args.source_topic:
                self._timer = self.create_timer(1.0 / 10.0, self._pull_and_process)
                self.get_logger().info("启动拉取模式...")

        def _pull_and_process():
            if args.source:
                if str(args.source).isdigit():
                    cap = cv2.VideoCapture(int(args.source))
                else:
                    cap = cv2.VideoCapture(str(args.source))
                if not cap.isOpened():
                    self.get_logger().error(f"无法打开输入源：{args.source}")
                    return
                while rclpy.ok():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._publisher.process_frame(frame)
                cap.release()

    try:
        node = RealtimeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()