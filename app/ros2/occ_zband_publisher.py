from __future__ import annotations
"""ROS2 publisher for Yosegment local 2.5D map (occ2d + z_band2d).

This module is designed to be imported from the realtime pipeline and can also
be used standalone.

Published topics (default)
--------------------------
- /yoseg/occ2d_grid     : nav_msgs/OccupancyGrid
- /yoseg/z_band_markers : visualization_msgs/MarkerArray

Notes
-----
- All ROS2 imports are done lazily so that importing this file in a non-ROS
  python environment will not immediately fail.
- Input arrays are in grid coords (x right, y down). OccupancyGrid is flipped
  in Y for RViz display.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _to_occupancygrid_data(occ2d: np.ndarray, *, threshold: int = 0) -> list[int]:
    # occ2d: uint8[H,W] in grid coords.
    # Convert to int8 OccupancyGrid convention: -1 unknown, 0 free, 100 occupied.
    # Here we don't have unknown => always 0/100.
    occ = (occ2d.astype(np.int32) > int(threshold)).astype(np.int32) * 100

    # Flip Y for RViz: grid(y=0) top row -> OccupancyGrid origin at bottom-left
    occ = np.flipud(occ)

    # Flatten row-major
    return occ.reshape(-1).astype(np.int8).tolist()


def _build_z_band_markers(
    *,
    z_band2d: np.ndarray,
    frame_id: str,
    resolution: float,
    step: int,
    alpha: float,
    scale: float,
):
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    H, W, _ = z_band2d.shape
    z_lo = z_band2d[..., 0]
    z_hi = z_band2d[..., 1]

    markers = MarkerArray()

    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "z_band"
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = float(scale)
    m.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=float(alpha))
    m.lifetime = Duration(sec=0, nanosec=0)

    st = max(1, int(step))
    for y in range(0, H, st):
        for x in range(0, W, st):
            lo = float(z_lo[y, x])
            hi = float(z_hi[y, x])
            if hi <= 0.0 or hi <= lo:
                continue

            xf = (x + 0.5) * resolution
            yf = (H - 1 - y + 0.5) * resolution  # flip y for RViz

            m.points.append(Point(x=xf, y=yf, z=lo))
            m.points.append(Point(x=xf, y=yf, z=hi))

    markers.markers.append(m)
    return markers


@dataclass(slots=True)
class OccZBandPublishConfig:
    frame_id: str = "map"
    resolution: float = 1.0

    occ_topic: str = "/yoseg/occ2d_grid"
    zband_topic: str = "/yoseg/z_band_markers"

    queue_size: int = 1
    publish_rate_hz: float = 5.0

    occupied_threshold: int = 1

    marker_step: int = 1
    marker_alpha: float = 0.8
    marker_scale: float = 0.08


class OccZBandPublisher:
    """Reusable publisher wrapper.

    Usage
    -----
    pub = OccZBandPublisher(node, OccZBandPublishConfig(...))
    pub.publish(occ2d, z_band2d)
    """

    def __init__(self, node, config: OccZBandPublishConfig):
        # Lazy ROS imports
        from nav_msgs.msg import OccupancyGrid
        from visualization_msgs.msg import MarkerArray

        self._node = node
        self._cfg = config
        self._H: Optional[int] = None
        self._W: Optional[int] = None

        self._pub_occ = node.create_publisher(OccupancyGrid, str(config.occ_topic), int(config.queue_size))
        self._pub_markers = node.create_publisher(MarkerArray, str(config.zband_topic), int(config.queue_size))

    def publish(self, occ2d: np.ndarray, z_band2d: np.ndarray):
        from nav_msgs.msg import OccupancyGrid

        if occ2d.ndim != 2:
            raise ValueError(f"occ2d must be 2D, got shape={occ2d.shape}")
        if z_band2d.ndim != 3 or z_band2d.shape[-1] != 2:
            raise ValueError(f"z_band2d must be [H,W,2], got shape={z_band2d.shape}")

        H, W = occ2d.shape
        if self._H is None:
            self._H, self._W = int(H), int(W)
        elif (int(H), int(W)) != (int(self._H), int(self._W)):
            # Allow dynamic resize but log once.
            self._node.get_logger().warn(f"occ2d size changed: {(self._H, self._W)} -> {(H, W)}")
            self._H, self._W = int(H), int(W)

        msg = OccupancyGrid()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = str(self._cfg.frame_id)

        msg.info.resolution = float(self._cfg.resolution)
        msg.info.width = int(W)
        msg.info.height = int(H)
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = _to_occupancygrid_data(occ2d, threshold=int(self._cfg.occupied_threshold))
        self._pub_occ.publish(msg)

        markers = _build_z_band_markers(
            z_band2d=z_band2d,
            frame_id=str(self._cfg.frame_id),
            resolution=float(self._cfg.resolution),
            step=int(self._cfg.marker_step),
            alpha=float(self._cfg.marker_alpha),
            scale=float(self._cfg.marker_scale),
        )
        markers.markers[0].header.stamp = msg.header.stamp
        self._pub_markers.publish(markers)


def main() -> None:
    """Standalone entry: load npz and publish at fixed rate.

    Example
    -------
    python -m app.ros2.occ_zband_publisher --npz runs/debug/map_v0.npz --rate 2
    """
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Publish occ2d + z_band2d (.npz) to ROS2")
    p.add_argument("--npz", type=str, required=True, help="Input npz path (from octomap_export)")
    p.add_argument("--frame-id", type=str, default="map", help="TF frame id")
    p.add_argument("--resolution", type=float, default=1.0, help="OccupancyGrid resolution (meters per cell)")
    p.add_argument("--rate", type=float, default=2.0, help="Publish rate (Hz)")

    p.add_argument("--occupied-threshold", type=int, default=1, help="occ2d>threshold -> occupied")

    p.add_argument("--marker-step", type=int, default=1, help="Downsample markers by step in x/y")
    p.add_argument("--marker-alpha", type=float, default=0.8)
    p.add_argument("--marker-scale", type=float, default=0.08, help="LINE_LIST width")
    p.add_argument("--occ-topic", type=str, default="/yoseg/occ2d_grid")
    p.add_argument("--zband-topic", type=str, default="/yoseg/z_band_markers")
    args = p.parse_args()

    import rclpy
    from rclpy.node import Node

    npz = Path(args.npz)
    if not npz.exists():
        raise FileNotFoundError(npz)
    data = np.load(npz)
    occ2d = data["occ2d"]
    z_band2d = data["z_band2d"]

    class PublisherNode(Node):
        def __init__(self):
            super().__init__("yoseg_occ_zband_publisher")
            cfg = OccZBandPublishConfig(
                frame_id=str(args.frame_id),
                resolution=float(args.resolution),
                occ_topic=str(args.occ_topic),
                zband_topic=str(args.zband_topic),
                publish_rate_hz=float(args.rate),
                occupied_threshold=int(args.occupied_threshold),
                marker_step=int(args.marker_step),
                marker_alpha=float(args.marker_alpha),
                marker_scale=float(args.marker_scale),
            )
            self._pub = OccZBandPublisher(self, cfg)
            self._timer = self.create_timer(1.0 / max(0.1, float(args.rate)), self._on_timer)

        def _on_timer(self):
            self._pub.publish(occ2d, z_band2d)

    rclpy.init()
    node = PublisherNode()
    node.get_logger().info(f"publishing occ2d/z_band2d from: {npz}")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()