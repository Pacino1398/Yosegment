from __future__ import annotations

"""ROS2 publisher for Phase2 v0 outputs: occ2d + z_band2d.

This node is intentionally kept in-repo (not a full ROS2 package) so you can
prototype quickly. Run it inside a ROS2 Python environment that has:
- rclpy
- nav_msgs
- std_msgs
- visualization_msgs

Inputs
------
- .npz file exported by `python -m app.mapping.octomap_export --out-npz ...`
  containing: occ2d(uint8[H,W]), z_band2d(float32[H,W,2]), grid_scale.

Published topics
----------------
- /yoseg/occ2d_grid   : nav_msgs/OccupancyGrid
- /yoseg/z_band_markers : visualization_msgs/MarkerArray (per-cell vertical segments)

Coordinate convention
---------------------
The exported arrays are in *grid coordinates* (x right, y down in image).
OccupancyGrid expects row-major with origin at lower-left for visualization.
We flip Y for better RViz alignment.

Note
----
This file is not imported anywhere by default to avoid hard dependency on ROS2.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish occ2d + z_band2d (.npz) to ROS2")
    p.add_argument("--npz", type=str, required=True, help="Input npz path from octomap_export")
    p.add_argument("--frame-id", type=str, default="map", help="TF frame id")
    p.add_argument("--resolution", type=float, default=1.0, help="OccupancyGrid resolution (meters per cell)")
    p.add_argument("--rate", type=float, default=2.0, help="Publish rate (Hz)")

    p.add_argument("--occupied-threshold", type=int, default=1, help="occ2d>threshold -> occupied")

    # Marker viz controls
    p.add_argument("--marker-step", type=int, default=1, help="Downsample markers by step in x/y")
    p.add_argument("--marker-alpha", type=float, default=0.8)
    p.add_argument("--marker-scale", type=float, default=0.08, help="LINE_LIST width")
    return p.parse_args()


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
    # Lazily import ROS messages (only available in ROS2 env)
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    H, W, _ = z_band2d.shape
    z_lo = z_band2d[..., 0]
    z_hi = z_band2d[..., 1]

    markers = MarkerArray()

    # One LINE_LIST marker is lighter than per-cell markers.
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

            # Convert to metric coords (cell center). Flip y for RViz.
            xf = (x + 0.5) * resolution
            yf = (H - 1 - y + 0.5) * resolution

            m.points.append(Point(x=xf, y=yf, z=lo))
            m.points.append(Point(x=xf, y=yf, z=hi))

    markers.markers.append(m)
    return markers


def main() -> None:
    args = parse_args()

    # ROS2 imports
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import OccupancyGrid

    npz = Path(args.npz)
    if not npz.exists():
        raise FileNotFoundError(npz)

    data = np.load(npz)
    occ2d = data["occ2d"]
    z_band2d = data["z_band2d"]

    if occ2d.ndim != 2:
        raise ValueError(f"occ2d must be 2D, got shape={occ2d.shape}")
    if z_band2d.ndim != 3 or z_band2d.shape[-1] != 2:
        raise ValueError(f"z_band2d must be [H,W,2], got shape={z_band2d.shape}")

    H, W = occ2d.shape

    class PublisherNode(Node):
        def __init__(self):
            super().__init__("yoseg_occ_zband_publisher")
            self.pub_occ = self.create_publisher(OccupancyGrid, "/yoseg/occ2d_grid", 1)
            from visualization_msgs.msg import MarkerArray

            self.pub_markers = self.create_publisher(MarkerArray, "/yoseg/z_band_markers", 1)

            period = 1.0 / max(0.1, float(args.rate))
            self.timer = self.create_timer(period, self._on_timer)

        def _on_timer(self):
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(args.frame_id)

            msg.info.resolution = float(args.resolution)
            msg.info.width = int(W)
            msg.info.height = int(H)
            msg.info.origin.position.x = 0.0
            msg.info.origin.position.y = 0.0
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0

            msg.data = _to_occupancygrid_data(occ2d, threshold=int(args.occupied_threshold))
            self.pub_occ.publish(msg)

            markers = _build_z_band_markers(
                z_band2d=z_band2d,
                frame_id=str(args.frame_id),
                resolution=float(args.resolution),
                step=int(args.marker_step),
                alpha=float(args.marker_alpha),
                scale=float(args.marker_scale),
            )
            markers.markers[0].header.stamp = msg.header.stamp
            self.pub_markers.publish(markers)

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
