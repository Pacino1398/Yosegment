from __future__ import annotations
"""Realtime ROS2 node: subscribe to segmentation results and publish occ2d + z_band2d.

Why this file exists
--------------------
The existing `app/mapping/ros2_publish_occ_zband.py` publishes from a .npz file.
For realtime usage we need to compute occ2d/z_band2d per-frame and publish them.

This node keeps ROS2 imports inside runtime so the repo still works without ROS2.

Expected input
--------------
A topic carrying per-frame mask/segmentation results. Because this repo currently
doesn't define a ROS message for segmentation masks, this node provides two modes:

1) "snapshot-json" mode (default):
   - Subscribe: std_msgs/String, JSON payload compatible with OctoMap.export_planner_snapshot()
     at least containing:
       - bounds: [grid_w, grid_h, max_z]
       - grid_scale: int
       - blocked_obstacles: list[[x,y], ...] or list["(x,y)"] (best-effort parse)
       - columns: optional (for z_band2d) as dict mapping "(x,y)" -> column dict

2) "external-hook" mode:
   - You integrate in your pipeline by calling OccZBandPublisher directly
     (recommended). See app/ros2/occ_zband_publisher.py.

If you already have your realtime pipeline in-process (python), prefer option (2)
and avoid ROS subscription overhead.

Published topics
----------------
- /yoseg/occ2d_grid     : nav_msgs/OccupancyGrid
- /yoseg/z_band_markers : visualization_msgs/MarkerArray
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.mapping.octomap import ColumnState, OctoMap
from app.ros2.occ_zband_publisher import OccZBandPublishConfig, OccZBandPublisher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime publish occ2d + z_band2d to ROS2 from JSON snapshots")
    p.add_argument("--sub-topic", type=str, default="/yoseg/octomap_snapshot_json", help="Input snapshot JSON topic")
    p.add_argument("--frame-id", type=str, default="map")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=10.0, help="Max publish rate (Hz); publishes on message arrival anyway")

    p.add_argument("--occ-topic", type=str, default="/yoseg/occ2d_grid")
    p.add_argument("--zband-topic", type=str, default="/yoseg/z_band_markers")

    p.add_argument("--occupied-threshold", type=int, default=1)
    p.add_argument("--marker-step", type=int, default=1)
    p.add_argument("--marker-alpha", type=float, default=0.8)
    p.add_argument("--marker-scale", type=float, default=0.08)
    return p.parse_args()


def _parse_cell(cell: Any) -> tuple[int, int] | None:
    if cell is None:
        return None
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        try:
            return int(cell[0]), int(cell[1])
        except Exception:
            return None
    if isinstance(cell, str):
        # support "(x, y)" or "x,y"
        s = cell.strip().lstrip("(").rstrip(")")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2:
            try:
                return int(float(parts[0])), int(float(parts[1]))
            except Exception:
                return None
    return None


def snapshot_to_maps(snapshot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = snapshot.get("bounds") or [0, 0, 0]
    grid_w = int(bounds[0])
    grid_h = int(bounds[1])
    grid_scale = int(snapshot.get("grid_scale", 1))

    octo = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)

    # Prefer columns (richer z_band), fallback to blocked_obstacles
    columns_raw = snapshot.get("columns")
    if isinstance(columns_raw, dict) and columns_raw:
        added: dict[tuple[int, int], ColumnState | dict[str, Any] | int | float | None] = {}
        for k, v in columns_raw.items():
            cell = _parse_cell(k)
            if cell is None:
                continue
            added[cell] = v  # normalized by OctoMap.update_columns()
        octo.update_columns(added, removed=None)
    else:
        blocked = snapshot.get("blocked_obstacles") or []
        cells = set(filter(None, (_parse_cell(c) for c in blocked)))
        octo.build_octomap(cells)

    occ2d = octo.build_occ2d(use_columns=bool(columns_raw))
    z_band2d = octo.build_z_band2d()
    return occ2d, z_band2d


def main() -> None:
    args = parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class NodeImpl(Node):
        def __init__(self):
            super().__init__("yoseg_realtime_occ_zband_node")

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
            self._sub = self.create_subscription(String, str(args.sub_topic), self._on_msg, 10)
            self.get_logger().info(f"subscribing snapshot json: {args.sub_topic}")

        def _on_msg(self, msg: String):
            try:
                snapshot = json.loads(msg.data)
                if not isinstance(snapshot, dict):
                    raise ValueError("snapshot must be a JSON object")
                occ2d, z_band2d = snapshot_to_maps(snapshot)
                self._pub.publish(occ2d, z_band2d)
            except Exception as e:
                self.get_logger().warn(f"failed to publish from snapshot: {e}")

    rclpy.init()
    node = NodeImpl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()