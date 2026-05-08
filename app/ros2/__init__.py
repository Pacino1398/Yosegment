"""ROS2 integration helpers.

This package is intentionally lightweight and optional: importing it should not
hard-require ROS2 dependencies (rclpy, message packages). Keep ROS2 imports
inside runtime code paths/functions to avoid breaking non-ROS environments.
"""