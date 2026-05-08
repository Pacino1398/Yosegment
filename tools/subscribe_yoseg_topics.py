#!/usr/bin/env python3
"""订阅 Yosegment 工程中 ROS 发布的话题

使用方法:
    # 需要 ROS2 环境
    ros2 topic echo /yoseg/occ2d_grid
    ros2 topic echo /yoseg/z_band_markers
    
    # 或使用此脚本订阅
    python tools/subscribe_yoseg_topics.py
"""

import argparse
import sys


def subscribe_occ2d_grid(node_name: str = "yoseg_occ2d_subscriber"):
    """订阅 /yoseg/occ2d_grid 话题 (OccupancyGrid)"""
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import OccupancyGrid
    except ImportError as e:
        print(f"错误：需要 ROS2 环境。{e}")
        print("请确保已安装 ROS2 并 source 环境，例如:")
        print("  Windows: 运行 'C:\\dev\\ros2_humble\\setup.bat' (根据实际安装路径)")
        print("  Linux: source /opt/ros/humble/setup.bash")
        sys.exit(1)
    
    class Occ2dSubscriber(Node):
        def __init__(self):
            super().__init__(node_name)
            self.subscription = self.create_subscription(
                OccupancyGrid,
                '/yoseg/occ2d_grid',
                self.listener_callback,
                10
            )
            self.get_logger().info("开始订阅 /yoseg/occ2d_grid 话题")
        
        def listener_callback(self, msg: OccupancyGrid):
            self.get_logger().info(
                f"收到 OccupancyGrid: 大小={msg.info.width}x{msg.info.height}, "
                f"分辨率={msg.info.resolution}m"
            )
            # 打印部分数据
            if msg.data:
                sample_size = min(10, len(msg.data))
                self.get_logger().info(f"  数据样本 (前{sample_size}个): {msg.data[:sample_size]}")
    
    return Occ2dSubscriber


def subscribe_z_band_markers(node_name: str = "yoseg_zband_subscriber"):
    """订阅 /yoseg/z_band_markers 话题 (MarkerArray)"""
    try:
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import MarkerArray
    except ImportError as e:
        print(f"错误：需要 ROS2 环境。{e}")
        sys.exit(1)
    
    class ZBandSubscriber(Node):
        def __init__(self):
            super().__init__(node_name)
            self.subscription = self.create_subscription(
                MarkerArray,
                '/yoseg/z_band_markers',
                self.listener_callback,
                10
            )
            self.get_logger().info("开始订阅 /yoseg/z_band_markers 话题")
        
        def listener_callback(self, msg: MarkerArray):
            self.get_logger().info(f"收到 MarkerArray: 包含 {len(msg.markers)} 个 marker")
            for i, marker in enumerate(msg.markers):
                self.get_logger().info(f"  Marker[{i}]: 命名空间={marker.ns}, ID={marker.id}")
    
    return ZBandSubscriber


def main():
    parser = argparse.ArgumentParser(description="订阅 Yosegment ROS 话题")
    parser.add_argument(
        "--topic",
        choices=["occ2d", "zband", "all"],
        default="all",
        help="选择要订阅的话题：occ2d, zband, 或 all"
    )
    parser.add_argument(
        "--ros2-command",
        action="store_true",
        help="仅打印 ROS2 命令行，不执行订阅"
    )
    args = parser.parse_args()
    
    if args.ros2_command:
        print("# ROS2 命令行订阅方式:")
        print("ros2 topic echo /yoseg/occ2d_grid")
        print("ros2 topic echo /yoseg/z_band_markers")
        print("")
        print("# 或一次性查看所有话题:")
        print("ros2 topic list | grep yoseg")
        return
    
    try:
        import rclpy
    except ImportError:
        print("错误：未检测到 ROS2 环境")
        print("")
        print("请按照以下步骤设置环境:")
        print("1. 安装 ROS2 Humble (https://docs.ros.org/en/humble/Installation.html)")
        print("2. 在终端中 source ROS2 环境:")
        print("   Windows: C:\\dev\\ros2_humble\\setup.bat (根据实际路径)")
        print("   Linux: source /opt/ros/humble/setup.bash")
        print("")
        print("3. 运行 ROS2 节点发布话题后，使用以下命令订阅:")
        print("   ros2 topic echo /yoseg/occ2d_grid")
        print("   ros2 topic echo /yoseg/z_band_markers")
        sys.exit(1)
    
    rclpy.init()
    
    try:
        if args.topic == "occ2d" or args.topic == "all":
            node_class = subscribe_occ2d_grid()
            node = node_class()
            print(f"已启动节点订阅 /yoseg/occ2d_grid 话题")
            print("按 Ctrl+C 停止订阅")
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()
    except KeyboardInterrupt:
        print("\n订阅已停止")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()