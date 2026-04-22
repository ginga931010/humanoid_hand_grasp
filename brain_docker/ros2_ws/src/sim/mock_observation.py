#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Int32

class MockObservationNode(Node):
    def __init__(self):
        super().__init__('mock_observation_node')
        
        # 建立三個對應的發布器
        self.joint_pub = self.create_publisher(JointState, '/current_angles', 10)
        self.obj_pos_pub = self.create_publisher(Point, '/target_3d_position', 10)
        self.obj_type_pub = self.create_publisher(Int32, '/object_type_id', 10)

        # 必須與推論節點的關節名稱完全一致
        self.joint_names = [
            "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
            "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
            "Revolute_7", "Revolute_8", "Revolute_29"
        ]
        
        # 設定為 40Hz (0.025 秒)
        self.timer = self.create_timer(0.025, self.timer_callback)
        self.start_time = time.time()
        
        self.get_logger().info("🚀 Mock Observation 節點已啟動！")
        self.get_logger().info("正在以 40Hz 發布平滑的模擬觀測資料...")
        
        self.print_counter = 0

    def timer_callback(self):
        # 經過的時間，用來產生連續的波型
        t = time.time() - self.start_time
        
        # --- 1. 發布假的關節狀態 (JointState) ---
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        # 利用 sin 波產生平滑的角度變化，振幅設為 0.2 弧度，每個關節給點相位差
        joint_msg.position = [math.sin(t * 2.0 + i * 0.5) * 0.2 for i in range(len(self.joint_names))]
        self.joint_pub.publish(joint_msg)

        # --- 2. 發布假的目標位置 (Point) ---
        pos_msg = Point()
        pos_msg.x = 0.3  # 假設物體在 X=0.3m 的地方
        pos_msg.y = 0.0
        pos_msg.z = 0.8
        self.obj_pos_pub.publish(pos_msg)

        # --- 3. 發布假的物體類別 (Int32) ---
        type_msg = Int32()
        type_msg.data = 2  # 2代表 cylinder，你可以隨便改
        self.obj_type_pub.publish(type_msg)

        # 每發布 40 次 (約 1 秒) 印出一次提示，避免洗版
        self.print_counter += 1
        if self.print_counter >= 40:
            self.get_logger().info(f"✅ 持續以 40Hz 發送資料中... (首個關節假角度: {joint_msg.position[0]:.3f})")
            self.print_counter = 0

def main(args=None):
    rclpy.init(args=args)
    node = MockObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('收到中斷訊號，關閉 Mock 節點...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()