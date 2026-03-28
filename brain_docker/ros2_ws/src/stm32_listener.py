import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class STM32Listener(Node):
    def __init__(self):
        super().__init__('stm32_listener_node')
        
        # 訂閱 STM32 發出來的 Topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/hand_sensor_data',
            self.sensor_callback,
            10  # QoS queue size
        )
        self.get_logger().info("🎧 STM32 監聽節點已啟動，等待機械手感測器資料...")

    def sensor_callback(self, msg):
        # 取出陣列中的資料
        data = msg.data
        if len(data) >= 5:
            thumb, index, middle, ring, pinky = data[:5]
            self.get_logger().info(
                f"收到手指資料 -> 拇指:{thumb:.1f}, 食指:{index:.1f}, 中指:{middle:.1f}, 無名指:{ring:.1f}, 小指:{pinky:.1f}"
            )
        else:
            self.get_logger().warn(f"收到的資料長度不對: {data}")
            

def main(args=None):
    rclpy.init(args=args)
    node = STM32Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()