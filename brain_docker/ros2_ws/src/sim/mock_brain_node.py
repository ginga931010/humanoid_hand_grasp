import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math
import time

class MockBrainNode(Node):
    def __init__(self):
        super().__init__('mock_brain_node')
        
        # 建立發布者，Topic 名稱設定為 '/target_angles'
        # STM32 端之後需要 subscribe 這個同名的 Topic
        self.publisher_ = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        
        # 設定發布頻率為 10 Hz (每 0.1 秒發布一次)
        timer_period = 0.0167
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.start_time = time.time()
        self.get_logger().info("🧠 模擬大腦已啟動！開始向 STM32 發布 11 個關節目標角度...")
        self.j = 0.0  # 用於生成變化的角度值
    def timer_callback(self):
        # 建立 Float32MultiArray 訊息物件
        msg = Float32MultiArray()
        
        # 計算經過的時間，用來生成平滑的正弦波
        t = time.time() - self.start_time
        
        # 準備一個空陣列來存放 11 個角度
        target_angles = []
        
        
        for i in range(11):
            
            self.j += 0.001
            angle = self.j
            target_angles.append(float(angle))
            
        # 在 Python 中，只需要把陣列直接賦值給 msg.data 即可
        msg.data = target_angles
        
        # 發布訊息
        self.publisher_.publish(msg)
        
        # 將數值格式化印在終端機，方便你即時監控
        angles_str = ", ".join([f"{a:.1f}" for a in target_angles])
        self.get_logger().info(f"發布 11 軸角度: [{angles_str}]")

def main(args=None):
    rclpy.init(args=args)
    node = MockBrainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()