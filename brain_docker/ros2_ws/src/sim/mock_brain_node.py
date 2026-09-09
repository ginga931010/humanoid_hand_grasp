# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray
# import math
# import time

# class MockBrainNode(Node):
#     def __init__(self):
#         super().__init__('mock_brain_node')
        
#         # 建立發布者，Topic 名稱設定為 '/target_angles'
#         # STM32 端之後需要 subscribe 這個同名的 Topic

#         self.publisher_ = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        
#         # 設定發布頻率為 10 Hz (每 0.1 秒發布一次)
#         timer_period = 0.0167
#         self.timer = self.create_timer(timer_period, self.timer_callback)
        
#         self.start_time = time.time()
#         self.get_logger().info("🧠 模擬大腦已啟動！開始向 STM32 發布 11 個關節目標角度...")
#         self.j = 0.0  # 用於生成變化的角度值
#     def timer_callback(self):
#         # 建立 Float32MultiArray 訊息物件
#         msg = Float32MultiArray()
        
#         # 計算經過的時間，用來生成平滑的正弦波
#         t = time.time() - self.start_time
        
#         # 準備一個空陣列來存放 11 個角度
#         target_angles = []
        
        
#         for i in range(11):
            
#             self.j += 0.001
#             angle = self.j
#             target_angles.append(float(angle))
            
#         # 在 Python 中，只需要把陣列直接賦值給 msg.data 即可
#         msg.data = target_angles
        
#         # 發布訊息
#         self.publisher_.publish(msg)
        
#         # 將數值格式化印在終端機，方便你即時監控
#         angles_str = ", ".join([f"{a:.1f}" for a in target_angles])
#         self.get_logger().info(f"發布 11 軸角度: [{angles_str}]")

# def main(args=None):
#     rclpy.init(args=args)
#     node = MockBrainNode()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

class MockBrainNode(Node):
    def __init__(self):
        super().__init__('mock_brain_node')
        
        # 建立發布者，Topic 名稱設定為 '/target_angle'
        self.publisher_ = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        
        # 設定發布頻率為 60 Hz (每 0.0167 秒發布一次)
        timer_period = 0.0167
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.start_time = time.time()
        self.get_logger().info("🧠 模擬大腦已啟動！開始向 STM32 發布 11 個關節的目標 PWM...")
        
        # ==========================================================
        # ★ 從 rl_inference_node 移植的轉換參數
        # ==========================================================
        self.action_scale = np.array([
            0.8726, 0.8726, 0.8726, 0.8726,  
            0.8726, 0.8726, 0.8726, 0.8726,  
            0.6981,                          
            0.8726,                          
            0.523599                         
        ], dtype=np.float32)

        self.action_offset = np.array([
            0.8726, 0.8726, 0.8726, 0.8726,
            0.8726, 0.8726, 0.8726, 0.8726,
            0.0,
            0.1745,
            -0.523599
        ], dtype=np.float32)

        self.rad_min = self.action_offset - self.action_scale
        self.rad_max = self.action_offset + self.action_scale
        
        self.pwm_min = np.array([
            1050.0, 1200.0, 975.0, 1100.0, 
            1075.0, 1100.0, 1050.0, 975.0, 
            950.0, 1150.0, 1250.0          
        ], dtype=np.float32)

        self.pwm_max = np.array([
            2400.0, 2000.0, 2450.0, 2000.0,
            2400.0, 2000.0, 2350.0, 2000.0,
            1950.0, 1875.0, 1700.0         
        ], dtype=np.float32)

    def rad_to_pwm(self, rad_array):
        # 核心轉換邏輯：將 Radian 映射到 PWM 並使用 clip 防止超出安全範圍
        ratio = (rad_array - self.rad_min) / (self.rad_max - self.rad_min)
        ratio = np.clip(ratio, 0.0, 1.0)
        return ratio * (self.pwm_max - self.pwm_min) + self.pwm_min

    def timer_callback(self):
        msg = Float32MultiArray()
        
        # 計算經過的時間
        t = time.time() - self.start_time
        
        # 產生介於 -1 到 1 之間的平滑正弦波，乘上時間係數可調整擺動速度 (目前為 1.0)
        sin_wave_factor = np.sin(t * 1.0) 
        
        # 根據各軸的偏移量與縮放比例，計算出實際的目標弧度 (Radian)
        target_angles = self.action_offset + (self.action_scale * sin_wave_factor)
        
        # 將弧度轉換為 STM32 實際需要的 PWM 數值
        target_pwm = self.rad_to_pwm(target_angles)
        
        # 將 numpy 陣列轉換為標準 Python list 後指派給 msg.data
        msg.data = target_pwm.tolist()
        
        # 發布訊息
        self.publisher_.publish(msg)
        
        # 將 PWM 數值格式化印在終端機 (四捨五入到整數位方便監控)
        pwm_str = ", ".join([f"{p:.0f}" for p in target_pwm])
        self.get_logger().info(f"發布 11 軸 PWM: [{pwm_str}]")

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