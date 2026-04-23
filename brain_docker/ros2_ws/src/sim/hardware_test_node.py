import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time

class HardwareTestNode(Node):
    def __init__(self):
        super().__init__('hardware_test_node')
        
        # --- 1. 完整複製你的硬體參數與關節名稱 ---
        self.joint_names = [
            "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
            "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
            "Revolute_7", "Revolute_8", "Revolute_29"
        ]
        self.num_joints = len(self.joint_names)

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

        self.adc_min = np.array([
            1950.0, 1615.0, 1940.0, 1530.0,   
            1970.0, 1595.0, 2000.0, 1595.0,   
            1465.0, 1240.0, 1470.0            
        ], dtype=np.float32)

        self.adc_max = np.array([
            3055.0, 2620.0, 3110.0, 2570.0, 
            3130.0, 2630.0, 3110.0, 2650.0, 
            2700.0, 2880.0, 2475.0  
        ], dtype=np.float32)

        # 儲存收到的即時狀態
        self.current_pwm_feedback = np.zeros(self.num_joints, dtype=np.float32)

        # --- 2. 建立通訊 ---
        self.target_pub = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        self.joint_sub = self.create_subscription(Float32MultiArray, '/current_angle', self.joint_state_callback, 10)
        
        # 設定發布頻率為 40Hz
        self.dt = 0.025
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.start_time = time.time()
        self.log_counter = 0
        
        self.get_logger().info("🔧 硬體測試節點已啟動！正在依序測試各關節 (0% -> 100% -> 0%)...")

    def adc_to_pwm(self, joint_idx, raw_adc):
        """將 ADC 轉換回標準 PWM 以便比較"""
        a_min = self.adc_min[joint_idx]
        a_max = self.adc_max[joint_idx]
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        
        ratio = (raw_adc - a_min) / (a_max - a_min)
        # 這裡故意不限制 0~1，讓你能看到 ADC 是否超出了你設定的邊界
        return ratio * (p_max - p_min) + p_min

    def joint_state_callback(self, msg):
        if len(msg.data) >= self.num_joints:
            for i in range(self.num_joints):
                # 收到 ADC，馬上轉換成 PWM 儲存起來
                self.current_pwm_feedback[i] = self.adc_to_pwm(i, msg.data[i])

    def timer_callback(self):
        t = time.time() - self.start_time
        
        # 測試邏輯：每 4 秒換下一個馬達
        test_duration = 4.0
        joint_idx = int((t / test_duration) % self.num_joints)
        phase = t % test_duration
        
        # 利用 cos 函數製造一個平滑的 0 -> 1 -> 0 比例
        # phase=0 時 ratio=0, phase=2 時 ratio=1, phase=4 時 ratio=0
        ratio = 0.5 * (1.0 - math.cos(phase * math.pi / (test_duration / 2.0)))
        
        # 準備要發送的目標陣列 (預設全部保持在 pwm_min，即 0% 位置)
        target_pwms = np.copy(self.pwm_min)
        
        # 只有當前正在測試的馬達，會套用掃描比例
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        target_pwms[joint_idx] = p_min + ratio * (p_max - p_min)
        
        # 發布指令給 STM32
        msg = Float32MultiArray()
        msg.data = target_pwms.tolist()
        self.target_pub.publish(msg)
        
        # 每隔大約 0.5 秒，印出一次對比 Log (避免畫面刷太快看不清楚)
        self.log_counter += 1
        if self.log_counter >= int(0.5 / self.dt):
            self.log_counter = 0
            
            target_val = target_pwms[joint_idx]
            actual_val = self.current_pwm_feedback[joint_idx]
            error = actual_val - target_val
            
            joint_name = self.joint_names[joint_idx]
            ratio_percent = ratio * 100
            
            self.get_logger().info(
                f"測試 [{joint_name}] {ratio_percent:5.1f}% | "
                f"目標 PWM: {target_val:6.1f} | 實際 PWM: {actual_val:6.1f} | 誤差: {error:6.1f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = HardwareTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()