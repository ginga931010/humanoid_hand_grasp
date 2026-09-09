import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class AutoTrajectoryNode(Node):
    def __init__(self):
        super().__init__('auto_trajectory_node')

        # --- 1. 定義關節名稱與實體參數 ---
        self.joint_names = [
            "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
            "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
            "Revolute_7", "Revolute_8", "Revolute_29"
        ]
        self.num_joints = len(self.joint_names)

        self.pwm_min = np.array([
            1050.0, 1200.0, 975.0, 1100.0,  
            1075.0, 1100.0, 1050.0, 975.0,  
            950.0,                       
            1150.0,                       
            1250.0                        
        ], dtype=np.float32)

        self.pwm_max = np.array([
            2400.0, 2000.0, 2450.0, 2000.0,  
            2400.0, 2000.0, 2350.0, 2000.0,  
            1950.0,                          
            1875.0,                          
            1700.0                           
        ], dtype=np.float32)

        self.adc_min = np.array([
            1950.0, 1615.0, 1940.0, 1530.0,   
            1970.0, 1595.0, 2000.0, 1595.0,   
            1465.0,                        
            1240.0,                        
            1470.0                         
        ], dtype=np.float32)

        self.adc_max = np.array([
            3055.0, 2620.0, 3110.0, 2570.0, 
            3130.0, 2630.0, 3110.0, 2650.0, 
            2700.0, 
            2880.0, 
            2475.0  
        ], dtype=np.float32)

        # ==========================================================
        # ★ 安全截斷 (Clipping) 設定區
        # ==========================================================
        self.safe_pwm_min = np.copy(self.pwm_min)
        self.safe_pwm_max = np.copy(self.pwm_max)
        
        # 針對 Revolute_7 (索引 8) 設定安全上下限
        self.safe_pwm_min[8] = 1300.0
        self.safe_pwm_max[8] = 1600.0
        # ==========================================================

        # ==========================================================
        # ★ 在此設定動作目標 (0.0 ~ 1.0)
        # ==========================================================
        self.target_normalized = np.array([
            0.7, 0.2, 0.7, 0.2,  # Revolute_12, 14, 16, 18
            0.7, 0.2, 0.7, 0.2,  # Revolute_20, 22, 24, 26
            0.6,                 # Revolute_7
            0.5,                 # Revolute_8
            0.5                  # Revolute_29
        ], dtype=np.float32)
        # ==========================================================

        # --- 2. 狀態與插值控制變數 ---
        self.current_pwm = np.zeros(self.num_joints, dtype=np.float32)
        self.target_pwm = np.zeros(self.num_joints, dtype=np.float32)
        self.start_pwm = np.zeros(self.num_joints, dtype=np.float32)
        
        # 5Hz 發布，2 秒完成動作 = 10 個插值步
        self.total_steps = 10  
        self.current_step = 0 
        self.has_initialized = False 

        # --- 3. 建立發布器、訂閱器與定時器 ---
        self.target_pub = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        self.state_sub = self.create_subscription(Float32MultiArray, '/current_angle', self.state_callback, 10)

        self.dt = 0.2  # 發布頻率 5Hz
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info("等待實體手指回傳初始角度以建立計算起點...")

    def adc_to_pwm(self, joint_idx, raw_adc):
        """將單一關節的 ADC 轉換為標準 PWM"""
        a_min = self.adc_min[joint_idx]
        a_max = self.adc_max[joint_idx]
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        
        ratio = (raw_adc - a_min) / (a_max - a_min)
        ratio = np.clip(ratio, 0.0, 1.0) 
        return ratio * (p_max - p_min) + p_min

    def state_callback(self, msg):
        """僅在初始化階段觸發一次，確保從物理當前位置出發"""
        if not self.has_initialized and len(msg.data) >= self.num_joints:
            for i in range(self.num_joints):
                raw_adc = msg.data[i]
                self.start_pwm[i] = self.adc_to_pwm(i, raw_adc)
            
            # 將 0~1 的設定值映射為實體的目標 PWM (使用原始的 pwm_max / pwm_min 計算真實比例)
            self.target_pwm = self.target_normalized * (self.pwm_max - self.pwm_min) + self.pwm_min
            
            self.has_initialized = True
            self.get_logger().info("✅ 初始位置抓取完畢，開始以 5Hz 執行插值動作。")

    def timer_callback(self):
        """每 0.2 秒執行一次 (5Hz)"""
        if not self.has_initialized:
            return

        # 若還沒走完插值步數，則依比例推進
        if self.current_step < self.total_steps:
            self.current_step += 1
            progress = self.current_step / self.total_steps
            
            # 線性插值：當下位置 = 起點 + (總距離 * 進度比例)
            self.current_pwm = self.start_pwm + (self.target_pwm - self.start_pwm) * progress
            self.get_logger().info(f"移動進度: {self.current_step}/{self.total_steps} (進度: {progress * 100:.0f}%)")
        else:
            self.current_pwm = np.copy(self.target_pwm)

        # ==========================================================
        # ★ 最終輸出前，將數值強制限制在安全範圍內
        # ==========================================================
        final_pwm = np.clip(self.current_pwm, self.safe_pwm_min, self.safe_pwm_max)

        # 封裝並發布被截斷後的最終安全數值
        msg = Float32MultiArray()
        msg.data = final_pwm.tolist()
        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()