import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort

from std_msgs.msg import Int32, Float32MultiArray, Bool
from sensor_msgs.msg import JointState
# from geometry_msgs.msg import Point  # 已經不需要了，已註解
from math import pi


class RLInferenceNode(Node):
    def __init__(self):
        super().__init__('rl_inference_node')
        
        # 1. 初始化 ONNX Runtime
        # model_path = '/ros2_ws/src/17-01-01-01-01_SH250-3900.onnx'
        # model_path = '/ros2_ws/src/18-01_Alpha01.onnx'
        # model_path = '/ros2_ws/src/39-01-01-02_LowForce.onnx'
        # model_path = '/ros2_ws/src/39-01-01-03-01-01_HighDelay.onnx'
        model_path = '/ros2_ws/src/42-01.onnx'
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': False,
                'trt_engine_cache_enable': False,
                'trt_engine_cache_path': '/ros2_ws/src/',
            }),
            'CUDAExecutionProvider'
        ]
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.get_logger().info(f"Active Providers: {self.ort_session.get_providers()}")
        self.input_name = self.ort_session.get_inputs()[0].name
        
        # --- 2. 定義關節名稱與實體參數 ---
        self.joint_names = [
            "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
            "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
            "Revolute_7", "Revolute_8", "Revolute_29"
        ]
        self.num_joints = len(self.joint_names)

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

        self.action_alpha = 0.5

        # ==========================================================
        # ★ PWM 與 Radian 轉換參數設定區
        # ==========================================================
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

        # --- 3. 狀態暫存區 ---
        self.current_pos = np.zeros(self.num_joints, dtype=np.float32)
        self.previous_pos = np.zeros(self.num_joints, dtype=np.float32)
        self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
        self.obs_vel_alpha = 1.0
        self.target_pos = np.zeros(self.num_joints, dtype=np.float32) 
        self.obj_type_one_hot = np.zeros(3, dtype=np.float32)

        # ★ 新增：Action History Buffer (12步 x 11關節)
        # 預設儲存方式為 chronologically: [t-12, t-11, ... t-1]
        self.history_length = 12
        self.action_history_buf = np.zeros((self.history_length, self.num_joints), dtype=np.float32)

        self.received_joint = False
        self.received_obj_type = False
        self.is_running = 0
        self.is_first_step = True  

        # --- 4. 建立訂閱器與發布器 ---
        self.joint_sub = self.create_subscription(Float32MultiArray, '/current_angle', self.joint_state_callback, 10)
        self.obj_type_sub = self.create_subscription(Int32, '/object_type_id', self.obj_type_callback, 10)
        self.cmd_sub = self.create_subscription(Int32, '/start_policy_topic', self.cmd_callback, 10)

        self.target_pub = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        
        self.dt = 0.025 # 40Hz
        self.timer = self.create_timer(self.dt, self.inference_loop)

    def pwm_to_rad(self, joint_idx, pwm_val):
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        r_min = self.rad_min[joint_idx]
        r_max = self.rad_max[joint_idx]
        ratio = (pwm_val - p_min) / (p_max - p_min)
        ratio = max(0.0, min(1.0, ratio)) 
        return ratio * (r_max - r_min) + r_min

    def adc_to_pwm(self, joint_idx, raw_adc):
        a_min = self.adc_min[joint_idx]
        a_max = self.adc_max[joint_idx]
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        ratio = (raw_adc - a_min) / (a_max - a_min)
        ratio = max(0.0, min(1.0, ratio)) 
        return ratio * (p_max - p_min) + p_min

    def rad_to_pwm(self, rad_array):
        ratio = (rad_array - self.rad_min) / (self.rad_max - self.rad_min)
        ratio = np.clip(ratio, 0.0, 1.0)
        return ratio * (self.pwm_max - self.pwm_min) + self.pwm_min

    def cmd_callback(self, msg):
        self.is_running = msg.data
        status = "啟動" if self.is_running == 1 else "停止"
        self.get_logger().info(f"🕹️ 收到控制指令：策略推論已 {status}")

    def joint_state_callback(self, msg):
        if len(msg.data) >= self.num_joints:
            for i in range(self.num_joints):
                raw_adc = msg.data[i]
                standard_pwm = self.adc_to_pwm(i, raw_adc)
                self.current_pos[i] = self.pwm_to_rad(i, standard_pwm)
            self.received_joint = True
        else:
            self.get_logger().warn(f"⚠️ 收到的關節資料長度不足！預期 {self.num_joints}，實際 {len(msg.data)}")

    def obj_type_callback(self, msg):
        type_id = msg.data
        self.obj_type_one_hot = np.zeros(3, dtype=np.float32)
        if 0 <= type_id < 3:
            self.obj_type_one_hot[type_id] = 1.0
        self.received_obj_type = True

    def build_tactile_proxy(self):
        joint_pos_error = self.target_pos - self.current_pos
        velocity_penalty = np.exp(-5.0 * np.abs(self.filtered_vel))
        contact_signal = joint_pos_error * velocity_penalty
        
        tactile_proxy = np.concatenate([
            joint_pos_error, 
            self.filtered_vel, 
            contact_signal
        ], axis=-1)
        return tactile_proxy

    def inference_loop(self):
        if self.is_running == 0:
            return
            
        # 移除 object_pos 的等待條件
        if not (self.received_joint and self.received_obj_type):
            self.get_logger().warn("等待資料齊全中...", throttle_duration_sec=2.0)
            return

        if self.is_first_step:
            self.previous_pos = np.copy(self.current_pos)
            self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
            self.target_pos = np.copy(self.current_pos) 
            self.action_history_buf.fill(0.0) # 初始化 Action History
            self.is_first_step = False
            
        raw_vel = (self.current_pos - self.previous_pos) / self.dt
        self.filtered_vel = (self.obs_vel_alpha * raw_vel) + ((1.0 - self.obs_vel_alpha) * self.filtered_vel)
        self.previous_pos = np.copy(self.current_pos)

        tactile_proxy = self.build_tactile_proxy()
        
        # --- 組合 Observation (維度總和: 179) ---
        # ⚠️ 請確保這裡的串接順序與你在 Isaac Lab 的 ObservationCfg 裡面定義的一模一樣！
        obs = np.concatenate([
            self.current_pos,                   # 11維
            self.action_history_buf.flatten(),  # 132維 (替換掉原本的 11維 last_action)
            self.obj_type_one_hot,              # 3維
            tactile_proxy                       # 33維
        ], axis=-1).astype(np.float32)

        obs_input = np.expand_dims(obs, axis=0)
        if np.isnan(obs_input).any():
            self.get_logger().error("⚠️ 抓到了！送進模型的 Observation 裡面含有 NaN！")
            return 

        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        action_output = self.ort_session.run([output_name], {input_name: obs_input})[0]
        
        action = np.clip(action_output[0], -1.0, 1.0)
        
        # ★ 更新 Action History Buffer (將舊陣列往前推一格，並把最新的 action 塞到最後面)
        self.action_history_buf = np.roll(self.action_history_buf, shift=-1, axis=0)
        self.action_history_buf[-1] = np.copy(action)

        raw_target_pos = (action * self.action_scale) + self.action_offset
        self.target_pos = ((self.action_alpha * raw_target_pos) + ((1.0 - self.action_alpha) * self.target_pos))
        
        target_pwm = self.rad_to_pwm(self.target_pos)
        
        msg = Float32MultiArray()
        msg.data = target_pwm.tolist() 
        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()