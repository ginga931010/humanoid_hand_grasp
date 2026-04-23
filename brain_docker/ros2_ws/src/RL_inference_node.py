# import rclpy
# from rclpy.node import Node
# import numpy as np
# import onnxruntime as ort

# from std_msgs.msg import Int32, Float32MultiArray, Bool
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import Point
# from math import pi


# class RLInferenceNode(Node):
#     def __init__(self):
#         super().__init__('rl_inference_node')
        
#         # 1. 初始化 ONNX Runtime
#         model_path = '/ros2_ws/src/policy.onnx'
#         # self.ort_session = ort.InferenceSession(
#         #     model_path, 
#         #     providers=['CPUExecutionProvider'] # 強迫只用 CPU 算
#         # )
#         providers = [
#             ('TensorrtExecutionProvider', {
#                 'device_id': 0,
#                 'trt_fp16_enable': False,
#                 'trt_engine_cache_enable': False,
#                 'trt_engine_cache_path': '/ros2_ws/src/',
                
                
#             }),
#             'CUDAExecutionProvider'
#         ]
#         self.ort_session = ort.InferenceSession(model_path, providers=providers)
#         print("Active Providers:", self.ort_session.get_providers())
#         self.input_name = self.ort_session.get_inputs()[0].name
        
#         # --- 2. 定義關節名稱與實體參數 ---
#         self.joint_names = [
#             "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
#             "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
#             "Revolute_7", "Revolute_8", "Revolute_29"
#         ]
#         self.num_joints = len(self.joint_names)

#         # 根據 ActionsCfg 定義 Scale 與 Offset (順序必須對齊 joint_names)
#         self.action_scale = np.array([
#             0.8726, 0.8726, 0.8726, 0.8726,  # 12, 14, 16, 18
#             0.8726, 0.8726, 0.8726, 0.8726,  # 20, 22, 24, 26
#             0.6981,                          # 7
#             0.8726,                          # 8
#             0.523599                         # 29
#         ], dtype=np.float32)

#         self.action_offset = np.array([
#             0.8726, 0.8726, 0.8726, 0.8726,
#             0.8726, 0.8726, 0.8726, 0.8726,
#             0.0,
#             0.1745,
#             -0.523599
#         ], dtype=np.float32)

#         self.action_alpha = 0.5  # 來自 EMAJointPositionActionCfg

#         # --- 3. 狀態暫存區 ---
#         self.current_pos = np.zeros(self.num_joints, dtype=np.float32)
#         self.previous_pos = np.zeros(self.num_joints, dtype=np.float32)
        
#         # 觀測值輸入用的濾波速度
#         self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
#         self.obs_vel_alpha = 0.2  
        
#         # 紀錄 RL 模型原始輸出的 action (用於 observation 裡的 last_action)
#         self.last_action = np.zeros(self.num_joints, dtype=np.float32)
#         # 紀錄經過 Scale, Offset 與 EMA 處理後的最終目標位置
#         self.target_pos = np.zeros(self.num_joints, dtype=np.float32) 

#         self.obj_pos = np.zeros(3, dtype=np.float32)
#         self.obj_type_one_hot = np.zeros(3, dtype=np.float32)

#         # 狀態標記
#         self.received_joint = False
#         self.received_obj_pos = False
#         self.received_obj_type = False
#         self.is_running = 0
#         self.is_first_step = True  

#         # --- 4. 建立訂閱器與發布器 ---
#         self.joint_sub = self.create_subscription(JointState, '/current_angles', self.joint_state_callback, 10)
#         self.obj_pos_sub = self.create_subscription(Point, '/target_3d_position', self.obj_pos_callback, 10)
#         self.obj_type_sub = self.create_subscription(Int32, '/object_type_id', self.obj_type_callback, 10)
#         self.cmd_sub = self.create_subscription(Int32, '/start_policy_topic', self.cmd_callback, 10)

#         # 這裡發布的會是處理完的「目標角度 (徑度)」，而不是原始的 -1~1 action
#         self.target_pub = self.create_publisher(Float32MultiArray, '/target_angles', 10)
        
#         self.dt = 0.025 # 50Hz
#         self.timer = self.create_timer(self.dt, self.inference_loop)

#     def cmd_callback(self, msg):
#         self.is_running = msg.data
#         status = "啟動" if self.is_running == 1 else "停止"
#         self.get_logger().info(f"🕹️ 收到控制指令：策略推論已 {status}")

#     def joint_state_callback(self, msg):
#         for i, name in enumerate(self.joint_names):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 self.current_pos[i] = msg.position[idx]
#         self.received_joint = True

#     def obj_pos_callback(self, msg):
#         # 修正：直接讀取真實資料，移除所有人工雜訊
#         self.obj_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
#         self.received_obj_pos = True

#     def obj_type_callback(self, msg):
#         type_id = msg.data
#         self.obj_type_one_hot = np.zeros(3, dtype=np.float32)
#         if 0 <= type_id < 3:
#             self.obj_type_one_hot[type_id] = 1.0
#         self.received_obj_type = True

#     def build_tactile_proxy(self):
#         # 這裡的 self.target_pos 已經是經過 Action EMA 處理後的正確物理目標位置了！
#         joint_pos_error = self.target_pos - self.current_pos
#         velocity_penalty = np.exp(-5.0 * np.abs(self.filtered_vel))
#         contact_signal = joint_pos_error * velocity_penalty
        
#         tactile_proxy = np.concatenate([
#             joint_pos_error, 
#             self.filtered_vel, 
#             contact_signal
#         ], axis=-1)
#         return tactile_proxy

#     def inference_loop(self):
#         if self.is_running == 0:
#             return
            
#         if not (self.received_joint and self.received_obj_pos and self.received_obj_type):
#             self.get_logger().warn("等待資料齊全中...", throttle_duration_sec=2.0)
#             return

#         # --- 啟動第一步保護機制 ---
#         if self.is_first_step:
#             self.previous_pos = np.copy(self.current_pos)
#             self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
#             # 這裡對齊你在 Isaac Lab 裡的 reset 邏輯：初始化時將目標值對齊當前物理角度
#             self.target_pos = np.copy(self.current_pos) 
#             self.is_first_step = False
            
#         # --- 計算速度 (輸入端) ---
#         raw_vel = (self.current_pos - self.previous_pos) / self.dt
#         self.filtered_vel = (self.obs_vel_alpha * raw_vel) + ((1.0 - self.obs_vel_alpha) * self.filtered_vel)
#         self.previous_pos = np.copy(self.current_pos)

#         # --- 組合 Observation ---
#         tactile_proxy = self.build_tactile_proxy()
        
#         obs = np.concatenate([
#             self.current_pos,      # 11維
#             self.last_action,      # 11維 (RL 輸出的原始值)
#             self.obj_pos,          # 3維
#             self.obj_type_one_hot, # 3維
#             tactile_proxy          # 33維
#         ], axis=-1).astype(np.float32)

#         obs_input = np.expand_dims(obs, axis=0)
#         if np.isnan(obs_input).any():
#             self.get_logger().error("⚠️ 抓到了！送進模型的 Observation 裡面含有 NaN！")
#             self.get_logger().error(f"詳細資料: {obs_input}")
#             return # 提早結束，保護模型

#         # --- 執行推論 ---
#         input_name = self.ort_session.get_inputs()[0].name
#         output_name = self.ort_session.get_outputs()[0].name
#         action_output = self.ort_session.run([output_name], {input_name: obs_input})[0]
        
#         # 取得 RL 輸出的正規化數值 (大約在 -1.0 ~ 1.0 之間)
#         action = np.clip(action_output[0], -1.0, 1.0)
#         self.last_action = np.copy(action) # 存起來給下一步的 Observation 用

#         # --- 計算並發布目標指令 (輸出端) ---
#         # 1. 將正規化 action 轉換為物理目標角度 (Scale & Offset)
#         raw_target_pos = (action * self.action_scale) + self.action_offset
        
#         # 2. 套用 Action EMA 低通濾波 (對齊你的 EMAJointPositionAction)
#         self.target_pos = ((self.action_alpha * raw_target_pos) + ((1.0 - self.action_alpha) * self.target_pos))
        
#         # 發布計算完成的實體目標角度
#         msg = Float32MultiArray()
#         msg.data = self.target_pos.tolist()
#         self.target_pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RLInferenceNode()
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
import numpy as np
import onnxruntime as ort

from std_msgs.msg import Int32, Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from math import pi


class RLInferenceNode(Node):
    def __init__(self):
        super().__init__('rl_inference_node')
        
        # 1. 初始化 ONNX Runtime
        # model_path = '/ros2_ws/src/policy.onnx'
        # model_path = '/ros2_ws/src/12_step4-300.onnx'
        
        model_path = '/ros2_ws/src/13_Step5Ar0.08L2-200.onnx'

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

        # 根據 ActionsCfg 定義 Scale 與 Offset
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

        self.action_alpha = 0.1

        # ==========================================================
        # ★ 新增：PWM 與 Radian 轉換參數設定區
        # ==========================================================
        # 1. 計算模型預期的實體弧度極限 (Action = -1 與 Action = 1 的結果)
        self.rad_min = self.action_offset - self.action_scale
        self.rad_max = self.action_offset + self.action_scale
        
        # 2. 定義每一顆馬達專屬的 PWM 上下限 (順序必須對齊 joint_names)
        # 預設先全部給 500，之後你可以依據硬體測試結果獨立修改對應位置的數值
        self.pwm_min = np.array([
            1050.0, 1200.0, 975.0, 1100.0,  # Revolute_12, 14, 16, 18
            1075.0, 1100.0, 1050.0, 975.0,  # Revolute_20, 22, 24, 26
            950.0,                       # Revolute_7
            1150.0,                       # Revolute_8
            1250.0                        # Revolute_29
        ], dtype=np.float32)

        # 預設先全部給 2500，針對每一顆馬達獨立修改上限
        self.pwm_max = np.array([
            2400.0, 2000.0, 2450.0, 2000.0,  # Revolute_12, 14, 16, 18
            2400.0, 2000.0, 2350.0, 2000.0,  # Revolute_20, 22, 24, 26
            1950.0,                          # Revolute_7
            1875.0,                          # Revolute_8
            1700.0                           # Revolute_29
        ], dtype=np.float32)
        # ==========================================================

        # ==========================================================
        # ★ 新增：STM32 回傳的 ADC (0~4095) 上下限映射參數
        # ==========================================================
        # 這是你在 STM32 端讀取到的實際最小/最大值。
        # 之後請手動把手指凹到極限，把讀到的數值填入對應的位置 (例如 100 ~ 1000)
        self.adc_min = np.array([
            1950.0, 1615.0, 1940.0, 1530.0,   # Revolute_12, 14, 16, 18
            1970.0, 1595.0, 2000.0, 1595.0,   # Revolute_20, 22, 24, 26
            1465.0,                        # Revolute_7
            1240.0,                        # Revolute_8
            1470.0                         # Revolute_29
        ], dtype=np.float32)

        self.adc_max = np.array([
            3055.0, 2620.0, 3110.0, 2570.0, 
            3130.0, 2630.0, 3110.0, 2650.0, 
            2700.0, 
            2880.0, 
            2475.0  
        ], dtype=np.float32)
        # ==========================================================

        # --- 3. 狀態暫存區 (維持 Radian 計算) ---
        self.current_pos = np.zeros(self.num_joints, dtype=np.float32)
        self.previous_pos = np.zeros(self.num_joints, dtype=np.float32)
        
        self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
        self.obs_vel_alpha = 0.2  
        
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.target_pos = np.zeros(self.num_joints, dtype=np.float32) 

        self.obj_pos = np.zeros(3, dtype=np.float32)
        self.obj_type_one_hot = np.zeros(3, dtype=np.float32)

        self.received_joint = False
        self.received_obj_pos = False
        self.received_obj_type = False
        self.is_running = 0
        self.is_first_step = True  

        # --- 4. 建立訂閱器與發布器 ---
        self.joint_sub = self.create_subscription(Float32MultiArray, '/current_angle', self.joint_state_callback, 10)
        self.obj_pos_sub = self.create_subscription(Point, '/target_3d_position', self.obj_pos_callback, 10)
        self.obj_type_sub = self.create_subscription(Int32, '/object_type_id', self.obj_type_callback, 10)
        self.cmd_sub = self.create_subscription(Int32, '/start_policy_topic', self.cmd_callback, 10)

        # 現在這裡發布的會是 PWM 值了，但依據你的需求維持 Float32MultiArray
        self.target_pub = self.create_publisher(Float32MultiArray, '/target_angle', 10)
        
        self.dt = 0.025 # 40Hz
        self.timer = self.create_timer(self.dt, self.inference_loop)

    # ==========================================================
    # ★ 新增：PWM 互轉函數
    # ==========================================================
    def pwm_to_rad(self, joint_idx, pwm_val):
        """將單一關節收到的 PWM 轉換回模型需要的 Radian"""
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        r_min = self.rad_min[joint_idx]
        r_max = self.rad_max[joint_idx]
        
        # 1. 算出佔比，並限制在 0~1 之間 (防止 STM32 送來奇怪的數值導致模型崩潰)
        ratio = (pwm_val - p_min) / (p_max - p_min)
        ratio = max(0.0, min(1.0, ratio)) 
        
        # 2. 映射回 Radian
        return ratio * (r_max - r_min) + r_min

    def adc_to_pwm(self, joint_idx, raw_adc):
        """將 STM32 讀取到的原始 ADC 值轉換為對應的標準 PWM 值"""
        a_min = self.adc_min[joint_idx]
        a_max = self.adc_max[joint_idx]
        p_min = self.pwm_min[joint_idx]
        p_max = self.pwm_max[joint_idx]
        
        # 1. 算出在 ADC 範圍內的比例，並限制在 0~1 之間 (防止雜訊超幅)
        ratio = (raw_adc - a_min) / (a_max - a_min)
        ratio = max(0.0, min(1.0, ratio)) 
        
        # 2. 映射到你定義的標準 PWM 範圍
        return ratio * (p_max - p_min) + p_min

    def rad_to_pwm(self, rad_array):
        """將模型算出的 11 維 Radian 陣列轉換為 PWM"""
        # 1. 算出佔比，並限制在 0~1 之間
        ratio = (rad_array - self.rad_min) / (self.rad_max - self.rad_min)
        ratio = np.clip(ratio, 0.0, 1.0)
        
        # 2. 映射到 PWM 範圍
        return ratio * (self.pwm_max - self.pwm_min) + self.pwm_min
    # ==========================================================

    def cmd_callback(self, msg):
        self.is_running = msg.data
        status = "啟動" if self.is_running == 1 else "停止"
        self.get_logger().info(f"🕹️ 收到控制指令：策略推論已 {status}")

    def joint_state_callback(self, msg):
        # 接收來自 STM32 的 ADC 數值 (Float32MultiArray)
        
        if len(msg.data) >= self.num_joints:
            for i in range(self.num_joints):
                raw_adc = msg.data[i]
                
                # 步驟 1: 先將 0-4095 區間的 ADC 轉換為標準 PWM
                standard_pwm = self.adc_to_pwm(i, raw_adc)
                
                # 步驟 2: 再將標準 PWM 轉回 Radian 給模型使用
                self.current_pos[i] = self.pwm_to_rad(i, standard_pwm)
                
            self.received_joint = True
        else:
            self.get_logger().warn(f"⚠️ 收到的關節資料長度不足！預期 {self.num_joints}，實際 {len(msg.data)}")

    def obj_pos_callback(self, msg):
        self.obj_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.received_obj_pos = True

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
            
        if not (self.received_joint and self.received_obj_pos and self.received_obj_type):
            self.get_logger().warn("等待資料齊全中...", throttle_duration_sec=2.0)
            return

        # --- 啟動第一步保護機制 ---
        if self.is_first_step:
            self.previous_pos = np.copy(self.current_pos)
            self.filtered_vel = np.zeros(self.num_joints, dtype=np.float32)
            self.target_pos = np.copy(self.current_pos) 
            self.is_first_step = False
            
        # --- 計算速度 (輸入端，依然保持 Radian/s) ---
        raw_vel = (self.current_pos - self.previous_pos) / self.dt
        self.filtered_vel = (self.obs_vel_alpha * raw_vel) + ((1.0 - self.obs_vel_alpha) * self.filtered_vel)
        self.previous_pos = np.copy(self.current_pos)

        # --- 組合 Observation ---
        tactile_proxy = self.build_tactile_proxy()
        
        obs = np.concatenate([
            self.current_pos,      
            self.last_action,      
            self.obj_pos,          
            self.obj_type_one_hot, 
            tactile_proxy          
        ], axis=-1).astype(np.float32)

        obs_input = np.expand_dims(obs, axis=0)
        if np.isnan(obs_input).any():
            self.get_logger().error("⚠️ 抓到了！送進模型的 Observation 裡面含有 NaN！")
            return 

        # --- 執行推論 ---
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        action_output = self.ort_session.run([output_name], {input_name: obs_input})[0]
        
        action = np.clip(action_output[0], -1.0, 1.0)
        self.last_action = np.copy(action) 

        # --- 計算目標角度 (Radian) ---
        raw_target_pos = (action * self.action_scale) + self.action_offset
        self.target_pos = ((self.action_alpha * raw_target_pos) + ((1.0 - self.action_alpha) * self.target_pos))
        
        # ==========================================================
        # ★ 新增：發布前，將目標 Radian 轉換為目標 PWM
        # ==========================================================
        target_pwm = self.rad_to_pwm(self.target_pos)
        
        msg = Float32MultiArray()
        msg.data = target_pwm.tolist() # 發送出去的會是 500~2500 左右的浮點數
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