#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Int32  # 👈 [新增] 匯入 Int32 訊息格式
from cv_bridge import CvBridge
import message_filters

import cv2
import numpy as np
import os
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_processing_node')
        self.bridge = CvBridge()
        
        # --- 1. 載入 YOLOv8 模型 (Jetson 效能優化版) ---
        engine_path = 'yolov8n-1.engine'
        if not os.path.exists(engine_path):
            self.get_logger().info("🚀 初次在 Jetson 運行，正在將 YOLO 模型轉換為 TensorRT 格式...")
            self.get_logger().info("⏳ 這可能需要 10 ~ 20 分鐘，請耐心等候，只需執行一次！")
            temp_model = YOLO('yolov8n-1.pt')
            temp_model.export(format='engine', half=True, workspace=2)
            self.get_logger().info("✅ 模型轉換完成！")
            
        self.model = YOLO(engine_path, task='detect')
        self.get_logger().info("✅ YOLOv8 TensorRT 引擎載入成功！")

        # --- 2. 相機內參變數初始化 ---
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.intrinsics_ready = False

        # --- 3. 建立訂閱器 (Subscribers) ---
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # 建立影像同步訂閱器
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        
        # 使用近似時間同步器
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        # --- 4. 建立發布器 (Publishers) ---
        # 發布 3D 座標給你的 RL 強化學習節點
        self.target_pub = self.create_publisher(Point, '/target_3d_position', 10)
        
        # 👈 [新增] 發布物體類別 ID 給 RL 節點
        self.obj_type_pub = self.create_publisher(Int32, '/object_type_id', 10)
        
        # 額外發布標註好的影像
        self.debug_img_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        self.get_logger().info("🎯 視覺處理節點已啟動，等待 RealSense 影像輸入...")

    def camera_to_world_transform(self, cam_x, cam_y, cam_z):
        """
        根據實體觀察：Isaac XYZ 對應 RealSense 的 X, Z, -Y
        """
        
        # ==========================================
        # 步驟 1：軸向對齊 (你的完美觀察)
        # ==========================================
        # 讓大腦的 X, Y, Z 對齊相機的軸
        aligned_x = cam_x
        aligned_y = cam_z
        aligned_z = -cam_y

        # ==========================================
        # 步驟 2：加上平移偏移量 (Translation Offset)
        # ==========================================
        # ⚠️ 這裡非常重要！軸向對齊了，但原點還沒對齊。
        # 這是「相機鏡頭中心」到「機械手 Base 原點」的直線距離 (單位：公尺)
        # 請拿尺實際量測後填入 (以下為假設值，請務必修改)
        OFFSET_X = 0.4   # 相機相對於 Base 在左右方向的偏移
        OFFSET_Y = -0.64 # 相機相對於 Base 在前後方向的偏移 (例如相機在手掌後方20公分)
        OFFSET_Z = 0.89  # 相機相對於 Base 在上下方向的偏移 (例如相機比手掌底部高15公分)

        world_x = aligned_x + OFFSET_X
        world_y = aligned_y + OFFSET_Y
        world_z = aligned_z + OFFSET_Z

        return float(world_x), float(world_y), float(world_z)

    def camera_info_callback(self, msg):
        if not self.intrinsics_ready:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.intrinsics_ready = True
            self.get_logger().info(f"✅ 已成功取得相機內參: fx={self.fx:.2f}, fy={self.fy:.2f}")

    def sync_callback(self, color_msg, depth_msg):
        # self.get_logger().info("🔄 收到一組同步影像！正在檢查內參...")
        
        if not self.intrinsics_ready:
            self.get_logger().warn("⚠️ 收到影像，但相機內參尚未就緒，捨棄此幀。")
            return

        # 將 ROS 影像轉換為 OpenCV 格式
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        # 執行 YOLO 推論
        results = self.model(color_image, verbose=False, conf=0.4)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 取得 2D Bounding Box 座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 計算中心點 (像素座標)
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # 確保座標在深度影像範圍內
                if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                    dist_mm = depth_image[v, u]
                    if dist_mm == 0:
                        continue
                    
                    z_3d = dist_mm / 1000.0
                    x_3d = (u - self.cx) * z_3d / self.fx
                    y_3d = (v - self.cy) * z_3d / self.fy

                    # 發布 3D 座標
                    target_point = Point()
                    # target_point.x = x_3d 
                    # target_point.y = y_3d 
                    # target_point.z = z_3d
                    target_point.x, target_point.y, target_point.z = self.camera_to_world_transform(x_3d, y_3d, z_3d)
                    self.target_pub.publish(target_point)


                    # 取得類別名稱
                    cls_id = int(box.cls)
                    cls_name = self.model.names[cls_id].lower() # 轉小寫保險一點

                    # 👈 [新增] 根據類別名稱發布對應的 ID
                    obj_id_msg = Int32()
                    if 'cube' in cls_name:
                        obj_id_msg.data = 0
                    elif 'sphere' in cls_name:
                        obj_id_msg.data = 1
                    elif 'cylinder' in cls_name:
                        obj_id_msg.data = 2
                    else:
                        continue # 如果抓到其他不相關的東西，就不發布 ID

                    self.obj_type_pub.publish(obj_id_msg)

                    # 終端機輸出 (順便印出 ID 方便除錯)
                    self.get_logger().info(f"[{cls_name} (ID:{obj_id_msg.data})] World XYZ: ({target_point.x:.3f}, {target_point.y:.3f}, {target_point.z:.3f})")

                    # --- 畫面繪製 ---
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                    label_text = f"{cls_name}"
                    # ✅ 讓螢幕上也顯示世界座標，方便你拿尺對照！
                    coord_text = f"W_X:{target_point.x:.2f} W_Y:{target_point.y:.2f} W_Z:{target_point.z:.2f}"
    
                    cv2.putText(color_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_image, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # 發布 debug 影像
        debug_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        self.debug_img_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()