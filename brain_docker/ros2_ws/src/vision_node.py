#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
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
        # 1. 訂閱相機內參 (加入 QoS)
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. 訂閱相機內參 (套用自訂 QoS)
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',  # ← 這裡修改
            self.camera_info_callback,
            10
        )

        # 2. 建立影像同步訂閱器
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'  # ← 這裡修改
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'  # ← 這裡修改
        )
        # self.info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/camera/camera/color/camera_info',
        #     self.camera_info_callback,
        #     best_effort_qos
        # )

        # # 2. 建立影像同步訂閱器 (套用自訂 QoS)
        # color_sub = message_filters.Subscriber(
        #     self, Image, '/camera/camera/color/image_raw', qos_profile=best_effort_qos
        # )
        # depth_sub = message_filters.Subscriber(
        #     self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=best_effort_qos
        # )
        
        # 使用近似時間同步器，稍微將容許誤差 slop 從 0.05 放寬到 0.1 秒
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        

        # --- 4. 建立發布器 (Publishers) ---
        # 發布 3D 座標給你的 RL 強化學習節點
        self.target_pub = self.create_publisher(Point, '/target_3d_position', 10)
        
        # 額外發布標註好的影像 (方便你在主機端用 rqt_image_view 觀看，免去 Docker 顯示 UI 的麻煩)
        self.debug_img_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        self.get_logger().info("🎯 視覺處理節點已啟動，等待 RealSense 影像輸入...")

    def camera_info_callback(self, msg):
        if not self.intrinsics_ready:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.intrinsics_ready = True
            # 加入這行綠色大字，確認我們真的有收到內參
            self.get_logger().info(f"✅ 已成功取得相機內參: fx={self.fx:.2f}, fy={self.fy:.2f}")

    def sync_callback(self, color_msg, depth_msg):
        # 只要有一組影像同步進來，就印出這行
        self.get_logger().info("🔄 收到一組同步影像！正在檢查內參...")
        
        if not self.intrinsics_ready:
            self.get_logger().warn("⚠️ 收到影像，但相機內參尚未就緒，捨棄此幀。")
            return

        # 將 ROS 影像轉換為 OpenCV 格式
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # 深度影像格式通常為 16-bit，單位是毫米 (mm)
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
                    # 獲取深度 (距離)，單位轉為公尺
                    dist_mm = depth_image[v, u]
                    if dist_mm == 0:
                        continue # 深度為 0 代表該點無效
                    
                    z_3d = dist_mm / 1000.0

                    # --- 2D 像素轉 3D 空間座標 (替代 rs2_deproject_pixel_to_point) ---
                    # 這是標準的針孔相機幾何運算
                    x_3d = (u - self.cx) * z_3d / self.fx
                    y_3d = (v - self.cy) * z_3d / self.fy

                    # 發布 3D 座標給 RL 節點
                    target_point = Point()
                    target_point.x = x_3d
                    target_point.y = y_3d
                    target_point.z = z_3d
                    self.target_pub.publish(target_point)

                    # 取得類別名稱
                    cls_id = int(box.cls)
                    cls_name = self.model.names[cls_id]

                    # 終端機輸出
                    self.get_logger().info(f"[{cls_name}] Dist: {z_3d:.3f}m | XYZ: ({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f})")

                    # --- 畫面繪製 (OpenCV) ---
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                    label_text = f"{cls_name} {z_3d:.2f}m"
                    coord_text = f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}"

                    cv2.putText(color_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_image, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 將畫好的畫面發布出去，以便在 Host 端直接觀看
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