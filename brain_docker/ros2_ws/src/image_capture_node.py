import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        
        # 1. 訂閱彩色影像 Topic (請確認與你的 rs_launch 輸出一致)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        
        # 2. 設定儲存路徑 (建議掛載到主機的資料夾，方便上傳 Roboflow)
        self.save_path = './dataset_capture3'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f'建立資料夾: {self.save_path}')

        # 3. 計時器控制：每 1.0 秒允許拍一張
        self.last_capture_time = time.time()
        self.capture_interval = 1.0 
        self.img_count = 0

        self.get_logger().info('📸 拍照節點啟動！每秒將自動儲存一張照片...')

    def listener_callback(self, msg):
        current_time = time.time()
        
        # 檢查是否過了 1 秒
        if current_time - self.last_capture_time >= self.capture_interval:
            try:
                # 將 ROS 影像轉換為 OpenCV 格式
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
                # 生成檔名 (使用時間戳避免重複)
                timestamp = int(current_time)
                filename = os.path.join(self.save_path, f'capture_{timestamp}_{self.img_count}.jpg')
                
                # 儲存照片
                cv2.imwrite(filename, cv_image)
                
                self.get_logger().info(f'✅ 已儲存照片: {filename}')
                
                # 更新計時器與計數
                self.last_capture_time = current_time
                self.img_count += 1
                
            except Exception as e:
                self.get_logger().error(f'轉換影像失敗: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()