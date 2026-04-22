#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32  # 假設你是用整數 1 和 0 來控制，如果不是請看下方說明

class PolicyController(Node):
    def __init__(self):
        super().__init__('policy_controller_node')
        
        # ⚠️ 注意：這裡的 '/start_policy_topic' 必須改成你 policy 節點實際訂閱的 Topic 名稱！
        self.command_topic = '/start_policy_topic' 
        self.publisher_ = self.create_publisher(Int32, self.command_topic, 10)
        self.get_logger().info("🎮 控制台節點已建立，準備發送指令！")

    def send_command(self, value):
        msg = Int32()
        msg.data = value
        self.publisher_.publish(msg)
        self.get_logger().info(f"發送指令: {value} 到 {self.command_topic}")

def main(args=None):
    rclpy.init(args=args)
    node = PolicyController()

    print("=====================================")
    print("🤖 策略控制終端已啟動！")
    print("▶️ 輸入 'start'  啟動 Policy (發送 1)")
    print("⏸️ 輸入 'stop'   停止 Policy (發送 0)")
    print("❌ 輸入 'quit'   退出控制台")
    print("=====================================")

    try:
        while rclpy.ok():
            user_input = input("請輸入指令 (start/stop/quit): ").strip().lower()
            
            if user_input == 'start':
                node.send_command(1)
            elif user_input == 'stop':
                node.send_command(0)
            elif user_input in ['quit', 'q', 'exit']:
                print("👋 關閉控制台...")
                break
            else:
                print("⚠️ 無效指令，請重新輸入！")
                
    except KeyboardInterrupt:
        print("\n強制中斷，關閉控制台...")
        
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()