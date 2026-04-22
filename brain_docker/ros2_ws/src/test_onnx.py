import onnxruntime as ort
import numpy as np

# 載入模型 (純 CPU 模式最保險)
session = ort.InferenceSession('/ros2_ws/src/policy.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 建立 1 筆 61 維的純零測試資料 (對齊你目前的新模型)
dummy_obs = np.zeros((1, 61), dtype=np.float32)

print("正在執行推論...")
output = session.run(None, {input_name: dummy_obs})[0]

print(f"模型輸出結果: \n{output}")
if np.isnan(output).any():
    print("❌ 完蛋了！ONNX 檔案本身就有毒，給它 0 它也吐 NaN！")
else:
    print("✅ 模型輸出正常！")