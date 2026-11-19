import json
import os
import numpy as np
from glob import glob

input_dir = 'my_results'
output_dir = 'my_results_txt'
if not os.path.exists(output_dir): os.makedirs(output_dir)

json_files = glob(os.path.join(input_dir, '*.json'))

for json_path in json_files:
    filename = os.path.basename(json_path).replace('.json', '.txt')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 讀取相機高度 (z0 是相機到地板的距離，通常為正數)
    z0 = data.get('z0', 1.6) 
    uv = np.array(data['uv'])
    
    # 只取地板點 (Index 為 1, 3, 5...)
    floor_uv = uv[1::2]
    
    with open(os.path.join(output_dir, filename), 'w') as out_f:
        for u, v in floor_uv:
            # 將歸一化座標轉為球面角度 (u: 0~1 -> -pi~pi, v: 0~1 -> -pi/2~pi/2)
            theta = (u - 0.5) * 2 * np.pi
            phi = (v - 0.5) * np.pi
            
            # 轉換為 3D 射線方向 (假設 Y 軸為垂直方向)
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi)
            z = np.cos(phi) * np.cos(theta)
            
            # 根據相機高度縮放到地板平面 (y 應該是負值，指向地面)
            if abs(y) > 1e-6:
                scale = -z0 / y
                # 這裡的 X, Z 是地板上的水平位置（單位：公尺）
                pos_x = x * scale
                pos_z = z * scale
                # 輸出格式：X, Z, Y (對齊專案通常預期 Y 為垂直，或是 X, Y 為平面)
                # 根據此專案慣例，輸出 X, Z 座標即可
                out_f.write(f"{pos_x}, {pos_z}\n")

print("\n--- 3D 座標轉換完成！請重新搬運至 layout_gt 並執行 Step 00 與 05 ---")