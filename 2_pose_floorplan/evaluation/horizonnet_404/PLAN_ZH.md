# HorizonNet 外框擴大評估：評分規格（在看任何結果之前寫）

- 預測：工具包 `2_pose_floorplan/pipeline/horizonnet.py`（官方 preprocess.py 消失點對齊 + inference.py，MP3D 權重，角點映回原圖座標），CPU 執行；輸入為原圖以 BICUBIC 縮成 1024×512（與 preprocess.py 內部相同）。
- 真值：room-stitching-systemV3/wall-solver-point-edge/data/group/*/layout_gt/*.txt（最新版，404 張、42 個樓層、24 棟），每張全景自己的房間外框，1024×512 像素座標。
- 指標（每張全景，與 HorizonNet 官方 eval_general.py 定義相同）：
  1. 2D IoU：地板角點投影到地面的多邊形 IoU（主要指標）。
  2. 3D IoU：2D 交集 × 較低天花板高度；天花板高度用官方 post_proc.get_z1。
  3. 深度 RMSE、δ1（官方 layout_2_depth；預測的天花板角點 x 對齊到地板角點 x，差距約 1 像素）。
  4. 逐欄牆距相對誤差（中位數）。
- 預測缺失或多邊形無效：IoU 記 0，另外列出筆數。
- 統計：全體平均；依真值角點數（4、6、8、10+）、房間性質（rooms.csv）、樓層分組；以「房屋」為單位 bootstrap 1,000 次的 95% 區間。
- 對照：官方 MatterportLayout 結果（2D 81.71、3D 79.11、RMSE 0.197、δ1 0.929）；2026-10-21 舊評估（拼接群組真值，94 張，IoU_J 0.567）列出差異原因，不直接比較數字。
