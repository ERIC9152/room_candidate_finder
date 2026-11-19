# interface.py
"""
core_roma_match 的統一外部介面。
上層只需要呼叫這裡的函式，底層具體使用哪個版本 (例如 v1_matcher 或 v2_matcher)
都可以在這裡快速更換，不影響測試端的呼叫。
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except ImportError:
    pass

from core_roma_match import RoMaScan

def run_roma_match(img_path_a, img_path_b, threshold=0.1, use_depth_filter=False, local_conf_threshold=0.65, model_type="roma_v2", batch_bidir=False):
    """
    統一呼叫介面：匹配兩張圖片。
    使用 RoMaScan 演算法，支援 ERP 圖的無扭曲匹配。

    Args:
        batch_bidir: 是否使用 batch=2 打包加速雙向匹配（需要較大 VRAM）。
                     預設 False（串行模式，VRAM 友善）。
    """
    # print(f"Interface: 接收到圖片路徑 A:{img_path_a}, B:{img_path_b}")

    # 目前使用 RoMaScan 版本 (K-means + Fixed FOV)
    result = RoMaScan.RomaScan(
        img_a=img_path_a,
        img_b=img_path_b,
        threshold=threshold,
        local_conf_threshold=local_conf_threshold,
        model_type=model_type,          # 支援模型版本選擇
        batch_bidir=batch_bidir,        # 雙向匹配策略：True=batch打包 / False=串行
        cluster_method="kmeans",        # 啟用 K-means
        fov_mode="fixed",               # 啟用 Fixed FOV
        export_dir=None                 # 關閉匯出功能
    )

    return result
# 可以繼續擴充其他統一介面，如 depth_inference 等
def run_depth_estimation(img_path):
    print(f"Interface: 執行深度估算於 {img_path}")
    # 這裡可以橋接至不同的深度模型
    return {"depth_map": "mock_data"}
