import os
import cv2
import numpy as np
from PIL import Image
from core_roma_match.proection import ProjectionEngine
from core_roma_match.visualization import visualize_slice_matches

class LocalRefiner:
    """執行局部透視重匹配的協調者 (SRP: Coordination of Local Task)"""
    def __init__(self, matcher, export_dir=None,
                 draw_lines=True, line_percent=0.05, line_alpha=0.5):
        self.matcher = matcher
        self.export_dir = export_dir
        # 視覺化設定：是否畫連線、連線取信心前百分之幾、連線透明度(可看見背景)
        self.draw_lines = draw_lines
        self.line_percent = line_percent
        self.line_alpha = line_alpha

    def prepare_crops(self, img_a_cv, img_b_cv, cluster, tmp_dir, fov_a, fov_b):
        """CPU 階段：生成透視裁切圖與 map_back 函數，不執行 GPU 匹配。

        Returns:
            (pil_a, pil_b, mb_a, mb_b, p_a_cv, p_b_cv)
            p_a_cv / p_b_cv 是 BGR numpy，供 export 視覺化用
        """
        H_a, W_a = img_a_cv.shape[:2]
        H_b, W_b = img_b_cv.shape[:2]
        engine_a = ProjectionEngine(W_a, H_a)
        engine_b = ProjectionEngine(W_b, H_b)
        mx_a, my_a, mb_a = engine_a.get_mapping(cluster['center_a'][0], cluster['center_a'][1], fov_a)
        mx_b, my_b, mb_b = engine_b.get_mapping(cluster['center_b'][0], cluster['center_b'][1], fov_b)

        p_a = cv2.remap(img_a_cv, mx_a, my_a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        p_b = cv2.remap(img_b_cv, mx_b, my_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        pil_a = Image.fromarray(cv2.cvtColor(p_a, cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(p_b, cv2.COLOR_BGR2RGB))

        # 若有開啟 export，寫出裁切圖以供 Debug
        if self.export_dir:
            path_a = os.path.join(tmp_dir, f"a_{cluster['id']}.jpg")
            path_b = os.path.join(tmp_dir, f"b_{cluster['id']}.jpg")
            cv2.imwrite(path_a, p_a)
            cv2.imwrite(path_b, p_b)

        return pil_a, pil_b, mb_a, mb_b, p_a, p_b

    @staticmethod
    def apply_map_back(matches, mb_a, mb_b):
        """將透視空間匹配點透過 map_back 轉回 ERP 座標。"""
        if not matches:
            return []
        m_np = np.array(matches)
        return np.hstack([mb_a(m_np[:, 0:2]), mb_b(m_np[:, 2:4]), m_np[:, 4:5]]).tolist()

    def refine(self, img_a_cv, img_b_cv, cluster, tmp_dir, fov_a, fov_b, conf_threshold=None):
        """原始逐一匹配介面（向後相容）。"""
        pil_a, pil_b, mb_a, mb_b, p_a, p_b = self.prepare_crops(
            img_a_cv, img_b_cv, cluster, tmp_dir, fov_a, fov_b)

        # 直接餵給 matcher PIL Image 物件
        matches, _ = self.matcher.match(pil_a, pil_b, conf_threshold=conf_threshold)

        if self.export_dir:
            vis = visualize_slice_matches(p_a, p_b, matches,
                                          draw_lines=self.draw_lines,
                                          line_percent=self.line_percent,
                                          line_alpha=self.line_alpha)
            cv2.imwrite(os.path.join(self.export_dir, f"c_{cluster['id']}_f{int(fov_a)}_{int(fov_b)}.jpg"), vis)

        if not matches: return []
        m_np = np.array(matches)
        return np.hstack([mb_a(m_np[:, 0:2]), mb_b(m_np[:, 2:4]), m_np[:, 4:5]]).tolist()
