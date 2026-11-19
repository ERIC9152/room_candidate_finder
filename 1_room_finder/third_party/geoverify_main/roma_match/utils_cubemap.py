"""
[轉換工具] 全景圖與六面體投影轉換工具 (ERP to Cubemap Projector)
功能說明：
1. 提供 CubemapProjector 類別，實現全景圖 (Equirectangular) 到立方體六面 (Cubemap) 的幾何投影。
2. 有效消除全景圖在南北極點的嚴重拉伸扭曲，為 RoMa 模型提供標準化的透視視點。
3. 提供雙向座標映射，支持將立方體面上的局部像素精準回推至全景圖全域座標。
"""
import numpy as np
from PIL import Image

class CubemapProjector:
    def __init__(self, face_size=512):
        self.face_size = face_size
        self.face_id = {"front": 0, "right": 1, "back": 2, "left": 3, "top": 4, "bottom": 5}

    def erp_to_cubemap(self, erp_img):
        """
        將 ERP 全景圖轉換為 6 張 Cubemap 面。
        Return: dict {"front": PIL.Image, ...}
        """
        if isinstance(erp_img, str):
            erp_img = Image.open(erp_img)
        
        erp_w, erp_h = erp_img.size
        erp_np = np.array(erp_img)
        faces = {}
        
        # 建立網格並正規化到 [-1, 1]
        grid = np.indices((self.face_size, self.face_size)).astype(np.float32)
        u, v = 2 * (grid[1] + 0.5) / self.face_size - 1, - (2 * (grid[0] + 0.5) / self.face_size - 1)

        for face_name in self.face_id.keys():
            # 依據面方向定義 3D 向量 (x, y, z)
            if face_name == 'front':  x, y, z = u, v, 1
            elif face_name == 'right': x, y, z = 1, v, -u
            elif face_name == 'back':  x, y, z = -u, v, -1
            elif face_name == 'left':  x, y, z = -1, v, u
            elif face_name == 'top':   x, y, z = u, 1, -v
            elif face_name == 'bottom': x, y, z = u, -1, v
            
            # 處理 broadcasting
            if isinstance(x, (float, int)): x = np.full_like(u, x)
            if isinstance(y, (float, int)): y = np.full_like(u, y)
            if isinstance(z, (float, int)): z = np.full_like(u, z)

            # 3D -> 球座標 (Lat, Lon) -> ERP UV
            r = np.sqrt(x**2 + y**2 + z**2)
            lat = np.arcsin(y / r)
            lon = np.arctan2(x, z)

            u_erp = (lon / (2 * np.pi)) + 0.5
            v_erp = 0.5 - (lat / np.pi) 
            
            px = (u_erp * erp_w).astype(np.int32) % erp_w
            py = np.clip((v_erp * erp_h).astype(np.int32), 0, erp_h - 1)
            
            faces[face_name] = Image.fromarray(erp_np[py, px])
            
        return faces

    def reassemble_erp(self, faces, erp_w, erp_h):
        """
        將 Cubemap 的 6 個面重新組合成 ERP 全景圖 (Inverse of erp_to_cubemap)。
        Input: 
            faces: dict {"front": PIL.Image/numpy array, ...}
            erp_w, erp_h: 目標 ERP 寬高
        Return:
            PIL.Image (ERP)
        """
        # 準備 ERP 的網格座標
        # u: [0, 1], v: [0, 1]
        y_inds, x_inds = np.indices((erp_h, erp_w))
        u_erp = (x_inds + 0.5) / erp_w
        v_erp = (y_inds + 0.5) / erp_h

        # UV -> 球座標 (Lat, Lon)
        # lon: [-pi, pi], lat: [-pi/2, pi/2]
        lon = (u_erp - 0.5) * 2 * np.pi
        lat = (0.5 - v_erp) * np.pi

        # 球座標 -> 3D 向量 (x, y, z)
        # x = cos(lat) * sin(lon)
        # y = sin(lat)
        # z = cos(lat) * cos(lon)
        # 注意: 這裡的座標系必須與 erp_to_cubemap 一致
        # erp_to_cubemap: lon = arctan2(x, z), lat = arcsin(y/r)
        # => y = r * sin(lat), x = r * cos(lat) * sin(lon), z = r * cos(lat) * cos(lon)
        
        cos_lat = np.cos(lat)
        x = cos_lat * np.sin(lon)
        y = np.sin(lat)
        z = cos_lat * np.cos(lon)

        # 找出每個像素對應哪一個面 (最大分量決定面)
        # faces: front(+z), back(-z), right(+x), left(-x), top(+y), bottom(-y)
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        # 為了向量化，用 mask 分別處理 6 個面
        # Mask 優先順序: 這裡簡單用最大值判斷
        max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
        
        # 定義輸出的畫布 (假設是 RGB 或是 單通道)
        # 先讀取一張面圖來決定通道數
        sample_face = np.array(list(faces.values())[0])
        if len(sample_face.shape) == 2: # Grayscale / Mask
            out_img = np.zeros((erp_h, erp_w), dtype=sample_face.dtype)
            channels = 1
        else:
            out_img = np.zeros((erp_h, erp_w, sample_face.shape[2]), dtype=sample_face.dtype)
            channels = sample_face.shape[2]

        face_configs = [
            ('front',  (z >= max_axis), lambda x,y,z: (x, y, z)),      # u=x/z, v=y/z
            ('back',   (-z >= max_axis), lambda x,y,z: (-x, y, -z)),   # u=-x/-z, v=y/-z
            ('right',  (x >= max_axis), lambda x,y,z: (-z, y, x)),     # u=-z/x, v=y/x (注意: erp_to_cubemap: right x=1, z=-u => u=-z)
            ('left',   (-x >= max_axis), lambda x,y,z: (z, y, -x)),    # u=z/-x, v=y/-x (注意: erp_to_cubemap: left x=-1, z=u => u=z)
            ('top',    (y >= max_axis), lambda x,y,z: (x, -z, y)),     # u=x/y, v=-z/y (注意: erp_to_cubemap: top y=1, z=-v => v=-z)
            ('bottom', (-y >= max_axis), lambda x,y,z: (x, z, -y)),    # u=x/-y, v=z/-y (注意: erp_to_cubemap: bottom y=-1, z=v => v=z)
        ]
        
        # 修正 face_configs 的坐標變換以匹配 erp_to_cubemap
        # erp_to_cubemap 逆變換邏輯:
        # Front (z=1): x=u, y=v => u_face = x/z, v_face = y/z
        # Right (x=1): z=-u, y=v => u_face = -z/x, v_face = y/x
        # Back (z=-1): x=-u, y=v => u_face = -x/-z, v_face = y/-z 
        # Left (x=-1): z=u, y=v => u_face = z/-x, v_face = y/-x
        # Top (y=1): x=u, z=-v => u_face = x/y, v_face = -z/y
        # Bottom (y=-1): x=u, z=v => u_face = x/-y, v_face = z/-y

        for face_name, mask, coords_func in face_configs:
            if not np.any(mask): continue
            
            # 取出該面的 mask 區域的 x, y, z
            x_m, y_m, z_m = x[mask], y[mask], z[mask]
            
            # 轉換為局部 3D 投影平面 (normalization plane)
            # 需依所在面將 x,y,z 轉為 u,v 的形式
            # 實際上 coords_func 應該返回 (u_numer, v_numer, denominator)
            # 簡化：直接在下面寫死轉換邏輯，確保與 erp_to_cubemap 對應
            
            if face_name == 'front':  u_raw, v_raw, div = x_m, y_m, z_m
            elif face_name == 'right': u_raw, v_raw, div = -z_m, y_m, x_m
            elif face_name == 'back':  u_raw, v_raw, div = -x_m, y_m, -z_m
            elif face_name == 'left':  u_raw, v_raw, div = z_m, y_m, -x_m
            elif face_name == 'top':   u_raw, v_raw, div = x_m, -z_m, y_m
            elif face_name == 'bottom': u_raw, v_raw, div = x_m, z_m, -y_m
            
            # Project to [-1, 1]
            u_proj = u_raw / div
            v_proj = v_raw / div

            # Map [-1, 1] to [0, face_size]
            # u = 2 * (x + 0.5) / size - 1  => x = (u + 1) * size / 2 - 0.5
            # v = -(2 * (y + 0.5) / size - 1) => y = (-v + 1) * size / 2 - 0.5
            
            f_img = np.array(faces[face_name])
            f_h, f_w = f_img.shape[:2]

            x_face = ((u_proj + 1) * f_w / 2 - 0.5).astype(np.int32)
            y_face = ((-v_proj + 1) * f_h / 2 - 0.5).astype(np.int32)
            
            # Clip coordinates
            x_face = np.clip(x_face, 0, f_w - 1)
            y_face = np.clip(y_face, 0, f_h - 1)
            
            # Assign values
            out_img[mask] = f_img[y_face, x_face]

        return Image.fromarray(out_img)

    def cubemap_to_erp_coords(self, face_name, x, y, erp_w, erp_h):
        """
        將 Cubemap 局部座標 (x, y) 轉回 ERP 全域座標 (px, py)。
        """
        u = 2 * (x + 0.5) / self.face_size - 1
        v = -(2 * (y + 0.5) / self.face_size - 1) # Flip Y
        
        if face_name == 'front':   vec = np.array([u, v, 1])
        elif face_name == 'right': vec = np.array([1, v, -u])
        elif face_name == 'back':  vec = np.array([-u, v, -1])
        elif face_name == 'left':  vec = np.array([-1, v, u])
        elif face_name == 'top':   vec = np.array([u, 1, -v])
        elif face_name == 'bottom': vec = np.array([u, -1, v])
        else: return 0, 0
            
        # Normalize -> Sphere -> UV
        vec = vec / np.linalg.norm(vec)
        lat = np.arcsin(vec[1])
        lon = np.arctan2(vec[0], vec[2])
        
        px = ((lon / (2 * np.pi)) + 0.5) * erp_w
        py = (0.5 - (lat / np.pi)) * erp_h
        
        return px, py

def batch_cubemap_to_erp(face_name, x_arr, y_arr, erp_w, erp_h, face_size):
    """
    向量化版本的座標轉換，一次處理整個陣列的點。
    """
    u = 2 * (x_arr + 0.5) / face_size - 1
    v = -(2 * (y_arr + 0.5) / face_size - 1)
    ones = np.ones_like(u)

    if face_name == 'front':   vec_x, vec_y, vec_z = u, v, ones
    elif face_name == 'right': vec_x, vec_y, vec_z = ones, v, -u
    elif face_name == 'back':  vec_x, vec_y, vec_z = -u, v, -ones
    elif face_name == 'left':  vec_x, vec_y, vec_z = -ones, v, u
    elif face_name == 'top':   vec_x, vec_y, vec_z = u, 1, -v
    elif face_name == 'bottom': vec_x, vec_y, vec_z = u, -1, v
    else: return np.zeros_like(u), np.zeros_like(u)

    norm = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
    vec_x, vec_y, vec_z = vec_x/norm, vec_y/norm, vec_z/norm
    lat = np.arcsin(vec_y)
    lon = np.arctan2(vec_x, vec_z)
    px = ((lon / (2 * np.pi)) + 0.5) * erp_w
    py = (0.5 - (lat / np.pi)) * erp_h
    return px, py
