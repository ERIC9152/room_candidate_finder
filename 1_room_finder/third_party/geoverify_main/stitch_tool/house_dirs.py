"""
房屋影像資料夾路徑解析（stitch_tool 專用、獨立實作，不依賴 coarse_filter_experiment）。

datasets/ 底下的房屋資料夾可能被同名巢狀重複任意層，例如：
  flat:   datasets/<house>/img.jpg
  double: datasets/<house>/<house>/img.jpg
  triple: datasets/<house>/<house>/<house>/img.jpg
resolve_task_dir() 由淺到深持續往下找同名子資料夾，直到找到含有全景影像的那一層，
或超過 max_depth 才放棄回傳 None。
"""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 套件未安裝時（直跑腳本、或標註工具把本目錄放進搜尋路徑）補上 src，僅為了找到
# roma_geo。能匯入就完全不動搜尋路徑，以免搶掉呼叫端的同名模組。
if importlib.util.find_spec("roma_geo") is None:
    sys.path.append(os.path.join(_ROOT, "src"))

from roma_geo.names import dir_has_images, is_image_filename  # noqa: E402


def resolve_task_dir(data_root: str, house: str, max_depth: int = 5):
    current = os.path.join(data_root, house)
    if not os.path.isdir(current):
        return None
    depth = 0
    while depth <= max_depth:
        if dir_has_images(current):
            if depth > 1:
                print(f"[house_dirs] '{house}' 巢狀 {depth} 層才找到 -> {current}")
            return current
        nested = os.path.join(current, house)
        if not os.path.isdir(nested):
            return None
        current = nested
        depth += 1
    return None


def list_house_jpgs(data_root: str, house: str, max_depth: int = 5):
    """列出一間房的全景影像檔名（已排序）。找不到資料夾時回 (None, [])。"""
    task_dir = resolve_task_dir(data_root, house, max_depth=max_depth)
    if task_dir is None:
        return None, []
    names = sorted(
        n for n in os.listdir(task_dir)
        if is_image_filename(n) and os.path.isfile(os.path.join(task_dir, n))
    )
    return task_dir, names
