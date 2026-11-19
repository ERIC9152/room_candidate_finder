"""以檔案路徑載入專案原始碼樹裡的腳本。

流程步驟要用到 `stitch_tool/` 與 `environments/` 底下的腳本，但那些目錄不在套件
清單內。不能為了載入它們去改呼叫端的模組搜尋路徑：那會讓 `config`、`names`、
`house_dirs` 這類泛用檔名變成頂層模組，搶掉呼叫端自己的同名模組。

因此一律以檔案路徑載入，模組名冠上 `roma_geo._project.` 前綴。呼叫端的搜尋路徑與
頂層模組名都不受影響；同一支腳本重複載入時沿用第一次的模組物件。
"""
import importlib.util
import os
import sys

#: 載入後的模組名前綴。不是真的套件，只用來避開裸檔名。
NAMESPACE = "roma_geo._project"


def load_project_module(directory, name):
    """載入 `directory/name.py` 並回傳模組物件；檔案不存在時回傳 None。

    錯誤訊息留給呼叫端：缺檔的成因各不相同（非編輯模式安裝缺的是整棵原始碼樹），
    在這裡統一講「找不到檔案」對使用者沒有用。
    """
    modname = "%s.%s" % (NAMESPACE, name)
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    path = os.path.join(directory, name + ".py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[modname]
        raise
    return module
