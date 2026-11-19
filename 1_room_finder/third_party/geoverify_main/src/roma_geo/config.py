r"""
[模組用途]
專案層級的路徑與門檻設定。所有「會隨部署環境改變」的位置統一由本模組解析，
呼叫端不寫死路徑。

外部交界處有三類：上游（連通性粗篩的矩陣）、資料集、以及本專案的產出位置。
每一項皆可由環境變數覆寫，無須修改程式碼。

| 項目 | 環境變數 | 預設 | 誰在用 |
|---|---|---|---|
| 全景資料集 | `ROMA_GEO_DATASETS` | `datasets/` | 匹配、幾何驗證、工具 |
| 門語意圖 | （跟隨資料集） | `<datasets>/semantic/` | `semantic_doors/`、工具 |
| 連通性矩陣（上游輸入） | `ROMA_GEO_MATRICES` | `matrices/<版本>/` | 挑配對、工具資料、排序評估 |
| 挑配對門檻 | `ROMA_GEO_PAIR_THRESHOLD` | `PAIR_THRESHOLD` | `environments/select_pairs.py` |
| 產出根目錄 | `ROMA_GEO_OUTPUT` | `Output/` | 幾何驗證產物、工具讀區域 |
| 匹配快取 | `ROMA_GEO_CACHE` | `roma_match/cache/` | 匹配、批次幾何驗證 |
| 內部房子 JSON | `ROMA_GEO_HOUSE_DATA` | 有產出根目錄時為 `<output>/house_data`，否則 `stitch_tool/data/` | 組房、匯出 |
| 語意門輸出 | `ROMA_GEO_SEMANTIC_OUT` | 有產出根目錄時為 `<output>/semantic`，否則 `stitch_tool/semantic/` | 語意門步驟 |
| 對外交付目錄 | `ROMA_GEO_PUBLIC_HOUSES` | `<output>/public_houses` | 對外匯出 |
| 上游連通 JSON | `ROMA_GEO_CONNECTIONS` | 無（缺席則略過轉矩陣） | 連通輸出轉矩陣 |

[用法]
  from config import datasets_root, semantic_root, regions_root
  img_dir = os.path.join(datasets_root(), house)

換路徑時不必改任何程式碼：

  PowerShell:  $env:ROMA_GEO_DATASETS = "D:\panorama_data"
  bash:        export ROMA_GEO_DATASETS=/mnt/data/panorama

  檢查目前全部解析到哪裡：  python config.py
"""
import os

# 套件位於 src/roma_geo/，專案根目錄上兩層。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: 覆寫資料集根目錄的環境變數名。pose_regions / roma_match 各自讀同一個變數名
#: 而不 import 本模組（維持可獨立匯入），改名時三處一併修改。
ENV_DATASETS = "ROMA_GEO_DATASETS"

#: 預設位置（未設環境變數時）
DEFAULT_DATASETS = os.path.join(PROJECT_ROOT, "datasets")

#: 連通性矩陣（粗篩輸出）來源的環境變數名。多個版本可並存於 `matrices/` 底下。
ENV_MATRICES = "ROMA_GEO_MATRICES"

#: 目前使用的版本目錄名。要換版本時改這裡，或設 ENV_MATRICES 指向任意路徑，
#: 或直接用 prepare_house_data.py 的 --oof_dir（優先權最高）。
MATRICES_VERSION = "candidate_val"
DEFAULT_MATRICES_ROOT = os.path.join(PROJECT_ROOT, "matrices")

#: 挑配對門檻：矩陣 ch0 的分數 ≥ 此值才送進 RoMa 匹配。
#: 單位由矩陣決定，不做 sigmoid 轉換；⚠ **換矩陣來源必須重新校準此值**。
#: 現值對應 `candidate_val`：ch0 存 logit，機率 ≥ 0.5 ⇔ logit ≥ 0.0。
ENV_PAIR_THRESHOLD = "ROMA_GEO_PAIR_THRESHOLD"
PAIR_THRESHOLD = 0.0

#: 產出根目錄（幾何驗證的 per-pair JSON 與各項 CSV）。
#: 獨立成一項的理由：批次作業常在其他機器上執行（Kaggle 或遠端機器），
#: 產物需能寫入掛載的共享位置，否則結果只能以人工複製方式取回。
ENV_OUTPUT = "ROMA_GEO_OUTPUT"
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "Output")

#: RoMa 匹配快取。屬暫存檔且可重新產生，但全量體積達數百 MB，常需指向外部位置。
ENV_CACHE = "ROMA_GEO_CACHE"
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, "roma_match", "cache")

ENV_CONNECTIONS = "ROMA_GEO_CONNECTIONS"
ENV_HOUSE_DATA = "ROMA_GEO_HOUSE_DATA"
ENV_SEMANTIC_OUT = "ROMA_GEO_SEMANTIC_OUT"
ENV_PUBLIC = "ROMA_GEO_PUBLIC_HOUSES"


def _from_env(name: str, default: str) -> str:
    p = os.environ.get(name)
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return default


def datasets_root() -> str:
    """ERP 全景資料集根目錄，底下是 `<house>/.../*.jpg` 與 `*_HOTSPOT.json`。"""
    return _from_env(ENV_DATASETS, DEFAULT_DATASETS)


def semantic_root() -> str:
    """門語意圖：`<datasets>/semantic/<taskid>/<image>_mask.png`。

    目前與全景圖置於同一棵目錄樹下。若語意圖改由其他位置提供，只需修改本函式，
    不必更動 semantic_doors/ 與 stitch_tool/ 的呼叫端。
    """
    return os.path.join(datasets_root(), "semantic")


def matrices_version_root() -> str:
    """連通性矩陣的版本目錄，底下是各矩陣變體的子目錄（`refined_matrices/` 等）。

    解析優先序：環境變數 `ROMA_GEO_MATRICES` > `matrices/<MATRICES_VERSION>/`。

    此為**上游交界點**：本變數指向連通性粗篩輸出矩陣的目錄，兩邊僅透過檔案往來，
    不互相 import。格式契約見 `matrices/README.md`。
    """
    return _from_env(ENV_MATRICES, os.path.join(DEFAULT_MATRICES_ROOT, MATRICES_VERSION))


def matrices_root(variant: str = "refined") -> str:
    """某一變體的矩陣目錄（`<house>.npy` / `<house>_images.json`；
    舊目錄可能還有 `<house>_doormap.npy`，現行上游已不產出、讀取端已移除）。

    variant：`refined`（正式流程）或 `oof`（僅 covis_ranking_eval.py 使用）。
    版本目錄底下沒有子目錄時，回退為版本目錄本身。

    優先序：`prepare_house_data.py --oof_dir` > 環境變數 > 預設值。
    格式契約見 `matrices/README.md`。
    """
    base = matrices_version_root()
    sub = os.path.join(base, "%s_matrices" % variant)
    if os.path.isdir(sub):
        return sub
    return base          # 相容：.npy 直接放在版本目錄底下


def matrices_versions() -> list:
    """列出 `matrices/` 底下可用的版本目錄名。"""
    try:
        return sorted(d for d in os.listdir(DEFAULT_MATRICES_ROOT)
                      if os.path.isdir(os.path.join(DEFAULT_MATRICES_ROOT, d)))
    except OSError:
        return []


def pair_threshold() -> float:
    """挑配對門檻。矩陣 ch0 的分數 ≥ 此值才進 RoMa 匹配。

    ⚠ 單位由矩陣決定（logit 或機率），本函式不做轉換。上游變更輸出型態時僅需調整門檻值。
    優先序：`select_pairs.py --threshold` > 環境變數 > `PAIR_THRESHOLD`。
    """
    v = os.environ.get(ENV_PAIR_THRESHOLD)
    if v:
        return float(v)
    return float(PAIR_THRESHOLD)


def output_root() -> str:
    """產出根目錄。批次在其他機器上執行時，指向掛載位置即可免去人工搬移檔案。"""
    return _from_env(ENV_OUTPUT, DEFAULT_OUTPUT)


def regions_root() -> str:
    """幾何驗證產物：per-pair JSON + `_summary.csv` / `_ranking_eval.csv`。

    批次配對來源為同目錄的 `_pairs.csv`。逐步呼叫的本次清單為 `_step_pairs.csv`。
    """
    return os.path.join(output_root(), "roma_regions")


def cache_root() -> str:
    """RoMa 匹配快取目錄（`*.npz`）。暫存檔、可重生，不屬於交付內容。"""
    return _from_env(ENV_CACHE, DEFAULT_CACHE)


class Settings:
    """呼叫端直接指定的路徑與門檻。未傳入的項目才讀環境變數與專案預設。"""

    def __init__(self, datasets=None, matrices=None, output=None, cache=None,
                 pair_threshold=None, connections=None, house_data=None,
                 semantic=None, public_houses=None, semantic_correct=False,
                 semantic_out=None):
        self._datasets = datasets
        self._matrices = matrices
        self._output = output
        self._cache = cache
        self._pair_threshold = pair_threshold
        self._connections = connections
        self._house_data = house_data
        self._semantic = semantic
        self._public_houses = public_houses
        self.semantic_correct = bool(semantic_correct)
        self._semantic_out = semantic_out

    def _abs(self, value):
        return os.path.abspath(os.path.expanduser(value))

    def _specified(self, ctor_value, env_name):
        if ctor_value:
            return True
        return bool(os.environ.get(env_name))

    def datasets_root(self):
        return self._abs(self._datasets) if self._datasets else datasets_root()

    def matrices_root(self, variant="refined"):
        if self._matrices:
            base = self._abs(self._matrices)
        else:
            return matrices_root(variant)
        sub = os.path.join(base, "%s_matrices" % variant)
        if os.path.isdir(sub):
            return sub
        return base

    def output_root(self):
        return self._abs(self._output) if self._output else output_root()

    def regions_root(self):
        return os.path.join(self.output_root(), "roma_regions")

    def cache_root(self):
        return self._abs(self._cache) if self._cache else cache_root()

    def pair_threshold(self):
        if self._pair_threshold is not None:
            return float(self._pair_threshold)
        return pair_threshold()

    def semantic_root(self):
        if self._semantic:
            return self._abs(self._semantic)
        if self._datasets:
            return os.path.join(self.datasets_root(), "semantic")
        return semantic_root()

    def house_data_dir(self):
        if self._house_data:
            return self._abs(self._house_data)
        env = os.environ.get(ENV_HOUSE_DATA)
        if env:
            return self._abs(env)
        if self._specified(self._output, ENV_OUTPUT):
            return os.path.join(self.output_root(), "house_data")
        return os.path.join(PROJECT_ROOT, "stitch_tool", "data")

    def public_dir(self):
        if self._public_houses:
            return self._abs(self._public_houses)
        env = os.environ.get(ENV_PUBLIC)
        if env:
            return self._abs(env)
        return os.path.join(self.output_root(), "public_houses")

    def connections_dir(self):
        if self._connections:
            return self._abs(self._connections)
        env = os.environ.get(ENV_CONNECTIONS)
        if env:
            return self._abs(env)
        return None

    def semantic_out_dir(self):
        if self._semantic_out:
            return self._abs(self._semantic_out)
        env = os.environ.get(ENV_SEMANTIC_OUT)
        if env:
            return self._abs(env)
        if self._specified(self._output, ENV_OUTPUT):
            return os.path.join(self.output_root(), "semantic")
        return os.path.join(PROJECT_ROOT, "stitch_tool", "semantic")


def _src(env_name: str, default_desc: str) -> str:
    return "環境變數 %s" % env_name if os.environ.get(env_name) else default_desc


def _count(d: str, ext: str) -> str:
    try:
        return "{:,}".format(sum(1 for f in os.listdir(d) if f.endswith(ext)))
    except OSError:
        return "-"


def describe() -> str:
    st = Settings()
    src = "環境變數 %s" % ENV_DATASETS if os.environ.get(ENV_DATASETS) else "預設（專案內 datasets/）"
    msrc = ("環境變數 %s" % ENV_MATRICES if os.environ.get(ENV_MATRICES)
            else "預設（matrices/%s）" % MATRICES_VERSION)
    d, sem, m = st.datasets_root(), st.semantic_root(), st.matrices_root()
    n_npy = len([f for f in os.listdir(m) if f.endswith(".npy")]) if os.path.isdir(m) else 0
    variants = [x for x in ("refined", "oof")
                if os.path.isdir(os.path.join(matrices_version_root(), "%s_matrices" % x))]
    conn = st.connections_dir()
    env_lines = []
    for name, label in (
            (ENV_DATASETS, "全景資料集"),
            (ENV_MATRICES, "連通矩陣"),
            (ENV_PAIR_THRESHOLD, "挑配對門檻"),
            (ENV_OUTPUT, "產出根目錄"),
            (ENV_CACHE, "匹配快取"),
            (ENV_CONNECTIONS, "上游連通 JSON"),
            (ENV_HOUSE_DATA, "內部房子 JSON"),
            (ENV_SEMANTIC_OUT, "語意門輸出"),
            (ENV_PUBLIC, "對外交付目錄"),
    ):
        val = os.environ.get(name)
        env_lines.append("  %s（%s）: %s" % (name, label, val if val else "未設"))
    return (
        "datasets_root : %s\n"
        "  來源        : %s\n"
        "  存在        : %s\n"
        "semantic_root : %s\n"
        "  存在        : %s\n"
        "matrices_root : %s\n"
        "  來源        : %s\n"
        "  存在        : %s（%d 個 .npy）\n"
        "  變體        : %s\n"
        "  可用版本    : %s\n"
        "pair_threshold: %s（單位由矩陣決定：logit 或機率，本模組不做轉換）\n"
        "  來源        : %s\n"
        "output_root   : %s\n"
        "  來源        : %s\n"
        "regions_root  : %s\n"
        "  存在        : %s（%s 個 .json）\n"
        "cache_root    : %s\n"
        "  來源        : %s\n"
        "  存在        : %s（%s 個 .npz）\n"
        "house_data_dir: %s\n"
        "  來源        : %s\n"
        "semantic_out_dir: %s\n"
        "  來源        : %s\n"
        "public_dir    : %s\n"
        "  來源        : %s\n"
        "connections_dir: %s\n"
        "環境變數：\n%s"
        % (d, src, os.path.isdir(d), sem, os.path.isdir(sem),
           m, msrc, os.path.isdir(m), n_npy,
           variants or "（無子目錄，.npy 直接置於版本目錄）",
           matrices_versions() or "（無）",
           st.pair_threshold(), _src(ENV_PAIR_THRESHOLD, "PAIR_THRESHOLD=%s" % PAIR_THRESHOLD),
           st.output_root(), _src(ENV_OUTPUT, "預設（專案內 Output/）"),
           st.regions_root(), os.path.isdir(st.regions_root()), _count(st.regions_root(), ".json"),
           st.cache_root(), _src(ENV_CACHE, "預設（roma_match/cache/）"),
           os.path.isdir(st.cache_root()), _count(st.cache_root(), ".npz"),
           st.house_data_dir(), _src(ENV_HOUSE_DATA, "跟隨產出根目錄或 stitch_tool/data/"),
           st.semantic_out_dir(), _src(ENV_SEMANTIC_OUT, "跟隨產出根目錄或 stitch_tool/semantic/"),
           st.public_dir(), _src(ENV_PUBLIC, "跟隨產出根目錄 public_houses"),
           conn if conn else "缺席",
           "\n".join(env_lines))
    )


if __name__ == "__main__":
    print(describe())
