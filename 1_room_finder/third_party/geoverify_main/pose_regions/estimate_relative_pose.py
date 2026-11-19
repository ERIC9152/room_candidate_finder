#!/usr/bin/env python3
"""
從 RoMa 匹配估計兩張全景(ERP)之間的相對位姿,並過濾幾何上不合理的匹配。
不需要 GT pose / 深度 / layout。

三種輸入:
  A) --csv      : 吃 export_correspondences.py 匯出的 CSV(x_a,y_a,x_b,y_b,confidence)
  B) --match    : 只對「這一對」直接跑 RoMa(免匯出整棟,需 GPU venv)
  C) --cache_npz: 直接讀 roma_match/cache/ 的 npz(WP1;免 GPU。檔名自帶 house 與
                  A/B 圖名,圖會自動從 datasets/ 找,--img_a/--img_b 可省略)

流程:ERP 像素 -> 單位方向向量(球面射線) -> RANSAC essential matrix
      -> 分解 R、t(尺度未知) -> 內點 = 合理匹配。
輸出:<out>/inliers_overlay.jpg、<out>/topdown.png,並印出內點比例、R、t、退化警告。

用法:
  # 單對直接跑(不必匯出整棟):
  python estimate_relative_pose.py --match --img_a a.jpg --img_b b.jpg --mode direct --conf 0.65 --out pose_out
  # 用已匯出的 CSV:
  python estimate_relative_pose.py --csv a_to_b.csv --img_a a.jpg --img_b b.jpg --out pose_out
  # 用快取 npz(座標=原圖像素,約定驗證見 README / spike_cache_consistency.py):
  python estimate_relative_pose.py --cache_npz ../roma_match/cache/DollhouseTask_xxx_..._face512.npz --out pose_out
  # 幾何核心自我驗證:
  python estimate_relative_pose.py --selftest
"""
import argparse, glob, os, re, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../pose_regions

# ---------- ERP <-> 球面射線 ----------
def erp_to_bearing(xy, W, H):
    x, y = xy[:, 0], xy[:, 1]
    lon = (x / W) * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (y / H) * np.pi
    cl = np.cos(lat)
    d = np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=1)  # Y 為上
    return d / np.linalg.norm(d, axis=1, keepdims=True)

# ---------- 8-point essential matrix ----------
def eight_point_E(da, db):
    A = np.einsum('ni,nj->nij', db, da).reshape(len(da), 9)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(E)
    return U @ np.diag([1, 1, 0]) @ Vt2

def sampson_residual(E, da, db):
    Eda = da @ E.T
    Etdb = db @ E
    num = np.abs(np.sum(db * Eda, axis=1))
    r1 = num / (np.linalg.norm(Eda, axis=1) + 1e-12)
    r2 = num / (np.linalg.norm(Etdb, axis=1) + 1e-12)
    return 0.5 * (r1 + r2)

def _stratified_samples(rng, k, pool, az, nbins=12):
    """方位分桶抽樣:每列 8 槽輪流取「不同方位桶」的點(桶內隨機),提升 minimal sample
    的空間分散度/條件數。只在「有點的桶」之間輪替。門洞 pair 支撐窄是常態,不強迫散開。
    同列抽到重複點的稀有情況退回均勻抽樣。"""
    b = ((az[pool] + 180.0) // (360.0 / nbins)).astype(int) % nbins
    occ = np.unique(b)
    nb = len(occ)
    perm = np.argsort(rng.random((k, nb)), axis=1)          # 每列一個桶排列
    slots = np.tile(perm, (1, -(-8 // nb)))[:, :8]          # 不足 8 桶就循環
    idx = np.empty((k, 8), dtype=int)
    for j in range(nb):
        members = pool[b == occ[j]]
        mask = slots == j
        cnt = int(mask.sum())
        if cnt:
            idx[mask] = members[rng.integers(0, len(members), cnt)]
    srt = np.sort(idx, axis=1)
    for i in np.where((srt[:, 1:] == srt[:, :-1]).any(axis=1))[0]:
        idx[i] = rng.choice(pool, 8, replace=False)
    return idx

def ransac_E(da, db, thresh, iters=2000, seed=0, chunk=256, conf_stop=0.999,
             conf=None, sampler="stratified"):
    """向量化 RANSAC(整塊 minimal sample 一起 batched SVD + 殘差),含標準早停:
    當目前最佳內點率 w 滿足 1-(1-w^8)^N >= conf_stop 即停。輸出契約同舊版(E, inl)。

    sampler:
      "stratified"(預設):PROSAC-lite + 方位分桶。有 conf 時按 conf 降冪排序,取樣池
        從 top 25% 逐塊放寬到 100%(廢點晚進池);每個 minimal sample 的 8 點再按 A 側
        方位分桶輪替(_stratified_samples)。conf 只影響「先抽誰」;內點判定仍對全部點
        算殘差、最終 refit 也用全部內點,不會被 conf 偏見污染。
      "uniform":完全均勻隨機(舊行為,供 A/B 對照)。"""
    rng = np.random.default_rng(seed)
    n = len(da); best_in = None; best_cnt = -1
    if n < 8: raise ValueError("need >=8 matches")
    d1 = np.array([1.0, 1.0, 0.0])
    if sampler == "stratified":
        az = np.degrees(np.arctan2(da[:, 0], da[:, 2]))
        order = np.arange(n) if conf is None else np.argsort(-np.asarray(conf))
        fracs = (0.25, 0.5, 0.75, 1.0)
    done = 0; nchunk = 0
    while done < iters:
        k = min(chunk, iters - done); done += k
        if sampler == "stratified":
            m_pool = n if conf is None else max(min(64, n), int(np.ceil(fracs[min(nchunk, 3)] * n)))
            idx = _stratified_samples(rng, k, order[:m_pool], az)
        else:
            idx = np.stack([rng.choice(n, 8, replace=False) for _ in range(k)])   # (k,8)
        nchunk += 1
        A = np.einsum('kni,knj->knij', db[idx], da[idx]).reshape(k, 8, 9)
        try:
            _, _, Vt = np.linalg.svd(A)                      # 批次 SVD
            E = Vt[:, -1, :].reshape(k, 3, 3)
            U, _, Vt2 = np.linalg.svd(E)
            E = np.einsum('kab,kbc->kac', U * d1[None, None, :], Vt2)   # 投影回 essential 流形
        except np.linalg.LinAlgError:
            continue
        Eda  = np.einsum('kab,nb->kna', E, da)               # E @ da
        Etdb = np.einsum('kab,na->knb', E, db)               # E^T @ db(= db @ E)
        num = np.abs(np.einsum('kna,na->kn', Eda, db))
        r = 0.5 * (num / (np.linalg.norm(Eda, axis=2) + 1e-12)
                 + num / (np.linalg.norm(Etdb, axis=2) + 1e-12))
        cnts = (r < thresh).sum(axis=1)
        j = int(np.argmax(cnts))
        if cnts[j] > best_cnt:
            best_cnt, best_in = int(cnts[j]), (r[j] < thresh)
        w = best_cnt / n
        if w > 0 and done >= np.log(max(1 - conf_stop, 1e-12)) / np.log(max(1 - w ** 8, 1e-12)):
            break
    E = eight_point_E(da[best_in], db[best_in])
    inl = sampson_residual(E, da, db) < thresh
    for _ in range(3):   # LO 式迭代 refit:全內點重擬合直到內點集不再變大(只在變好時接受)
        if inl.sum() < 8:
            break
        E2 = eight_point_E(da[inl], db[inl])
        inl2 = sampson_residual(E2, da, db) < thresh
        if int(inl2.sum()) <= int(inl.sum()):
            break
        E, inl = E2, inl2
    return E, inl

# ---------- 分解 E -> R,t + cheirality ----------
def decompose_E(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    R1 = U @ W @ Vt; R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def triangulate(da, db, R, t):
    Rda = da @ R.T
    lam = np.zeros((len(da), 2))
    for i in range(len(da)):
        M = np.stack([Rda[i], -db[i]], axis=1)
        sol, *_ = np.linalg.lstsq(M, -t, rcond=None)
        lam[i] = sol
    return lam[:, 0], lam[:, 1]

def select_pose(E, da, db, inl):
    best = None; best_cnt = -1
    for R, t in decompose_E(E):
        la, lb = triangulate(da[inl], db[inl], R, t)
        cnt = int(np.sum((la > 0) & (lb > 0)))
        if cnt > best_cnt: best_cnt, best = cnt, (R, t)
    return best

# ---------- 退化(近零基線/純旋轉)偵測 ----------
def rotation_only_fit(da, db):
    Hm = db.T @ da
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    cos = np.clip(np.sum((da @ R.T) * db, axis=1), -1, 1)
    return R, np.degrees(np.arccos(cos))

# ---------- 載入匹配:兩種來源 ----------
def load_csv(csv, conf_min, max_pts):
    import pandas as pd
    df = pd.read_csv(csv)
    if conf_min is not None and "confidence" in df:
        df = df[df["confidence"] >= conf_min]
    if max_pts and len(df) > max_pts:
        df = df.sample(max_pts, random_state=0)
    conf = df["confidence"].to_numpy(float) if "confidence" in df else None
    return df[["x_a", "y_a"]].to_numpy(float), df[["x_b", "y_b"]].to_numpy(float), conf

def match_pair(img_a, img_b, mode, conf, max_pts):
    """只對這一對跑 RoMa(需含 GPU 的 venv)。免匯出整棟。"""
    import torch
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))   # 把專案根目錄加進 path 以匯入 roma_match
    from roma_match.matcher import RoMaMatcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = RoMaMatcher(device=device, conf_threshold=conf)
    fn = {"core": matcher.match_core, "cubemap": matcher.match_cubemap, "direct": matcher.match_direct}[mode]
    matches, _ = fn(img_a, img_b)
    m = np.asarray(matches, float)
    if m.size == 0: raise SystemExit("這一對沒有匹配點")
    if max_pts and len(m) > max_pts:
        m = m[np.random.default_rng(0).choice(len(m), max_pts, replace=False)]
    print(f"RoMa({mode}) matched {len(m)} points on this pair")
    return m[:, :2], m[:, 2:4], m[:, 4]

# ---------- 載入匹配:roma_match/cache 的 npz() ----------
# 快取檔名約定:
#   {house}_{imgA}_{imgB}_direct_roma_v2_std0.1_face512.npz
# 內容:matches (K,5) = (x_a, y_a, x_b, y_b, conf),座標 = 原圖 ERP 像素
# (連續值,雙線性 warp 取樣;x 觀測範圍 [5.12, 8186.88] = 內部網格半格 margin,W=8192。
#  約定驗證見 spike_cache_consistency.py 與 README)。
# conf 底床 0.1(cache 存的是 0.1 門檻的完整結果,供不同門檻共享)。
CACHE_SUFFIX = "_direct_roma_v2_std0.1_face512.npz"
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_CACHE_RE = re.compile(r"^(?P<house>.+?)_(?P<na>" + _UUID + r")_(?P<nb>" + _UUID + r")"
                       + re.escape(CACHE_SUFFIX) + r"$")

def parse_cache_name(npz_path):
    """從 cache 檔名解析 (house, imgA名, imgB名);不符約定就報錯。

    既有資料的影像識別碼為 UUID，優先用 UUID 規則解析，與既有快取檔名一致。
    UUID 對不上時，從右側拆兩段影像識別碼，其餘為房子名。
    """
    base = os.path.basename(npz_path)
    m = _CACHE_RE.match(base)
    if m:
        return m.group("house"), m.group("na"), m.group("nb")
    if not base.endswith(CACHE_SUFFIX):
        raise SystemExit(f"cache 檔名不符約定 {{house}}_{{imgA}}_{{imgB}}{CACHE_SUFFIX}: {npz_path}")
    rest = base[:-len(CACHE_SUFFIX)]
    parts = rest.rsplit("_", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"cache 檔名不符約定 {{house}}_{{imgA}}_{{imgB}}{CACHE_SUFFIX}: {npz_path}")
    return parts[0], parts[1], parts[2]

def find_house_image(house, img_name, max_depth=5, datasets_root=None):
    """在資料集根目錄找圖。房子資料夾可能同名巢狀任意層(同 stitch_tool/house_dirs.py 的約定)。

    datasets_root 由呼叫端傳入時優先使用。未傳入才讀環境變數 ROMA_GEO_DATASETS
    或專案內 datasets/。此處直接讀環境變數而不 import config,以維持 pose_regions 可獨立匯入。
    """
    if datasets_root:
        root = os.path.abspath(os.path.expanduser(datasets_root))
    else:
        root = os.environ.get("ROMA_GEO_DATASETS")
        root = os.path.abspath(os.path.expanduser(root)) if root else \
            os.path.join(os.path.dirname(SCRIPT_DIR), "datasets")
    cur = os.path.join(root, house)
    try:
        from roma_geo.names import find_image_file, dir_has_images
    except ImportError:
        find_image_file = dir_has_images = None
    for _ in range(max_depth + 1):
        if not os.path.isdir(cur):
            return None
        if find_image_file is not None:
            p = find_image_file(cur, img_name)
            if p:
                return p
            if dir_has_images(cur):
                return None
        else:
            for ext in (".jpg", ".JPG", ".jpeg", ".JPEG"):
                p = os.path.join(cur, img_name + ext)
                if os.path.exists(p):
                    return p
            if glob.glob(os.path.join(cur, "*.jpg")) or glob.glob(os.path.join(cur, "*.JPG")) \
                    or glob.glob(os.path.join(cur, "*.jpeg")) or glob.glob(os.path.join(cur, "*.JPEG")):
                return None
        cur = os.path.join(cur, house)
    return None


def _list_image_stems(datasets_root, house, max_depth=5):
    """列出一間房的影像識別碼，供快取檔名對回已知房子與影像。"""
    try:
        from roma_geo.names import is_image_filename
    except ImportError:
        def is_image_filename(name):
            return os.path.splitext(name)[1].lower() in (".jpg", ".jpeg")
    cur = os.path.join(datasets_root, house)
    for _ in range(max_depth + 1):
        if not os.path.isdir(cur):
            return []
        names = [os.path.splitext(n)[0] for n in os.listdir(cur) if is_image_filename(n)]
        if names:
            return names
        cur = os.path.join(cur, house)
    return []


def identify_cache_file(npz_path, datasets_root=None):
    """確定快取檔對應的 (house, na, nb)。UUID 優先，其次用資料集核對，推測須通過驗證。"""
    base = os.path.basename(npz_path)
    m = _CACHE_RE.match(base)
    if m:
        return m.group("house"), m.group("na"), m.group("nb")
    ident = match_cache_to_dataset(npz_path, datasets_root)
    if ident:
        return ident
    if not base.endswith(CACHE_SUFFIX):
        return None
    rest = base[:-len(CACHE_SUFFIX)]
    parts = rest.rsplit("_", 2)
    if len(parts) != 3 or not all(parts):
        return None
    house, na, nb = parts
    if not datasets_root:
        return None
    if find_house_image(house, na, datasets_root=datasets_root) and \
            find_house_image(house, nb, datasets_root=datasets_root):
        return house, na, nb
    return None


def match_cache_to_dataset(npz_path, datasets_root):
    """用資料集裡已知的房子名與影像名拆快取檔名。對不上回 None。"""
    if not datasets_root or not os.path.isdir(datasets_root):
        return None
    base = os.path.basename(npz_path)
    if not base.endswith(CACHE_SUFFIX):
        return None
    rest = base[:-len(CACHE_SUFFIX)]
    try:
        houses = [n for n in os.listdir(datasets_root)
                  if os.path.isdir(os.path.join(datasets_root, n))]
    except OSError:
        return None
    houses.sort(key=len, reverse=True)
    for house in houses:
        prefix = house + "_"
        if not rest.startswith(prefix):
            continue
        rem = rest[len(prefix):]
        stems = set(_list_image_stems(datasets_root, house))
        for na in stems:
            token = na + "_"
            if rem.startswith(token):
                nb = rem[len(token):]
                if nb in stems:
                    return house, na, nb
    return None

def load_cache_npz(npz_path, conf_min, max_pts, img_a=None, img_b=None):
    """回傳 (pa, pb, img_a, img_b)。若傳入的 --img_a/--img_b 與檔名 A/B 相反,自動交換座標
    (同 roma_match/matcher.py 讀快取時的 is_swapped 行為)。"""
    house, na, nb = parse_cache_name(npz_path)
    m = np.load(npz_path)["matches"].astype(float)
    if m.size == 0:
        raise SystemExit("這個 cache 沒有匹配點")

    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    if img_a and img_b:
        sa, sb = stem(img_a), stem(img_b)
        if (sa, sb) == (nb, na):
            m[:, [0, 1, 2, 3]] = m[:, [2, 3, 0, 1]]
            na, nb = nb, na
            print("[cache] --img_a/--img_b 與檔名 A/B 相反,已交換座標")
        elif (sa, sb) != (na, nb):
            raise SystemExit(f"--img_a/--img_b({sa},{sb})與 cache 檔名({na},{nb})不符")
    else:
        img_a, img_b = find_house_image(house, na), find_house_image(house, nb)
        if not (img_a and img_b):
            raise SystemExit(f"在 datasets/{house} 找不到 {na}.jpg / {nb}.jpg;請手動給 --img_a --img_b")

    if conf_min is None:
        conf_min = 0.65   # cache 底床 0.1 的低段大多是雜訊;沿用 --match 模式預設
        print(f"[cache] 未指定 --conf,用預設 {conf_min}(cache 底床 0.1)")
    k_raw = len(m)
    m = m[m[:, 4] >= conf_min]
    k_conf = len(m)
    if k_conf < 8:
        raise SystemExit(f"conf>={conf_min} 只剩 {k_conf} 點(<8),請調低 --conf")
    if max_pts and len(m) > max_pts:
        m = m[np.random.default_rng(0).choice(len(m), max_pts, replace=False)]
    print(f"[cache] {house}: {na[:8]}... -> {nb[:8]}... | 原始 {k_raw} 點, conf>={conf_min} 後 {k_conf} 點, 取樣後 {len(m)} 點")
    return m[:, :2], m[:, 2:4], m[:, 4], img_a, img_b

# ---------- 主流程 ----------
def run(pa, pb, img_a, img_b, out, thresh_deg, max_lines=200, conf=None, sampler="stratified"):
    import cv2, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    def imread_u(p):   # Windows 中文路徑安全(cv2.imread 吃不了非 ASCII 路徑)
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise SystemExit(f"讀不到圖: {p}")
        return img
    A = imread_u(img_a); B = imread_u(img_b)
    Ha, Wa = A.shape[:2]; Hb, Wb = B.shape[:2]
    da = erp_to_bearing(pa, Wa, Ha); db = erp_to_bearing(pb, Wb, Hb)

    thr = np.sin(np.radians(thresh_deg))
    E, inl = ransac_E(da, db, thr, conf=conf, sampler=sampler)
    R, t = select_pose(E, da, db, inl)
    _, rot_err = rotation_only_fit(da[inl], db[inl])

    n = len(da); ni = int(inl.sum())
    print(f"matches: {n} | inliers: {ni} ({100*ni/n:.1f}%)")
    print("R =\n", np.round(R, 4)); print("t (unit, scale-free) =", np.round(t, 4))
    Cb = -R.T @ t
    print(f"baseline dir |t_xz|={np.linalg.norm(t[[0,2]]):.3f}  median rot-only err={np.median(rot_err):.2f} deg")
    if np.median(rot_err) < thresh_deg * 1.5:
        print("⚠ 疑似近零基線/純旋轉:平移方向不可靠(A、B 可能幾乎同位置,例如同房近距離拍攝)")

    H = max(Ha, Hb); canvas = np.zeros((H, Wa + Wb, 3), np.uint8)
    canvas[:Ha, :Wa] = A; canvas[:Hb, Wa:Wa + Wb] = B
    overlay = canvas.copy()
    rng = np.random.default_rng(0)
    idx_in = np.where(inl)[0]; idx_out = np.where(~inl)[0]
    sel_in  = rng.choice(idx_in,  min(len(idx_in),  max_lines),      replace=False) if len(idx_in)  else np.array([], int)
    sel_out = rng.choice(idx_out, min(len(idx_out), max_lines // 3), replace=False) if len(idx_out) else np.array([], int)
    for i in np.concatenate([sel_in, sel_out]):
        col = (0, 220, 0) if inl[i] else (0, 0, 255)
        x1, y1 = int(pa[i, 0]), int(pa[i, 1]); x2, y2 = int(pb[i, 0]) + Wa, int(pb[i, 1])
        cv2.line(overlay, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)   # 半透明,背景看得到
    cv2.imencode(".jpg", canvas)[1].tofile(os.path.join(out, "inliers_overlay.jpg"))   # 中文路徑安全

    la, lb = triangulate(da[inl], db[inl], R, t)
    good = (la > 0) & (lb > 0)
    X = da[inl][good] * la[good, None]
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 2], s=4, c="#0D9488", alpha=0.5, label="inlier pts (triangulated)")
    plt.scatter([0], [0], c="k", marker="^", s=120, label="Cam A")
    plt.scatter([Cb[0]], [Cb[2]], c="r", marker="^", s=120, label="Cam B")
    plt.gca().set_aspect("equal"); plt.legend(); plt.title("Top-down (scale-free)")
    plt.xlabel("X"); plt.ylabel("Z (forward)")
    plt.savefig(os.path.join(out, "topdown.png"), dpi=130, bbox_inches="tight")
    print(f"saved -> {out}/inliers_overlay.jpg , {out}/topdown.png")

# ---------- 合成資料自我驗證 ----------
def selftest():
    rng = np.random.default_rng(1)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = 0.5
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    R_gt = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*K@K
    t_gt = np.array([1.0, 0.05, 0.3]); t_gt /= np.linalg.norm(t_gt)
    X = rng.normal(size=(300, 3)) * np.array([3, 2, 3]) + np.array([0, 0, 6])
    da = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xb = (R_gt @ X.T).T + t_gt * 4.0
    db = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    m = len(da); k = m // 5
    db[:k] = rng.normal(size=(k, 3)); db[:k] /= np.linalg.norm(db[:k], axis=1, keepdims=True)
    # 合成 conf:與內點身分相關但有重疊(外點 0.4~0.8、內點 0.7~1.0),仿 RoMa conf 的先驗性質
    conf = np.concatenate([rng.uniform(0.4, 0.8, k), rng.uniform(0.7, 1.0, m - k)])
    ok = True
    for sampler, c in (("uniform", None), ("stratified", conf)):
        E, inl = ransac_E(da, db, np.sin(np.radians(0.5)), conf=c, sampler=sampler)
        R, t = select_pose(E, da, db, inl)
        # clip 必要:R 完全正確時引數恰為 1.0,浮點誤差超過 1 會使 arccos 回 nan。
        rot_deg = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1)))
        t_ang = np.degrees(np.arccos(np.clip(abs(t @ t_gt), -1, 1)))
        good = rot_deg < 2 and t_ang < 3 and abs(int(inl.sum()) - (m - k)) < 0.1 * m
        ok &= good
        print(f"[selftest][{sampler:>10}] inliers {int(inl.sum())}/{m} (true ~{m-k}) | "
              f"R err {rot_deg:.2f} deg | t-dir err {t_ang:.2f} deg | {'PASS' if good else 'CHECK'}")
    print("[selftest]", "PASS" if ok else "CHECK")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--img_a"); ap.add_argument("--img_b")
    ap.add_argument("--match", action="store_true", help="只對這一對直接跑 RoMa(免匯出整棟,需 GPU venv)")
    ap.add_argument("--mode", default="direct", choices=["core", "cubemap", "direct"],
                    help="--match 用的 RoMa 模式。預設 direct，與 verify_pair 及基準產出相同")
    ap.add_argument("--cache_npz", help="讀 roma_match/cache 的 npz(免 GPU;--img_a/--img_b 可省略,會自動從 datasets/ 找)")
    ap.add_argument("--out", default="pose_out")
    ap.add_argument("--conf", type=float, default=None, help="confidence 門檻(--match/--cache_npz 預設 0.65)")
    ap.add_argument("--thresh_deg", type=float, default=0.5, help="對極內點角度門檻")
    ap.add_argument("--max_pts", type=int, default=4000, help="超過就抽樣,加速 RANSAC")
    ap.add_argument("--max_lines", type=int, default=200, help="overlay 最多畫幾條內點線(越小越看得到背景)")
    ap.add_argument("--sampler", default="stratified", choices=["stratified", "uniform"],
                    help="RANSAC 取樣:stratified=conf 排序漸進池+方位分桶(預設);uniform=舊版均勻")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.cache_npz:
        pa, pb, cf, a.img_a, a.img_b = load_cache_npz(a.cache_npz, a.conf, a.max_pts, a.img_a, a.img_b)
    elif not (a.img_a and a.img_b):
        ap.error("need --img_a --img_b(或改用 --cache_npz)")
    elif a.match:
        pa, pb, cf = match_pair(a.img_a, a.img_b, a.mode, a.conf if a.conf is not None else 0.65, a.max_pts)
    else:
        if not a.csv:
            ap.error("CSV 模式需 --csv(或改用 --match / --cache_npz)")
        pa, pb, cf = load_csv(a.csv, a.conf, a.max_pts)
    run(pa, pb, a.img_a, a.img_b, a.out, a.thresh_deg, a.max_lines, conf=cf, sampler=a.sampler)
