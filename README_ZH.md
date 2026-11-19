# room_candidate_finder：全景同房篩選 → 相機姿態 → 平面圖

本文件是這個工具包唯一的說明文件，2026-11-17 由原本的 `README`、`NOTICE`、`SPEC`、`OUTPUT_FORMAT` 合併而成。英文版：[README.md](README.md)。

這個工具包把一批 360° 全景圖整理成「每個房間的相機配置與平面圖」。所有路徑都相對於本資料夾，所以整個資料夾可以單獨複製到另一台電腦上執行。

**目錄**

| § | 內容 |
|---|---|
| [0](#0-交接與使用前必讀) | 交接與使用前必讀 |
| [1](#1-系統概要) | 系統概要、流程、模型 |
| [2](#2-資料夾結構) | 資料夾結構 |
| [3](#3-執行環境) | 硬體、環境、SAM3 |
| [4](#4-安裝與驗證) | 安裝與驗證（含複製到其他電腦後） |
| [5](#5-使用方式) | 指令與選項 |
| [6](#6-輸入格式) | 兩個資料夾的**輸入格式** |
| [7](#7-輸出格式1_room_finder) | `1_room_finder` 的**輸出格式** |
| [8](#8-輸出格式2_pose_floorplan) | `2_pose_floorplan` 的**輸出格式** |
| [9](#9-單位座標與俯視圖) | 單位、座標與俯視圖 |
| [10](#10-運算速度) | 運算速度 |
| [11](#11-演算法與參數) | 演算法與參數 |
| [12](#12-模型與權重) | 模型與權重 |
| [13](#13-如何訓練-covispose) | 如何訓練 CoVisPose |
| [14](#14-如何訓練或微調-horizonnet) | 如何訓練或微調 HorizonNet |
| [15](#15-geoverify-與-roma-v2) | GeoVerify 與 RoMa v2 |
| [16](#16-測試與驗證紀錄) | 測試與驗證紀錄 |
| [17](#17-授權) | 授權 |
| [18](#18-已知條件限制與尚未解決的事項) | 已知條件、限制與尚未解決的事項 |
| [19](#19-修改紀錄) | 修改紀錄 |
| [20](#20-其他文件) | 其他文件 |

---

## 0. 交接與使用前必讀

**從 GitHub 取得**

Git 倉庫包含程式、文件、9 張範例全景、訓練配對清單與結果檔。以下內容**不在**倉庫內：

| 不在倉庫內 | 大小 | 取得方式 |
|---|---|---|
| `2_pose_floorplan\models\covispose\epoch_015.pt` | 1.41 GB | Release `v1.0` 附件（見 `models\covispose\README.md`） |
| `2_pose_floorplan\models\horizonnet\resnet50_rnn__mp3d.pth` | 327 MB | Release `v1.0` 附件 |
| RoMa v2 權重、DINOv3 程式、ImageNet ResNet-50（`models\torch_home\`） | 1.2 GB | `Setup.ps1` 自動下載 |
| Python 環境 | — | `Setup.ps1` 建立 |
| SAM3（`1_room_finder\third_party\sam3-pano-seg\`） | 約 9 GB | 不散布；沒有時審查程式改用一般楔形（§3.3） |
| Dollhouse 全景（`1_room_finder\data\dollhouse`）與 viewer 的全景檔（`runs\*\viewer\images`） | 約 14 GB | 不散布；掃描結果、審查紀錄與 104 組結果都在倉庫內，`viewer.html` 可看平面圖但不能看全景 |
| ZInD 真值 `evaluation\zind_test_posegraph\gt.json` 與 ZInD 預測 | — | `Run_ZInD_PoseGraph_Eval.ps1` 會用你自己的 ZInD 重新產生（§11.3） |

步驟：`git clone` 到只有英文字母的路徑 → 把兩個 Release 附件下載到上表的資料夾（`gh release download v1.0 --repo ERIC9152/room_candidate_finder --dir <資料夾> --pattern <檔名>`，或從 Release 頁面下載）→ `Setup.ps1` → `2_pose_floorplan\Verify.ps1`。

**在其他電腦上使用**

1. 資料夾要放在**只有英文字母的路徑**，例如 `C:\room_candidate_finder`。HorizonNet 與 SAM3 環境在中文路徑下會失敗。
2. 複製或搬動資料夾後，先執行一次 `Setup.ps1`，再執行 `2_pose_floorplan\Verify.ps1`（§4）。Windows 的 Python 虛擬環境會記住建立時的絕對路徑，複製過來的環境無法啟動；`Setup.ps1` 會偵測並重建。最後出現 `VERIFY PASSED`，就代表這台電腦可以使用。
3. `Setup.ps1` 需要網路，約 10–30 分鐘、下載 8–10 GB，完成後可離線使用。
4. GPU：掃描（步驟 A）與 SAM3 需要 NVIDIA GPU，驅動程式要支援 CUDA 13.0。步驟 1–6 沒有 GPU 時會改用 CPU。RTX 50 系列執行 HorizonNet 時要加 `-HnCpu`（§18）。

**授權**（§17）

5. CoVisPose 權重與 `training\pairs\` 衍生自 ZInD，**只限學術、非商業用途**。
6. SAM3 採用 Meta SAM License。交付時**不可以移除** `1_room_finder\third_party\sam3-pano-seg\third_party\sam3\LICENSE`。
7. Dollhouse 全景圖（`1_room_finder\data\dollhouse`，約 15 GB）已內含。交付前請依資料提供者的規定確認對方可以取得這份資料；不需要時，刪掉 `1_room_finder\data\dollhouse` 與 `1_room_finder\scans\dollhouse`。

**解讀輸出**（§6–§9）

8. **角度有兩種單位。** JSON 裡沒有 `_deg` 字尾的角度（`yaw`、`geo_yaw`、`corr_*`、`bon_*`）是**弧度**。有 `_deg` 字尾的欄位、CSV 的 `yaw_deg`、`1_room_finder` 的所有楔形，以及 GeoVerify 的方位角都是**度**（§9.1）。
9. **長度單位是「相機高度」**：相機到地板的距離 = 1。乘上實際相機高度就是公尺（ZInD 約 1.5 m；Dollhouse 未知）。面積單位是相機高度²。
10. **座標是 CoVisPose 原生座標。直接以 x 向右、y 向上畫，會左右相反。** 畫俯視圖（bird's-eye view）請畫 **(−x, y)**。流程輸出的 PNG 與 viewer 都已經這樣畫（§9.3）。
11. **位姿**：相對位姿是 `p_B = R(yaw) · p_A + t`，相機位姿是 `p_root = R(yaw) · p_u + t`。GeoVerify 的 3D 位姿只有方向（`t` 是單位向量），流程只用 GeoVerify 判斷是否同房。
12. **沒有姿態真值。** 所有位姿與平面圖都標 `pose_verified: false`；紅圈「!」代表值得檢查，不代表確定錯。
13. **Dollhouse 有兩份分組檔**：`room_groups_dollhouse104.json` 是 104 組正式結果的輸入；`room_groups_test_20261026.json` 只是 6 組的測試匯出（§7.10）。
14. **run 資料夾會凍結輸入與程式雜湊。** 改過 `scripts\manual_room_groups_v2.py` 或 `covispose\` 的程式後，舊的 run 無法續跑，要換一個 `-Name` 重新 prepare。
15. `viewer\viewer.html` 必須和 `viewer\images\` 放在一起。`covispose_gpu_*.log` 是 UTF-16 編碼（PowerShell 5.1 寫出）。

---

## 1. 系統概要

輸入是一批 360° 等距長方投影全景圖，每棟房子一個資料夾。輸出是每個房間的相機相對位置、合併平面圖，以及可以在平面圖上走動看全景的互動網頁。

```
全景資料夾
   │
   ▼  1_room_finder
   A. 找同房候選（GeoVerify＋RoMa v2，GPU）
   │   └─ A'. SAM3 門洞偵測 → 去門楔形（內含，可略過）
   B. 人工審查（審查程式，逐對判定同房／不同房）
   │
   ▼  room_groups.json（加上旁邊的掃描 manifest.json）
   │
   ▼  2_pose_floorplan
   1. prepare（凍結輸入）           CPU
   2. CoVisPose 相對姿態           GPU（無 GPU 時改用 CPU）
   3. HorizonNet 房間外框          GPU（無 GPU 時改用 CPU）
   4. poses（採用規則、位姿圖最佳化） CPU
   5. plans（合併平面圖）           CPU
   6. viewer（互動網頁）            CPU
```

| 步驟 | 做什麼 | 主要程式 | 硬體 |
|---|---|---|---|
| A. 找同房候選 | 同一棟房子內的全景兩兩做幾何驗證，算出內點數與兩側共視楔形角度 | `finder/scan.py`（GeoVerify＋RoMa v2） | GPU |
| A'. 門洞偵測（可略過） | SAM3「open doorway」門洞遮罩 → 剔除隔著門洞看到的內點 → 去門楔形 | `Start_Openings.ps1`、`finder/openings.py` | GPU |
| B. 人工審查 | 依楔形門檻列出候選，人工判定同房／不同房；同房配對自動串成房間群組，匯出 `room_groups.json` | `finder/app.py`（Tkinter） | — |
| 1. prepare | 讀群組與影像、逐張檢查、凍結輸入（影像 sha256、配對清單、權重與程式雜湊）；群組內有「不同房」判定就拒絕 | `pipeline/prepare.py` | CPU |
| 2. CoVisPose | 群組內每一對全景的相對位姿與逐欄輸出 | `scripts/manual_room_groups_v2.py run` | GPU；沒有時用 CPU |
| 3. HorizonNet | 每張全景的房間外框 | `pipeline/horizonnet.py` → `third_party/HorizonNet` | GPU；可改 CPU |
| 4. poses | 採用規則、以生成樹為初始值、對有迴圈的分量做位姿圖最佳化（§11.3）、原點、姿態疑慮標記 | `pipeline/pose_graph.py`（`-NoPoseGraph`：`pipeline/poses.py`，只用生成樹） | CPU |
| 5. plans | 合併平面圖 | `pipeline/plans.py` | CPU |
| 6. viewer | 互動網頁 | `pipeline/viewer.py` | CPU |

步驟 1–6 由 `2_pose_floorplan\Run_Pipeline.ps1` 一次執行，可中斷續跑。

**使用的模型**

| 模型 | 用途 | 權重 | 是否由本專案訓練 |
|---|---|---|---|
| CoVisPose（自行重新實作，ECCV 2022 架構） | 兩張全景 → 共視、逐欄對應、牆地交界、相對平移與 yaw | `models/covispose/epoch_015.pt` | **是**（ZInD，§13） |
| HorizonNet（官方） | 單張全景 → 房間外框 | `models/horizonnet/resnet50_rnn__mp3d.pth` | 否，官方 MP3D 權重（微調見 §14） |
| RoMa v2（GeoVerify 使用） | 稠密特徵匹配 | `1_room_finder\models\torch_home`（`Setup.ps1` 下載） | 否 |
| SAM3（sam3-pano-seg） | 門洞分割（open doorway） | `third_party/sam3-pano-seg/checkpoints/sam3.pt` | 否 |

---

## 2. 資料夾結構

工具包分成兩個子資料夾，依執行順序編號。兩者之間只透過審查程式匯出的 `room_groups.json` 與旁邊的掃描 `manifest.json` 銜接，Python 程式互不引用。主環境只建一套，放在 `1_room_finder` 底下，`2_pose_floorplan` 也使用它。

```
room_candidate_finder\
├─ README.md, README_ZH.md                                     本文件（英文／中文）
├─ Setup.ps1                                                   建立所有環境（兩個子資料夾共用）
├─ tools\_env.ps1                                              共用函式：找環境、找 SAM3、檢查 GPU
│
├─ 1_room_finder\          ── 篩選相同房間（步驟 A、A'、B）
│   ├─ CODE_OVERVIEW.md, CODE_OVERVIEW_ZH.md                    本資料夾的程式與函式說明
│   ├─ Import_Dataset.ps1                                       匯入＋掃描（＋門洞偵測）
│   ├─ Start_Openings.ps1                                       門洞偵測（Import_Dataset.ps1 也會呼叫）
│   ├─ Start_Review.bat                                         審查程式
│   ├─ finder\             掃描、去門楔形、審查程式；config.py 是預設門檻；
│   │                      calibration_dollhouse.json 是 P/R 校準表；i18n.py 是英文／中文介面文字
│   ├─ tools\              fix_sam3_env.ps1、recompute_inliers.py、import_openings.py
│   ├─ models\torch_home\                                       RoMa v2 權重與 DINOv3 程式（TORCH_HOME）
│   ├─ third_party\geoverify_main\                              GeoVerify（commit b5897093）；主環境 venv 建在其下
│   ├─ third_party\sam3-pano-seg\                               SAM3：現成環境 .runtime\、權重、原始碼、sam3_pano（約 9 GB）
│   ├─ scans\dollhouse\                                         Dollhouse 60 棟的掃描結果、門洞遮罩、人工審查紀錄
│   ├─ data\                                                    匯入的全景（data\dollhouse：Dollhouse 資料集，約 15 GB）
│   ├─ tests\test_core.py
│   └─ docs\REVIEW_APP.md, docs\benchmark\                      審查程式說明、同房分類器比較結果
│
└─ 2_pose_floorplan\       ── 產生全景相對姿態與平面圖（步驟 1–6）
    ├─ CODE_OVERVIEW.md, CODE_OVERVIEW_ZH.md                    本資料夾的程式與函式說明
    ├─ Run_Pipeline.ps1  Verify.ps1                             步驟 1–6、範例驗證
    ├─ requirements_horizonnet.txt  requirements_main_extra.txt  constraints_main.txt
    ├─ pipeline\           步驟 1–6（步驟 4：pose_graph.py 為預設、poses.py 只用生成樹），verify.py（範例比對）、envcheck.py（環境檢查）
    ├─ covispose\          CoVisPose 模型、資料集、損失、幾何、評估（推論與訓練共用）
    ├─ scripts\
    │   ├─ manual_room_groups_v2.py, run_scene_graph.py        推論（步驟 2）
    │   ├─ make_zind_room_groups.py, eval_pose_graph_zind.py    步驟 4 的 ZInD 評估（§11.3）
    │   └─ train.py, build_pair_index.py, evaluate_split.py,
    │      tune_ransac.py, benchmark_vram.py, verify_targets.py, plot_curves.py   訓練與評估
    ├─ models\covispose\epoch_015.pt                            1.41 GB（或 epoch_015_ema.pt，0.35 GB，§12）
    ├─ models\horizonnet\resnet50_rnn__mp3d.pth                 327 MB
    ├─ models\torch_home\                                       ImageNet ResNet-50（HorizonNet 建模用）
    ├─ third_party\HorizonNet\                                  HorizonNet 官方程式（含 train.py）；HorizonNet venv 建在其下
    ├─ training\           訓練設定、ZInD 配對清單（44 MB）、訓練環境規格、原始 log、設計文件
    ├─ evaluation\horizonnet_404\                               HorizonNet 外框 404 張評估
    ├─ evaluation\zind_test_posegraph\                          步驟 4 ZInD test 評估的輸入與啟動腳本（§11.3）
    ├─ sample\                                                  9 張全景、掃描 manifest、room_groups.json、參考結果
    ├─ runs\                                                    Run_Pipeline.ps1 的輸出
    └─ tests\covispose\、tests\pipeline\
```

以下各節的路徑，步驟 A、A'、B 的檔案相對於 `1_room_finder\`；步驟 1–6、CoVisPose、HorizonNet 與訓練相關的相對於 `2_pose_floorplan\`。另有標示者除外。

---

## 3. 執行環境

### 3.1 硬體與作業系統

| 項目 | 需求 |
|---|---|
| 作業系統 | Windows 10／11 64 位元（腳本是 PowerShell 5.1 與 .bat） |
| GPU | NVIDIA。主環境用 CUDA 13.0 版 PyTorch，需要支援它的驅動程式；GeoVerify 以 Ada（RTX 40，sm_89）與 Blackwell（RTX 50，sm_120）驗證過。舊架構（Pascal、Volta 等）不在 cu130 支援範圍 |
| 顯示記憶體 | 一次只跑一個 GPU 程式；CoVisPose 推論為 FP32、88M 參數，建議 8 GB 以上 |
| 磁碟 | 不含 Dollhouse 約 11 GB（其中 SAM3 現成環境與權重約 9 GB）；Dollhouse 全景另約 15 GB（整個資料夾約 26 GB）；`Setup.ps1` 建立的環境另需 8–10 GB |
| 路徑 | **只能有英文字母**（`Setup.ps1` 會警告） |
| 網路 | 建環境時需要；之後可離線 |

### 3.2 三個推論環境（都建在本資料夾內）

| 環境 | 位置 | 用途 | 版本 |
|---|---|---|---|
| 主環境 | `1_room_finder\third_party\geoverify_main\venv` | 步驟 A、A' 的楔形計算、步驟 1、2、4、5、6 | Python 3.11（uv）、torch 2.10.0+cu130、numpy 1.26.4、opencv-python 4.11.0.86、scikit-learn 1.8.0、pillow 12.0.0、tqdm 4.70.0、romav2（GitHub 原始碼）；另加 shapely ≥ 2.0、matplotlib ≥ 3.8、scipy ≥ 1.11（numpy 以 `constraints_main.txt` 固定在 1.26.4） |
| HorizonNet 環境 | `2_pose_floorplan\third_party\HorizonNet\venv` | 步驟 3 | Python 3.9、torch 1.12.0+cu113、torchvision 0.13.0+cu113、numpy 1.21.0、opencv-python 4.7.0.72、pylsd-nova 1.2.0、scipy 1.7.3、scikit-learn 1.0.2、Shapely 1.8.5、pillow 11.3.0、matplotlib 3.5.3（`requirements_horizonnet.txt`） |
| 審查環境 | `1_room_finder\.venv` | 步驟 B | 任何含 Tkinter 的 Python 3.9 以上＋ pillow ≥ 9 |

環境變數（執行腳本前設定）：

| 環境變數 | 說明 |
|---|---|
| `RCF_MAIN_PYTHON`、`RCF_HN_PYTHON`、`RCF_REVIEW_PYTHON` | 改用電腦上已有的 python.exe 當主環境／HorizonNet 環境／審查環境 |
| `RCF_SAM3_ROOT` | sam3-pano-seg 資料夾（預設 `1_room_finder\third_party\sam3-pano-seg`） |
| `RCF_GEOVERIFY_MAIN` | GeoVerify 程式位置（預設 `1_room_finder\third_party\geoverify_main`） |
| `RCF_DEVICE` | `cpu` 強制 CoVisPose 用 CPU（預設：有 CUDA 就用） |
| `RCF_LANG` | 審查程式與 viewer 的介面語言：`en`（預設）或 `zh` |
| `TORCH_HOME` | 已設定時不變；否則 `finder\config.py` 指向 `1_room_finder\models\torch_home` |
| `ZIND_ROOT`、`ZIND_CODE` | ZInD 資料與 ZInD 官方程式位置，訓練與部分測試使用 |

### 3.3 SAM3 環境（內含，現成）

| 位置（在 `1_room_finder\third_party\sam3-pano-seg\` 內） | 內容 |
|---|---|
| `.runtime\python312\` | Python 3.12.10（基底直譯器） |
| `.runtime\sam3-env\` | 虛擬環境：torch 2.7.1+cu126、torchvision 0.22.1+cu126 等；`sam3` 與 `sam3_pano` 以 editable 方式安裝 |
| `checkpoints\sam3.pt` | SAM3 權重（3.45 GB） |
| `third_party\sam3\` | SAM3 原始碼與授權書 `LICENSE` |
| `sam3_pano\` | 全景分割程式；流程執行 `python -m sam3_pano.seg.unified --categories door --door-prompt-override "open doorway" --pano-vote --mask-only` |

- 不需要另外安裝，但虛擬環境會記住絕對路徑（`pyvenv.cfg` 的 `home`、兩個 `__editable___*_finder.py`）。`tools\fix_sam3_env.ps1` 會把它們改成目前的位置；`Setup.ps1` 與 `Start_Openings.ps1` 會自動呼叫，路徑已經正確時不做任何事。
- 每張全景約 3–4 秒（RTX GPU）；第一次載入模型約 1–2 分鐘。
- 可用 `-Sam3Root`／`RCF_SAM3_ROOT` 改用其他位置的 sam3-pano-seg。刪掉 `third_party\sam3-pano-seg`（省約 9 GB）時，掃描會略過門洞偵測，審查程式改用一般楔形（45°）。

---

## 4. 安裝與驗證

在根目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File Setup.ps1                     # 建立所有環境（需網路，約 10–30 分鐘）
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Verify.ps1   # 用附帶的 9 張範例跑步驟 1–6 並比對參考結果（約 2–4 分鐘）
```

`Setup.ps1` 依序執行：

1. **主環境**：執行 GeoVerify 的 `1_room_finder\third_party\geoverify_main\environments\setup_env.ps1`（用 uv 安裝 Python 3.11、torch 2.10.0 cu130、核心套件、romav2，再重新固定版本），再補裝 `2_pose_floorplan\requirements_main_extra.txt`。這個安裝程式含中文、存成沒有 BOM 的 UTF-8，Windows PowerShell 5.1 在中文系統上無法解析，所以 `Setup.ps1` 改執行一份加了 BOM 的暫存複本（原檔不修改）。RoMa v2 權重會從本機快取複製，或下載到 `1_room_finder\models\torch_home`。
2. **HorizonNet 環境**：`py -3.9 -m venv`，安裝 `requirements_horizonnet.txt`，下載 ImageNet ResNet-50 到 `2_pose_floorplan\models\torch_home`。電腦上沒有 Python 3.9 時，用 uv 下載一份（`uv venv --python 3.9 --seed`）。
3. **審查環境**：依序嘗試 `py -3.11`、`py -3`、`python` 建立 `1_room_finder\.venv`，安裝 `pillow>=9`，檢查 Tkinter 與 Pillow。電腦上沒有含 Tkinter 的 Python 時會停下來：從 python.org 安裝 Python 3（保留「tcl/tk and IDLE」），再執行一次 `Setup.ps1`。
4. **SAM3**：執行 `1_room_finder\tools\fix_sam3_env.ps1`，再做匯入檢查（`import torch, sam3, sam3_pano`）。
5. **檢查**：`python -m pipeline.envcheck`（套件、CUDA、權重 sha256）與 HorizonNet 環境匯入測試。

已存在但無法啟動的環境（例如從其他電腦複製過來）會自動重建，不需要手動刪除。

`Verify.ps1` 用 `sample\` 的 9 張全景、3 個群組跑完步驟 1–6，再用 `pipeline\verify.py` 和原始 Dollhouse 結果比對。通過標準：CoVisPose 每對 yaw ≤ 0.5°、平移 ≤ 0.02；HorizonNet 角點數相同、地板角點 ≤ 0.05；poses 的分量、生成樹、標記相同，相機位置 ≤ 0.05、yaw ≤ 1°；plans 面積差 ≤ 5%。結果寫在 `2_pose_floorplan\runs\_verify_<時間>\verify_result.json`（`passed`、`checks`、`at`）。每次驗證都會建立新資料夾，驗證完可以刪除。

執行 GPU 步驟前，`tools\_env.ps1` 的 `Assert-GpuFree` 會檢查 GPU：只有**其他 Python 程式**正在使用 GPU 時才會詢問 `(y/N)`（輸入 `y` 繼續）。Windows 上 `nvidia-smi` 列出的桌面程式不算。沒有 `nvidia-smi` 時會提示並略過檢查。

---

## 5. 使用方式

以下指令都在根目錄執行。

```powershell
# A（＋A'）匯入並掃描；可中斷續跑（重跑同一指令就會接續）
powershell -ExecutionPolicy Bypass -File 1_room_finder\Import_Dataset.ps1 -Source D:\new_panos -Name new_panos
#   -NoCopy      全景已在 1_room_finder\data\new_panos：只掃描（不給 -Source）
#   -Limit 5     只做 5 對新配對就停（試跑）
#   -NoOpenings  不做 SAM3 門洞偵測

# A' 之後補做門洞偵測（-Scan 相對於 1_room_finder；-Sam3Root 可省略）
powershell -ExecutionPolicy Bypass -File 1_room_finder\Start_Openings.ps1 -Scan scans\new_panos

# B 審查：逐對判定，最後按「匯出房間群組 JSON」（不給參數時會開啟資料夾選擇視窗）
1_room_finder\Start_Review.bat 1_room_finder\scans\new_panos

# 1–6 姿態、平面圖、viewer；可中斷續跑
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Run_Pipeline.ps1 -Groups 1_room_finder\scans\new_panos\room_groups.json -Name new_panos
```

完成後打開 `2_pose_floorplan\runs\new_panos\viewer\viewer.html`。

`Run_Pipeline.ps1` 的選項：

| 選項 | 說明 |
|---|---|
| `-Groups`（必要） | 房間群組檔（§6.3）；相對於執行時所在的資料夾 |
| `-Name`（必要） | run 名稱，輸出在 `2_pose_floorplan\runs\<Name>`。已有 `manifest.json` 的 run 會續跑 |
| `-ScanDir` | 群組檔不在掃描資料夾內時，指向有 `manifest.json` 的掃描資料夾 |
| `-Index` | Dollhouse 的 `data_index.json`，只用來顯示房間名稱 |
| `-Title` | viewer 網頁標題 |
| `-Cpu` | CoVisPose 與 HorizonNet 都用 CPU，不做 GPU 檢查 |
| `-HnCpu` | 只有 HorizonNet 用 CPU（RTX 50 系列） |
| `-NoPoseGraph` | 步驟 4 只用生成樹（`pipeline.poses`，2026-11-19 以前的做法）。預設為位姿圖最佳化（`pipeline.pose_graph`，§11.3） |

審查畫面標示的逐點內點，目前的掃描會直接存下。較舊的掃描可以手動重算（GPU，每對約 3 秒，可中斷續跑），在 `1_room_finder` 內用主環境執行：

```powershell
third_party\geoverify_main\venv\Scripts\python.exe tools\recompute_inliers.py --scan scans\dollhouse [--house 名稱] [--reviewed] [--groups <room_groups.json>]
```

**介面語言**：審查程式與 viewer 預設為英文。可以按語言按鈕切換、在啟動前設定 `RCF_LANG=zh`，或用 `python -m finder.app --lang zh`、`python -m pipeline.viewer --run <run> --lang zh`。viewer 會在瀏覽器裡記住上次的選擇。

**重跑 Dollhouse 104 組**（§16.2）：

```powershell
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Run_Pipeline.ps1 -Groups 1_room_finder\scans\dollhouse\room_groups_dollhouse104.json -ScanDir 1_room_finder\scans\dollhouse -Index 1_room_finder\scans\dollhouse\data_index.json -Name dollhouse104 -Title "Dollhouse 104 groups"
```

---

## 6. 輸入格式

### 6.1 步驟 A（`Import_Dataset.ps1`、`finder.scan`）：全景資料夾

```
<來源資料夾>\
├─ <房屋 1>\            每棟房子一個子資料夾；資料夾名稱就是房屋名稱
│   ├─ pano_001.jpg
│   └─ floor2\pano_002.png      更深的子資料夾也會讀
└─ <房屋 2>\ ...
```

| 規則 | 說明（`finder\dataset.py`、`finder\config.py`） |
|---|---|
| 檔案類型 | `.jpg`、`.jpeg`、`.png`（不分大小寫） |
| 略過的資料夾 | `bi_layout_predictions`、`horizonnet_predictions`、`semantic`、`preview`、`thumbs` |
| 投影 | 等距長方投影。寬／高與 2 的差距要在 ±0.02 以內，否則略過（`not 2:1`）。**步驟 1 會再要求寬度剛好等於 2 倍高度**（§6.3），所以請使用剛好 2:1 的影像 |
| 全景編號（id） | 檔名（不含副檔名）。同一棟裡檔名相同、內容不同時，自動加上 `_2`、`_3`… |
| 重複影像 | 同一棟裡內容相同（前 1 MiB＋檔案大小）只留一張，其他記在 `skipped` |
| 房屋 | 每棟至少要有 2 張全景。來源資料夾沒有含影像的子資料夾時，整個資料夾視為一棟 |
| 配對 | 同一棟的全景兩兩配對（n 張 → n(n−1)/2 對），不分樓層 |
| 無法讀取的檔案 | 略過，原因記在 `manifest.json` 的 `skipped` |

`-Source` 會以 robocopy 複製到 `1_room_finder\data\<Name>`；`-NoCopy` 直接掃描 `data\<Name>`。輸出在 `1_room_finder\scans\<Name>\`（§7）。

### 6.2 步驟 A' 與 B：掃描資料夾

`Start_Openings.ps1 -Scan` 與 `Start_Review.bat` 讀步驟 A 產生的掃描資料夾。裡面必須有 `manifest.json`；全景經由它的 `root`／`root_abs` 找到（§7.1），所以 `data\<Name>` 必須還在。審查程式另外讀 `scores.jsonl`、`pairs\`（或 `pairs_inliers\`），有的話也讀 `open_scores.jsonl` 與 `decisions.jsonl`。

### 6.3 步驟 1（`Run_Pipeline.ps1 -Groups`）：房間群組檔

可以是審查程式的匯出檔（§7.9），也可以是任何具備下列欄位的 JSON；`room_groups_dollhouse104.json`（欄位較多的 `room_classification.json` 格式）也照樣接受。檢查由 `pipeline\prepare.py` 進行：

| 欄位 | 必要 | 規則 |
|---|---|---|
| `groups` | 是 | 不可為空 |
| `groups[].room_id` | 是 | 不可重複 |
| `groups[].task` | 是 | 房屋名稱；必須等於掃描 `manifest.json` 中的某個房屋 |
| `groups[].panorama_uuids` | 是 | 至少 2 張、不重複，每張只能出現在一個群組；每張都必須是掃描 manifest 中該房屋的 `id` |
| `groups[].status` | 是 | 必須是 `human_verified` |
| `groups[].floor`、`space_kind` | 否 | 照抄（預設 `""`、`room`） |
| `pair_reviews[]` | 否 | `uuid_a`、`uuid_b`、`relation`。同一組內有 `different_room` 的配對會拒絕整批；`same_room` 會讓該配對標記為 `directly_reviewed` |
| `conflicts` | 否 | 非空時拒絕整批 |

步驟 1 的其他輸入：

| 輸入 | 預設 | 規則 |
|---|---|---|
| 掃描 `manifest.json` | 群組檔所在的資料夾；或 `-ScanDir` | 使用 `houses[房屋][].id` 與 `path`；先試 `root`（相對於掃描資料夾），再試 `root_abs` |
| 影像 | — | 必須能完整解碼，而且剛好 2:1；失敗的影像在 `manifest.json` 與 `preflight.json` 中逐張記錄（`status: input_error`） |
| `-Index` | 無 | JSON `{"panos": {"<uuid>": {"asteroom_name": "<房間名稱>"}}}`；只用房間名稱 |

不符規則時會在 GPU 步驟之前印出 `Input rejected:`。請在審查程式中修正後重新匯出。

### 6.4 訓練輸入

CoVisPose 訓練需要 ZInD（§13.2）：`zind_data\<房號>\zind_data.json`、`panos\*.jpg`，以及配對清單 `training\pairs\*.csv`（欄位 `house_id, floor_id, room_a, pano_a, room_b, pano_b, covis_score, baseline, dyaw_deg, kind`）。HorizonNet 訓練資料格式見 §14。

---

## 7. 輸出格式：`1_room_finder`

輸出位置：`1_room_finder\scans\<資料集名稱>\`。

### 7.1 `manifest.json`：資料集清單

| 欄位 | 說明 |
|---|---|
| `root` | 全景資料夾，相對於掃描資料夾（例如 `../../data/dollhouse`） |
| `root_abs` | `root` 找不到時使用的絕對路徑 |
| `geoverify_commit`、`mode`、`params` | GeoVerify 版本、模式（固定 `direct`）、參數 |
| `houses` | `{房屋: [{id, path, w, h}]}`；`id` 是全景編號（§6.1），`path` 相對於 `root`，`w`／`h` 是影像像素寬高 |
| `skipped` | 略過的檔案：`[{path, reason}]` |
| `total_pairs`、`created` | 配對數、建立時間 |

Dollhouse 的掃描是從舊實驗匯入的：`w`、`h` 記為 0，另有 `note` 欄位說明來源。

### 7.2 `scores.jsonl`：每一對的 GeoVerify 摘要

每行一對：`{"house", "a", "b", "score": {...}}`。同一對出現多次時以最後一行為準。

| `score` 欄位 | 單位 | 說明 |
|---|---|---|
| `has_pose` | — | GeoVerify 有解出位姿 |
| `n_inlier` | 點 | 本質矩陣內點數 |
| `inlier_rate` | 0–1 | 內點／使用的匹配點 |
| `wedge_a`、`wedge_b` | 度 | 共視楔形寬：包住 80% 內點方位角的最窄圓弧 |
| `wedge_min` | 度 | 兩側較小者；分類器用這個值 |
| `n_regions` | 個 | 共視區域數 |

### 7.3 `pairs\<房屋>\<a>__<b>.json`：GeoVerify 完整輸出

GeoVerify `verify_pair` 的原樣輸出（`schema_version` = `wp1-step3-v2`），完整欄位見 `third_party\geoverify_main\API.md` 與 `pose_regions\README.md`。交接重點：

- 像素座標都在**原圖尺寸**（`img_size_a`、`img_size_b`，例如 8192 × 4096）。
- 方位角：`az = (x / W) × 360 − 180`（度），影像中央 = 0，往右增加。
- `pose.R`、`pose.t`：3D 相對位姿 `x_b = R · x_a + t`，**`t` 是單位向量**；`cam_b_in_a` 是 B 相機在 A 座標中的方向。流程不使用。
- `quality.az_wedge_a_deg`／`az_wedge_b_deg`：即 `scores.jsonl` 的楔形（度）。
- `regions`：共視區域，多邊形頂點是原圖像素。
- 新掃描含 `inliers`；從舊實驗匯入的 Dollhouse 配對沒有。

### 7.4 `pairs_inliers\<房屋>\<a>__<b>.json`：重算後含內點的版本

由 `tools\recompute_inliers.py` 產生，格式同 §7.3，另含 `inliers`：`[x_a, y_a, x_b, y_b, conf]`（原圖像素；最多 2000 點）。審查程式與門洞偵測會優先讀這裡。`consistency.jsonl` 記錄重算結果與原掃描是否一致。

### 7.5 `roma_cache\*.npz`：RoMa 原始匹配

`matches`：N × 5，`[x_a, y_a, x_b, y_b, conf]`，原圖像素。只用來在重算時省掉 RoMa 推論；Dollhouse 只有重算過的 301 對有快取。

### 7.6 `openings\<房屋>\<id>_mask.png`：門洞遮罩

SAM3「open doorway」結果，2048 × 1024、1 位元黑白圖，白色 = 開口（看得穿的門洞）。存的是**未外擴**的遮罩；計分時才外擴（2048 寬度下左右 4 px、上下 10 px）。

### 7.7 `open_scores.jsonl`、`open_scores_meta.json`：去門楔形

每行：`{"house", "a", "b", "open": {...}}`，只有「有位姿且內點 ≥ 30」的配對。

| `open` 欄位 | 單位 | 說明 |
|---|---|---|
| `n_keep` | 點 | 去掉門洞內的內點後剩下的數量（已換算回完整內點數） |
| `door_frac` | 0–1 | 落在任一張遮罩內的內點比例 |
| `wedge_a`、`wedge_b`、`wedge_min` | 度 | 用剩下的內點重算的楔形 |

`open_scores_meta.json`：提示詞、遮罩尺寸、外擴量、處理數量。

### 7.8 `decisions.jsonl`：人工審查紀錄

只增不改，每次判定一行；同一對（不論 `a`、`b` 順序）以最後一行為準。

| 欄位 | 說明 |
|---|---|
| `house`、`a`、`b` | 配對 |
| `relation` | `same_room`、`different_room`、`uncertain`；`null` = 清除先前的判定 |
| `note`、`reviewer`、`at` | 備註、審查者、時間 |
| `score` | 判定當時的分數（同 §7.2，含 `open`） |
| `thresholds` | 判定當時審查程式的門檻與模式 |

### 7.9 `room_groups.json`：匯出的房間群組（交給 `2_pose_floorplan`）

由審查程式「匯出房間群組 JSON」產生，檔名可自訂。

| 欄位 | 說明 |
|---|---|
| `schema_version` | 1 |
| `generator` | `room_candidate_finder` |
| `classifier` | 匯出時的規則文字、門檻、GeoVerify 版本 |
| `exported_at` | 匯出時間 |
| `groups` | 每組：`room_id`（`<房屋>_room_001`…）、`task`（房屋）、`floor`（空字串）、`room_label`（空字串）、`space_kind`（固定 `room`）、`panorama_uuids`、`status`（`human_verified`）、`notes`、`source_pairs`、`created_at` |
| `conflicts` | 同一組內有「不同房」判定的配對；非空時步驟 1 會拒絕整批 |
| `pair_reviews` | 每一對的人工判定（`pair_id`、`uuid_a`、`uuid_b`、`relation`…） |

群組是「同房」判定的連通分量，至少 2 張；只有 1 張的全景不會出現在任何群組。

### 7.10 Dollhouse 專用檔案（`scans\dollhouse\`）

| 檔案 | 說明 |
|---|---|
| `room_groups_dollhouse104.json` | **104 組正式結果的輸入**（43 棟、278 張、301 對）。來自 `room_stitch_mvp` 的人工分類（`room_classification.json` 格式，欄位比 §7.9 多，`room_id` 為 `task_<編號>_<樓層>_room_<序號>`），可直接給 `Run_Pipeline.ps1` |
| `data_index.json` | Dollhouse 標籤索引；流程只讀 `panos[uuid].asteroom_name`。其他欄位含原開發機路徑，不使用 |
| `room_groups_test_20261026.json` | 2026-10-26 審查程式的測試匯出：6 組、10 筆審查、一般楔形 20°。**不是** 104 組 |

### 7.11 其他

| 檔案 | 說明 |
|---|---|
| `status.json`、`scan.log`、`RUNNING.lock` | 掃描進度、紀錄、執行中的鎖定檔（確定沒有掃描在跑時才手動刪除） |
| `openings.log`、`openings_todo.txt` | 門洞偵測紀錄；暫存資料夾 `openings_stage\`、`openings_raw\` 完成後會刪除 |

---

## 8. 輸出格式：`2_pose_floorplan`

輸出位置：`2_pose_floorplan\runs\<名稱>\`。步驟 4、5 每次執行都會建立新的時間戳資料夾，不覆寫舊的；後續步驟讀最新的一個。

### 8.1 凍結的輸入（步驟 1）

| 檔案 | 內容 |
|---|---|
| `classification_source.json` | 輸入群組檔的原樣複本 |
| `manifest.json` | `groups`（room_id、task、floor、space_kind、panorama_uuids、status）；`images`（每張的 `id`、`label`、`house`、`status`、`path`、`sha256`、`size` [寬, 高]，失敗時有 `error`）；`pairs`；`duplicate_images`（內容完全相同的影像） |
| `run_config.json` | 推論設定（EMA、TTA、FP32）、權重路徑與 sha256、凍結程式的雜湊、採用規則參數、輸入檔的雜湊 |
| `preflight.json` | 群組數、房屋數、影像數、配對數、群組大小分布、輸入錯誤、`status`（`ready_for_gpu` 或 `input_errors_preserved`） |

`pairs` 的每一對：`pair_id`（`sha256("v2|" + room_id + "|" + a + "|" + b)` 前 24 碼，`a`、`b` 依字母排序）、`group_id`、`a`、`b`、`directly_reviewed`、`duplicate_content`、`blind_review`（保留欄位）。

### 8.2 CoVisPose 結果（步驟 2）：`predictions\`

`<pair_id>.json`：

| 欄位 | 單位 | 說明 |
|---|---|---|
| `status` | — | `ok` 或 `error`（附 `reason`） |
| `yaw` | **弧度** | 解碼器的相對旋轉 |
| `t` | 相機高度 | 解碼器的相對平移 [x, y]；A 相機在 B 座標中的位置 |
| `geo_yaw`、`geo_t` | 弧度、相機高度 | 幾何分支（逐欄對應 Procrustes）的位姿 |
| `geo_weight` | — | 幾何分支使用的對應權重總和；約 0 代表沒有可用的對應 |
| `heads_sha256` | — | 對應 `.npz` 的雜湊（續跑時檢查） |
| `seconds` | 秒 | 該對推論時間；啟用特徵快取時，第一次用到某張全景的配對較慢 |

位姿慣例：`p_B = R(yaw) · p_A + t`，4 視角 TTA 平均。

`<pair_id>.npz`：6 個長度 1024 的 float32 陣列；第 j 欄的方位角 `θ_j = ((j + 0.5) / 1024 − 0.5) × 2π`（第 0 欄是影像最左邊）。

| 陣列 | 單位 | 說明 |
|---|---|---|
| `covis_a`、`covis_b` | 0–1 | 該欄內容在另一張全景中看得到的機率 |
| `corr_a`、`corr_b` | 弧度，[−π, π) | 該欄內容在另一張全景中的方位角 |
| `bon_a`、`bon_b` | 弧度 | 地板與牆交界的俯角 b（> 0）；距離 = 1 / tan(b) 相機高度 |

`inference_status.json`：完成與失敗的配對。`run_env_<時間>.json`：Python、torch、CUDA、使用裝置（`cuda`／`cpu`）、GPU 名稱、CPU 執行緒數。`covispose_gpu_<時間>.log`：步驟 2 的畫面輸出（UTF-16）。

### 8.3 HorizonNet 結果（步驟 3）：`horizonnet\`

`<全景>.json`：

| 欄位 | 說明 |
|---|---|
| `uv` | 房間角點，**正規化影像座標**（0–1，u 向右、v 向下），已映回**原圖**；每個角點兩點，天花板（v < 0.5）與地板（v > 0.5） |
| `z0`、`z1` | HorizonNet 內部高度：天花板固定 50，地板為負值；`z0 / −z1` = 天花板到相機 ÷ 相機到地板 |
| `room_stitch_alignment` | 消失點對齊資訊：`vp`、旋轉 `R`、`tilt_deg`、`yaw_deg`、`det` |

`provenance_<時間>_<批次>.json`：指令、權重雜湊、每張的對齊診斷。`status.json`：沒有結果的全景（這些全景改用 CoVisPose 的外框）。

### 8.4 相機位姿（步驟 4）：`poses_<時間>\`

**`groups\<群組>.json`**

| 欄位 | 說明 |
|---|---|
| `convention` | `p_root = R(yaw) p_u + t; camera height = 1; root = most-connected camera` |
| `method` | `pose_graph`（預設）或 `spanning_forest`（`-NoPoseGraph`）。2026-11-19 以前產生的資料夾沒有這個欄位，即 `spanning_forest` |
| `rule` | 採用規則與求解方式（G 規則＋方向連線 f = 0.30；最大權重生成樹；之後做位姿圖最佳化，`-NoPoseGraph` 時不做） |
| `components` | 連通分量（全景 uuid 清單） |
| `cameras` | 每台相機：`label`、`component_root`、`is_root`、`yaw`（**弧度**）、`t`（相機高度，原生座標）、`flags`、`weak_edges`、`bad_loop` |
| `tree_edges` | 生成樹使用的配對 |
| `pairs` | 每一對：`status`、`in_tree`、`inliers`、`inlier_fraction`、`correspondences`、`decoder_geometric_yaw_deg`（**度**）、`decoder_geometric_t`、`yaw`（**弧度**）、`t`、`duplicate_content` |
| `loops` | 不在樹上的採用連線：`yaw_deg`（**度**）、`translation`（相機高度）= 該連線的預測與最終相機位姿的差（生成樹的位姿，或位姿圖最佳化後的位姿） |
| `flag_stats` | 這個方法下各標記在 ZInD test 上的錯誤率（位姿圖為 `pose_graph_zind_test_v1`，生成樹為 `pose_flags_v1`；§11.3）；viewer 顯示這組數字 |
| `summary`、`pose_verified` | 群組摘要；固定 `false` |

配對狀態 `status`：`accepted`（完整連線）、`accepted_direction_only`（方向連線：方向可信，位置可能偏約 0.6 相機高）、`low_inlier_fraction`、`too_few_inliers`、`no_correspondences`、`decoder_geometric_disagree`。

相機標記 `flags`：`direction_link`（路徑上有方向連線）、`weak_edge`（路徑上有弱連線）、`loop_not_closing`（所在迴圈殘差 > 5° 或 > 0.30）、`unverified`（不在任何迴圈上，只是資訊）。前三種在 viewer 中以橘色或紅圈「!」顯示。

**CSV（UTF-8 含 BOM）**

| 檔案 | 欄位 |
|---|---|
| `cameras.csv` | group、house、camera、label、component_root、`yaw_deg`（**度**）、`x`、`y`（原生座標）、direction_link、weak_edge、loop_not_closing |
| `pairs.csv` | group、pair_id、a、b、status、in_tree、inliers、inlier_fraction、correspondences、decoder_geometric_yaw_deg、decoder_geometric_t、duplicate_content（不含位姿） |
| `groups.csv` | group、house、n_images、pairs、accepted_full、accepted_direction、component_sizes、fully_connected、loop_max_yaw_deg、loop_max_translation、flagged_weak_or_loop |

`summary.json`：全部群組的合計，含 `method`（位姿圖另有 `components_optimised`）。`plots\<群組>.png`：相機位置圖（已用 (−x, y) 畫成俯視圖）。

`pose_graph_report.json`（只有位姿圖）：設定值（`sigma_yaw_deg`、`sigma_t`、`sigma_t_direction`、`huber_f_scale`），以及每個群組的 `components`（每個最佳化的分量：`root`、`cameras`、`pairs`、`optimised`、`nfev`、`cost_initial`、`cost_final`、`message`）、`pairs_forest` 與 `pairs_pose_graph`（每一對採用連線與生成樹位姿、與最佳化後位姿的差：`yaw_deg`、`translation`）。相機的 `flags` 由原始的配對預測計算，兩種方法相同。

### 8.5 平面圖（步驟 5）：`plans_<時間>\`

**`groups\<群組>.json`** 的 `components`，每個分量：

| 欄位 | 說明 |
|---|---|
| `panoramas`、`root` | 分量內的全景、原點相機 |
| `method` | `HN union`（≥ 2 張：HorizonNet 外框聯集）或 `MX (single panorama)` |
| `outline_source` | 每張全景在 `plan` 中使用的外框：`HN` 或 `MX` |
| `plan` | 合併平面圖：多邊形清單，每個是外框頂點 `[[x, y], …]`（不重複首點；**孔洞已丟棄**），原點相機座標，單位相機高度 |
| `area` | `plan` 的面積，相機高度²（扣掉孔洞） |
| `covispose_plan` | 只用 CoVisPose 外框的聯集（對照用） |
| `outlines` | 每張全景的三種外框，**各自的相機座標**（未放到原點座標）：`CV`（CoVisPose 地板邊界，各配對取中位數）、`HN`（HorizonNet）、`MX`（逐欄取兩者較遠者）；無法取得時為 `null` |

把 `outlines` 放到原點座標：`p_root = R(yaw) · p + t`，`yaw`、`t` 取自 `poses` 中該相機的值。

`index.json`：每組的 `n`、各分量大小、`hn_missing`。`plots\<群組>.png`：平面圖（俯視圖）。

### 8.6 互動 viewer（步驟 6）：`viewer\`

| 檔案 | 說明 |
|---|---|
| `viewer.html` | 單一網頁，資料與縮圖（JPEG quality 70）直接嵌在頁面中；不需要伺服器 |
| `images\<全景>.jpg.js` | 全景圖（4096 × 2048、JPEG quality 82、base64），由頁面載入 |

### 8.7 驗證結果

`verify_result.json`（`Verify.ps1`）：每項檢查的結果與 `passed`。

---

## 9. 單位、座標與俯視圖

### 9.1 角度單位對照

| 單位 | 欄位 |
|---|---|
| 弧度 | `predictions\*.json` 的 `yaw`、`geo_yaw`；`predictions\*.npz` 的 `corr_*`、`bon_*`；`poses_*\groups\*.json` 的 `cameras.*.yaw` 與 `pairs[].yaw` |
| 度 | 名稱有 `…_deg` 的欄位；`cameras.csv` 的 `yaw_deg`；`loops[].yaw_deg`；`1_room_finder` 的所有楔形；GeoVerify 的 `az_*`、`rot_angle_deg` |

### 9.2 座標定義

- **全景欄位與方位角**：第 j 欄（寬 W）的方位角 θ = ((j + 0.5) / W − 0.5) × 2π；中央 = 0，往右增加。GeoVerify 用度，方向相同。
- **單張全景的平面座標**（CoVisPose 原生，與 ZInD 相同）：方位角 θ 的方向是 `(−sin θ, cos θ)`。影像中央在 +y；影像右半邊的東西在 x < 0。
- **相對位姿**（`predictions`、`poses` 的 `pairs`）：`p_B = R(yaw) · p_A + t`，`R(yaw) = [[cos, −sin], [sin, cos]]`；`t` 是 A 相機在 B 座標中的位置。
- **相機位姿**（`poses` 的 `cameras`）：`p_root = R(yaw) · p_u + t`；`t` 是相機 u 在原點相機座標中的位置，相機正前方在該座標中的方向是 `(−sin yaw, cos yaw)`。原點是分量中連線最多的相機；沒有絕對尺度與方位。
- **平面圖**：原點相機座標，與 `poses` 相同。

### 9.3 畫俯視圖要用 (−x, y)

直接以 x 向右、y 向上畫原生座標，會得到左右鏡像的圖。正確的俯視圖是 (−x, y)，`poses`、`plans` 的 PNG 與 viewer 都是這樣畫。

確認方式（2026-11-17）：

1. **推導**：站在相機位置面向影像中央（+y），影像右側的東西在原生座標的 x < 0，畫成 (−x, y) 後才會在右邊。
2. **以 ZInD 官方平面圖比對**：ZInD 0000 號房屋的房間外框經 ZInD 自己的座標轉換，放到 Zillow 繪製的平面圖（`floor_plans\floor_01.png`）上完全對齊，條件是 x 向右、y 向下；這等同 (x, −y)，與 (−x, y) 只差 180° 旋轉，兩者都不是鏡像。

前提：全景圖是一般的等距長方投影（往影像右側 = 從上方看順時針）。ZInD 是如此；Dollhouse 沒有獨立的平面圖可比對，但流程中 CoVisPose、HorizonNet、viewer 用同一套方位角定義，彼此一致。

```python
import json, numpy as np, matplotlib.pyplot as plt
g = json.load(open(r'runs\<名稱>\plans_<時間>\groups\<群組>.json', encoding='utf-8'))
for comp in g['components']:
    for ring in comp['plan']:
        p = np.array(ring + ring[:1])
        plt.plot(-p[:, 0], p[:, 1])          # 俯視圖：(-x, y)
plt.gca().set_aspect('equal'); plt.show()
```

---

## 10. 運算速度

以下都是實際執行紀錄的數字。GPU 型號不同時，時間大致與 GPU 效能成反比。

| 步驟 | 單位 | 每單位時間 | 測量環境 | 依據 |
|---|---|---|---|---|
| A. GeoVerify 掃描 | 每對 | **中位數 2.59 秒**，平均 2.80 秒（RoMa v2 2.54 秒＋幾何驗證 0.05 秒）；實務上**約 3 秒** | RTX 4060、torch 2.10 cu130，120 對 | `1_room_finder\third_party\geoverify_main\environments\timing_torch210_n120.md` |
| A. 模型載入 | 每次掃描 | 6.4 秒 | 同上 | 同上 |
| A'. SAM3 門洞偵測 | 每張全景 | **中位數 3.19 秒**，p90 3.75 秒；第一次載入模型 1–2 分鐘 | RTX 3060，828 張 | sam3-pano-seg `outputs\samroom_opendoorway.log` |
| A'. 去門楔形計算 | 每份掃描 | 數秒到數十秒（CPU） | — | — |
| 1. prepare | 每次執行 | 約 7 秒（主要是計算權重的 sha256） | 2 核 CPU | 2026-11-05 打包驗證 |
| 2. CoVisPose | 每對 | **約 0.48 秒**（4 視角 TTA、FP32）；載入約 7 秒 | RTX 3060；301 對 144 秒（加入特徵快取前） | `C:\covispose\outputs\manual_room_groups_v2\` |
| 2. CoVisPose，ZInD test | 2,536 對 | 約 8.5 分鐘（每對約 0.2 秒：3–11 張的房間重複使用編碼器特徵） | RTX 3060 | 2026-11-19，`runs\zind_test_posegraph` |
| 2. CoVisPose（CPU） | 每對 | 未快取約 10 秒；快取後每對 2.8 秒＋每張全景一次約 1.5 秒 | 2 核 CPU，範例 | 2026-11-05、2026-11-15 驗證 |
| 3. HorizonNet | 每張全景 | 約 6.7 秒（消失點對齊約 5.4 秒，主要吃 CPU＋推論約 1.3 秒） | RTX 3060，13 張 | `provenance_*.json` |
| 3. HorizonNet（CPU） | 每張全景 | 前處理 6.64 秒＋推論 1.26 秒 | 2 核 CPU，404 張 | `evaluation\horizonnet_404\REPORT_ZH.md` §5 |
| 4. poses／5. plans | 每次執行 | 各 1.7 秒（範例） | 2 核 CPU | 打包驗證 |
| 6. viewer | 每次執行 | 5.7 秒（9 張全景，每張約 0.6 秒） | 2 核 CPU | 打包驗證 |

估算一批資料的時間：

- 步驟 A：一棟 n 張 → n(n−1)/2 對 × 約 3 秒。30 張 → 435 對 → 約 19–22 分鐘。Dollhouse 60 棟、7,985 對 → 約 6–7 小時。
- 步驟 A'：每張約 3.2 秒，加上載入 1–2 分鐘。
- 步驟 2：只算群組內的配對，k 張的群組有 k(k−1)/2 對。Dollhouse 104 組、301 對 → GPU 約 2.4 分鐘。
- 步驟 3：每張約 6.7–7.9 秒。Dollhouse 278 張 → 約 31–37 分鐘。
- 步驟 1、4、5、6：通常一分鐘以內。
- 一次只能跑一個 GPU 步驟；掃描和流程不能同時跑。

訓練速度：3 張 GPU（A5000×2＋A6000）每秒約 36.8 對；每個 epoch 約 4 小時 13 分；16 個 epoch 共 46 小時 07 分。

---

## 11. 演算法與參數

### 11.1 同房候選分類器（步驟 A、A'）

依據 `1_room_finder\docs\benchmark\RESULTS_ZH.md`（2026-10-23）與 `RESULTS_open_wedge_ZH.md`（2026-10-27）。資料為 Dollhouse 60 棟、1,232 張室內全景、7,985 對同樓層配對；GT 是人工群組加盲審（148 組、410 張）。

| 規則 | 定義 | 預設門檻（`finder\config.py`） |
|---|---|---|
| 一般楔形 | GeoVerify `verify_pair` direct 模式、預設參數：有位姿、E 內點 ≥ 30、兩側 80% 共視楔形都 ≥ 門檻 | 45° |
| 去門楔形 | 內點在 A 或 B 任一側落在 SAM3「open doorway」遮罩內就剔除（2048×1024 下左右外擴 4 px ≈ 0.7°、上下 10 px）；剩餘內點 ≥ 30 時重算 80% 楔形，兩側都 ≥ 門檻 | 40° |

| 規則 | 門檻 | Precision | Recall | F1 | 候選對數 |
|---|---:|---:|---:|---:|---:|
| 一般楔形 | 45° | 0.507 | 0.538 | 0.522 | 445 |
| 去門楔形 | 40° | 0.559 | 0.584 | 0.571 | 439 |

留一棟交叉驗證：F1 0.535 → 0.569。其他門檻的 P/R 見 `1_room_finder\docs\REVIEW_APP_ZH.md` §5；審查程式會依 `finder\calibration_dollhouse.json` 顯示預估值。

### 11.2 CoVisPose 推論（步驟 2）

- 權重：`server_c10neg` 第 15 個 epoch（餘弦退火最後一個 epoch）的 **EMA 權重**。
- FP32、關閉 TF32；4 視角 TTA（原圖、轉半圈、鏡射、鏡射加轉半圈），還原座標後平均。
- 輸入 512 × 1024；姿態慣例 `p_B = R(yaw) p_A + t`，相機高 = 1。
- 逐欄對應的自我檢查參數（`RULES`）：共視門檻 0.5、最遠距離 10、殘差 ≤ 0.30、內點 ≥ 20、內點比例 ≥ 0.40。
- 特徵快取：每張全景只編碼一次（原圖＋鏡射；轉半圈直接位移特徵），同一群組的所有配對重複使用。每次執行的第一對會同時用不快取的做法計算並比對（yaw ≤ 0.05°、平移 ≤ 0.002）；不一致時改回不快取的做法繼續。`--no-feature-cache` 可關閉。
- 裝置：`--device auto|cuda|cpu`（預設 auto）、`RCF_DEVICE`、`Run_Pipeline.ps1 -Cpu`。

### 11.3 採用規則、位姿圖與疑慮標記（步驟 4）

- **採用**：內點 ≥ 20（殘差 ≤ 0.30）、內點比例 ≥ 0.40，且解碼器與幾何分支的位姿一致（≤ 5°、≤ 0.25）。
- **方向連線**：內點比例 0.30–0.40 且兩分支一致。方向可信，位置可能偏約 0.6 相機高（viewer 中為橘色）。
- **生成樹**：以採用的連線建最大權重生成森林（完整連線權重 1 + f，方向連線 f），沿樹推出每台相機的位姿；原點是連線最多的相機。`-NoPoseGraph` 時這就是結果（2026-11-19 以前的做法）。
- **位姿圖最佳化**（2026-11-19 起為預設，`pipeline\pose_graph.py`）：以生成樹為初始值。凡是採用的連線多於生成樹所需（至少有一個迴圈）的分量，把所有採用的連線一起做穩健最小平方（`scipy.optimize.least_squares`，Huber 損失）。連線 (a, b) 的預測位姿為 (yaw, t)，殘差為 `wrap((yaw_a − yaw_b) − yaw) / 1°` 與 `(R(−yaw_b)(t_a − t_b) − t) / 0.06`（方向連線除以 0.6），各乘上 √權重（權重與生成樹相同）；Huber 損失在 3σ 以外改為線性，一對錯誤的連線不會拖歪整個迴圈。配對位姿使用解碼器的位姿（4 視角 TTA），也就是採用規則依據的估計；1° 與 0.06 約為它在 ZInD test 上的誤差中位數（`training\PHASE5_NOTES.md`）。沒有迴圈的分量維持生成樹的位姿不變。不需要重新訓練，只使用已存的預測。
- **紅圈「!」標記**：到原點的路上有弱連線，或迴圈不閉合（> 5° 或 > 0.30）。由原始的配對預測計算，兩種方法標記的相機相同；標記代表「值得檢查」，不代表一定錯。

**ZInD test 評估（2026-11-19）**：ZInD test 中所有至少 3 張全景的完整房間（219 間、158 棟、1,087 張全景、2,536 對；輸入在 `evaluation\zind_test_posegraph\`，由 `scripts\make_zind_room_groups.py` 產生），經 `prepare` 與未修改的 CoVisPose 步驟（RTX 3060），再以 `scripts\eval_pose_graph_zind.py` 評估。上述參數在測試前就已固定。落在同一分量的任兩台相機（2,294 組，兩種方法相同）都與 ZInD 的相對位姿比對；183 間房有迴圈。

| 指標 | 生成樹 | 位姿圖 |
|---|---|---|
| yaw 中位數／平均 | 1.96° / 4.54° | **1.49° / 3.36°** |
| 平移中位數／平均（相機高度） | 0.148 / 0.297 | **0.132 / 0.249** |
| 準確率 ≤ 0.1、1° | 17.7% | 19.8% |
| 準確率 ≤ 0.3、5° | 68.7% | **76.2%** |
| 準確率 ≤ 0.5、10° | 87.0% | 89.7% |
| 準確率 ≤ 1.0、20° | 94.7% | 95.7% |
| 錯誤相機（多數配對 > 0.5 或 > 10°） | 7.05%（74 / 1,050） | **6.10%（64 / 1,050）** |

- 有迴圈的房間，以房間為單位 bootstrap（位姿圖 − 生成樹，95% 信賴區間）：平移平均 −0.050 [−0.088, −0.023]；yaw 平均 −1.25° [−2.11°, −0.60°]；準確率 ≤ 0.3、5° +7.9 個百分點 [+4.2, +11.4]；準確率 ≤ 0.1、1° +2.1 個百分點 [−0.5, +4.7]（不顯著）。116 間變好、67 間變差。
- 房間越大改善越多（準確率 ≤ 0.3、5°）：3 張 84.1% → 86.4%，5 張 72.7% → 79.3%，7 張以上 65.5% → 74.7%。
- 不是每一處都變好：125 組相機從錯變對，64 組從對變錯（錯 = > 0.5 或 > 10°）。
- 標記（有標記／無標記的錯誤率）：W 生成樹 22.8% / 4.5%，位姿圖 21.4% / 3.7%；L 10.3% / 5.0% → 7.1% / 5.5%；D（21 台）14.3% → 28.6%。W 仍能分辨錯誤相機；L 的分辨力大幅下降，因為它標記的迴圈多半已被最佳化修正。位姿圖的輸出帶有這組數字（`flag_stats`）。
- 這組資料上被採用的配對位姿：yaw 中位數 1.16°、平移中位數 0.093（1,475 條完整連線）。比 `PHASE5_NOTES.md` 的數字差，因為這裡包含房間內所有配對（含重疊很少的）；不影響兩種方法的比較。
- 每組相機的結果：`runs\zind_test_posegraph\pose_graph_zind_eval_20261119_173053.json`。重做：`powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\evaluation\zind_test_posegraph\Run_ZInD_PoseGraph_Eval.ps1`（ZInD 在 `C:\zind_data`；RTX 3060 約 10 分鐘）。若要調整參數，請在 validation split（`make_zind_room_groups.py --split val`）上調，不要用 test。

**Dollhouse 104 組**（沒有真值）：22 個分量有迴圈；278 台相機中 65 台移動（中位數 0.031 相機高度／0.73°；最大 0.93／23.7°，在 `task_69734_floor2_room_001`，其中洗衣間–食物儲藏室–廚房 2–廚房的迴圈本身互相矛盾）。留一對檢查（以其他連線預測迴圈上的每一條連線）看不出明顯差異，這符合迴圈少且小的情況。套件內的 `runs\dollhouse104\` 是用生成樹產生的；要取得位姿圖版本，在這個 run 的複本上執行 `python -m pipeline.pose_graph`、`pipeline.plans`、`pipeline.viewer`（試驗用的 run 已移出套件）。

### 11.4 HorizonNet（步驟 3）

用官方 `preprocess.py`（LSD 消失點、完整 3D 旋轉對齊）→ `inference.py`（MP3D 權重）推論對齊後的影像 → 角點映回原圖座標。`pipeline\horizonnet.py` 使用純英文的暫存資料夾（`HN_ASCII_TMP`），並在 HorizonNet 環境沒有 CUDA 時自動加 `--no_cuda`。

在 404 張 Dollhouse 全景上的準確度（2026-11-05，`evaluation\horizonnet_404\REPORT_ZH.md`）：

| 指標 | Dollhouse 404 張 | 官方：MatterportLayout |
|---|---:|---:|
| 2D IoU（對每張全景自己的 `layout_gt`） | 51.4%（95% 區間 46.7–56.5） | 81.71% |
| 3D IoU | 44.9% | 79.11% |
| 排除尺度後的 2D IoU（形狀） | 66.8%（63.9–70.2；中位數 77.2%） | — |
| 逐欄牆距相對誤差（中位數） | 21.9% | — |

- 臥室、車庫、浴室、儲藏的形狀 IoU 是 0.70–0.78；開放式客廳、廚房、餐廳、走廊只有約 0.5。常見失誤是外框切過家具，或穿過門洞延伸到隔壁空間。
- 絕對 IoU 是低估值：`layout_gt` 有整棟性的上下偏移。58498、59868、58599 三棟經目視確認真值牆線有誤；可能有 7 棟受影響。以這份真值做的評估（例如 §11.5 的平面圖規則）也可能受影響。
- 角點後處理對浮點誤差敏感：換 GPU 或驅動後，個別全景的角點數可能不同（§16.2）。

### 11.5 平面圖（步驟 5）

多張全景的群組取 HorizonNet 外框的聯集；單張取 CoVisPose 與 HorizonNet 逐欄較遠者。規則是在 Dollhouse 36 個有真值的群組上比較多種方法後選定的。聯集會偏大，開口處可能穿進隔壁房間。

---

## 12. 模型與權重

| 檔案（在 `2_pose_floorplan` 內） | sha256 | 來源 | 訓練資料與授權 |
|---|---|---|---|
| `models\covispose\epoch_015.pt` | `d2daeb2af602a61c0c0d610c12d1f721fcc2b578fd59a45252cb295095775bed` | 本專案訓練（`training\configs\server_c10neg.yaml`）；含原始權重、EMA、AdamW 狀態、模型設定 | ZInD：**只限學術、非商業** |
| `models\covispose\epoch_015_ema.pt`（選用，0.35 GB） | `f87c3122…2436c` | 同一個 epoch 的 EMA 匯出檔（`C:\covispose\weights\covispose_c10neg_epoch015_ema.pt`） | 同上 |
| `models\horizonnet\resnet50_rnn__mp3d.pth` | `bdef2e337133f8f2937f99e7643cbef7be56a745a1fe3dad00d855bdcadd0c96` | HorizonNet 官方釋出 | Matterport3D 布局標註；依 Matterport3D 與 HorizonNet 授權 |

- `epoch_015.pt` 不存在而 `epoch_015_ema.pt` 存在時，流程改用 EMA 匯出檔（`pipeline\common.py`）。只有續訓、使用原始（非 EMA）權重或重新匯出時，才需要完整檢查點。
- `pipeline\envcheck.py` 的 `CKPT_SHA`（兩個雜湊都認得）與 `HN_SHA` 會在 Setup 與每次執行前檢查。

---

## 13. 如何訓練 CoVisPose

### 13.1 原始訓練的環境（`training\env\`）

| 項目 | 值 |
|---|---|
| 作業系統 | Ubuntu 20.04（Linux 5.15） |
| GPU | RTX A5000 24 GB ×2、RTX A6000 48 GB ×1（Ampere） |
| 驅動／CUDA | 570.181／CUDA 12.8，cuDNN 9.19 |
| Python | 3.11.16（conda） |
| 主要套件 | torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 1.26.4、scipy 1.17.1、pillow 12.3.0、PyYAML 6.0.3、shapely 2.1.2、matplotlib 3.11.2、tqdm 4.70.1（完整清單 `training\env\pip_freeze.txt`；最嚴格的重建 `conda_explicit.txt`） |
| 時間與資源 | 16 個 epoch，共 46 小時 07 分；每 epoch 16,456 步；rank 0 峰值 16.56 GiB |

```bash
# 在 2_pose_floorplan 資料夾內執行（Linux＋NVIDIA）
conda create -y -p ~/envs/covispose python=3.11
conda activate ~/envs/covispose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training/env/requirements-train.txt
# 要求版本完全相同時：pip install -r training/env/pip_freeze.txt
```

這是訓練環境。Windows 上的推論用主環境（§3.2），兩者讀同一份 `covispose\` 程式與同一種 checkpoint 格式。

### 13.2 資料（ZInD，不在包內）

到 https://github.com/zillow/zind 申請下載（學術、非商業授權）。

| 檔案 | 用途 | 位置 |
|---|---|---|
| `zind_data\<4 位數房號>\zind_data.json`、`panos\*.jpg` | 全景與標註 | `ZIND_ROOT` 或 `--data-root` |
| `covis_scores.csv` | ZInD 提供的全景配對共視分數 | `--covis-csv` |
| `zind_partition.json` | 官方 train／val／test 分割 | `--partition` |

原始訓練使用的配對清單已附上（`training\pairs\`）：

| 檔案 | 配對數 |
|---|---:|
| `train_c10neg.csv` | 394,963 |
| `val_c10neg.csv` | 44,852 |
| `test_c10neg.csv` | 47,271 |
| `test_c10neg_adjacent.csv` | 6,208 |

欄位：`house_id, floor_id, room_a, pano_a, room_b, pano_b, covis_score, baseline, dyaw_deg, kind`（`kind` 為 `pos` 或零共視負例）。重建方法：

```bash
python scripts/build_pair_index.py --data-root $ZIND_ROOT --covis-csv <covis_scores.csv> --partition <zind_partition.json> \
    --covis-min 0.10 --neg-ratio 0.4 --adj-per-house 40 --tag _c10neg      # 輸出到 training/pairs/
```

規則：共視 ≥ 0.10、基線 ≥ 0.10 相機高，正負例比 2.5:1，每棟加入 40 組相鄰負例；所有正例都在同一個 `complete_room` 內。

### 13.3 訓練指令（在 `2_pose_floorplan` 內執行）

```bash
export ZIND_ROOT=/path/to/zind_data
python scripts/benchmark_vram.py --config training/configs/server_c10neg.yaml --data-throughput   # 換機器時先量
torchrun --standalone --nproc_per_node=3 scripts/train.py --config training/configs/server_c10neg.yaml   # 3 張 GPU（原始設定）
python scripts/train.py --config training/configs/server_c10neg.yaml                                    # 1 張 GPU
python scripts/train.py --config training/configs/smoke_cpu.yaml                                        # CPU 冒煙測試（仍需要 ZInD）
```

checkpoint 寫到 `2_pose_floorplan/ckpt/server_c10neg/`；`resume: auto` 會在中斷後接續。其他選項：`--max-epochs`、`--out-dir`、`--resume`、`--val-subsample`、`--quickval-every`、`--save-every-steps`、`--limit-steps`、`--data-root`。

### 13.4 設定重點（`training\configs\server_c10neg.yaml`）

| 類別 | 值 |
|---|---|
| 模型 | ResNet-50（IMAGENET1K_V2，需網路下載）、輸入 512×1024、每張 1024 個欄 token、d_model 384、8 層 Transformer、8 heads、dim_ff 1536、drop_path 0.1；attention 姿態解碼器（4 query、2 層）＋可微分加權 Procrustes 幾何分支（IRLS 2 次，c = 0.3）；共 88.2 M 參數 |
| 損失權重 | covis 1.0、corr 1.0、bon 2.0、pose_t 1.0、pose_yaw 1.0、geo_t 0.5、geo_yaw 0.5（幾何項在第 9,000–18,000 步之間逐漸加入） |
| 最佳化 | AdamW、lr 3e-4（backbone ×0.1）、weight decay 0.05（norm／bias／embedding 不衰減）、梯度裁切 5.0、warmup 4,000 步、餘弦退火到 1%、EMA 0.9998 |
| 批次 | 每張 GPU 8 對 × 3 = 24 對／步；bf16 AMP、TF32、channels_last、fused AdamW |
| 資料增強 | 精確水平旋轉、配對互換、gamma、50% 雙圖鏡射；`fast_jpeg` 在 DCT 域做 1/2 縮小 |
| 驗證 | 每個 epoch 跑完整 val split（原始與 EMA）；每 2,000 步 quickval |

- lr 5e-4（依 GPU 數開根號放大）實測不穩定，已改回 3e-4（`training\SERVER_OPTIMIZATION.md`）。
- bf16 需要 Ampere 以上；較舊的 GPU 要改 `amp_dtype`，並關掉 `tf32`、`fused_adamw`。
- 單張 GPU 時有效批次從 24 變 8，lr 與步數需要依此調整。

### 13.5 選 checkpoint 與評估

- `best.pt`（EMA val loss 最低）選到 epoch 13；**流程使用 epoch 15**，它在高共視配對上較好（`training\PAPER_COMPARISON.md`）。
- epoch 15 驗證結果（EMA，44,852 對）：共視準確率 0.940、IoU 0.626、對應誤差 15.9°、邊界誤差 1.71°、平移誤差 0.248、yaw 誤差 4.69°。
- 測試（`test_c10neg`，47,271 對）：

| 姿態來源 | 平移中位數 | yaw 中位數 | 準確率 (0.5, 10°) |
|---|---:|---:|---:|
| 姿態解碼器 | 0.085 | 1.23° | 0.888 |
| RANSAC @0.3 | 0.084 | 1.09° | 0.899 |
| 幾何分支（Procrustes） | 0.082 | 1.08° | 0.890 |

```bash
python scripts/evaluate_split.py --ckpt ckpt/server_c10neg/epoch_015.pt --split test \
    --pair-index training/pairs/test_c10neg.csv --data-root $ZIND_ROOT --weights ema --batch-size 16 --dump
python scripts/tune_ransac.py --ckpt <checkpoint> --pair-index training/pairs/val_c10neg.csv    # 換模型後重調 RANSAC 門檻
```

`run_scene_graph.py`、`tune_ransac.py`、`plot_curves.py` 的預設路徑指向原開發機的訓練資料夾；手動執行時請明確給 `--ckpt models\covispose\epoch_015.pt`。

### 13.6 把新權重換進流程

1. 把新的 checkpoint（`train.py` 存的、含 `ema` 的完整檔案）放到 `models\covispose\epoch_015.pt`，或修改 `pipeline\common.py` 的 `CHECKPOINT`。
2. 更新 `pipeline\envcheck.py` 的 `CKPT_SHA`。
3. `sample\reference\` 是舊權重產生的，換權重後 `Verify.ps1` 的 CoVisPose 比對不會通過，這是預期結果。用新權重跑一次範例，確認合理後再更新參考結果。
4. §11.2、§11.3 的門檻是以 epoch 15 訂定的，換權重後要重新確認。

### 13.7 重現性

AMP 開啟時，cuDNN 卷積反向傳播使用 atomics，兩次相同的反向傳播會有約 1e-3 的差異，所以訓練無法逐位元重現；續跑本身是精確的（`training\ASSUMPTIONS.md` A13）。`training\reference_run\` 的原始 log 可用來比對訓練曲線。

---

## 14. 如何訓練或微調 HorizonNet

本專案直接用官方 MP3D 權重，沒有重新訓練 HorizonNet。要訓練或微調時，使用 `third_party\HorizonNet\` 的官方程式。

- **環境**：HorizonNet 環境再加 `pip install tensorboardX`。官方測試版本是 PyTorch 1.8.1／Python 3.7.6，本包的 torch 1.12.0／Python 3.9 可以推論。建議在 Linux 上訓練（`train.py` 沒有 `inference.py` 那段處理中文暫存路徑的程式）。
- **資料格式**（`README_PREPARE_DATASET.md`）：`<dataset>/img/*.png`（1024×512，已做消失點對齊）與 `<dataset>/label_cor/*.txt`（每行 `x y`，依序為各角點的天花板、地板座標）。ZInD 用 `misc/zind_prepare_dataset.py` 轉換；Structured3D 見 `README_ST3D.md`。
- **訓練**（在 `2_pose_floorplan\third_party\HorizonNet` 內執行）：

```bash
python train.py --id my_model --train_root_dir data/my/train --valid_root_dir data/my/valid --epochs 50
python train.py --id my_ft --pth ../../models/horizonnet/resnet50_rnn__mp3d.pth --train_root_dir ... --valid_root_dir ... --epochs 50
```

主要選項：`--batch_size_train`（8）、`--lr`（1e-4）、`--epochs`（300）、`--backbone`（resnet50）、`--freeze_earlier_blocks`、`--no_cuda`、`--multi_gpu`。評估用 `eval_general.py`、`eval_cuboid.py`。

- **換進流程**：放到 `models\horizonnet\`，修改 `pipeline\common.py` 的 `HN_CHECKPOINT` 與 `pipeline\envcheck.py` 的 `HN_SHA`，並更新 Verify 的參考結果。

---

## 15. GeoVerify 與 RoMa v2

- 程式：`1_room_finder\third_party\geoverify_main`，GitHub main 匯出，commit `b5897093ce2f7d98f3540b93c5c0f93b87e83a5e`（`SOURCE_COMMIT.txt`）。只用 direct 模式的 `verify_pair` 與預設參數。
- RoMa v2 是預訓練模型，本專案不訓練；權重與 DINOv3 程式存放在 `1_room_finder\models\torch_home`。
- 環境由 `environments\setup_env.ps1` 建立（說明見 `environments\README.md`）。

---

## 16. 測試與驗證紀錄

### 16.1 測試

| 測試 | 指令（主環境） | 條件 |
|---|---|---|
| finder 核心 | 在 `1_room_finder` 內：`python -m pytest tests\test_core.py` | 不需要 GPU 與視窗 |
| CoVisPose 單元測試 | 在 `2_pose_floorplan` 內：`python -m pytest tests\covispose` | 部分測試需要 `ZIND_ROOT`、ZInD 官方程式（`transformations`、`ZIND_CODE`）或下載 ImageNet 權重的網路；不符合時自動略過（`tests\covispose\_requires.py`） |
| 環境 | 在 `2_pose_floorplan` 內：`python -m pipeline.envcheck` | —（沒有 CUDA 時只提示） |
| 位姿圖（步驟 4） | 在 `2_pose_floorplan` 內：`python -m pytest tests\pipeline` | 不需資料；合成群組 |
| 端到端 | `2_pose_floorplan\Verify.ps1` | GPU（或 `-Cpu` 路徑） |
| 步驟 4 對真值 | `2_pose_floorplan\evaluation\zind_test_posegraph\Run_ZInD_PoseGraph_Eval.ps1` | ZInD 在 `C:\zind_data`、GPU（§11.3） |

### 16.2 驗證紀錄

| 日期 | 內容 | 結果 |
|---|---|---|
| 2026-11-05 | 打包驗證（雲端 CPU）：finder 測試；CoVisPose 測試；範例步驟 1、2、4、5、6 在 CPU 上執行（HorizonNet 以參考結果代入）；PowerShell 語法；權重雜湊 | finder 8/8；VERIFY PASSED，12 對差異 0.000°／0.0000；腳本全部可解析，雜湊相符 |
| 2026-11-15 | 原開發機（RTX 3060、環境已存在）執行 `Setup.ps1`＋`Verify.ps1` | VERIFY PASSED |
| 2026-11-15 | Linux、無 GPU、無 ZInD、只有 EMA 匯出檔：範例在 CPU 上執行；`tests\covispose` | VERIFY PASSED（差異 0）；30 通過、11 略過、0 失敗 |
| 2026-11-15 | 特徵快取，CPU | 與未快取逐位元相同；VERIFY PASSED |
| 加入特徵快取後 | 原開發機以 GPU、啟用特徵快取執行 `Verify.ps1` | VERIFY PASSED |
| 2026-11-19 | 以 `pipeline.poses`（生成樹路徑，未受位姿圖修改影響）重算 104 組 | 與 2026-11-17 的輸出相同（最大差 4e-16，浮點誤差） |
| 2026-11-19 | ZInD test 上以真值比較位姿圖與生成樹（219 間房、2,536 對，RTX 3060） | 準確率 ≤ 0.3、5°：68.7% → 76.2%；詳見 §11.3 |
| 2026-11-19 | `Verify.ps1` 參考結果：加入位姿圖的位姿與平面圖（`groups`），保留生成樹的參考（`groups_spanning_forest`）；以 2026-11-15 GPU run 的 CoVisPose 與 HorizonNet 結果，在 Linux CPU 上確認 | 位姿圖的範例 run 與 2026-11-15 生成樹 run 都 VERIFY PASSED |
| 2026-11-19 | `tests\pipeline\test_pose_graph.py` | 4/4 通過 |
| 2026-11-17 | 套件的 `prepare` 讀 `room_groups_dollhouse104.json` | 與原始實驗的輸入完全相同：301 個配對編號、278 張影像的 sha256、標籤、104 組成員 |
| 2026-11-17 | 以原始 CoVisPose 與 HorizonNet 結果重算位姿與平面圖 | 104/104 組相同（相機位置差 0、面積差 0%、標記相同） |
| 2026-11-17 | 104 組從頭以 GPU 重跑（RTX 3060，啟用特徵快取） | CoVisPose 301/301 對差異 0；位姿與平面圖 104/104 組在 `Verify.ps1` 容許範圍內；HorizonNet 277/278 張相同。例外（`08dded67…`）的角點後處理剛好落在臨界：10 個角點變 8 個，該組平面圖面積 −0.5%。判定為浮點臨界情況，不是可攜性問題 |

104 組的檢查程式與參考 run 不在套件內：`C:\covispose\scripts\check_rcf_dollhouse104.py`、`C:\covispose\outputs\rcf_dollhouse104_check\`。

---

## 17. 授權

| 項目 | 來源 | 限制 |
|---|---|---|
| `2_pose_floorplan\models\covispose\*.pt`、`2_pose_floorplan\training\pairs\` | 以 ZInD（Zillow Indoor Dataset）訓練、產生 | **只限學術、非商業研究**（ZInD 授權） |
| `2_pose_floorplan\models\horizonnet\resnet50_rnn__mp3d.pth` | HorizonNet 官方，Matterport3D 布局標註 | 依 Matterport3D 與 HorizonNet 授權 |
| `1_room_finder\third_party\sam3-pano-seg\` 的 SAM3 程式與權重 | Meta | SAM License：可再散布，必須附上 `third_party\sam3\LICENSE`；用 SAM3 的結果發表研究時，需要註明使用了 SAM Materials |
| Dollhouse 資料集（`1_room_finder\data\dollhouse`） | — | 依資料提供者的規定 |

交付對象必須知道 CoVisPose 權重只能用於學術、非商業用途；商業用途需要另外取得授權，或改用其他資料重新訓練（§13）。公開的 GitHub 倉庫與 Release 同樣受此限制：CoVisPose 權重與 `training\pairs` 衍生自 ZInD，須遵守 ZInD Terms of Use（https://bridgedataoutput.com/zillowterms）；ZInD 資料本身（影像、標註、`gt.json`）不再散布。

---

## 18. 已知條件、限制與尚未解決的事項

**尚未解決的事項（腳本無法自動處理）**

| 事項 | 說明 | 處理方式 |
|---|---|---|
| 資料夾很大 | Dollhouse 全景約 15 GB；不含環境時整個資料夾約 26 GB | 不需要時刪掉 `1_room_finder\data\dollhouse` 與 `1_room_finder\scans\dollhouse` |
| 只有在另一台電腦才會走到的路徑 | 「環境無法啟動時自動重建」「以 uv 下載 Python 3.9」「沒有 `nvidia-smi` 時略過 GPU 檢查」尚未實際執行過 | 在目標電腦以 `Setup.ps1`＋`Verify.ps1` 確認 |
| 2026-11-17 的 GPU 檢查修改 | `Assert-GpuFree` 改成只回報 Python 程式；已檢查內容，尚待在 Windows 上實際執行 | `Verify.ps1` 應該不會再出現 `(y/N)` 詢問 |
| 步驟 A 在目前的資料夾結構下沒跑過 | 2026-11-05 拆成兩個子資料夾後，匯入、掃描、門洞偵測尚未實際執行 | 先用 `Import_Dataset.ps1 -Limit 5` 試跑 |
| 部分手動腳本含原開發機路徑 | `run_scene_graph.py`、`tune_ransac.py`、`plot_curves.py` | 明確給 `--ckpt models\covispose\epoch_015.pt` |
| ZInD 不在包內 | 訓練與 11 個測試需要；ZInD 不可再散布 | 見 §13.2 |
| 位姿圖版本的 `Verify.ps1` 尚未在 Windows 上執行 | 新的參考結果是用 Linux CPU 路徑確認的（§16.2） | 在目標電腦上執行一次 `Verify.ps1` |

修改前的檔案備份在原開發機的 `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\`（紀錄在 `C:\HorizonNet\_待刪除\搬移紀錄.tsv`）。

**環境**

1. 路徑必須是純英文。
2. CoVisPose 預設用 CUDA，沒有時改用 CPU（結果相同）。GeoVerify 掃描（步驟 A）與 SAM3 需要 GPU；沒有合適的 GPU 時，只能用既有的群組檔執行步驟 1–6。
3. HorizonNet 環境是 torch 1.12＋CUDA 11.3，支援到 RTX 40。RTX 50 請加 `-HnCpu`（較慢）。
4. 主環境需要支援 CUDA 13.0 的驅動程式。
5. 同一張 GPU 一次只跑一個程式。`Run_Pipeline.ps1` 會檢查；手動執行 `tools\recompute_inliers.py` 時請自行確認。
6. `Setup.ps1` 需要網路；審查程式需要含 Tkinter 的 Python 3。
7. 複製或搬動資料夾後，先執行一次 `Setup.ps1`（重建環境、修正 SAM3 路徑）。刪掉 `1_room_finder\third_party\sam3-pano-seg` 時自動改用一般楔形。

**資料**

8. 全景必須剛好是 2:1 等距長方投影（§6.1）；同一棟的全景會兩兩配對。
9. `scans\dollhouse\manifest.json` 先找 `data\dollhouse`，找不到時才讀 `root_abs`（`C:/datasets`）。
10. 匯出的群組都標成 `room`，不區分開放空間，也沒有樓層。

**方法**

11. **穿過門口的相鄰房間解不出姿態**：CoVisPose 訓練時沒有「跨門可見」的正例（所有正例都在同一個 `complete_room` 內，A8）；在 ZInD test 的門口相連配對中，834 組裡有 814 組判定為零共視。所以流程以同房群組為單位，不拼接整層。
12. 同房分類器主要的剩餘錯誤來自開放式空間（客廳、餐廳、廚房相連），這種情況沒有門洞，去門楔形也幫不上。
13. 你的資料沒有姿態真值，輸出只代表內部一致性。步驟 4（生成樹與位姿圖）只在 ZInD test 上以真值量測過（§11.3）；位姿圖平均較好，但個別房間可能變差，多半發生在迴圈本身互相矛盾時（這些相機仍有 `loop_not_closing` 標記）。
14. 平面圖用外框聯集，會偏大，開口處可能穿進隔壁房間。
15. 楔形門檻的 P/R 表來自 Dollhouse，換到風格差很多的資料時只能當參考。
16. ZInD 是空屋，而且是 CoVisPose 的訓練來源，ZInD 上的數字偏樂觀；有家具的資料以 Dollhouse 結果為準。
17. HorizonNet 形狀 IoU 在 Dollhouse 上約 0.67，開放空間約 0.5；以 `layout_gt` 為基準的數字在真值修正前只能當參考。

**程式**

18. `scripts\manual_room_groups_v2.py` 只保留 `run`。它與 `covispose\` 的程式雜湊會寫進 `run_config.json`，程式改過後，之前 prepare 的 run 無法續跑（§0 第 14 點）。
19. 訓練與評估程式的 ZInD 預設路徑（`C:\zind_data` 等）只是預設值，請用 `ZIND_ROOT` 或參數指定。

---

## 19. 修改紀錄

- **整合**：本資料夾原本只有找同房掃描與審查程式。從 2026-10-26 的 room_pose_pipeline 打包版加入 CoVisPose（程式與 epoch 15 權重）、HorizonNet（官方程式與 MP3D 權重）、步驟 1–6、範例與驗證，另加 CoVisPose 訓練程式、設定、配對清單與環境規格。本資料夾改為根目錄，所有腳本路徑都已調整。
- **2026-11-05**：拆成 `1_room_finder\` 與 `2_pose_floorplan\`；根目錄只留 `Setup.ps1`、`tools\_env.ps1` 與文件。`Run_Pipeline.ps1` 改以執行時所在的資料夾解析 `-Groups`、`-ScanDir`、`-Index`。與最新配置無關的設定（備份檔、benchmark 匯入工具、包外的後備環境、舊的 SAM3 位置、未使用的設定檔）移到 `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\`。SAM3 改為內含的精簡複本。HorizonNet 暫存資料夾預設值與訓練配對清單路徑（`training\pairs`）也一併調整。
- **2026-11-09**：改以英文為主、保留中文（審查程式、viewer、文件 `*_ZH.md`）。合併啟動程序（`Start_Scan.ps1` → `Import_Dataset.ps1 -NoCopy`；各安裝腳本 → `Setup.ps1`；移除 `Start_Recompute_Inliers.ps1`）。`manual_room_groups_v2.py` 只保留 `run`。第三方程式沒有修改。
- **2026-11-15（可攜性）**：`Get-MainPy`／`Get-HnPy`／`Get-ReviewPy` 改成實際試跑環境；`Setup.ps1` 遇到無法啟動的環境會自動重建；`Start_Review.bat` 也會檢查環境。RoMa v2 權重與 DINOv3 程式移到 `1_room_finder\models\torch_home`。沒有 Python 3.9 時以 uv 下載；沒有 `nvidia-smi` 時略過 GPU 檢查。Dollhouse 全景圖複製回 `data\dollhouse`。
- **2026-11-15（CPU 後備、EMA 匯出檔、測試）**：加入 `--device auto|cuda|cpu`、`RCF_DEVICE`、`Run_Pipeline.ps1 -Cpu`；HorizonNet 自動加 `--no_cuda`；`envcheck` 沒有 CUDA 時只提示。可以只放 EMA 匯出檔。需要 ZInD 或網路的測試改為略過。
- **2026-11-15（特徵快取）**：步驟 2 每張全景只編碼一次（§11.2）。CPU 上範例 108 秒 → 55 秒；輸出逐位元相同。
- **2026-11-17**：整理 Dollhouse 分組檔（新增 `room_groups_dollhouse104.json` 與 `data_index.json`；6 組測試匯出改名為 `room_groups_test_20261026.json`）。記錄輸出格式並對照程式確認；確認俯視圖畫法。`Assert-GpuFree` 改成只回報 Python 程式（Windows 上 `nvidia-smi` 列出的桌面程式不再觸發詢問）。README、NOTICE、SPEC、OUTPUT_FORMAT 合併為本文件。
- **2026-11-19（位姿圖）**：步驟 4 改為對每個有迴圈的分量做位姿圖最佳化（`pipeline\pose_graph.py`，§11.3），不再只用生成樹；`Run_Pipeline.ps1 -NoPoseGraph` 保留舊做法。`pipeline\poses.py` 可傳入求解方法、規則文字、標記統計與方法名稱（預設輸出不變），並寫出 `method`；viewer 從 poses 資料夾讀取方法與標記統計。`Verify.ps1` 的參考結果包含兩種方法（`pipeline\verify.py` 依 `method` 選擇；`--make-reference` 保留另一種方法與配對位姿，`--pairs` 才重寫配對位姿）。新增 `scripts\make_zind_room_groups.py`、`scripts\eval_pose_graph_zind.py`、`evaluation\zind_test_posegraph\`、`tests\pipeline\test_pose_graph.py`。模型與步驟 2 沒有改變；既有的 run 可以續跑。
- **2026-11-19（GitHub）**：公開倉庫用的 `.gitignore`／`.gitattributes`（不含環境、權重、SAM3、範例以外的全景、ZInD 真值與預測；§0「從 GitHub 取得」）；`models\covispose\`、`models\horizonnet\` 加上下載說明；`envcheck` 會提示缺少的權重去哪裡下載；`Run_ZInD_PoseGraph_Eval.ps1` 在缺少 `gt.json` 時從 ZInD 重新產生（`-ZindData`、`-Partition`；已確認與 2026-11-19 使用的檔案逐位元相同）。

每次修改前的備份都在原開發機的 `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\`。

---

## 20. 其他文件

| 文件 | 內容 |
|---|---|
| `1_room_finder\CODE_OVERVIEW_ZH.md`、`2_pose_floorplan\CODE_OVERVIEW_ZH.md` | 資料夾內每支程式在流程中的角色、主要函式、使用的函式庫、測試 |
| `1_room_finder\docs\REVIEW_APP_ZH.md` | 審查程式操作、門檻選擇、常見問題 |
| `1_room_finder\docs\benchmark\RESULTS_ZH.md`、`RESULTS_open_wedge_ZH.md`、`EVALUATION_STANDARD_ZH.md` | 同房分類器的比較、去門楔形、評估標準 |
| `2_pose_floorplan\evaluation\horizonnet_404\REPORT_ZH.md`、`PLAN_ZH.md` | HorizonNet 外框 404 張評估與事前評分規格 |
| `2_pose_floorplan\runs\zind_test_posegraph\pose_graph_zind_eval_*.json` | 步驟 4 的 ZInD test 評估：摘要，以及兩種方法下每組相機的誤差（§11.3） |
| `2_pose_floorplan\training\SERVER_OPTIMIZATION.md` | 伺服器訓練設定的每項變更與理由 |
| `2_pose_floorplan\training\PAPER_COMPARISON.md` | 與 CoVisPose 論文的比較、選 epoch 15 的依據 |
| `2_pose_floorplan\training\ASSUMPTIONS.md` | 所有設計假設（A1–A18） |
| `2_pose_floorplan\training\PHASE1_NOTES.md`、`PHASE5_NOTES.md` | ZInD 資料管線、伺服器訓練結果 |
| `2_pose_floorplan\third_party\HorizonNet\README*.md` | HorizonNet 官方說明 |
| `1_room_finder\third_party\geoverify_main\README.md`、`API.md` | GeoVerify 說明 |

以上文件（第三方除外）都有英文版（去掉 `_ZH` 的同名檔案）。訓練筆記（`SERVER_OPTIMIZATION.md`、`PAPER_COMPARISON.md`、`ASSUMPTIONS.md`、`PHASE*_NOTES.md`）只有英文。
