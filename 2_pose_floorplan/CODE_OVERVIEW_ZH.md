# 2_pose_floorplan 程式說明

本文件概述 `2_pose_floorplan\` 的每支程式：它在流程中的位置、主要函式與功能。使用方式、輸入與輸出格式見根目錄的 [README_ZH.md](../README_ZH.md)（§5、§6.3、§8、§9）。英文版：[CODE_OVERVIEW.md](CODE_OVERVIEW.md)。

這個資料夾讀入 `1_room_finder` 匯出的房間群組，依序產生 CoVisPose 相對姿態、HorizonNet 房間外框、相機位姿、合併平面圖與互動網頁（步驟 1–6），另外包含 CoVisPose 的訓練與評估程式。

> 程式內的 docstring 沿用舊版編號（prepare = Step 1、CoVisPose = Step 2、HorizonNet = Step 2b、poses = Step 3、plans = Step 4、viewer = Step 5）。本文件與 README 使用 1–6 的編號。

---

## 1. 呼叫關係

```
Run_Pipeline.ps1（主環境；HorizonNet 用自己的環境）
  1 python -m pipeline.prepare      群組檔＋掃描 manifest → runs\<名稱>\ 凍結輸入
  ─ Assert-GpuFree（tools\_env.ps1）
  2 python scripts\manual_room_groups_v2.py run   CoVisPose → predictions\
       └─ covispose.evaluate（load_model_from_ckpt、forward_tta）、covispose.model、covispose.geo_pose
  3 python -m pipeline.horizonnet   → 子程序：HorizonNet 環境的 preprocess.py、inference.py → horizonnet\
  4 python -m pipeline.pose_graph   → poses_<時間>\     （預設；先用 pipeline.poses 的生成樹，再做位姿圖最佳化）
    或 python -m pipeline.poses     → poses_<時間>\     （-NoPoseGraph：只用生成樹；使用 covispose.ransac.build_correspondences）
  5 python -m pipeline.plans        → plans_<時間>\     （使用 covispose.geometry、shapely）
  6 python -m pipeline.viewer       → viewer\viewer.html、images\

Verify.ps1 = Run_Pipeline.ps1 -Groups sample\room_groups.json -ScanDir sample\scan -Index sample\data_index_sample.json
             -Name _verify_<時間>，再執行 python -m pipeline.verify
```

---

## 2. 啟動腳本

| 檔案 | 功能 |
|---|---|
| `Run_Pipeline.ps1` | 參數 `-Groups`、`-Name`、`-ScanDir`、`-Index`、`-Title`、`-Cpu`、`-HnCpu`、`-NoPoseGraph`（步驟 4 改用 `pipeline.poses`，不用 `pipeline.pose_graph`）。先以執行時所在的資料夾解析路徑，再切換到 `2_pose_floorplan`。`runs\<Name>\manifest.json` 不存在時執行 prepare，存在時直接續跑。之後依序執行步驟 2–6。CoVisPose 用 `inference.lock` 防止重複執行，畫面輸出以 `Tee-Object` 寫入 `covispose_gpu_<時間>.log`。任何一步失敗就停下，修正後重跑同一指令即可接續 |
| `Verify.ps1` | 以 `sample\` 的 9 張全景、3 個群組（`sample\room_groups.json`、`sample\scan`、`sample\data_index_sample.json`）呼叫 `Run_Pipeline.ps1`，再用 `pipeline.verify` 和 `sample\reference\` 比對。流程失敗時顯示 `Pipeline failed on the sample.` |

---

## 3. `pipeline\`：流程步驟 1–6

### 3.1 `common.py`：共用工具

| 名稱 | 功能 |
|---|---|
| `CHECKPOINT`、`CHECKPOINT_FULL`、`CHECKPOINT_EMA`、`HN_CHECKPOINT` | 權重位置；只有 EMA 匯出檔時自動改用它 |
| `local(p)`、`rel_or_abs(p)` | 讀取時把相對路徑解析到套件根目錄；寫入時，套件內的路徑存成相對路徑，搬動後仍可用 |
| `read`、`write`、`csvwrite` | JSON 讀寫（UTF-8）；CSV 寫成 UTF-8 含 BOM |
| `digest(path)`、`stamp()` | sha256、時間戳（用於資料夾名稱） |
| `rot(yaw)`、`wrap(a)`、`err(p, q)` | 2D 旋轉矩陣、角度包到 [−π, π)、兩個相對位姿的差（度、平移） |
| `latest(run, prefix)` | 找最新的 `poses_*`／`plans_*` 資料夾 |
| `mpl_fonts(matplotlib)` | 設定 matplotlib 的中文字型 |

### 3.2 `prepare.py`：步驟 1，凍結輸入（CPU）

| 函式 | 功能 |
|---|---|
| `prepare(groups, output, scan_dir, index)` | ① 讀群組檔與掃描 `manifest.json`（先找 `root`，再找 `root_abs`）；② 檢查：狀態必須是 `human_verified`、每組至少 2 張、成員不可重複、`room_id` 不可重複、群組內不可有 `different_room` 或 `conflicts`，違反時印出 `Input rejected` 並停止；③ 逐張以 Pillow 完整解碼，確認剛好 2:1，再計算 sha256；④ 找出內容相同的影像，產生群組內所有配對與 `pair_id`；⑤ 寫入 `classification_source.json`、`manifest.json`、`run_config.json`（包含權重雜湊、`manual_room_groups_v2.sources()` 的程式雜湊、`RULES`）、`preflight.json` |
| `join(*parts)` | Windows 與 Linux 都適用的路徑組合 |

### 3.3 `scripts\manual_room_groups_v2.py`：步驟 2，CoVisPose 推論（GPU／CPU）

這支程式在 `scripts\` 底下，但它是流程的一部分，也是「凍結來源」：它和 `covispose\` 的 8 個檔案（`FROZEN_SOURCES`）的雜湊都記錄在 `run_config.json`。

| 函式 | 功能 |
|---|---|
| `validate(run)` | 執行前檢查：manifest、凍結程式、權重、影像雜湊都要和 prepare 時相同，否則停止 |
| `completed(run, pair)` | 這一對是否已有完整且雜湊相符的結果；用於續跑 |
| `pick_device(choice)` | `auto`／`cuda`／`cpu`（也讀 `RCF_DEVICE`）；auto 在 CUDA 可用時用 GPU，否則用 CPU |
| `encode_views(model, img)` | 一張全景只跑一次影像編碼器，得到原圖與鏡射兩種欄特徵 |
| `cache_usable(...)` | 確認 TTA 的轉半圈剛好是編碼器步距（32 px）的整數倍，才能直接位移特徵 |
| `forward_cols(model, fa, fb)` | 從已編碼的欄特徵執行 CoVisPose 後半段（pair Transformer、各輸出頭、姿態解碼器、幾何分支） |
| `forward_tta_cached(...)` | 與 `covispose.evaluate.forward_tta` 相同的 4 視角 TTA，但重複使用特徵 |
| `same_prediction(x, y)` | 比較快取與不快取的結果（yaw ≤ 0.05°、平移 ≤ 0.002） |
| `infer(args)` | 主迴圈：載入 EMA 權重（`load_model_from_ckpt`）、FP32、關閉 TF32；對每一對做 TTA，第一對同時用兩種方法計算並比對，不一致時改回不快取；寫入 `predictions\<pair_id>.json`（解碼器與 `geometric_pose` 幾何分支的位姿）、`.npz`（共視、對應、牆地交界），以及 `inference_status.json`、`run_env_<時間>.json` |
| `sources()`、`local`、`read`、`write`、`digest`、`stamp` | 凍結程式雜湊與輸入輸出工具（`MRG_PATH_MAP` 可把 Windows 路徑映射到其他作業系統） |
| `main()` | 子指令 `run --run <資料夾> [--device] [--no-feature-cache]` |

### 3.4 `horizonnet.py`：步驟 3，房間外框（GPU／CPU）

HorizonNet 在自己的 Python 3.9 環境中以子程序執行；這支程式負責準備輸入與轉換座標。

| 函式 | 功能 |
|---|---|
| `horizonnet(run, hn_python, no_cuda)` | 把還沒有結果的全景複製到工作資料夾（檔名 `<id>.jpg`），並把 `TEMP`／`TMP`／`HN_ASCII_TMP` 設成純英文路徑（使用者名稱是中文時 scipy 等套件會失敗）；執行官方 `preprocess.py`（LSD 消失點、3D 旋轉對齊）與 `inference.py --pth <MP3D 權重>`；結果經 `unalign` 轉回原圖座標，寫入 `horizonnet\<id>.json`。HorizonNet 環境沒有 CUDA 時自動加 `--no_cuda`；可續跑 |
| `vp_rotation(vp)` | 由消失點重建 `preprocess.py` 使用的旋轉矩陣 R |
| `unalign(...)` | 對齊後影像上的角點 → 射線 → 乘上 Rᵀ → 原圖的 uv（使用 `equirect.floor_uv_to_xy`、`ray_to_uv`） |
| `rotation_diagnostics(R)` | 記錄傾斜角、yaw、行列式，寫入 provenance |
| `_ours_to_lsd`、`_lsd_to_ours` | 本流程與 HorizonNet／LSD 座標軸之間的轉換 |
| `run_cmd(cmd, cwd)` | 執行子程序並記錄指令、耗時、輸出，寫入 `provenance_*.json` |

### 3.5 `equirect.py`：等距長方投影幾何

`uv_to_angles`、`angles_to_uv`、`uv_to_ray`、`ray_to_uv`（像素 ↔ 方位角／仰角 ↔ 3D 射線）；`floor_uv_to_xy`、`ceiling_uv_to_xy`、`xy_to_floor_uv`、`xy_to_ceiling_uv`（像素 ↔ 地板／天花板平面上的點）；`ceiling_ratio_from_pair`（從同一角點欄的天花板與地板像素算出天花板高度比）；`circ_du`、`split_at_seam`（處理全景左右接縫）。

### 3.6 `poses.py`：步驟 4，相機位姿（CPU）

| 函式 | 功能 |
|---|---|
| `load_group_edges(run, m, g)` | 讀一個群組的所有 `predictions`（先核對雜湊），以 `covispose.ransac.build_correspondences` 把逐欄輸出轉成兩組對應的 2D 點 P、Q，並收集每張全景的牆地交界 |
| `classify(e)` | 以解碼器位姿計算 P、Q 的殘差，數出內點（殘差 ≤ 0.30）與內點比例 f；再檢查解碼器與幾何分支是否一致（≤ 5°、≤ 0.25）。結果為 `full`（≥ 20 點且 f ≥ 0.40）、`dir`（f 在 0.30–0.40）或不採用 |
| `forest(nodes, edges, acc)` | 最大權重生成森林（完整連線權重 1 + f，方向連線權重 f），沿樹推出每台相機的 `(yaw, t)`，不在樹上的採用連線記為迴圈 |
| `solve(nodes, edges)` | 呼叫 `classify`、`forest`；選連線最多的相機當原點；計算 W（弱連線）、L（迴圈不閉合 > 5° 或 > 0.30）、D（方向連線）等標記與迴圈殘差 |
| `inv`、`rel` | 位姿反轉、兩台相機的相對位姿 |
| `reason(c, e, dy, dt)` | 每一對的狀態文字（`accepted`、`low_inlier_fraction`、`decoder_geometric_disagree`…） |
| `poses(run, solver, rule, flag_stats, method)` | 確認步驟 2 已全部完成，逐群組計算後寫入 `groups\*.json`、`cameras.csv`、`pairs.csv`、`groups.csv`、`summary.json` |
| `plot(...)` | matplotlib 畫相機位置與連線（俯視圖，(−x, y)） |
| `FLAG_STATS` | 生成樹位姿下各標記在 ZInD test 的錯誤率（`pose_flags_v1`），寫入輸出，也給 viewer 顯示 |

`poses()` 的參數預設為 `solve`、生成樹的規則文字、`FLAG_STATS`、`spanning_forest`；`pose_graph` 會傳入自己的版本。

### 3.6b `pose_graph.py`：步驟 4，位姿圖最佳化（CPU，預設）

採用規則與生成森林都和 `poses.py` 相同，生成森林作為初始值。參數與 ZInD test 結果見 README §11.3。

| 名稱 | 功能 |
|---|---|
| `optimise(nodes, edges, cls, p1)` | 對採用連線多於生成樹所需的每個分量，以所有採用的連線做穩健最小平方（`scipy.optimize.least_squares`、Huber、`F_SCALE` = 3）。未知數是森林根以外每台相機的 (yaw, x, y)。每條連線的殘差：yaw ／ `SIGMA_YAW`（1°）、平移 ／ `SIGMA_T`（0.06；方向連線用 `SIGMA_T_DIR` 0.6），再乘 √權重。沒有迴圈的分量原樣返回；最佳化失敗時保留生成樹的位姿。回傳位姿與每個分量的資訊（最佳化前後的代價、計算次數） |
| `solve_graph(nodes, edges)` | `poses.solve`（採用、生成樹、原點、標記）＋ `optimise`；`loops` 改為不在樹上的連線與最佳化後位姿的差。可直接取代 `poses.solve` |
| `run_pose_graph(run)` | `poses.poses(run, solver=solve_graph, rule=RULE, flag_stats=FLAG_STATS_GRAPH, method='pose_graph')`，再寫 `pose_graph_report.json`，並在 `summary.json` 加上 `components_optimised` |
| `evaluate(run)`（`--eval`） | 不需要真值：迴圈上的每條完整連線輪流拿掉，用其他連線分別以生成樹與位姿圖預測它，再和它自己的解碼器與幾何分支位姿比較；寫入 `pose_graph_eval_<時間>.json` |
| `FLAG_STATS_GRAPH` | 位姿圖位姿下各標記在 ZInD test 的錯誤率（`pose_graph_zind_test_v1`） |

### 3.7 `plans.py`：步驟 5，平面圖（CPU）

| 函式 | 功能 |
|---|---|
| `outlines(bons, hn_json)` | 每張全景的三種外框：CV（各配對牆地交界的逐欄中位數，`bfloor_to_dist`）、HN（`hn_footprint`）、MX（逐欄取兩者較遠者） |
| `hn_footprint(path)` | HorizonNet 的 `uv` → 相機座標中的地板多邊形；角點超過 10 個相機高度、或資料不完整時回傳 None |
| `coldist(poly)` | 多邊形 → 1024 欄的徑向距離（`covispose.geometry.polygon_to_column_distance`） |
| `simplify(poly)` | shapely 簡化（容差 0.05） |
| `placed(poly, cam)` | 用相機位姿把外框放到原點座標 |
| `rings(geom)` | shapely 幾何 → 外框頂點清單（丟棄孔洞） |
| `plans(run)` | 讀最新的 `poses_*`；≥ 2 張的分量取 HN 外框聯集（`unary_union`，HN 不可用時改用 MX），單張用 MX；同時算出只用 CV 的聯集；寫入 `groups\*.json`、`index.json`、`plots\*.png` |

### 3.8 `viewer.py`：步驟 6，互動網頁（CPU）

| 函式 | 功能 |
|---|---|
| `viewer(run, title, lang)` | 讀最新的 poses 與 plans（位姿方法與標記錯誤率取自 poses 資料夾的 `method` 與 `flag_stats`），把相機、外框、平面圖、標記與縮圖（JPEG quality 70）嵌入單一 `viewer.html`；全景縮成 4096×2048（quality 82），以 base64 存成 `images\<id>.jpg.js`，由頁面按需載入。介面有英文與中文 |
| `overlap_rings(polys)` | 計算各全景外框的重疊區域（shapely），在「重疊」模式中顯示 |

### 3.9 `verify.py`、`envcheck.py`

| 函式 | 功能 |
|---|---|
| `verify.verify(run)` | 與 `sample\reference\reference.json` 比對：CoVisPose 每對 yaw／平移、HorizonNet 角點數與位置（使用 `plans.hn_footprint`）、poses 的分量／生成樹／標記／相機位置、plans 面積；結果寫入 `verify_result.json`。位姿與平面圖依 run 的 `method` 和 `groups`（位姿圖）或 `groups_spanning_forest`（生成樹）比對 |
| `verify.make_reference(run, pairs)` | 維護用（`--make-reference [--pairs]`）：把 run 所用方法的位姿與平面圖寫入參考結果，保留另一種方法；CoVisPose 配對位姿只有加 `--pairs` 時才重寫（換權重後使用） |
| `envcheck.main()` | 檢查主環境套件能否匯入、CUDA 是否可用（沒有時只提示）、兩個權重的 sha256 |

---

## 4. `covispose\`：CoVisPose 模型（推論與訓練共用）

### 4.1 流程與訓練會用到的模組

| 模組 | 主要函式／類別 | 功能 |
|---|---|---|
| `model.py` | `CoVisPose`、`build_model`、`config_from_dict`、`CoVisPoseConfig`、`count_params` | 完整模型：`ColumnEncoder`（ResNet-50＋HorizonNet 式高度壓縮 `ConvCompressH`／`GlobalHeightConv`，左右以 `LRPad`／`wrap_lr_pad` 環狀補邊）→ 每張 1024 個欄特徵＋`azimuth_encoding` → `PairTransformer`（`PairEncoderLayer`，含 `drop_path`）融合兩張 → `ColumnHeads`（共視、對應方位、牆地交界）＋`AttnPoseDecoder`（4 個 query cross-attention 輸出 dx、dy、dyaw；舊版 `PoseDecoder` 保留）＋幾何分支 |
| `geo_pose.py` | `geometric_pose`、`weighted_procrustes_2d`、`bon_to_dist`、`sample_circular` | 可微分的幾何位姿：把逐欄輸出轉成兩組 2D 點，以加權 Procrustes（IRLS 2 次）求 yaw 與 t |
| `geometry.py` | `col_centers_theta`、`theta_to_col`、`dir_of_theta`、`theta_of`、`bfloor_to_dist`、`dist_to_bfloor`、`wrap_pi`、`relative_pose`、`transform_points`、`ray_cast_polygon`、`polygon_to_column_distance`、`PanoPose` | 全景欄位 ↔ 方位角、牆地交界 ↔ 距離、ZInD 位姿、射線與多邊形相交等基礎幾何（§9 的座標慣例就定義在這裡） |
| `ransac.py` | `build_correspondences`、`ransac_rigid`、`kabsch_2d`、`pose_from_predictions`、`sample_boundary_at_azimuth` | 逐欄輸出 → 對應點 → RANSAC 剛體配準；步驟 4 的 `classify` 使用 `build_correspondences` |
| `evaluate.py` | `load_model_from_ckpt`、`forward_tta`、`tta_views`、`invert_view`、`collect_predictions`、`CachedPredictions`、`head_metrics`、`pose_errors`、`summarize_pose`、`run_ransac_over_cache` | 依 checkpoint 內的設定重建模型並載入 EMA 或原始權重；4 視角 TTA 與還原；評估指標 |
| `checkpoint.py` | `save_checkpoint`、`load_checkpoint`、`find_resume`、`torch_load`、`prune_epoch_checkpoints` | 完整續訓（模型、EMA、最佳化器、排程器、AMP scaler、亂數狀態），原子寫入 |
| `ema.py` | `ModelEMA` | 權重指數移動平均（0.9998） |
| `losses.py` | `covispose_loss`、`covispose_metrics`、`LossWeights`、`circular_l1` | 共視、對應、牆地交界、解碼器位姿、幾何位姿的多任務損失 |
| `dataset.py` | `PairDataset`、`make_loader`、`augment_targets` | ZInD 配對資料集；水平旋轉、配對互換、鏡射、gamma 等增強 |
| `targets.py` | `compute_pair_targets`、`PairTargets` | 從 ZInD 標註算出逐欄真值（牆地交界、共視、對應方位） |
| `pairs.py` | `read_covis_scores`、`write_pair_index`、`read_pair_index`、`room_connections`、`PairRef` | 由 `covis_scores.csv` 建立與讀取配對清單 |
| `zind_parse.py` | `load_house`、`load_partition`、`PanoRecord`、`HouseRecord`、`house_scale_meters` | 解析 ZInD `zind_data.json` |
| `health.py` | `HealthMonitor`、`check_against_baselines`、`trivial_baselines`、`vram_guard` | 訓練監控：NaN、發散、沒有贏過簡單基準、顯示記憶體超標時提早停止 |
| `scene_graph.py` | `spanning_tree`、`global_poses`、`edge_residuals`、`boundary_to_polygon`、`to_world`、`align_rigid` | 舊版場景組裝（`run_scene_graph.py` 使用；它是凍結來源之一） |
| `viz.py` | `plot_pair_targets`、`plot_prediction_vs_gt` | 真值與預測的視覺化（`verify_targets.py` 使用） |

### 4.2 研究階段的實驗模組（流程與訓練都沒有使用）

`corners.py`（`CornerNet` 牆角偵測）、`corner_fit.py`（固定角點方向的多邊形擬合）、`floor_boundary.py`（`FloorBoundaryNet` 單張牆地交界）、`joint_layout.py`（`JointLayoutNet`，LGT-Net 式布局）、`layout_integration.py`（`LayoutAdapter`，把上述布局模型接到固定位姿）、`manhattan_fit.py`（曼哈頓房型擬合）、`wall_ba.py`（牆面擬合與多視角 bundle adjustment）。保留供參考；修改它們不會影響流程，也不會使舊的 run 無法續跑。

---

## 5. `scripts\`：訓練與評估

| 檔案 | 主要函式 | 功能 |
|---|---|---|
| `train.py` | `main`、`load_config`（支援 `_base_` 繼承）、`setup_distributed`、`build_scheduler`（warmup＋餘弦）、`ramped_weights`（幾何項逐步加入）、`validate`、`JsonlLogger` | 單卡或 torchrun 多卡訓練；bf16 AMP、EMA、自動續訓；每個 epoch 完整驗證，並寫入 `log.jsonl` |
| `build_pair_index.py` | `main`、`sample_cross_room` | 由 ZInD 建立 train／val／test 配對清單（正例、零共視負例、相鄰房負例） |
| `evaluate_split.py` | `main`、`bucket_report` | 在一個 split 上評估 checkpoint（逐欄指標、三種位姿來源、依共視分組） |
| `tune_ransac.py` | `main` | 以模型預測重調 RANSAC 內點門檻 |
| `benchmark_vram.py` | `bench`、`bench_config`、`data_throughput` | 量測顯示記憶體用量、每步時間、資料讀取速度 |
| `verify_targets.py` | `main` | 畫出幾組配對的真值疊圖，檢查資料管線 |
| `plot_curves.py` | `read_log`、`smooth`、`main` | 從 `log.jsonl` 畫訓練與驗證曲線 |
| `run_scene_graph.py` | `load_pano`、`predict_edges`、`edge_confidence`、`run_scene`、`plot_floorplan` | 舊版整棟場景組裝。流程只用到其中的 `load_pano`（讀圖並縮放到 512×1024）；手動執行時要給 `--ckpt` |
| `make_zind_room_groups.py` | `main` | ZInD split → 流程輸入：每個至少 `--min-panos` 張全景的完整房間一組；寫出 `room_groups.json`、`manifest.json`、`index.json`、`gt.json`（ZInD 位姿）。`evaluation\zind_test_posegraph\` 由它產生 |
| `eval_pose_graph_zind.py` | `evaluate`、`stats`、`camera_wrong`、`gt_rel` | 在這些群組的 run 上，以 ZInD 真值比較生成樹與位姿圖：每組相機的誤差與準確率、錯誤相機、各方法的標記錯誤率、以房間為單位的 bootstrap；寫入 `pose_graph_zind_eval_<時間>.json` |

---

## 6. 使用的外部程式與函式庫

| 來源 | 使用的部分 | 用途 |
|---|---|---|
| PyTorch（主環境 torch 2.10 cu130） | 模型、推論、訓練 | CoVisPose |
| torchvision | `resnet50(IMAGENET1K_V2)` | CoVisPose 編碼器骨幹 |
| HorizonNet 官方程式（`third_party\HorizonNet`，獨立的 Python 3.9 環境） | `preprocess.py`、`inference.py`（訓練：`train.py`） | 消失點對齊、房間外框 |
| numpy | — | 幾何、RANSAC、統計 |
| scipy | `optimize.least_squares` | `pipeline\pose_graph.py`（步驟 4）與 §4.2 的實驗模組 |
| shapely | `Polygon`、`unary_union`、`simplify`、`intersection` | 外框聯集、重疊、簡化 |
| matplotlib | — | `poses`、`plans` 的 PNG 圖 |
| Pillow | `Image.open`、縮放、JPEG 編碼 | 影像檢查、viewer 圖片 |
| PyYAML | — | 訓練設定檔 |

HorizonNet 官方程式沒有修改。

---

## 7. 測試

`tests\covispose\`（`python -m pytest tests\covispose`）：

| 檔案 | 內容 |
|---|---|
| `test_geometry.py` | 射線與多邊形相交、牆地交界與距離互換、方位角與俯角和 ZInD 一致、相對位姿與反轉、逐欄距離 |
| `test_model.py` | 方位編碼週期性、輸出形狀、編碼器的環狀等變性、損失遮罩、RANSAC 與幾何分支能還原真值、旋轉與鏡射增強的正確性、TTA 還原、參數數量、舊 checkpoint 仍可載入 |
| `test_checkpoint.py` | 續訓後狀態完全還原且接續結果相同、AdamW 步數留在 CPU、EMA 存取、`find_resume` |
| `test_negatives.py` | 共用牆兩側不算共視、門口相連房間的判定、配對清單讀寫、負例不計姿態損失、共視與 Zillow 分數一致 |
| `test_bn_stats.py`、`test_overfit.py` | 建模後仍保留 ImageNet 預訓練的 BatchNorm 統計與權重；5 對資料過擬合（確認模型學得動，需要 ZInD） |
| `_requires.py` | 缺 ZInD 資料、ZInD 官方程式或網路時自動略過相關測試 |

`tests\pipeline\test_pose_graph.py`（`python -m pytest tests\pipeline`，不需要資料）：沒有迴圈的分量維持不變；在有雜訊的合成房間上，位姿圖比生成樹更接近真值；一條錯誤但被採用的連線不會拖歪迴圈；生成樹、原點與標記和 `poses.solve` 相同。
