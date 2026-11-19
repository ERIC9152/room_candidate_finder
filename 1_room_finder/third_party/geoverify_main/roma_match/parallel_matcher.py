"""
[模組用途]
多 GPU 平行匹配。偵測到多張 GPU 時，每張卡開一個 worker（以 `CUDA_VISIBLE_DEVICES` 隔離），
由 `environments/batch_match.py` 呼叫。
"""
import multiprocessing as mp
import torch
import os
import time
from tqdm import tqdm


def _timing_append_row(csv_path, fieldnames, row):
    """選配計時鉤子用的附加式 CSV 寫入：UTF-8，檔案不存在或為空時才寫表頭。

    全程以例外包住。計時寫入失敗一律靜默略過，不得影響匹配本身。
    刻意只用標準函式庫：worker 是 spawn 出來的行程，不應相依本專案的 sys.path 設定。
    """
    try:
        import csv
        write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass


def _gpu_worker(gpu_id, task_queue, result_queue, stage, mode, model_type, conf_threshold, exp_name=None, use_cache=True):
    """
    階段式 GPU Worker：根據 stage 參數只載入必要的模型。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # 初始化指定階段所需的模型
    matcher = None
    depth_est = None
    roi_predictor = None
    segmentor = None
    model_dispatcher = None

    try:
        if stage == "match":
            from roma_match.matcher import RoMaMatcher
            matcher = RoMaMatcher(device=device, conf_threshold=conf_threshold, model_type=model_type, use_cache=use_cache)
        elif stage == "inference":
            from depth.depth_estimator import DepthAnythingEstimator
            from roi_prediction.inference.predictor import ROIPredictor
            from segmentation.segmentor import SemanticSegmentor
            depth_est = DepthAnythingEstimator(device=device, use_cache=use_cache)
            roi_predictor = ROIPredictor(device=device)
            segmentor = SemanticSegmentor(device=device, use_cache=use_cache)
        elif stage == "score":
            from roma_match.matcher import RoMaMatcher, resolve_path
            from depth.depth_estimator import DepthAnythingEstimator
            from roi_prediction.inference.predictor import ROIPredictor
            from segmentation.segmentor import SemanticSegmentor
            from scoring.scorer import (
                PointConfidenceScorer, AggregateConfidenceScorer,
                AggregateROIScorer, GridDepthVarianceScorer,
                AggregateWeightedDepthScorer, AggregateSemanticWeightedDepthScorer,
                ModelFoldDispatcher
            )
            # 所有 GPU 模型只初始化一次，避免每個 task 重複載入權重
            matcher = RoMaMatcher(device=device, conf_threshold=conf_threshold, model_type=model_type, workers=1, use_cache=use_cache)
            depth_est = DepthAnythingEstimator(device=device, use_cache=use_cache)
            roi_predictor = ROIPredictor(device=device)
            segmentor = SemanticSegmentor(device=device, use_cache=use_cache)
            p_conf = PointConfidenceScorer()
            agg_conf = AggregateConfidenceScorer()
            agg_roi = AggregateROIScorer()
            grid_scorer = GridDepthVarianceScorer()
            agg_depth = AggregateWeightedDepthScorer()
            agg_sem = AggregateSemanticWeightedDepthScorer()
            if exp_name:
                model_dispatcher = ModelFoldDispatcher(exp_name=exp_name)

        print(f"  [Worker GPU:{gpu_id}] Stage '{stage}' initialized.")
    except Exception as e:
        import traceback
        result_queue.put(("ERROR", f"Init Error: {e}\n{traceback.format_exc()}"))
        return

    # --- 選配的逐對計時：父行程在 spawn worker 前設了 BM2_TIMING_DIR 才會啟用。
    # 每個 worker 寫自己的 pid 後綴 CSV，避免跨行程寫入競爭；父行程在 run_stage 回傳後
    # 彙整成 per-house 的 roma_times.csv。全程以例外包住：計時失敗即自行停用，不影響匹配。---
    _timing_csv = None
    _timing_cache_mode = {"core": "romascan", "cubemap": "cubemap"}.get(mode, "direct")
    try:
        _t_dir = os.environ.get("BM2_TIMING_DIR")
        if stage == "match" and _t_dir and os.path.isdir(_t_dir):
            _run_id = os.environ.get("BM2_ROMA_RUN_ID", "run")
            _timing_csv = os.path.join(_t_dir, f"roma_worker_times_{_run_id}_{os.getpid()}.csv")
    except Exception:
        _timing_csv = None

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        try:
            if stage == "match":
                idx, img_a, img_b = task
                # Optional timing pre-check (guarded): flag whether this pair's matches were
                # already cached BEFORE calling the matcher, so downstream statistics can
                # exclude cache hits from compute-time sums.
                _t0 = None
                _cache_hit = False
                if _timing_csv is not None:
                    try:
                        _cp, _ = matcher._get_cache_info(img_a, img_b, _timing_cache_mode)
                        _cache_hit = bool(use_cache and os.path.exists(_cp))
                    except Exception:
                        _cache_hit = False
                    _t0 = time.perf_counter()
                if mode == "core":
                    matcher.match_core(img_a, img_b)
                elif mode == "cubemap":
                    matcher.match_cubemap(img_a, img_b)
                else:
                    matcher.match_direct(img_a, img_b)
                if _timing_csv is not None and _t0 is not None:
                    _timing_append_row(
                        _timing_csv,
                        ["path_a", "path_b", "seconds", "cache_hit"],
                        {"path_a": os.path.basename(img_a),
                         "path_b": os.path.basename(img_b),
                         "seconds": round(time.perf_counter() - _t0, 6),
                         "cache_hit": _cache_hit},
                    )
                result_queue.put(("SUCCESS", {"idx": idx, "done": True}))

            elif stage == "inference":
                idx, img_path = task
                depth_est.predict(img_path)
                roi_predictor.predict(img_path)
                segmentor.predict(img_path, mode="direct")
                result_queue.put(("SUCCESS", {"idx": idx, "done": True}))

            elif stage == "score":
                idx, img_a, img_b, label, target_id, house_name, q_id = task

                # 使用初始化區塊中已載入的 matcher（不再每次重建）
                if mode == "cubemap": matches, info = matcher.match_cubemap(img_a, img_b)
                elif mode == "core": matches, info = matcher.match_core(img_a, img_b)
                else: matches, info = matcher.match_direct(img_a, img_b)

                if not matches:
                    res = {
                        "idx": idx, "House": house_name, "Query": q_id, "Gallery": os.path.basename(img_b),
                        "Is_Connected": label, "RawScore_RoMa": 0.0, "GridScore_Depth": 0.0, "ROIScore": 0.0,
                        "WeightedDepthScore": 0.0, "SemanticScore": 0.0, "ModelBasedScore": 0.0,
                        "MatchPoints": 0, "Is_Target": 0
                    }
                else:
                    # 使用初始化區塊中已載入的模型（不再每次重建）
                    w, h = info["w"], info["h"]
                    depth_maps = {"A": depth_est.predict(img_a), "B": depth_est.predict(img_b)}
                    roi_maps = {"A": roi_predictor.predict(img_a), "B": roi_predictor.predict(img_b)}
                    lmap_a, lbl_a, _ = segmentor.predict(img_a, mode="direct")
                    lmap_b, lbl_b, _ = segmentor.predict(img_b, mode="direct")

                    s_raw = agg_conf.compute(matches)
                    s_roi = agg_roi.compute(matches, roi_maps, w, h)
                    s_grid = grid_scorer.compute(matches, p_conf.compute(matches), depth_maps, w, h)
                    s_depth = agg_depth.compute(matches, depth_maps, w, h)
                    s_sem = agg_sem.compute(matches, depth_maps, lmap_a, lbl_a, lmap_b, lbl_b, w, h)

                    s_model = 0.0
                    if model_dispatcher:
                        scorer = model_dispatcher.get_scorer(house_name, device)
                        global_matches = info.get("global_matches", matches)
                        s_model = scorer.compute(img_a, global_matches, depth_maps["A"], depth_maps["B"])

                    res = {
                        "idx": idx, "House": house_name, "Query": q_id, "Gallery": os.path.basename(img_b),
                        "Is_Connected": label,
                        "RawScore_RoMa": round(float(s_raw), 4),
                        "GridScore_Depth": round(float(s_grid), 4),
                        "ROIScore": round(float(s_roi), 4),
                        "WeightedDepthScore": round(float(s_depth), 4),
                        "SemanticScore": round(float(s_sem), 4),
                        "ModelBasedScore": round(float(s_model), 4),
                        "Is_Target": 1 if target_id and target_id in os.path.basename(img_b) else 0,
                        "MatchPoints": len(matches)
                    }
                result_queue.put(("SUCCESS", res))

        except Exception as e:
            import traceback
            result_queue.put(("ERROR", f"Task Error: {e}\n{traceback.format_exc()}"))
        finally:
            task_queue.task_done()

class ParallelScorer:
    """
    通用批次評分引擎：支援階段式執行 (match, inference, score)。
    每個 Worker 獨立處理一個 task，由多 GPU 分攤負載。
    """
    def __init__(self, workers='auto', mode="direct", model_type="roma_v2", conf=0.5, exp_name=None, use_cache=True):
        self.num_gpus = torch.cuda.device_count()
        self.n_workers = self.num_gpus if str(workers).lower() == 'auto' else int(workers)
        self.mode = mode
        self.model_type = model_type
        self.conf = conf
        self.exp_name = exp_name
        self.use_cache = use_cache

    @staticmethod
    def _precache_hub_models(model_type="roma_v2"):
        """在主進程預先下載 torch.hub 模型，避免多 Worker 同時下載時的競爭條件。"""
        # 只預抓 RoMa v2 的 backbone；match 階段用不到 DINOv2。

        # RoMa v2 的 backbone (DINOv3)
        try:
            if model_type == "roma_v2":
                import urllib.request
                import zipfile
                # 先觸發一次 Roma v2 的權重下載，避免多進程同時搶下載
                print("  [PreCache] Roma v2 / DINOv3 pre-caching...")
                from core_roma_match.RoMaScan import MatcherFactory
                _ = MatcherFactory.create(model_type="roma_v2", device="cpu")
                print("  [PreCache] Roma v2 hub model cached.")
        except Exception as e:
            print(f"  [PreCache] Roma v2 pre-cache warning: {e}")

    def run_stage(self, tasks, stage="score"):
        """統一的階段性並行入口: stage=['match', 'inference', 'score']"""
        if not tasks: return []

        # 預先在主進程下載 torch.hub 的模型，避免多進程同時 load 產生檔案系統競爭
        self._precache_hub_models(self.model_type)

        ctx = mp.get_context('spawn')
        task_queue = ctx.JoinableQueue()
        result_queue = ctx.Queue()

        processes = []
        for i in range(self.n_workers):
            gpu_id = i % self.num_gpus
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, task_queue, result_queue, stage,
                      self.mode, self.model_type, self.conf, self.exp_name, self.use_cache))
            p.daemon = True
            p.start()
            processes.append(p)

        for i, t in enumerate(tasks):
            task_queue.put((i,) + t)

        results = [None] * len(tasks)
        with tqdm(total=len(tasks), desc=f"Parallel {stage.capitalize()}", leave=False) as pbar:
            for _ in range(len(tasks)):
                status, data = result_queue.get()
                if status == "SUCCESS":
                    results[data["idx"]] = data
                else:
                    print(f"\n  [Error] {data}")
                pbar.update(1)

        for _ in range(self.n_workers): task_queue.put(None)
        for p in processes: p.join()

        return [r for r in results if r is not None]

    def run_batch(self, tasks):
        """向後相容介面"""
        return self.run_stage(tasks, stage="score")
