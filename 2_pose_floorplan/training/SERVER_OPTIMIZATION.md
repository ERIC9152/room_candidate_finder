# Server optimisation — what changed, why, and how to run it

Target machine: **2× RTX A5000 (24 GB) + 1× RTX A6000 (48 GB)**, Ampere,
driver 570 / CUDA 12.8.

The RTX 1070 run (README, `docs/PHASE*_NOTES.md`) was sized around 8 GB of
Pascal memory. This branch undoes those hardware compromises and adds the
architecture and training changes that the Phase 3–4 evidence pointed at.

**Status: trained.** The run took 17 h 56 m over 16 epochs and improved every
test metric over the RTX 1070 run — results and the run's own story are in
[`PHASE5_NOTES.md`](PHASE5_NOTES.md). What is *not* established is which of the
changes below deserves the credit: `configs/server_legacy_arch.yaml` (the
original architecture with the new training recipe) is the control that would
answer it, and it has not been run.

---

## 1. Summary of changes

| area | RTX 1070 run | server config | evidence / reason |
|---|---|---|---|
| backbone | ResNet-34, V1 weights | **ResNet-50, IMAGENET1K_V2** | the 1070 README documents R34 as a memory compromise |
| columns per pano | 256 (1.41°) | **1024 (0.35°)** | per request; see §3 for what that does and does not buy |
| transformer | 6 × d256, stock layer | **8 × d384**, flash/mem-efficient SDPA, drop-path 0.1 | train/val gap grew to 0.34; sequence is now 2048 tokens |
| pose decoder | 1D conv over A\|B | **attention pooling** (4 learned queries) | conv straddles the A\|B junction and zero-pads a circular signal |
| geometric pose | RANSAC, post-hoc only | **differentiable weighted Procrustes + IRLS**, trained | heads were never asked to be geometrically consistent (A14b: RANSAC is limited by head accuracy, not outliers) |
| augmentation | roll, swap, gamma | **+ exact mirror** of both panos | doubles the geometric variety, exact targets |
| precision | fp16 + GradScaler | **bf16**, TF32, channels_last, fused AdamW | Ampere has tensor cores; bf16 needs no loss scaling |
| optimiser | AdamW wd 1e-4 on everything | AdamW **wd 0.05**, none on norms/biases/embeddings | standard transformer recipe |
| schedule | 6 epochs, cosine → 1e-11 | **16 epochs**, cosine with 1 % floor, 2k warmup | the 1070 run's last epoch trained at ~0 LR (A14a confound) |
| weights evaluated | raw | **EMA** (0.9998) | free for regression targets |
| batch | 8, one GPU | **8 × 3 GPUs (DDP) = 24** | √3-scaled lr **failed** — see below; trained at 3e-4 |
| validation | 2 000-pair subset | **full val split**, sharded across GPUs, raw + EMA | subset noise was the size of the epoch-4/5 margin |
| data loading | full JPEG decode + resize | **DCT-domain ½ decode** (`fast_jpeg`) | ZInD panos are 2048×1024; decode bounds the loader |
| checkpoint selection | val total at the ramped weights | val total at **full** weights, on EMA | comparable across epochs |
| evaluation | fp32, one view | optional **TTA** (column rolls + mirror), EMA weights, geo pose reported | — |

Backwards compatibility: every new option defaults to the old behaviour. The
old `configs/proto_rtx1070.yaml` still trains the same model, and the
RTX 1070 `epoch_005.pt` loads **strictly** and produces **bit-identical**
outputs under the new code (max abs difference 0.0 on every output, measured).

---

## 2. The changes in detail

### 2.1 Differentiable geometric pose (`covispose/geo_pose.py`)

The RANSAC estimator turns the three per-column heads into 2D–2D floor-plane
correspondences and fits a rigid transform — but only *after* training, so no
gradient ever tells the heads that their outputs must agree geometrically.
Phase 4 found RANSAC's accuracy surface flat in the threshold: what limits it
is head accuracy, not outliers.

The new branch builds the same correspondences from **both** directions
(A's columns and B's columns, ~2 × n_cols points), with differentiable circular
interpolation of the other pano's boundary, and solves a closed-form weighted
2D Procrustes in float32. Weights are predicted co-visibility × an
inverse-variance range term `1/(1 + r̄)²`. Two rounds of Cauchy IRLS stand in
for RANSAC; the robust weights are detached.

Measured on **ground-truth** head values (house 0000, 102 pairs, 256 columns):

| | t median | t p90 | yaw median | yaw p90 |
|---|---|---|---|---|
| RANSAC (reference) | 0.0153 | 0.102 | 0.035° | — |
| Procrustes, no IRLS | 0.0553 | 0.184 | 0.140° | 2.55° |
| **Procrustes, 2 × IRLS** | **0.0375** | **0.113** | **0.046°** | **0.92°** |

Its loss (`geo_t`, `geo_yaw`, weight 0.5 each) is ramped in linearly over the
second epoch (`geo_start_steps`, `geo_ramp_steps`): Procrustes over untrained
heads is noise. Evaluation reports it as a third pose estimator next to the
decoder and RANSAC. The decoder remains the primary output.

Caveat worth knowing: the GT floor of this estimator (0.0375) is below the
1070 decoder's *test* error (0.094) but not by an enormous factor, so on easy
pairs the geo loss is partly asking for precision the labels cannot provide.
If the column heads get *worse* with the geo loss on (compare `corr_err_deg`
and `bon_err_deg` against the legacy run), lower `geo_t`/`geo_yaw`.

### 2.2 Attention pose decoder

Four learned queries cross-attend to the full fused 2 × n_cols sequence (two
blocks), and their outputs are concatenated in a fixed order before the MLP, so
the pair order stays distinguishable. Output initialised to the identity pose,
as before. The conv decoder remains available as `pose_decoder: conv`.

### 2.3 Mirror augmentation

Mirroring *both* panoramas reflects the whole scene through the camera
y-axis: columns reverse, azimuths negate, `t → (−tx, ty)`, `yaw → −yaw`
(`M·Rot(a)·M = Rot(−a)`). Mirroring only one pano would not be rigid, so it
is never done. Verified three ways: the closed-loop geometric consistency test
(max angle error 7.9e-5°, identical radial error with and without mirror on
the same pairs), a pixel-exact image check, and an exact round-trip against the
TTA inverse.

### 2.4 Numerics and throughput

* One backbone call over `cat([A, B])` instead of two.
* `_masked_mean` no longer calls `.item()` — it was a host–device sync six times
  per step.
* The transformer layer is re-implemented with the stock layer's parameter names
  (so old checkpoints load) plus stochastic depth, and `nn.MultiheadAttention`
  dispatches to flash / memory-efficient SDPA on torch ≥ 2.0. This matters at
  2 048 tokens.
* `torch.load(weights_only=False)` for our own checkpoints: torch ≥ 2.6 would
  otherwise refuse every checkpoint in this repo.

### 2.5 The learning rate the scaling argument gave was wrong

`lr = 5e-4` came from √-scaling the 1070 run's 3e-4 to the 3× batch. It
destabilised the moment warmup ended: the boundary loss regressed from 0.049 to
0.11, gradient norms spiked to 170 with ~100 % clipping, and co-visibility sat
at the majority class through step 4 200. At **3e-4 with 4 000 warmup steps**
the same model trained cleanly, gradient norms 10–25 throughout. The config
carries this measurement so the scaling argument is not re-derived.

### 2.6 Things deliberately *not* changed

* **Co-visibility `pos_weight`** stays 1.0. A14a is still open and nothing new
  settles it; `loss.covis_pos_weight: 0.31` is the ablation if a GPU is free.
* **Loss weights** for the original five terms are unchanged (A9).
* **Pair sampling** is unchanged. The hardest covis bucket lost 6.4 points on
  test, and a covis-stratified sampler is the next thing to try if it still
  lags — but not before seeing whether the new model closes that gap alone.
* **Input resolution** in the main config stays 512×1024; see §3.

---

## 3. About 1 024 columns

`n_cols: 1024` at 512×1024 gives one token per pixel column, 0.35° apart,
which removes column quantisation from every per-column head and from the
correspondence grid. What it does *not* do is add image detail: the
backbone's finest stage is stride 4, so the 1 024 tokens are resampled from
256 distinct feature columns. The transformer can still refine sub-feature
positions (it attends across both panos at full token resolution), but the
gain is bounded by the backbone.

For tokens with more real pixels behind them, `configs/server_hires.yaml`
feeds the native 1024×2048 panorama (the stride-4 grid is then 512 wide, a 2×
resample, 2 pixels per token). It costs ~3–4× the compute, and is written for
the A6000 alone with gradient checkpointing and accumulation.

Cost of 1 024 columns in the main config: the transformer sequence is 2 048
tokens. With SDPA its memory is linear in sequence length; compute is
quadratic but small next to a ResNet-50 at 512×1024.

---

## 4. Running it on the server

### 4.1 Environment

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements-server.txt
```

```bash
export ZIND_ROOT=/path/to/zind_data
```

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

The second line matters on this mixed machine: CUDA's default order is
fastest-first, not `nvidia-smi`'s PCI order, so without it
`CUDA_VISIBLE_DEVICES=1` is not guaranteed to be the A6000.

### 4.2 Pair index

`data/pairs/*.csv` is git-ignored. Either copy the six CSVs from the 1070
machine into `data/pairs/`, or rebuild them:

```bash
python scripts/build_pair_index.py --covis-csv /path/to/covis_scores.csv --partition /path/to/zind_partition.json --out-dir data/pairs
```

### 4.3 Gates before a long run (≈10 minutes)

```bash
python tests/test_model.py
```

```bash
python tests/test_checkpoint.py
```

```bash
python tests/test_overfit.py --server-options
```

### 4.4 Measure, then set `batch_size` and `num_workers`

The per-GPU batch of 8 and `num_workers: 10` are **not measured**. On one
A5000 (the smaller card bounds the DDP batch):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_vram.py --config configs/server_a5000.yaml --batch-sizes 4,6,8,10,12 --data-throughput
```

Pick the largest batch whose *reserved* memory stays under ~22 GB. If it
changes, scale `lr` with the square root of the total batch and put the new
value in the config. The loader must deliver ≥ 3 × the measured pairs/s.

### 4.5 Train

All three GPUs, one DDP run (recommended for the main model):

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py --config configs/server_a5000.yaml
```

DDP steps at the speed of the slowest card; the A6000's extra memory goes
unused in this mode. Resume is automatic from `ckpt/<dir>/last.pt` — rerun the
same command.

Alternatively, use the cards for parallel experiments, e.g. the main model on
the two A5000s and the architecture control on the A6000:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/server_a5000.yaml
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/server_legacy_arch.yaml
```

(With `CUDA_DEVICE_ORDER=PCI_BUS_ID` the indices match `nvidia-smi`, where
GPU 1 is the A6000 on this machine.
With 2 GPUs instead of 3, the batch per step drops to 16; lower `lr` to 4e-4
or set `grad_accum` to keep 24.)

### 4.6 Evaluate

```bash
python scripts/tune_ransac.py --ckpt ckpt/server_r50_c1024/best.pt
```

```bash
python scripts/evaluate_split.py --ckpt ckpt/server_r50_c1024/best.pt --split val --batch-size 16
```

```bash
python scripts/evaluate_split.py --ckpt ckpt/server_r50_c1024/best.pt --split test --batch-size 16
```

With test-time augmentation (4 rolls × mirror = 8 forward passes per pair):

```bash
python scripts/evaluate_split.py --ckpt ckpt/server_r50_c1024/best.pt --split test --batch-size 16 --tta-rolls 4 --tta-flip --tag _tta
```

As before: tune the RANSAC threshold on **val only**, choose TTA on **val only**,
then report test once.

---

## 5. What it achieved, and what is still unattributed

The RTX 1070 test baseline was: IoU 0.8887, correspondence 7.55°, boundary
1.89°, decoder t median 0.0944, yaw median 1.671°, acc@(0.5, 10°) 0.8911, and
0.6821 on the hardest covis bucket.

The server run: IoU **0.9276**, correspondence **3.94°**, boundary **1.64°**,
t median **0.0616**, yaw median **0.869°**, acc@(0.5, 10°) **0.9549**, and
**0.8622** on that hardest bucket.

**Which change earned that is still unknown.** Backbone, column count,
transformer size, the attention decoder, the geometric branch, mirror
augmentation, EMA, the longer schedule and the bigger batch all moved at once.
`configs/server_legacy_arch.yaml` — the 1070 architecture under the new recipe —
is the one run that would split "recipe" from "architecture", and it is still
the most useful next experiment after the cross-room retraining.

### Measured on the server (for sizing a future run)

| | |
|---|---|
| throughput | 58 pairs/s across 3 GPUs (411–420 ms/step at 24 pairs/step) |
| epoch | 66–68 min, including full validation of raw and EMA weights |
| batch | 8/GPU → 13.3 GiB; flat past 8 (23.1 vs 23.7 pairs/s at 12) |
| loader | one loader gives 141 / 195 / 194 / 89 pairs/s at 4 / 8 / 16 / 32 workers |
| peak VRAM | 16.33 GiB of 24 |
