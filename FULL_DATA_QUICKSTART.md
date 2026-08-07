# MaSE-Net — Full Data Quick Start (Windows)

**MaSE-Net** = **M**asked **S**elective **E**nsemble Network  

**Repo:** https://github.com/una-sariel/deepyeast-mase  
**Method:** Masked selector + PLCNN (VGG/ResNet/DenseNet) + 4-head ensemble  
**Framework:** PyTorch only (no JAX / no TensorFlow for training)  
**Python:** 3.11 or 3.12  

**5% pilot (seed=42):** val **82.0%**, test **85.0%** (Keras baseline test 80.6%).  
**Full data v2 (seed=42):** test **89.1%** (Keras 88.4%).  
**Full data v3 (learnable):** test **87.75%** (weight collapse).  
**Recommended now:** **v4** fused-CE + **frozen uniform** weights (see [V4_RUN.md](V4_RUN.md)).  
**New — learnable fusion:** **v6 TP-AHF** ([V6_RUN.md](V6_RUN.md); accuracy push: [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md)).

---

## Download full data (`deepyeast_full`)

Use this when you need **all** images for full-data MaSE training (v4 / v5 / etc.).

### One command (from repo root)

```powershell
cd path\to\deepyeast-mase
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

**Do not omit `--fraction 1.0`.** Without it, the script uses the default `--fraction 0.05` and writes **`deepyeast_5pct`** (~4,499 images), not `deepyeast_full`.

### What the script does

1. **Step 1** — Download manifest files (cached after first run).
2. **Step 2** — Stratified sampling per split (with `--fraction 1.0`, keeps every image).
3. **Step 3** — Download `main.tar.gz` (~398 MB) from DeepYeast server (cached after first run).
4. **Step 4** — Extract selected PNGs into `deepyeast_full/train|val|test/<class>/`, write `labels.csv`, `class_map.json`, `dataset_stats.json`.

**Download cache (reused):** `%USERPROFILE%\.deepyeast\cache\`  
(manifests + `main.tar.gz`; safe to keep — re-runs skip re-download if checksums match).

**Output folder:** `deepyeast_full/` next to the script (or whatever you pass to `--out-dir`).

If `deepyeast_full` already exists, the script **deletes and recreates** it (full re-extract).

### Expected result (full data, seed=42)

```text
deepyeast_full/
  train/          65,000 images
  val/            12,500 images
  test/           12,500 images
  labels.csv
  class_map.json
  dataset_stats.json
```

(`metadata.csv` is auto-created from `labels.csv` on first training load.)

Verify counts:

```powershell
python -c "import json; s=json.load(open('deepyeast_full/dataset_stats.json')); print(s['total_images'], s['splits'])"
```

You should see **`total_images`: 90000**.

### Re-download / start over

Same command as above — it will reuse cached tar if present, wipe `deepyeast_full/`, and extract again:

```powershell
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

To force re-download the tar, delete `%USERPROFILE%\.deepyeast\cache\main.tar.gz` first.

**Time:** First run ~10–30+ minutes depending on network and disk (download ~400 MB + extract ~90k PNGs). Later re-extracts are faster if tar is cached.

### Common mistake

| You ran | You got | Training expects |
|---------|---------|------------------|
| `python prepare_deepyeast_subset.py` (no args) | `deepyeast_5pct`, ~4499 images | — |
| Full training `--data-dir deepyeast_full` | folder missing | **90000** images |

**Fix:** Run the full command with `--fraction 1.0 --out-dir deepyeast_full`, then point `--data-dir` at that folder.

---

## Prerequisites

You already have full DeepYeast data, for example:

```text
D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full\
  train\ ...
  val\ ...
  test\ ...
  labels.csv
```

Set that path as `DATA_DIR` below.  
If you do **not** have full data yet, see **[Download full data](#download-full-data-deepyeast_full)** above (not the 5% default).

---

## Step 1 — Clone & install (one time)

**Important:** init weights are in **Git LFS** (~70 MB). After clone you **must** run `git lfs pull` or training will fail.

Install [Git LFS](https://git-lfs.com) first.

```powershell
cd C:\dy
git lfs install
git clone https://github.com/una-sariel/deepyeast-mase.git
cd deepyeast-mase
git pull
git lfs pull

# Verify init checkpoints are real files (~28MB + ~44MB), NOT ~1KB pointers:
dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt

python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use CUDA:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(Adjust CUDA version to match your driver.)

Verify:

```powershell
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

**GPU strongly recommended.** Full data on CPU is very slow (days).

---

## Step 2 — Optional smoke test (5%, 1 epoch)

```powershell
python prepare_deepyeast_subset.py --fraction 0.05 --out-dir deepyeast_5pct --seed 42

python pytorch\train_mase_lite.py `
  --data-dir deepyeast_5pct `
  --epochs 1 --patience 0 --batch-size 32 `
  --checkpoint-name smoke_test
```

Expect a `results.json` under `deepyeast_5pct\checkpoints\smoke_test\`.

---

## Step 3 — Full-data training (recommended = Lite **v4**)

**v4** = fused-CE (like v3) + **frozen uniform** ensemble weights (like v2 diversity).  
Candidate to beat full-data v2 (**89.1%**). Detailed steps: [V4_RUN.md](V4_RUN.md).

```powershell
cd C:\dy\deepyeast-mase
.venv\Scripts\activate
git pull
git lfs pull

python pytorch\train_mase_lite.py `
  --data-dir "D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full" `
  --freeze-ensemble `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --aux-head-weight 0.5 `
  --distill-weight 0.1 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v4
```

Replace `--data-dir` with your actual `deepyeast_full` folder.

Log checks: progress **`MaSELiteV4`**, every epoch **`w=[0.25,0.25,0.25,0.25]`**.

### Recipe comparison

| Item | v2 | v3 | **v4 (recommended)** |
|------|----|----|----------------------|
| Loss | mean per-head CE | fused CE + aux + KL | **fused CE + aux + KL** |
| Ensemble weights | fixed 0.25 | learnable (collapsed on full) | **frozen 0.25** |
| Full test | **89.1%** | 87.75% | **TBD (goal > 89.1%)** |

### What this does

| Item | Setting |
|------|---------|
| Architecture | MaSE-Net Lite (~7M params) — **same as v2/v3** |
| Mask | hard top-k=40 → coverage **0.625** |
| Init | `artifacts/checkpoints_5pct/plcnn_triple` + `masked_v3k60_pytorch` (auto) |
| Regularization | strong augment, label smoothing 0.1, sparsity 0.05, dropout |
| Early stop | patience 15 on val accuracy |
| Output | `deepyeast_full\checkpoints\mase_lite_full_v4\` |

### Outputs to send back

Please return **these two JSON files** (see [V4_RUN.md](V4_RUN.md) for full steps including UQ):

```text
1) ...\checkpoints\mase_lite_full_v4\results.json
2) ...\checkpoints\mase_lite_full_v4\uq_mc_dropout\summary.json
```

| File | Purpose |
|------|---------|
| `results.json` | Accuracy, ensemble weights, mask coverage |
| `uq_mc_dropout\summary.json` | UAUC + τ sweep (after Step 3 UQ in V4_RUN.md) |

Key fields in `results.json`:

- `method` → `mase_lite_v4`
- `best_val_accuracy`
- `test.accuracy` (success if **> 0.891**)
- `test.mask_coverage` (should stay ~0.625)
- `test.ensemble_weights` (v4 should stay **~0.25**)
- `best_epoch`

---

## Optional variants

### A) Reproduce v2 exactly (89.1% full-data run)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "YOUR\deepyeast_full" `
  --legacy-v2-loss `
  --epochs 60 --patience 15 --top-k 40 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2
```

### B) Lite v3 learnable (known full-data collapse; keep for reference)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 60 --patience 15 --top-k 40 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --min-ensemble-weight 0.05 `
  --seed 42 --checkpoint-name mase_lite_full_v3
```

### C) No init (train from scratch)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "YOUR\deepyeast_full" `
  --freeze-ensemble `
  --no-init `
  --epochs 60 --patience 15 --top-k 40 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v4_noinit
```

### D) Full MSMM backbone (needs GPU; slower / heavier)

```powershell
python pytorch\train_mase.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 50 --patience 10 --top-k 38 `
  --seed 42 --checkpoint-name mase_full
```

### E) Lite v1 baseline (no PLCNN/selector init)

```powershell
python pytorch\train_mase_lite_v1.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 30 --patience 8 --top-k 38 `
  --seed 42 --checkpoint-name mase_lite_full_v1
```

---

## Comparison context (for reporting)

| Setting | Official Keras | Lite v2 | Lite v3 | Lite **v4** |
|---------|----------------|---------|---------|-------------|
| 5% test | 80.6% | 85.0% | 86.1% | TBD |
| Full test | 88.4% | **89.1%** | 87.75% | **TBD (goal > 89.1%)** |

Goal: **v4** test accuracy **> 89.1%**, with `ensemble_weights` staying at 0.25.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `git clone` tiny / no `.pt` | `git lfs install` then `git lfs pull` |
| `Checkpoint too small` / `torch.load` error | Git LFS pointers — run `git lfs pull`; see [COLLABORATOR_V2_RUN.md](COLLABORATOR_V2_RUN.md) |
| Init skipped | Check `artifacts\checkpoints_5pct\*\best.pt` are ~28MB / ~44MB |
| OOM on GPU | `--batch-size 32` or `16` |
| Slow on CPU | Use CUDA machine; full run not practical on CPU |
| Wrong data path | Folder must contain `metadata.csv` + `images\` |

---

## Contact

Questions / `results.json` → send back to Qiwu.
