# MaSE-Net — Full Data Quick Start (Windows)

**MaSE-Net** = **M**asked **S**elective **E**nsemble Network  

**Repo:** https://github.com/una-sariel/deepyeast-mase  
**Method:** Masked selector + PLCNN (VGG/ResNet/DenseNet) + 4-head ensemble  
**Framework:** PyTorch only (no JAX / no TensorFlow for training)  
**Python:** 3.11 or 3.12  

**5% pilot (seed=42):** val **82.0%**, test **85.0%** (Keras baseline test 80.6%).  
**Full data v2 (professor, seed=42):** test **89.1%** (Keras 88.4%).  
**Recommended now:** **v3** fused-CE + `min_ensemble_weight=0.05` (see Step 3).

---

## Prerequisites

You already have full DeepYeast data, for example:

```text
D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full\
  metadata.csv
  images\...
```

Set that path as `DATA_DIR` below.  
If you do **not** have full data yet:

```powershell
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

(`--fraction 1.0` = all images; download can take a while.)

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

## Step 3 — Full-data training (recommended = Lite v3)

**v3** fixes ensemble learning (fused CE + weight floor). Same architecture as v2 (89.1% test).

```powershell
cd C:\dy\deepyeast-mase
.venv\Scripts\activate
git pull
git lfs pull

python pytorch\train_mase_lite.py `
  --data-dir "D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full" `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --aux-head-weight 0.5 `
  --distill-weight 0.1 `
  --min-ensemble-weight 0.05 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v3
```

Replace `--data-dir` with your actual `deepyeast_full` folder.

Defaults already match v3 — you can omit the `--aux-head-weight` / `--distill-weight` / `--min-ensemble-weight` flags if you use a fresh `git pull`.

### What v3 adds over v2

| Item | v2 (89.1% run) | v3 (recommended) |
|------|----------------|------------------|
| Loss | mean per-head CE | **fused CE** + aux + KL |
| Ensemble weights | stuck at 0.25 | **learnable** (watch `w=[...]` in log) |
| Weight floor | none | **min_w = 0.05** (anti-collapse) |
| 5% pilot | 85.0% | **86.1%** |

### What this does

| Item | Setting |
|------|---------|
| Architecture | MaSE-Net Lite (~7M params) |
| Mask | hard top-k=40 → coverage **0.625** |
| Init | `artifacts/checkpoints_5pct/plcnn_triple` + `masked_v3k60_pytorch` (auto) |
| Regularization | strong augment, label smoothing 0.1, sparsity 0.05, dropout |
| Early stop | patience 15 on val accuracy |
| Output | `deepyeast_full\checkpoints\mase_lite_full_v3\` |

### Outputs to send back

```text
...\checkpoints\mase_lite_full_v3\
  results.json     ← best_val_accuracy + test.accuracy + ensemble_weights
  best.pt
```

Key fields in `results.json`:

- `best_val_accuracy`
- `test.accuracy`
- `test.mask_coverage` (should stay ~0.625)
- `test.ensemble_weights` (v3 should **not** stay at 0.25)
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

### B) No init (train from scratch)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "YOUR\deepyeast_full" `
  --no-init `
  --epochs 60 --patience 15 --top-k 40 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2_noinit
```

### C) Full MSMM backbone (needs GPU; slower / heavier)

```powershell
python pytorch\train_mase.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 50 --patience 10 --top-k 38 `
  --seed 42 --checkpoint-name mase_full
```

### D) Lite v1 baseline (no PLCNN/selector init)

```powershell
python pytorch\train_mase_lite_v1.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 30 --patience 8 --top-k 38 `
  --seed 42 --checkpoint-name mase_lite_full_v1
```

---

## Comparison context (for reporting)

| Setting | Official Keras | MaSE Lite v2 | MaSE Lite v3 |
|---------|----------------|--------------|--------------|
| 5% test | 80.6% | 85.0% | **86.1%** (pilot) |
| Full test | 88.4% | **89.1%** | TBD |

Goal: v3 should match or beat v2 **89.1%**, with non-uniform `ensemble_weights`.

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
