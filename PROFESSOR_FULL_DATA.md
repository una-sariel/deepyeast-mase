# Triple Fusion — Full Data Quick Start (Windows)

**Repo:** https://github.com/una-sariel/deepyeast-triple-fusion  
**Method:** Masked selector + PLCNN (VGG/ResNet/DenseNet) + 4-head ensemble  
**Framework:** PyTorch only (no JAX / no TensorFlow for training)  
**Python:** 3.11 or 3.12  

**5% pilot (seed=42):** val **82.0%**, test **85.0%** (Keras baseline test 80.6%).  
**Full data:** please run the command below and share `results.json`.

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

Install [Git LFS](https://git-lfs.com) first (needed for init checkpoints ~70 MB).

```powershell
cd C:\dy
git lfs install
git clone https://github.com/una-sariel/deepyeast-triple-fusion.git
cd deepyeast-triple-fusion
git lfs pull

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

python pytorch\train_triple_fusion_lite.py `
  --data-dir deepyeast_5pct `
  --epochs 1 --patience 0 --batch-size 32 `
  --checkpoint-name smoke_test
```

Expect a `results.json` under `deepyeast_5pct\checkpoints\smoke_test\`.

---

## Step 3 — Full-data training (recommended recipe = Lite v2)

Same hyperparameters as the 5% run that reached **85.0% test**.

```powershell
cd C:\dy\deepyeast-triple-fusion
.venv\Scripts\activate

python pytorch\train_triple_fusion_lite.py `
  --data-dir "D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full" `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --seed 42 `
  --checkpoint-name triple_fusion_lite_full_v2
```

Replace the `--data-dir` path with your actual `deepyeast_full` folder.

### What this does

| Item | Setting |
|------|---------|
| Architecture | Lite Triple Fusion v2 (~7M params) |
| Mask | hard top-k=40 → coverage **0.625** |
| Init | `artifacts/checkpoints_5pct/plcnn_triple` + `masked_v3k60_pytorch` (auto) |
| Regularization | strong augment, label smoothing 0.1, sparsity 0.05, dropout |
| Early stop | patience 15 on val accuracy |
| Output | `deepyeast_full\checkpoints\triple_fusion_lite_full_v2\` |

### Outputs to send back

```text
...\checkpoints\triple_fusion_lite_full_v2\
  results.json     ← best_val_accuracy + test.accuracy
  best.pt
  train.log        (if you tee the console)
```

Key fields in `results.json`:

- `best_val_accuracy`
- `test.accuracy`
- `test.mask_coverage` (should stay ~0.625)
- `best_epoch`

---

## Optional variants

### A) No init (train from scratch)

```powershell
python pytorch\train_triple_fusion_lite.py `
  --data-dir "YOUR\deepyeast_full" `
  --no-init `
  --epochs 60 --patience 15 --top-k 40 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name triple_fusion_lite_full_v2_noinit
```

### B) Full MSMM backbone (needs GPU; slower / heavier)

```powershell
python pytorch\train_triple_fusion.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 50 --patience 10 --top-k 38 `
  --seed 42 --checkpoint-name triple_fusion_full
```

### C) Lite v1 baseline (no PLCNN/selector init)

```powershell
python pytorch\train_triple_fusion_lite_v1.py `
  --data-dir "YOUR\deepyeast_full" `
  --epochs 30 --patience 8 --top-k 38 `
  --seed 42 --checkpoint-name triple_fusion_lite_full_v1
```

---

## Comparison context (for reporting)

| Setting | Official Keras | Masked v5full (prior) | Triple Fusion Lite v2 |
|---------|----------------|------------------------|------------------------|
| 5% test | 80.6% | — | **85.0%** |
| Full test | **88.4%** | 87.5% | **please fill** |

Goal: see whether Triple Fusion beats Keras **88.4%** on full data.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `git clone` tiny / no `.pt` | `git lfs install` then `git lfs pull` |
| Init skipped | Check `artifacts\checkpoints_5pct\*\best.pt` exist; or use `--no-init` |
| OOM on GPU | `--batch-size 32` or `16` |
| Slow on CPU | Use CUDA machine; full run not practical on CPU |
| Wrong data path | Folder must contain `metadata.csv` + `images\` |

---

## Contact

Questions / `results.json` → send back to Qiwu.
