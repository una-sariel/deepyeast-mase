# DeepYeast MaSE-Net

**MaSE-Net** = **M**asked **S**elective **E**nsemble Network  
Masked patch selector + PLCNN branches + MSMM-style multi-head ensemble (PyTorch only).

**Repo:** https://github.com/una-sariel/deepyeast-mase  
**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) · **Full-data:** [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md) · **v4:** [V4_RUN.md](V4_RUN.md) · **v5:** [V5_RUN.md](V5_RUN.md) · **v6 TP-AHF:** [V6_RUN.md](V6_RUN.md) · **v6 Phase 2.5:** [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md) · **v7:** [V7_RUN.md](V7_RUN.md) · **v8 ID-Gate:** [V8_ID_GATE_RUN.md](V8_ID_GATE_RUN.md) · **v9 RSB:** [V9_RUN.md](V9_RUN.md) · **Fine-tune log:** [FINE_TUNING_V4_TO_V7.md](FINE_TUNING_V4_TO_V7.md) · **Results JSON:** [results/](results/) · **TypeError fix:** [FIX_LABEL_SMOOTHING_RERUN.md](FIX_LABEL_SMOOTHING_RERUN.md)

## Results

JSON snapshots by version: [results/README.md](results/README.md) (`results/v2` … `results/v8`).

### Full data (seed=42, 5% init)

| Method | Test | Notes / source |
|--------|------|----------------|
| Keras baseline | 88.4% | official |
| MaSE Lite v2 | 89.1% | uniform; head-mean CE |
| MaSE Lite v3 | 87.75% | learnable; **collapsed** |
| MaSE Lite v4 | ≈89.58% | fused-CE + frozen uniform (**best train, no TTA**) |
| v5 | ~89.16% | learnable w; below v4 |
| v6 Phase 1 | 88.95% | `results/v6/phase1_results.json` |
| v6 Phase 2.5 | 89.07% | `results/v6/phase25_results.json` |
| **v6 Phase 2.5 + TTA** | **89.82%** | `results/v6/phase25_tta_summary.json` (**best overall**) |
| v7 | 89.06% | `results/v7/results.json` |
| **v7 + TTA** | **89.82%** | `results/v7/tta_summary.json` |
| v8 ID-Gate | 89.05% | `results/v8/results.json` |
| v8 + TTA | 89.69% | `results/v8/tta_summary.json` |

Report **train** and **TTA** as separate rows. Goal 90% still open (~0.18pp vs best TTA).

### 5% pilot (seed=42)

| Method | Val | Test |
|--------|-----|------|
| Official Keras DeepYeast | 76.8% | 80.6% |
| MaSE-Net Lite v2 | 82.0% | 85.0% |
| MaSE-Net Lite v3 (learnable) | — | 86.1% |
| MaSE-Net Lite v4 (frozen) | 82.7% | **85.9%** |

**Run docs:** v6 [V6_RUN.md](V6_RUN.md) / [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md); v7 [V7_RUN.md](V7_RUN.md).

## Download DeepYeast data

**Important:** `prepare_deepyeast_subset.py` defaults to a **5% subset** (`deepyeast_5pct`, ~4,500 images).
Running it with **no flags does NOT create `deepyeast_full`.**

| Goal | Command |
|------|---------|
| **Full dataset** (~90,000 images) | `python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42` |
| 5% smoke test only | `python prepare_deepyeast_subset.py --fraction 0.05 --out-dir deepyeast_5pct --seed 42` |

After full download you should see **train 65,000 / val 12,500 / test 12,500** under `deepyeast_full/`.
Training uses `--data-dir /path/to/deepyeast_full`.

Details (cache location, re-download, verification): **[FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)** → *Download full data*.

## Recommended scripts

### v4 (accuracy SOTA)

```bash
python pytorch/train_mase_lite.py \
  --data-dir /path/to/deepyeast_full \
  --freeze-ensemble \
  --epochs 60 --patience 15 --top-k 40 \
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 \
  --seed 42 --checkpoint-name mase_lite_full_v4
```

### v5 (learnable anti-collapse + UQ)

```bash
python pytorch/train_mase_lite.py \
  --data-dir /path/to/deepyeast_full \
  --v5 \
  --epochs 60 --patience 15 --top-k 40 \
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 \
  --seed 42 --checkpoint-name mase_lite_full_v5
```

Reports `uq.test_AUROC_PE` (PE → AUROC). Softmax baseline is secondary only.

- Reproduce **v2** (89.1%): `--legacy-v2-loss --checkpoint-name mase_lite_full_v2`
- Reproduce **v3**: default flags without `--freeze-ensemble` / `--v5`

Init weights (5% PLCNN + masked selector) ship under `artifacts/checkpoints_5pct/` and load automatically unless `--no-init`.

## Layout

```
prepare_deepyeast_subset.py
pytorch/
  train_mase_lite.py      # ★ Lite v2 / v3 / v4 (flags)
  train_mase_lite_v1.py
  train_mase.py
  eval_mase_uq.py         # MC Dropout UQ (optional)
artifacts/checkpoints_5pct/
V4_RUN.md
V6_RUN.md
FULL_DATA_QUICKSTART.md
ARCHITECTURE.md
UQ_MC_DROPOUT.md
```

## Requirements

- Python 3.11 or 3.12
- PyTorch ≥ 2.0 (CUDA strongly recommended for full data)
- Git LFS for init checkpoints
