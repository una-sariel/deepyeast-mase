# DeepYeast MaSE-Net

**MaSE-Net** = **M**asked **S**elective **E**nsemble Network  
Masked patch selector + PLCNN branches + MSMM-style multi-head ensemble (PyTorch only).

**Repo:** https://github.com/una-sariel/deepyeast-mase  
**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) · **Full-data guide:** [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md) · **v4 professor steps:** [PROFESSOR_V4_RUN.md](PROFESSOR_V4_RUN.md) · **TypeError fix:** [FIX_LABEL_SMOOTHING_RERUN.md](FIX_LABEL_SMOOTHING_RERUN.md)

## Results

### Full data (seed=42, 5% init)

| Method | Test | Notes |
|--------|------|-------|
| Keras baseline | 88.4% | official |
| **MaSE Lite v2** | **89.1%** | uniform heads; head-mean CE |
| MaSE Lite v3 | 87.75% | fused-CE, learnable weights **collapsed** |
| **MaSE Lite v4** | TBD | fused-CE + **frozen** uniform — **recommended** |

### 5% pilot (seed=42)

| Method | Val | Test |
|--------|-----|------|
| Official Keras DeepYeast | 76.8% | 80.6% |
| MaSE-Net Lite v2 | 82.0% | 85.0% |
| MaSE-Net Lite v3 (learnable) | — | 86.1% |

**Recommended training:** Lite **v4** (`--freeze-ensemble`). See [PROFESSOR_V4_RUN.md](PROFESSOR_V4_RUN.md).

## Recommended script (Lite v4)

```bash
python pytorch/train_mase_lite.py \
  --data-dir /path/to/deepyeast_full \
  --freeze-ensemble \
  --epochs 60 --patience 15 --top-k 40 \
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 \
  --seed 42 --checkpoint-name mase_lite_full_v4
```

- Reproduce **v2** (89.1%): `--legacy-v2-loss --checkpoint-name mase_lite_full_v2`
- Reproduce **v3** (learnable): omit `--freeze-ensemble`, use `--checkpoint-name mase_lite_full_v3`

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
PROFESSOR_V4_RUN.md
FULL_DATA_QUICKSTART.md
ARCHITECTURE.md
UQ_MC_DROPOUT.md
```

## Requirements

- Python 3.11 or 3.12
- PyTorch ≥ 2.0 (CUDA strongly recommended for full data)
- Git LFS for init checkpoints
