# DeepYeast MaSE-Net

**MaSE-Net** = **M**asked **S**elective **E**nsemble Network  
Masked patch selector + PLCNN branches + MSMM-style multi-head ensemble (PyTorch only).

**Repo:** https://github.com/una-sariel/deepyeast-mase  
**Full-data guide:** [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)

## Pilot result (5%, seed=42)

| Method | Val | Test |
|--------|-----|------|
| Official Keras DeepYeast | 76.8% | 80.6% |
| **MaSE-Net Lite v2** | **82.0%** | **85.0%** |

Mask coverage stays at **0.625** (top-k=40 / 64 patches). Full-data numbers: TBD.

## Recommended script (Lite v2)

```bash
python pytorch/train_mase_lite.py \
  --data-dir /path/to/deepyeast_full \
  --epochs 60 --patience 15 --top-k 40 \
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 \
  --seed 42 --checkpoint-name mase_lite_full_v2
```

Init weights (5% PLCNN + masked selector) ship under `artifacts/checkpoints_5pct/` and load automatically unless `--no-init`.

## Layout

```
prepare_deepyeast_subset.py   # download / sample DeepYeast
pytorch/
  train_mase_lite.py      # ★ Lite v2 (recommended)
  train_mase_lite_v1.py   # Lite v1 baseline
  train_mase.py           # Full MSMM backbone (GPU)
artifacts/checkpoints_5pct/        # init .pt (Git LFS)
results/                           # 5% summary JSON
FULL_DATA_QUICKSTART.md            # Windows full-data steps
```

## Requirements

- Python 3.11 or 3.12
- PyTorch ≥ 2.0 (CUDA strongly recommended for full data)
- See `requirements.txt`
