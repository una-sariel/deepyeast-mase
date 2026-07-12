# DeepYeast Triple Fusion

**Masked patch selector + PLCNN triple branches + MSMM-style multi-head ensemble** (PyTorch only).

**Repo:** https://github.com/una-sariel/deepyeast-triple-fusion  
**Professor full-data guide:** [PROFESSOR_FULL_DATA.md](PROFESSOR_FULL_DATA.md)

## Pilot result (5%, seed=42)

| Method | Val | Test |
|--------|-----|------|
| Official Keras DeepYeast | 76.8% | 80.6% |
| **Triple Fusion Lite v2** | **82.0%** | **85.0%** |

Mask coverage stays at **0.625** (top-k=40 / 64 patches). Full-data numbers: TBD (for professor run).

## Recommended script (Lite v2)

```bash
python pytorch/train_triple_fusion_lite.py \
  --data-dir /path/to/deepyeast_full \
  --epochs 60 --patience 15 --top-k 40 \
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 \
  --seed 42 --checkpoint-name triple_fusion_lite_full_v2
```

Init weights (5% PLCNN + masked selector) ship under `artifacts/checkpoints_5pct/` and load automatically unless `--no-init`.

## Layout

```
prepare_deepyeast_subset.py   # download / sample DeepYeast
pytorch/
  train_triple_fusion_lite.py      # ★ Lite v2 (recommended)
  train_triple_fusion_lite_v1.py   # Lite v1 baseline
  train_triple_fusion.py           # Full MSMM backbone (GPU)
artifacts/checkpoints_5pct/        # init .pt (Git LFS)
results/                           # 5% summary JSON
PROFESSOR_FULL_DATA.md             # Windows full-data steps
```

## Requirements

- Python 3.11 or 3.12
- PyTorch ≥ 2.0 (CUDA strongly recommended for full data)
- See `requirements.txt`
