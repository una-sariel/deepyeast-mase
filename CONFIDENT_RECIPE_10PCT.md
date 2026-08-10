# MaSE Lite — higher-confidence recipe (10% pilot)

Goal: lower PE / sharper Softmax vs `triple_fusion_lite_10pct_v2`  
(same data, then same `eval_mase_uq.py`).

## Changes vs v2

| Knob | v2 | confident |
|------|----|-----------|
| label_smoothing | 0.1 | **0** |
| branch_dropout | 0.4 | **0.25** |
| head_dropout | 0.5 | **0.3** |
| start | from PLCNN/masked init | **resume v2 best.pt** (finetune) |
| epochs | 60 | **20** (patience 8) |
| backbone lr | 1e-3 | **3e-4** |

Loss: **`--legacy-v2-loss`** (head-mean CE, same family as 10% v2).

## Train (from repo `pytorch/`)

```powershell
python train_mase_lite.py `
  --data-dir C:\Users\unaliuqw\deepyeast_10pct `
  --resume C:\Users\unaliuqw\deepyeast_10pct\checkpoints\triple_fusion_lite_10pct_v2\best.pt `
  --checkpoint-name mase_lite_10pct_confident `
  --legacy-v2-loss `
  --label-smoothing 0 `
  --branch-dropout 0.25 `
  --head-dropout 0.3 `
  --epochs 20 --patience 8 `
  --backbone-lr 3e-4 --selector-lr 1e-3 `
  --seed 42
```

## UQ after train

```powershell
python eval_mase_uq.py `
  --data-dir C:\Users\unaliuqw\deepyeast_10pct `
  --checkpoint C:\Users\unaliuqw\deepyeast_10pct\checkpoints\mase_lite_10pct_confident\best.pt `
  --mc-samples 30 --split both --seed 42 `
  --top-k 40 --soft-alpha 0.5 `
  --branch-dropout 0.25 --head-dropout 0.3
```

Compare `pe_mean` and `UAUC` vs:
- `...\triple_fusion_lite_10pct_v2\uq_mc_dropout\summary.json`
- `...\deepyeast_baseline_10pct\uq_mc_dropout\summary.json`
