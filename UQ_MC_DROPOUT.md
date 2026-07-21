# MaSE Lite — MC Dropout UQ evaluation

Architecture and training are **unchanged**. This script only changes **test-time
inference**: keep Dropout on, sample `T` times, average probabilities, compute
predictive entropy (PE).

## What you get

| Output | Meaning |
|--------|---------|
| `per_image_{split}.csv` | Per image: `y_true`, `y_pred`, `pe`, `max_prob` |
| `summary.json` | Accuracy, **UAUC**, PE stats, full τ sweep → UAcc/USen/USpe/UPre |

- **UAUC**: how well PE ranks wrong predictions above correct ones (no τ needed).
- **UQ confusion matrix**: `PE < τ` → certain; `PE ≥ τ` → uncertain.

## Run (5% example)

```bash
python pytorch/eval_mase_uq.py \
  --data-dir /path/to/deepyeast_5pct \
  --checkpoint /path/to/checkpoints/abl_min05/best.pt \
  --mc-samples 30 \
  --split both \
  --out-dir /path/to/checkpoints/abl_min05/uq_mc_dropout
```

Config is auto-loaded from `results.json` next to the checkpoint when present.

## Full-data (professor) example

```powershell
python pytorch\eval_mase_uq.py `
  --data-dir "D:\...\deepyeast_full" `
  --checkpoint "D:\...\checkpoints\mase_lite_full_v3\best.pt" `
  --mc-samples 30 `
  --split both `
  --out-dir "D:\...\checkpoints\mase_lite_full_v3\uq_mc_dropout"
```

Use `T=30` for reporting; smoke tests can use `T=10`.
