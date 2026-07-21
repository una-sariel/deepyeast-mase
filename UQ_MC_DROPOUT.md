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

## Full-data (professor) — after Lite **v4** training

Train first with [PROFESSOR_V4_RUN.md](PROFESSOR_V4_RUN.md) (`--freeze-ensemble` →
`mase_lite_full_v4`). Then evaluate UQ on that checkpoint (no retrain):

```powershell
python pytorch\eval_mase_uq.py `
  --data-dir "D:\...\deepyeast_full" `
  --checkpoint "D:\...\deepyeast_full\checkpoints\mase_lite_full_v4\best.pt" `
  --mc-samples 30 `
  --split both `
  --out-dir "D:\...\deepyeast_full\checkpoints\mase_lite_full_v4\uq_mc_dropout"
```

Send back: `...\mase_lite_full_v4\uq_mc_dropout\summary.json`  
(UAUC + τ sweep → UAcc/USen/USpe/UPre).

Use `T=30` for reporting; smoke tests can use `T=10`.
