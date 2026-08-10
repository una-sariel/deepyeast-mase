# Baseline vs MaSE UQ (same split, same metrics)

Compare **official DeepYeastNet** (paper architecture) vs **MaSE-Net Lite** under MC Dropout UQ.

## Protocol

| Item | Setting |
|------|---------|
| Data | Same local split (e.g. `deepyeast_10pct`) |
| Metrics | Acc, **UAUC = AUROC(PE)**, AUROC(1−max_prob), τ sweep |
| MC samples | `T=30` (match MaSE UQ docs) |
| Seed | 42 |

**Primary comparison:** `UAUC` / `AUROC(PE)` — not “who is more confident.”

## 1. Train DeepYeast baseline

```powershell
cd C:\Users\unaliuqw\deepyeast-mase\pytorch
python train_deepyeast_baseline.py `
  --data-dir C:\Users\unaliuqw\deepyeast_10pct `
  --out-dir C:\Users\unaliuqw\deepyeast_10pct\checkpoints\deepyeast_baseline_10pct `
  --epochs 40 --patience 12 --seed 42
```

## 2. UQ on baseline

```powershell
python eval_deepyeast_uq.py `
  --data-dir C:\Users\unaliuqw\deepyeast_10pct `
  --checkpoint C:\Users\unaliuqw\deepyeast_10pct\checkpoints\deepyeast_baseline_10pct\best.pt `
  --mc-samples 30 --split both --seed 42
```

→ `...\deepyeast_baseline_10pct\uq_mc_dropout\summary.json`

## 3. UQ on MaSE (same data)

Use existing MaSE / Triple-Fusion Lite checkpoint (state dict loads into `MaSELiteNet`):

```powershell
python eval_mase_uq.py `
  --data-dir C:\Users\unaliuqw\deepyeast_10pct `
  --checkpoint C:\Users\unaliuqw\deepyeast_10pct\checkpoints\triple_fusion_lite_10pct_v2\best.pt `
  --out-dir C:\Users\unaliuqw\deepyeast_10pct\checkpoints\triple_fusion_lite_10pct_v2\uq_mc_dropout `
  --mc-samples 30 --split both --seed 42 `
  --top-k 40 --soft-alpha 0.5 --branch-dropout 0.4 --head-dropout 0.5
```

## 4. Read results

From each `summary.json` → `splits.test`:

- `accuracy`
- `UAUC` (= AUROC of PE ranking errors)
- `UAUC_via_1_minus_maxprob`
- `pe_mean` (descriptive only; lower ≠ better)

Scripts: `train_deepyeast_baseline.py`, `eval_deepyeast_uq.py`, `eval_mase_uq.py`.
