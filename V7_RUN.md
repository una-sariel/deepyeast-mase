# MaSE-Net Lite v7 — tuned Phase 2.5 + TTA

**v7** = **tuned TP-AHF Phase 2.5** (lower head LR, slower ensemble LR, tighter early stop) + optional **TTA** at eval time.

| Step | What | Output folder |
|------|------|----------------|
| Prerequisite | Phase 1 ([V6_RUN.md](V6_RUN.md) Step 2) | `mase_lite_full_v6_phase1` |
| **Train v7** | `--v7` from Phase-1 `best.pt` | `mase_lite_full_v7` |
| **TTA** | `eval_mase_tta.py` or `--with-tta` | `mase_lite_full_v7/tta_eval` |

**Repo:** https://github.com/una-sariel/deepyeast-mase  
See also [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md) (default Phase 2.5 hyperparams).

---

## What to send back

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v7\results.json
2) <deepyeast_full>\checkpoints\mase_lite_full_v7\tta_eval\summary.json
```

Report **both** rows in your table:

| Row | Field |
|-----|-------|
| v7 (train) | `results.json` → `test.accuracy` |
| v7 + TTA | `tta_eval/summary.json` → `test_tta_accuracy` |

---

## Prerequisites

1. Full data `deepyeast_full` — [V6_RUN.md](V6_RUN.md) Step 1.
2. Phase 1 complete with `best.pt` — [V6_RUN.md](V6_RUN.md) Step 2.
3. `git pull` (needs `--v7` and `eval_mase_tta.py`).

```powershell
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast_full"
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast-mase"
cd $REPO
git pull
.venv\Scripts\activate
```

---

## Step 1 — Train v7 (~10–45 min)

Auto-resumes Phase-1 `best.pt`.

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v7 `
  --seed 42
```

### v7 defaults (vs v6 Phase 2.5)

| Knob | v6 Phase 2.5 | **v7** |
|------|----------------|--------|
| `phase25_head_lr` | 2e-5 | **1e-5** |
| `ensemble_lr_ratio` | 0.25 | **0.15** |
| `patience` | 5 | **3** |
| checkpoint | `mase_lite_full_v6_phase25` | **`mase_lite_full_v7`** |
| progress bar | `MaSELiteV6P25` | **`MaSELiteV7`** |

Everything else same as Phase 2.5 (15 epochs max, min_w=0.20, heads+w trainable, branches frozen).

Output:

```text
$DATA\checkpoints\mase_lite_full_v7\
  best.pt
  results.json
```

---

## Step 2 — TTA (~15–30 min, no retrain)

### Option A — separate script (any checkpoint)

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v7\best.pt" `
  --split both `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v7\tta_eval"
```

### Option B — run TTA right after training

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v7 `
  --with-tta `
  --seed 42
```

TTA modes:

| `--tta-mode` | Views |
|--------------|-------|
| `flip` | 4 (identity + H/V/HV flip) |
| `flip_rot` | **8** (+ 90°/180°/270°, default) |

---

## Also run TTA on existing Phase 2.5 checkpoint

If you already have `mase_lite_full_v6_phase25/best.pt` (test **89.07%**):

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt" `
  --split test `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase25\tta_eval"
```

This can run **in parallel** with Step 1 v7 training.

---

## Expected log lines

**Train v7:**

```text
MaSELiteV7
  v7 heads_lr=1e-05 ens_lr=0.00015 ...
  ep02  val=0.889  wh=ok  w=[0.24,0.25,0.25,0.26]
Test:     0.89xx
```

**TTA:**

```text
test: baseline=0.8907  tta=0.894x  gain=+0.3xpp
```

---

## Paper table (example)

| Method | Test acc |
|--------|----------|
| Phase 2.5 (v6) | 89.07% |
| **v7 train** | TBD |
| Phase 2.5 + TTA | TBD |
| **v7 + TTA** | TBD |

Keep train and TTA as **separate rows** (do not merge into one number without noting TTA).
