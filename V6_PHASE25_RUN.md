# MaSE-Net Lite v6 — Phase 2.5 (heads + weights fine-tune)

**TP-AHF Phase 2.5** = resume Phase-1 checkpoint, then train **`ensemble_logits` + 4 progressive heads** while **selector + PLCNN branches stay frozen**.

| Phase | Trains | Frozen | Checkpoint folder |
|-------|--------|--------|-------------------|
| **Phase 1** | full model, frozen w=0.25 | — | `mase_lite_full_v6_phase1` |
| **Phase 2** | `ensemble_logits` only | everything else | `mase_lite_full_v6` |
| **Phase 2.5** | `ensemble_logits` + `heads` | selector + branches | `mase_lite_full_v6_phase25` |

**Why:** Phase 2 (+0.06pp) shows global weight tuning alone is too weak. Phase 2.5 lets strong heads (H2/H3) co-adapt with mixture weights.

**Repo:** https://github.com/una-sariel/deepyeast-mase  
See also [V6_RUN.md](V6_RUN.md) (Phase 1/2), [V4_RUN.md](V4_RUN.md).

---

## What to send back

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v6_phase25\results.json
```

Optional: `mase_lite_full_v6_phase25\best.pt` (large; keep on GPU machine if possible).

---

## Prerequisites

1. **Full data** at `deepyeast_full` (90,000 images). See [V6_RUN.md](V6_RUN.md) Step 1.
2. **Phase 1 complete** ([V6_RUN.md](V6_RUN.md) Step 2) **and** a weight file to resume from (`mase_lite_full_v6_phase1\best.pt`, or Phase 2 shortcut below).
3. Latest code with `--v6-phase25`:

```powershell
cd path\to\deepyeast-mase
git pull
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('phase25 OK' if '--v6-phase25' in t else 'git pull for phase25')"
```

Set data path (edit for your machine):

```powershell
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v8\deepyeast_full"
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v8\deepyeast-mase"
cd $REPO
.venv\Scripts\activate
```

---

## Start without Phase-1 `best.pt`

Phase 1 finished and you sent `results.json`, but **`mase_lite_full_v6_phase1\best.pt` was not kept**. Weights cannot be rebuilt from JSON — use one of the paths below.

### Check what you still have

```powershell
dir "$DATA\checkpoints\mase_lite_full_v6_phase1\"
dir "$DATA\checkpoints\mase_lite_full_v6\best.pt"
```

### Path A — Phase 2 `best.pt` exists (shortcut, ~30–45 min)

Phase 2 training **only updates `ensemble_logits`**; selector, branches, and heads are the same as at the end of Phase 1. You can start Phase 2.5 from the Phase 2 checkpoint.

Read Phase-1 val gate from your existing `results.json`:

```powershell
python -c "import json; d=json.load(open(r'$DATA\checkpoints\mase_lite_full_v6_phase1\results.json')); print('phase1_val_gate=', d['best_val_accuracy'])"
```

Run Phase 2.5 (replace `0.8875` with the printed value):

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase25 `
  --resume "$DATA\checkpoints\mase_lite_full_v6\best.pt" `
  --phase1-checkpoint-name mase_lite_full_v6_phase1 `
  --phase1-val-acc 0.8875 `
  --seed 42
```

Default output folder is still `mase_lite_full_v6_phase25\`.

### Path B — Re-run Phase 1 (~2 h)

If there is **no** usable `.pt` under Phase 1 or Phase 2, re-run [V6_RUN.md Step 2](V6_RUN.md) (same hyperparams, `--seed 42`).  
When training finishes, **keep** `mase_lite_full_v6_phase1\best.pt`, then run Phase 2.5 normally:

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase25 `
  --seed 42
```

---

## Run Phase 2.5 (~30–45 min)

Auto-resumes Phase-1 checkpoint from [V6_RUN.md](V6_RUN.md) Step 2.

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase25 `
  --seed 42
```

Default output:

```text
$DATA\checkpoints\mase_lite_full_v6_phase25\
  best.pt
  results.json
  meta.json
```

### Phase 2.5 defaults (built into `--v6-phase25`)

| Knob | Value |
|------|-------|
| Trainable | `ensemble_logits` + **4 heads** |
| Frozen | `selector`, `branch_vgg/resnet/densenet` |
| `phase25_head_lr` | **2e-5** |
| `ensemble_lr` | **0.25 × backbone_lr** (= 2.5e-4) |
| `min_ensemble_weight` | **0.20** |
| `ensemble_entropy_weight` | **0.02** |
| Temperature | **1.5 → 1.0** linear anneal |
| epochs / patience | **15 / 5** |
| aux / distill / sparsity | **off** |
| Progress bar | **`MaSELiteV6P25`** |

### Check log

- Lines show **`wh=ok`** or **`wh=BAD`**
- **`best.pt` saved only if** val ≥ Phase-1 best val **and** `wh=ok`
- `method` in results → `mase_lite_v6_phase25`

### Success criteria

| Metric | Target |
|--------|--------|
| `test.accuracy` | **> Phase-1 test**; aim **≥ 0.895** |
| `weight_health.ok` | **true** |
| `max(w)` | **≤ 0.40** |

---

## Overrides

Custom Phase-1 resume path:

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase25 `
  --resume "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt" `
  --phase1-checkpoint-name mase_lite_full_v6_phase1 `
  --seed 42
```

Tighter head LR:

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase25 `
  --phase25-head-lr 1e-5 `
  --ensemble-lr-ratio 0.15 `
  --seed 42
```

---

## After Phase 2.5

| Phase 2.5 test | Next step |
|----------------|-----------|
| **≥ 90%** | Done — report numbers |
| **89.5% – 90%** | TTA eval on checkpoint (no retrain) |
| **< 89.5%** | 3-seed ensemble or ID-Gate (separate experiments) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `--v6-phase25 needs Phase-1 best.pt` | See § *Start without Phase-1 `best.pt`* above |
| All epochs `wh=BAD` | `--min-ensemble-weight 0.22` or `--ensemble-lr-ratio 0.15` |
| Tiny `.pt` / load error | `git lfs pull` on repo |

---

## Paper one-liner

> Phase 2.5 extends TP-AHF: after uniform Phase-1 training, we jointly fine-tune constrained mixture weights and progressive classification heads while keeping the masked backbone fixed, enabling head–fusion co-adaptation beyond global weight calibration alone.
