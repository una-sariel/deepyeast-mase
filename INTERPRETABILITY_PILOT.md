# Interpretability & UQ pilot scripts

Run on **full data** after Phase1/Phase2 (or on `deepyeast_5pct` for a quick smoke test).

**Prerequisites:** `pip install matplotlib` (visualization only).

Set paths (edit for your machine):

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v8"
$DATA = "$REPO\deepyeast_full"
$P1   = "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt"
$P2   = "$DATA\checkpoints\mase_lite_full_v6\best.pt"   # after Phase2
cd $REPO
```

---

## 1. Mask + pred visualization (100 test images, by class)

```powershell
python pytorch\visualize_mase_interpret.py `
  --data-dir $DATA `
  --checkpoint $P1 `
  --n-samples 100 `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase1\interpret_test100"
```

**Output:** `<out-dir>/<class_name>/*.png` + `manifest.json`  
Each PNG: mCherry | GFP | GFP + red mask overlay, title = true/pred.

Repeat with `$P2` after Phase2 to compare learned weights on hard cases.

---

## 2. Mask ablation table (same checkpoint, no retrain)

```powershell
python pytorch\eval_mase_mask_ablation.py `
  --data-dir $DATA `
  --checkpoint $P1 `
  --split test `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase1\mask_ablation"
```

| mode | meaning |
|------|---------|
| `learned` | trained ViT selector (default MaSE) |
| `none` | no masking (full image) |
| `random` | random top-k patches, same k |

**Output:** `ablation.json`, `ablation_table.md`

**Paper story:** if `learned` > `none` and `learned` > `random`, the selector adds real signal.

---

## 3. UQ comparison (Phase1 vs Phase2 vs baseline)

```powershell
python pytorch\eval_uq_compare.py `
  --data-dir $DATA `
  --split test `
  --mc-samples 30 `
  --out-dir "$DATA\checkpoints\uq_compare_v6" `
  --tag phase1 --checkpoint $P1 `
  --tag phase2 --checkpoint $P2 `
  --tag baseline --deepyeast-checkpoint "$DATA\checkpoints\deepyeast_baseline_full\best.pt"
```

If baseline checkpoint missing, omit the last two flags and compare Phase1 vs Phase2 only.

**Output:** `comparison.json`, `comparison_table.md` with:

- **UAUC (PE)** — ranks incorrect predictions (higher = better UQ)
- **AURC** — area under risk-coverage (lower = better selective prediction)
- **Acc@60/80/90%** — accuracy on the most confident fraction

Also works per-checkpoint via existing script:

```powershell
python pytorch\eval_mase_uq.py `
  --data-dir $DATA `
  --checkpoint $P1 `
  --mc-samples 30 --split test `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase1\uq_mc_dropout"
```

---

## Is this a good paper direction? (10% pilot snapshot)

From workspace notes (`docs/UQ_10PCT_BASELINE_VS_MASE.md`, 10% split):

| Model | Acc | UAUC |
|-------|-----|------|
| DeepYeast baseline | 81.3% | **0.883** |
| MaSE v2 | **86.0%** | 0.872 |
| MaSE confident | 85.2% | 0.877 |

**Takeaway:** MaSE wins on **accuracy**; **UAUC** can trail baseline on small splits but **Acc@coverage / AURC** often favor MaSE after tuning (see `CONFIDENT_RECIPE_10PCT.md`). Full-data UQ compare (script 3) confirms whether Phase2 improves selective prediction even if acc is flat.

**Interpretability angle:** mask ablation + class-wise overlays are independent of +0.5% acc — strong for localization biology narrative.

---

## Send back after running

```text
1) ...\v6_phase1\interpret_test100\manifest.json  (+ a few PNGs)
2) ...\v6_phase1\mask_ablation\ablation_table.md
3) ...\uq_compare_v6\comparison_table.md
```
