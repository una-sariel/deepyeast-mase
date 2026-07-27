# MaSE-Net Lite **v5** — Full-data run (Windows)

**Goal:** beat full-data **v4 test ≈ 89.58%** with **learnable** mixture weights that do **not** collapse (unlike v3).

| Version | Idea | Full-data test |
|---------|------|----------------|
| v2 | head-mean CE, w stuck 0.25 | 89.1% |
| v3 | fused-CE, learnable (unconstrained) | 87.75% (collapsed) |
| **v4** | fused-CE, **frozen** w=0.25 | **≈89.58%** (current best) |
| **v5** | fused-CE, learnable + **anti-collapse** | **TBD** ← this run |

**Same architecture** (`MaSELiteNet`). v2/v3/v4 flags remain.

**Anti-collapse defaults (`--v5`):**

| Knob | Value |
|------|-------|
| `min_ensemble_weight` | **0.15** |
| `ensemble_entropy_weight` | **0.01** |
| `ensemble_lr` | **0.5 × backbone_lr** (not 5×) |
| UQ | **on by default** → PE → **AUROC(PE)=UAUC** |

Weight health at best.pt: prefer `min(w)≥0.10` and `max(w)≤0.55`.

---

## Files to send back

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v5\results.json
2) <deepyeast_full>\checkpoints\mase_lite_full_v5\uq_mc_dropout\summary.json
```

In `results.json` look for:

- `test.accuracy` (success if **> 0.8958**)
- `test.ensemble_weights` + `weight_health.ok`
- `uq.test_AUROC_PE` ← **primary UQ** (PE → AUROC)
- Do **not** treat `UAUC_via_1_minus_maxprob` as primary (Softmax baseline only)

---

## Step 0 — Update

```powershell
cd "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v7"
git pull
git lfs pull
.venv\Scripts\activate
```

Check:

```powershell
python -c "t=open('pytorch/train_mase_lite.py',encoding='utf-8').read(); print('v5 OK' if '--v5' in t else 'MISSING')"
```

---

## Step 1 — Train Lite v5 (full data)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full" `
  --v5 `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v5
```

Log checks:

- Progress: **`MaSELiteV5`**
- Line contains `v5 learnable anti-collapse` and `min_w=0.15`
- `w=[...]` may move, but should **not** look like v3 (`~0.78` on one head)
- End: prints **`AUROC(PE) test=...`**

`--v5` already turns on MC Dropout UQ (T=30). Skip with `--no-uq` if needed.

---

## Optional — still available

```powershell
# v4 (current SOTA ~89.58%)
python pytorch\train_mase_lite.py --data-dir "..." --freeze-ensemble --checkpoint-name mase_lite_full_v4

# v2 (89.1%)
python pytorch\train_mase_lite.py --data-dir "..." --legacy-v2-loss --checkpoint-name mase_lite_full_v2
```

---

## UQ note (PE → AUROC)

Primary metric:

```text
AUROC(PE) = UAUC = results.json → uq.test_AUROC_PE
```

Built from MC Dropout mean probs → predictive entropy per image → ROC vs incorrect/correct.
