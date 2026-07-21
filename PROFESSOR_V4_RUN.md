# MaSE-Net Lite v4 — Full-data run (Windows / professor)

**Goal:** beat full-data **v2 test 89.1%** without deleting v2/v3.

| Version | Idea | Full-data test (known) |
|---------|------|------------------------|
| **v2** | head-mean CE, weights stuck at 0.25 | **89.1%** (baseline to beat) |
| **v3** | fused-CE, **learnable** weights | **87.75%** (collapsed ~0.78 on one head) |
| **v4** | fused-CE, **frozen uniform** w=0.25 | **TBD** ← run this |

**Same architecture** as v2/v3 (`MaSELiteNet`). Only the training recipe changes.

**Repo:** https://github.com/una-sariel/deepyeast-mase  

---

## Step 0 — Update code (every time)

Open **PowerShell** in your MaSE folder (example — change to yours):

```powershell
cd "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v6"
```

```powershell
git pull
git lfs install
git lfs pull
```

Check latest commit includes **v4** (`--freeze-ensemble`):

```powershell
git log -1 --oneline
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('v4 OK' if 'mase_lite_full_v4' in t else 'MISSING v4 — git pull again')"
```

### Verify Git LFS init weights (~28 MB + ~44 MB, not ~1 KB)

```powershell
dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt
```

If files are ~1 KB, run `git lfs pull` again. See also [COLLABORATOR_V2_RUN.md](COLLABORATOR_V2_RUN.md).

Activate venv:

```powershell
.venv\Scripts\activate
```

---

## Step 1 — Recommended: train Lite **v4** (full data)

Replace `--data-dir` with your real `deepyeast_full` path (same as v2/v3 runs).

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "D:\UG Research\DeepYeast\Qiwu\...\deepyeast_full" `
  --freeze-ensemble `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --aux-head-weight 0.5 `
  --distill-weight 0.1 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v4
```

### What you should see in the log

- Progress bar name: **`MaSELiteV4`**
- Line like: `fused-CE ... | v4 frozen_uniform w=0.25`
- Every epoch: **`w=[0.25,0.25,0.25,0.25]`** (must stay uniform; if it drifts, stop and `git pull`)

Wall time: similar to v3 (~1–2 h on GPU).

### Output folder

```text
<deepyeast_full>\checkpoints\mase_lite_full_v4\
  best.pt
  results.json
  meta.json
```

---

## Step 2 — After training, please send back

```text
<deepyeast_full>\checkpoints\mase_lite_full_v4\results.json
```

Useful fields:

- `method` → should be `mase_lite_v4`
- `best_val_accuracy`, `best_epoch`
- `test.accuracy` ← **success if > 0.891**
- `test.ensemble_weights` ← should be ~`[0.25,0.25,0.25,0.25]`
- `test.mask_coverage` ← ~0.625

---

## Optional — still available (not deleted)

### A) Reproduce v2 (89.1% reference)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<deepyeast_full>" `
  --legacy-v2-loss `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v2
```

### B) Re-run v3 learnable (known collapse on full data)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<deepyeast_full>" `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --min-ensemble-weight 0.05 `
  --seed 42 `
  --checkpoint-name mase_lite_full_v3
```

Do **not** use B for the “beat v2” attempt.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Tiny `.pt` / load error | `git lfs pull` |
| `label_smoothing` TypeError | [FIX_LABEL_SMOOTHING_RERUN.md](FIX_LABEL_SMOOTHING_RERUN.md) |
| `w` not all 0.25 on v4 | Need `--freeze-ensemble`; `git pull` latest |
| Wrong folder | Check `checkpoint-name mase_lite_full_v4` |

Full install / clone: [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)
