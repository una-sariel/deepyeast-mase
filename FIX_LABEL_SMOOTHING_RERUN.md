# Pull fix and re-run (Windows)

If you hit this error:

```text
TypeError: run_epoch() got multiple values for keyword argument 'label_smoothing'
```

it is a small bug in `pytorch/train_mase_lite.py` (already fixed on GitHub). Follow the steps below.

---

## 1. Update the repo

Open **PowerShell** in your MaSE folder (example path — change to yours):

```powershell
cd "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v6"
```

If this folder is a **git clone** of https://github.com/una-sariel/deepyeast-mase :

```powershell
git pull
git lfs install
git lfs pull
```

You need at least commit **`8e291cf`** (fix for `label_smoothing`).

Check:

```powershell
git log -1 --oneline
```

Expected something like:

```text
8e291cf Fix duplicate label_smoothing kwarg in val/test run_epoch.
```

(or any newer commit after that).

### If you downloaded a ZIP (no git)

1. Download / clone the latest repo again from GitHub, **or**
2. Replace only this file with the latest version from GitHub:
   - `pytorch/train_mase_lite.py`

Then continue with step 2.

---

## 2. Confirm init checkpoints (Git LFS)

```powershell
dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt
```

| File | Expected size |
|------|----------------|
| `plcnn_triple\best.pt` | ~**28 MB** |
| `masked_v3k60_pytorch\best.pt` | ~**44 MB** |

If either is ~**1 KB**, run `git lfs pull` again (or the init `torch.load` error will come back).

---

## 3. Activate env and train (Lite v3, recommended)

```powershell
.venv\Scripts\activate

python pytorch\train_mase_lite.py `
  --data-dir "<YOUR real deepyeast_full path>" `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v3
```

Replace `--data-dir` with your actual full dataset folder (must contain `metadata.csv` + images).

**Do not** add `--no-init` unless init files are missing on purpose.

Startup log should look like:

```text
Init PLCNN branches: ... tensors from ...
Init selector: ... tensors from ...
MaSE Lite ... fused-CE aux=0.5 distill=0.1 ...
```

Then epochs should start without the `label_smoothing` TypeError.

---

## 4. Optional: reproduce previous v2 (89.1% test)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<YOUR deepyeast_full>" `
  --legacy-v2-loss `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2
```

---

## 5. After training, please send back

```text
<deepyeast_full>\checkpoints\mase_lite_full_v3\results.json
```

Useful fields: `best_val_accuracy`, `test.accuracy`, `test.ensemble_weights`, `best_epoch`.

---

## What caused the bug (for reference)

Val/test called `run_epoch(..., label_smoothing=0.0, **epoch_kw)` while `epoch_kw` already contained `label_smoothing`. Python then raised “multiple values for keyword argument”. Fixed by merging: `**{**epoch_kw, "label_smoothing": 0.0}`.

More detail: [COLLABORATOR_V2_RUN.md](COLLABORATOR_V2_RUN.md) · [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)
