# MaSE-Net — Collaborator run notes

Use this for quick commands and LFS / init failures.

**Detailed v4 steps (recommended):** [V4_RUN.md](V4_RUN.md)

---

## Before every run

```powershell
cd <your deepyeast-mase folder>
git pull
git lfs install
git lfs pull
```

Verify init files are **real** checkpoints (not LFS pointers):

```powershell
dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt
```

Expected sizes: **~28 MB** and **~44 MB**. If you see **~1 KB**, run `git lfs pull` again.

---

## Recommended: Lite **v4** (fused-CE + frozen uniform weights)

Candidate to beat full-data **v2 89.1%**. Log must keep `w=[0.25,0.25,0.25,0.25]`.

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your real deepyeast_full path>" `
  --freeze-ensemble `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --aux-head-weight 0.5 --distill-weight 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v4
```

Send back: `<deepyeast_full>\checkpoints\mase_lite_full_v4\results.json`

---

## Still available (not removed)

### Reproduce v2 (89.1% full-data run)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your deepyeast_full>" `
  --legacy-v2-loss `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2
```

### Lite v3 learnable (full-data collapsed to ~87.75%; reference only)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your deepyeast_full>" `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --min-ensemble-weight 0.05 `
  --seed 42 --checkpoint-name mase_lite_full_v3
```

---

## After training

Useful fields in `results.json`: `method`, `best_val_accuracy`, `test.accuracy`, `test.ensemble_weights`, `best_epoch`.

If you see `TypeError: ... multiple values for keyword argument 'label_smoothing'`, follow **[FIX_LABEL_SMOOTHING_RERUN.md](FIX_LABEL_SMOOTHING_RERUN.md)**.

Full guide: [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)
