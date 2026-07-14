# MaSE-Net — Collaborator run notes

Use this if training fails at init (`torch.load`, checkpoint too small, etc.).

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

## Recommended: Lite v3 (fused-CE)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your real deepyeast_full path>" `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v3
```

v3 defaults include `--aux-head-weight 0.5 --distill-weight 0.1 --min-ensemble-weight 0.05`.

Watch the log: `w=[...]` should **move away from 0.25** during training.

---

## Reproduce v2 (89.1% full-data run)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your deepyeast_full>" `
  --legacy-v2-loss `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2
```

---

## After training, please send back

```text
<deepyeast_full>\checkpoints\mase_lite_full_v3\results.json
```

Useful fields: `best_val_accuracy`, `test.accuracy`, `test.ensemble_weights`, `best_epoch`.

If you see `TypeError: ... multiple values for keyword argument 'label_smoothing'`, follow **[PROFESSOR_RERUN.md](PROFESSOR_RERUN.md)** (`git pull` then re-run).

Full guide: [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md)
