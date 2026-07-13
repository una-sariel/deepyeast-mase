# MaSE-Net Lite v2 — Collaborator run notes

Use this if training fails at `torch.load(...)` / `load_partial_state` when initializing PLCNN / selector weights.

**You should run the v2 recipe** (already validated on another machine).

---

## Cause

v2 auto-loads two init checkpoints from Git LFS:

- `artifacts/checkpoints_5pct/plcnn_triple/best.pt` (~28 MB)
- `artifacts/checkpoints_5pct/masked_v3k60_pytorch/best.pt` (~44 MB)

A normal `git clone` may only download tiny pointer files. You must pull LFS content.

---

## Fix (required)

Install [Git LFS](https://git-lfs.com) if needed, then:

```powershell
cd <your deepyeast-mase folder>
git lfs install
git lfs pull

dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt
```

Those two `.pt` files should be **~28MB and ~44MB** (not ~1KB pointers).

---

## Run v2 on full data

```powershell
python pytorch\train_mase_lite.py `
  --data-dir "<your real deepyeast_full path>" `
  --epochs 60 --patience 15 --batch-size 64 `
  --top-k 40 --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 --label-smoothing 0.1 `
  --seed 42 --checkpoint-name mase_lite_full_v2
```

Replace `--data-dir` with your actual `deepyeast_full` folder (must contain `metadata.csv` + images).

**Do not use `--no-init`** — we want the same validated v2 setup.

GPU is strongly recommended for full data.

---

## After training, please send back

```text
<deepyeast_full>\checkpoints\mase_lite_full_v2\results.json
```

Useful fields: `best_val_accuracy`, `test.accuracy`, `test.mask_coverage`, `best_epoch`.

Also see `FULL_DATA_QUICKSTART.md` for the full Windows guide.
