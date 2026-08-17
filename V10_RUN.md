# MaSE Lite **v10** — Shared-Fixed-Region Mask (SFRM) 5% pilot

**v10** is the shared-region version of the professor sketch: each train epoch samples one \(S\times S\) window and zeros that region on **all** training images. Val/test stay unmasked. Optional eval: average softmax over the full image + \(M\) random windows.

`--v10` is an alias for `--sfrm`. Contrast with **v9 RSB** (per-image random top-k, not one shared window).

First 5% recipe: **v4 frozen \(w=0.25\)** + SFRM, **learned selector bypassed** (so SFRM is the only mask).

Compare to 5% v4 with selector: test **85.87%** (`results/v4/5pct_results.json`).

---

## 5% train (this machine)

```powershell
$PY = "C:\Users\unaliuqw\dp\.venv\Scripts\python.exe"
$DATA = "C:\Users\unaliuqw\deepyeast_5pct"
cd C:\Users\unaliuqw\deepyeast-mase

# once: build 5% from ~/.deepyeast/cache
& $PY prepare_deepyeast_subset.py --fraction 0.05 --out-dir $DATA --seed 42

& $PY pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v10 --sfrm-size 24 --sfrm-windows 7 `
  --freeze-ensemble `
  --epochs 60 --patience 15 --batch-size 64 `
  --seed 42 `
  --checkpoint-name mase_lite_5pct_v10
```

Progress bar: **`MaSELiteV10`**. Log line includes `sfrm=(r,c,S)`.

Send back:

```text
1) <deepyeast_5pct>\checkpoints\mase_lite_5pct_v10\results.json
2) <deepyeast_5pct>\checkpoints\mase_lite_5pct_v10\sfrm_vote\summary.json
```

---

## Defaults

| Knob | Value |
|------|--------|
| Recipe | v4 fused-CE, frozen \(w=0.25\) |
| Selector | **bypass** (identity mask) |
| SFRM size | 24 |
| When | one window / train epoch, all train images |
| Val / test | no SFRM |
| Vote | 7 windows + full image |
| Init | PLCNN 5% branches; selector init skipped |

`--sfrm-keep-selector` stacks learned top-k on top of SFRM (not the first 5% run).

---

## 5% results (this machine, 2026-08-13)

JSON: `results/v10/5pct_results.json`, `results/v10/5pct_vote_summary.json`

| Metric | SFRM | 5% v4 (selector) |
|--------|------|------------------|
| Best val | **83.22%** @ ep49 | 82.7% |
| Unmasked test | **85.09%** | **85.87%** |
| 7-window vote test | 84.27% (−0.64pp vs vote-eval baseline 84.91%) | — |
| Weights | frozen 0.25×4 | frozen 0.25×4 |
| Val mask | 1.000 (bypass) | learned top-k |
| Elapsed | 257.8 min CPU | — |

Trainer unmasked test (batch-mean) is 85.09%; the vote script’s own full-image baseline is 84.91% (micro-average). Report **unmasked 85.09%** as the SFRM accuracy; do not use vote.

**Readout:** healthy (not collapsed; `sfrm=(r,c,24)` changed every epoch). Unmasked test is **−0.78pp vs 5% v4**. Shared-region occlusion did not beat the learned selector on 5%. Multi-window vote **hurt**. Do not promote SFRM as a 90% accuracy weapon; optional full-data run is a method story only.

---

## 5% stacked (selector + S=16, 2026-08-13)

JSON: `results/v10/5pct_sel_results.json`

Same v4 frozen-\(w\) recipe, but **keep selector** and smaller window (`--sfrm-keep-selector --sfrm-size 16 --sfrm-windows 0`).

| Metric | SFRM bypass S=24 | SFRM+selector S=16 | 5% v4 |
|--------|------------------|--------------------|-------|
| Best val | 83.22% @ ep49 | **83.68%** @ ep49 | 82.7% |
| Unmasked test | 85.09% | **85.71%** | **85.87%** |
| Val mask | 1.000 | 0.625 | learned top-k |
| Elapsed | 257.8 min CPU | 264.9 min CPU | — |

**Readout:** stacking recovered **+0.62pp** vs bypass SFRM, still **−0.16pp vs v4**. Val beat v4; test did not. Stop iterating SFRM for 5% accuracy.
