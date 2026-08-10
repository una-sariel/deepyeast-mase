# MaSE-Net Lite v7 — tuned Phase 2.5 + TTA

**v7** = lower head/ensemble LR Phase 2.5 fine-tune + **TTA** at eval (report train and TTA separately).

**Repo:** https://github.com/una-sariel/deepyeast-mase

If you **just finished Phase 2.5** (test ≈ 89.07%) on the v9 machine, follow **Steps 0–5** below.  
No data re-download. No Phase 1 retrain.

---

## Paths (v9)

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast-mase"
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast_full"
```

---

## 0. Update code (~1 min)

```powershell
cd $REPO
git pull
.venv\Scripts\activate
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('v7 OK' if '--v7' in t else 'git pull again')"
```

---

## 1. Confirm local files (~30 sec)

```powershell
dir "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt"
dir "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt"
dir "$DATA\checkpoints\mase_lite_full_v6_phase25\results.json"
```

If all three exist → continue.

---

## 2. TTA on Phase 2.5 (~15 min, no retrain)

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt" `
  --split test `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase25\tta_eval"
```

Example log line:

```text
test: baseline=0.8907  tta=0.894x  gain=+0.3xpp
```

---

## 3. Train v7 (~10–45 min)

Resumes Phase-1 `best.pt` automatically.

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v7 `
  --seed 42
```

Progress bar: **`MaSELiteV7`**  
Output: `$DATA\checkpoints\mase_lite_full_v7\`

### v7 defaults (vs Phase 2.5)

| Knob | Phase 2.5 | **v7** |
|------|-----------|--------|
| `phase25_head_lr` | 2e-5 | **1e-5** |
| `ensemble_lr_ratio` | 0.25 | **0.15** |
| `patience` | 5 | **3** |
| checkpoint | `mase_lite_full_v6_phase25` | **`mase_lite_full_v7`** |

---

## 4. TTA on v7 (~15 min)

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v7\best.pt" `
  --split both `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v7\tta_eval"
```

Optional one-shot (train + TTA): `--v7 --with-tta`

### Related paper (TTA)

We use the standard practice of averaging class probabilities over geometric transforms of each test image (flips / 90° rotations). For a focused study of this **test-time augmentation** setup (and of simple averaging as the default aggregator), see:

- Shanmugam, Blalock, Balakrishnan, Guttag. **Better Aggregation in Test-Time Augmentation.** ICCV 2021.  
  [CVF open access](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html) · [arXiv:2011.11156](https://arxiv.org/abs/2011.11156)

Our `eval_mase_tta.py` implements the common **uniform mean** over views (their baseline aggregator), not their learned AugTTA/ClassTTA weights.

---

## 5. Send back these 3 files

```text
checkpoints\mase_lite_full_v6_phase25\tta_eval\summary.json
checkpoints\mase_lite_full_v7\results.json
checkpoints\mase_lite_full_v7\tta_eval\summary.json
```

| Table row | Field |
|-----------|-------|
| Phase 2.5 | phase25 `results.json` → `test.accuracy` |
| Phase 2.5 + TTA | phase25 `tta_eval/summary.json` → `test_tta_accuracy` |
| v7 (train) | v7 `results.json` → `test.accuracy` |
| v7 + TTA | v7 `tta_eval/summary.json` → `test_tta_accuracy` |

Keep **train** and **TTA** as separate rows.

---

## Order

```text
git pull  →  TTA(phase25)  →  train v7  →  TTA(v7)  →  send 3 json files
```

**Not needed:** re-download `deepyeast_full`, re-train Phase 1, or download any `.pt` from GitHub.

See also [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md) for the original Phase 2.5 recipe.
