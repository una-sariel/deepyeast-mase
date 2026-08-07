# v7 — You just finished Phase 2.5: run this

**For you if:** full-data Phase 2.5 is done (test ≈ **89.07%**) and Phase 1 `best.pt` is on disk.

**Your paths (v9):**

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast-mase"
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast_full"
```

**Repo:** https://github.com/una-sariel/deepyeast-mase

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

If all three exist → **no Phase 1 retrain**, **no data re-download**.

---

## 2. TTA on Phase 2.5 checkpoint (~15 min, no retrain)

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt" `
  --split test `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase25\tta_eval"
```

Check the last line, e.g.:

```text
test: baseline=0.8907  tta=0.894x  gain=+0.3xpp
```

---

## 3. Train v7 (~10–45 min)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v7 `
  --seed 42
```

Progress bar: **`MaSELiteV7`**  
Output: `$DATA\checkpoints\mase_lite_full_v7\`

---

## 4. TTA on v7 checkpoint (~15 min)

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v7\best.pt" `
  --split both `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v7\tta_eval"
```

---

## 5. Send back these 3 files

```text
checkpoints\mase_lite_full_v6_phase25\tta_eval\summary.json
checkpoints\mase_lite_full_v7\results.json
checkpoints\mase_lite_full_v7\tta_eval\summary.json
```

| Table row | Read this field |
|-----------|-----------------|
| Phase 2.5 | phase25 `results.json` → `test.accuracy` |
| Phase 2.5 + TTA | phase25 `tta_eval/summary.json` → `test_tta_accuracy` |
| v7 + TTA | v7 `tta_eval/summary.json` → `test_tta_accuracy` |

Report **train** and **TTA** as separate rows (do not merge into one number).

---

## Order

```text
git pull  →  TTA(phase25)  →  train v7  →  TTA(v7)  →  send 3 json files
```

**You do NOT need to:** re-download `deepyeast_full`, re-train Phase 1, or download any `.pt` from GitHub.
