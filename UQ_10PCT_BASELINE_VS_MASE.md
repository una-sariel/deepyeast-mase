# 10% UQ comparison: DeepYeast baseline vs MaSE

**Data:** `deepyeast_10pct` (train 6500 / val 1250 / test 1250), seed=42, MC Dropout \(T=30\).  
**Primary UQ metric:** AUROC(PE) = UAUC. PE mean = average confidence sharpness (lower → more confident), not a quality score by itself.

Code / training live in [deepyeast-mase](https://github.com/una-sariel/deepyeast-mase). This workspace keeps **result snapshots + notes**.

Raw JSON: `docs/mase_results/uq_10pct/`.

---

## Models

| Name | What |
|------|------|
| **DeepYeast baseline** | Official DeepYeastNet (Pärnamaa & Parts), trained on 10% |
| **MaSE v2** | Triple-Fusion Lite / MaSE Lite v2 (`triple_fusion_lite_10pct_v2`) |
| **MaSE confident** | Finetune from v2: `label_smoothing=0`, branch/head dropout `0.25/0.3`, `--legacy-v2-loss`, 20 epochs |

Architecture of MaSE **unchanged** for confident; recipe + `--resume` only.

---

## Test results (MC Dropout UQ)

| Model | Acc | AUROC(PE) | AUROC(1−maxp) | PE mean |
|-------|-----|-----------|---------------|---------|
| DeepYeast baseline | 81.3% | **0.883** | 0.890 | 0.58 |
| MaSE v2 | **86.0%** | 0.872 | 0.876 | 0.92 |
| MaSE confident | 85.2% | 0.877 | 0.876 | **0.39** |

### vs baseline (test)

- MaSE v2: Acc **+4.7 pp**, PE mean higher (less confident), AUROC(PE) slightly lower.
- MaSE confident: Acc **+3.9 pp**, PE mean **lower than baseline** (more confident), AUROC(PE) close to baseline.

---

## Temperature sweep (MaSE confident, no retrain)

Applied \(p \propto \exp(\log p / T)\). Deterministic Acc invariant to \(T\).

**Test:**

| T | det Acc | det PE | det maxp | mc Acc | mc PE | mc UAUC |
|---|---------|--------|----------|--------|-------|---------|
| 1.0 | 85.12% | 0.369 | 0.873 | 85.12% | 0.388 | 0.879 |
| 0.9 | 85.12% | 0.324 | 0.887 | 84.96% | 0.343 | 0.881 |
| 0.8 | 85.12% | 0.281 | 0.900 | 84.96% | 0.301 | 0.880 |
| 0.7 | 85.12% | 0.241 | 0.913 | 85.12% | 0.260 | 0.879 |
| 0.5 | 85.12% | 0.165 | 0.939 | 85.12% | 0.186 | 0.878 |

**Practical:** deploy with **T ≈ 0.7–0.8** for higher confidence without Acc drop; UQ metrics nearly flat.

Full sweep: `docs/mase_results/uq_10pct/mase_confident/temp_sweep.json`.

---

## Related docs

- `docs/BASELINE_UQ_COMPARE.md` — how to reproduce baseline vs MaSE UQ
- `docs/CONFIDENT_RECIPE_10PCT.md` — confident finetune recipe
- `docs/CNN_UQ_concepts_bilingual.md` / `docs/Native_Multiclass_concepts_bilingual.md` — concepts
- Scripts (run from **deepyeast-mase/pytorch** for MaSE deps): `train_deepyeast_baseline.py`, `eval_deepyeast_uq.py`, `eval_mase_uq.py`, `temp_sweep_mase.py` (copies also under `masked/pytorch/`)
