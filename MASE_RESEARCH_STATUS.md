# MaSE-Net research status (synced to workspace)

**Date:** 2026-07-27  
**Code repo (source of truth for training):** https://github.com/una-sariel/deepyeast-mase  
**This workspace:** concepts, notes, and result snapshots for migration / reporting.

---

## 1. Method summary

**MaSE-Net Lite** ≈ Patch region selector (top-k mask) + PLCNN (VGG/ResNet/DenseNet) + 4-head progressive ensemble.  
Same architecture across v2–v5; recipes differ mainly in **loss** and **ensemble weight training**.

| Version | Recipe | Full-data test | Notes |
|---------|--------|----------------|-------|
| Keras baseline | official DeepYeastNet | ~**88.4%** | reference |
| **v2** | mean per-head CE; w stuck at 0.25 | **89.1%** | first full SOTA |
| **v3** | fused-CE; learnable w (`min_w=0.05`, fast ens_lr) | **87.75%** | collapsed ~`[0.06,0.06,0.78,0.10]` |
| **v4** | fused-CE; **frozen** w=0.25 | **≈89.58%** | **current accuracy SOTA** |
| **v5** | fused-CE; learnable + anti-collapse (`min_w=0.15`, `ent_w=0.01`, ens_lr=0.5×bb) + PE→AUROC | TBD | running / for professor |

Flags in `deepyeast-mase`:

- v2: `--legacy-v2-loss`
- v4: `--freeze-ensemble`
- v5: `--v5` (UQ on by default)

Docs in MaSE repo: `V4_RUN.md`, `V5_RUN.md`, `UQ_MC_DROPOUT.md`, `ARCHITECTURE.md`.

---

## 2. Why v3 failed on full data / why v4 worked

- Fused CE gives gradients to `ensemble_logits` → one head can dominate.
- Full data amplified collapse; diversity of 4-head average was lost → below v2.
- **v4** keeps fused CE but **freezes** mixture at uniform 0.25 → recovered and beat v2 (~+0.5 pp).

---

## 3. 5% pilots (seed=42)

| Method | Test | Source |
|--------|------|--------|
| Lite v2 | **85.03%** | `docs/mase_results/v2/5pct_results.json` |
| Lite v3 / abl_min05 | **86.1%** | early ablation (see ranking below) |
| Lite v4 (local) | **85.87%** | `docs/mase_results/v4/5pct_results.json` |
| Lite v5 (local) | in progress | `deepyeast_5pct/checkpoints/mase_lite_5pct_v5/` |

Ablation ranking snapshot (`full_friendly_ablation_summary.jsonl` / master log, 2026-07-13):

1. fused_ce_60ep_ref — 86.3% (high acc, weak floor / collapse risk)  
2. abl_min05 — 86.1% → informed v3 defaults  
3. v2_uniform — 85.0%  
4. stronger entropy / aux / “full_safe” — worse  

---

## 4. Uncertainty quantification (CNN UQ)

**Confidence/UQ achievements (EN then ZH):** [`docs/CONFIDENCE_UQ_ACHIEVEMENTS_bilingual.md`](CONFIDENCE_UQ_ACHIEVEMENTS_bilingual.md)

- Concepts: `docs/CNN_UQ_concepts_bilingual.md`, `docs/Native_Multiclass_concepts_bilingual.md`
- Anti-collapse learnable plan: `docs/MASE_LEARNABLE_ENSEMBLE_NO_COLLAPSE.md`
- Implementation in MaSE: `pytorch/eval_mase_uq.py`; training `--with-uq` / `--v5`

**Primary metric:** **AUROC(PE) = UAUC**  
= MC Dropout → mean probs μ → predictive entropy PE per image → ROC AUC ranking incorrect vs correct.

**Not primary:** `UAUC_via_1_minus_maxprob` (Softmax confidence baseline only).

v5 `results.json` fields: `uq.test_AUROC_PE`, `uq.test_UAUC`, plus `uq_mc_dropout/summary.json`.

Weight health for v5: prefer `min_w ≥ 0.10` and `max_w ≤ 0.55` (`weight_health.ok`).

### 10% local UQ vs DeepYeast baseline (2026-07)

Same split (`deepyeast_10pct`), MC \(T=30\). Summary: `docs/UQ_10PCT_BASELINE_VS_MASE.md` · JSON: `docs/mase_results/uq_10pct/`.

| Model | Test Acc | AUROC(PE) | PE mean |
|-------|----------|-----------|---------|
| DeepYeastNet baseline | 81.3% | 0.883 | 0.58 |
| MaSE v2 | **86.0%** | 0.872 | 0.92 |
| MaSE confident (LS=0, lower dropout, resume v2) | 85.2% | 0.877 | **0.39** |

Temperature sweep on confident (Acc flat, PE↓): prefer deploy **T≈0.7–0.8**. See `mase_confident/temp_sweep.json`.

---

## 5. Full-data v4 (professor terminal, 2026-07)

From run log (awaiting formal `results.json` copy):

- Best val **89.47%** @ epoch 56  
- Test **89.58%**, mask 0.625, w=`[0.25×4]`  
- Checkpoint dir: `...\deepyeast_full\checkpoints\mase_lite_full_v4`

Please archive when available:

```text
mase_lite_full_v4/results.json
mase_lite_full_v4/uq_mc_dropout/summary.json   # if UQ was run
```

---

## 6. What “success” means going forward

| Track | Success |
|-------|---------|
| Accuracy | Prefer **v4 ~89.6%**; v5 only if test **> 89.58%** and `weight_health.ok` |
| UQ | Report **AUROC(PE)**; optional τ sweep UPre/USen |
| Fallback | If v5 collapses on full data → keep **v4** as main result |

---

## 7. Files in this sync

| Path | Content |
|------|---------|
| `docs/MASE_RESEARCH_STATUS.md` | this summary |
| `docs/MASE_LEARNABLE_ENSEMBLE_NO_COLLAPSE.md` | v5 design notes |
| `docs/mase_results/*` | JSON / ablation snapshots |
| `docs/CNN_UQ_concepts_bilingual.md` | UQ concepts |
| `docs/Native_Multiclass_concepts_bilingual.md` | Softmax vs OvA |
| `scripts/sync_mase_progress_to_workspace.sh` | optional daily sync helper |

Training code and large `.pt` weights live in **deepyeast-mase** (Git LFS), not duplicated here.
