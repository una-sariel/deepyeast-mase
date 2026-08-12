# MaSE Lite fine-tuning log: v4 → v7 (+ TTA)

**Scope:** full-data (90k), seed=42, same `MaSELiteNet` architecture.  
**Goal:** push test accuracy from v4 (~89.6%) toward **90%**, with learnable but healthy head fusion.

JSON evidence: [results/README.md](results/README.md).

---

## Context before v4

| Version | Idea | Full-data test | Outcome |
|---------|------|----------------|---------|
| v2 | head-mean CE; w stuck 0.25 | 89.1% | solid |
| v3 | fused-CE; unconstrained learnable w | 87.75% | **weight collapse** (~0.78 on one head) |

---

## Stage map (v4 → v7)

```text
v4          fused-CE + frozen uniform w=0.25  (accuracy SOTA, no TTA)
    ↓
v5          single-phase learnable w + anti-collapse
    ↓
v6 Phase1   = v4 recipe again (frozen w) — TP-AHF base
v6 Phase2   resume P1; train ONLY ensemble_logits
v6 Phase2.5 resume P1; train ensemble_logits + 4 heads
    ↓
v7          Phase2.5 with smaller LR / tighter patience
    ↓
TTA         flip + rot90 at eval (no retrain) on Phase2.5 and v7
```

---

## 1. v4 — frozen uniform fusion (starting point)

| Item | Detail |
|------|--------|
| Flag | `--freeze-ensemble` (also `--v6-phase1` = same recipe) |
| Train | full model; **frozen** w=0.25×4 |
| Loss | fused-CE + aux + distill + sparsity |
| Full-data test | **≈89.58%** |
| Weights | `[0.25, 0.25, 0.25, 0.25]` |
| Lesson | Avoids v3 collapse; strong accuracy baseline for all later fine-tunes. |

Doc: [V4_RUN.md](V4_RUN.md)

---

## 2. v5 — learnable fusion, one-shot anti-collapse

| Item | Detail |
|------|--------|
| Flag | `--v5` |
| Train | full model; learnable `ensemble_logits` |
| Anti-collapse | `min_w=0.15`, `ent_w=0.01`, `ens_lr=0.5×` backbone |
| UQ | MC Dropout on by default |
| Reported test | **~89.16%** (below v4) |
| Weights | max w ≈ **0.44** (healthier than v3, still peaky vs v4) |
| UQ | test UAUC(PE) **0.9249** (MC T=30) |
| Lesson | Learnable w can avoid hard collapse, but **did not beat frozen v4**. Need stronger constraints / two-phase training. |

Doc: [V5_RUN.md](V5_RUN.md)

---

## 3. v6 Phase 1 — TP-AHF base (= v4 recipe)

| Item | Detail |
|------|--------|
| Flag | `--v6-phase1` |
| Train | full model; **frozen** w=0.25 |
| Loss | fused-CE + aux + distill + sparsity (same as v4) |
| Checkpoint | `mase_lite_full_v6_phase1` |
| Test | **88.95%** (`results/v6/phase1_results.json`) |
| Best epoch | 56 |
| Weights | `[0.25, 0.25, 0.25, 0.25]` |
| Lesson | Reproducible frozen base for Phase 2 / 2.5. Slightly below historical v4 89.58% (seed/run variance). |

Doc: [V6_RUN.md](V6_RUN.md)

---

## 4. v6 Phase 2 — calibrate global w only

| Item | Detail |
|------|--------|
| Flag | `--v6-phase2` |
| Resume | Phase-1 `best.pt` |
| Trainable | **`ensemble_logits` only** (backbone + heads frozen) |
| Constraints | `min_w=0.20`, `ent_w=0.02`, T: 1.5→1.0, gate vs Phase-1 val |
| Test (approx) | **~89.01%** (+~0.06pp vs Phase 1) |
| Lesson | Global 4-scalar fusion has **too little capacity**. Need to move heads or w(x). |

---

## 5. v6 Phase 2.5 — co-adapt heads + w

| Item | Detail |
|------|--------|
| Flag | `--v6-phase25` |
| Resume | Phase-1 `best.pt` |
| Trainable | `ensemble_logits` + **4 progressive heads** |
| Frozen | selector + VGG/ResNet/DenseNet branches |
| Defaults | `head_lr=2e-5`, `ens_lr_ratio=0.25`, patience=5 |
| Test | **89.07%** (`results/v6/phase25_results.json`) |
| Best epoch | **2** (early overfit: train ~95.8% vs val ~88.9%) |
| Weights | `[0.242, 0.245, 0.249, 0.264]` — still near-uniform |
| vs Phase-1 gate | PASS |
| Lesson | Small gain over Phase 2; **best @ ep2** → LR too high / heads adapt too fast. w did not specialize to strong heads (cf. 10% head-contrib: H2/H3 stronger). |

Doc: [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md)  
Analysis: `results/head_contrib_10pct/`

---

## 6. v7 — tuned Phase 2.5

| Item | Detail |
|------|--------|
| Flag | `--v7` |
| Same unfreeze | heads + `ensemble_logits` |
| Changes vs 2.5 | `head_lr=1e-5`, `ens_lr_ratio=0.15`, `patience=3` |
| Test | **89.06%** (`results/v7/results.json`) |
| Best epoch | **2** again |
| Weights | `[0.245, 0.247, 0.249, 0.260]` |
| Lesson | Lower LR **did not help** accuracy vs Phase 2.5. Training-side fine-tune of heads+w is near a plateau (~89.07%). |

Doc: [V7_RUN.md](V7_RUN.md)

---

## 7. TTA — inference-only boost (no retrain)

| Checkpoint | Baseline test | + TTA (`flip_rot`, 7 views) | Gain |
|------------|---------------|----------------------------|------|
| Phase 2.5 | 89.08% | **89.82%** | **+0.74pp** |
| v7 | 89.07% | **89.82%** | **+0.74pp** |

Sources: `results/v6/phase25_tta_summary.json`, `results/v7/tta_summary.json`  
Script: `pytorch/eval_mase_tta.py`  
Paper: Shanmugam et al., ICCV 2021 — [arXiv:2011.11156](https://arxiv.org/abs/2011.11156) (uniform mean aggregator).

**Lesson:** Largest single step after v4 toward 90%. Still **~0.18pp short** of 90%. Report train and TTA as separate rows.

---

## Accuracy ladder (full data)

| Step | Test | Δ vs previous meaningful base |
|------|------|-------------------------------|
| **v4** (historical best train) | **≈89.58%** | arc start |
| v5 | ~89.16% | −0.42pp vs v4 (failed to beat) |
| v6 Phase 1 | 88.95% | TP-AHF base (≈ v4 recipe) |
| v6 Phase 2 | ~89.01% | +0.06pp vs P1 |
| v6 Phase 2.5 | 89.07% | +0.12pp vs P1 |
| v7 | 89.06% | ≈ Phase 2.5 |
| **Phase 2.5 / v7 + TTA** | **89.82%** | **+0.74pp** vs train; **+0.24pp** vs v4 train |
| v8 ID-Gate | 89.05% | +0.10pp vs P1; ≈ Phase 2.5 (see [V8_ID_GATE_RUN.md](V8_ID_GATE_RUN.md)) |
| v8 + TTA | 89.69% | below P2.5/v7 TTA 89.82% |

```text
v4 ≈89.58%
        │
        ▼
v5 ~89.16% ──► P1 88.95% ──P2──► 89.01% ──P2.5──► 89.07% ──v7──► 89.06%
                     │                                            │
                     └── v8 ID-Gate 89.05% ──TTA──► 89.69%        TTA ▼
                                                               89.82%  ··· goal 90%
```

---

## What worked / what did not

| Worked | Did not work (for +0.4pp+) |
|--------|---------------------------|
| **v4 frozen uniform** as strong train baseline | v5 single-phase learnable beating v4 |
| Constrained w (no v3-style collapse) | Phase 2 (w-only) alone |
| Phase 2.5 small train gain + paper story | Expecting global w to leave 0.25×4 |
| **TTA +0.74pp** (beats v4 train when stacked) | v7 LR retune vs Phase 2.5 |

---

## File index

| Artifact | Path |
|----------|------|
| v4 run doc | [V4_RUN.md](V4_RUN.md) |
| Phase 1 results | `results/v6/phase1_results.json` |
| Phase 2.5 results | `results/v6/phase25_results.json` |
| Phase 2.5 TTA | `results/v6/phase25_tta_summary.json` |
| v7 results | `results/v7/results.json` |
| v7 TTA | `results/v7/tta_summary.json` |
| Head contribution (10%) | `results/head_contrib_10pct/` |
| Train flags | `pytorch/train_mase_lite.py` (`--freeze-ensemble`, `--v5`, `--v6-phase*`, `--v7`) |
| TTA eval | `pytorch/eval_mase_tta.py` |
