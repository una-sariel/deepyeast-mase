# MaSE Lite fine-tuning log: v5 → v7 (+ TTA)

**Scope:** full-data (90k), seed=42, same `MaSELiteNet` architecture.  
**Goal:** push test accuracy from ~89.6% toward **90%**, with learnable but healthy head fusion.

JSON evidence: [results/README.md](results/README.md).

---

## Baseline before this arc

| Version | Idea | Full-data test | Outcome |
|---------|------|----------------|---------|
| v2 | head-mean CE; w stuck 0.25 | 89.1% | solid |
| v3 | fused-CE; unconstrained learnable w | 87.75% | **weight collapse** (~0.78 on one head) |
| **v4** | fused-CE; **frozen** w=0.25×4 | **≈89.58%** | accuracy SOTA (single model, no TTA) |

---

## Stage map (what we changed)

```text
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

## 1. v5 — learnable fusion, one-shot anti-collapse

| Item | Detail |
|------|--------|
| Flag | `--v5` |
| Train | full model; learnable `ensemble_logits` |
| Anti-collapse | `min_w=0.15`, `ent_w=0.01`, `ens_lr=0.5×` backbone |
| UQ | MC Dropout on by default |
| Reported test | **~89.16%** (below v4) |
| Weights | max w ≈ **0.44** (healthier than v3, still peaky vs v4) |
| Lesson | Learnable w can avoid hard collapse, but **did not beat frozen v4**. Need stronger constraints / two-phase training. |

Doc: [V5_RUN.md](V5_RUN.md)

---

## 2. v6 Phase 1 — TP-AHF base (= v4 recipe)

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

## 3. v6 Phase 2 — calibrate global w only

| Item | Detail |
|------|--------|
| Flag | `--v6-phase2` |
| Resume | Phase-1 `best.pt` |
| Trainable | **`ensemble_logits` only** (backbone + heads frozen) |
| Constraints | `min_w=0.20`, `ent_w=0.02`, T: 1.5→1.0, gate vs Phase-1 val |
| Test (approx) | **~89.01%** (+~0.06pp vs Phase 1) |
| Lesson | Global 4-scalar fusion has **too little capacity**. Need to move heads or w(x). |

---

## 4. v6 Phase 2.5 — co-adapt heads + w

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

## 5. v7 — tuned Phase 2.5

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

## 6. TTA — inference-only boost (no retrain)

| Checkpoint | Baseline test | + TTA (`flip_rot`, 7 views) | Gain |
|------------|---------------|----------------------------|------|
| Phase 2.5 | 89.08% | **89.82%** | **+0.74pp** |
| v7 | 89.07% | **89.82%** | **+0.74pp** |

Sources: `results/v6/phase25_tta_summary.json`, `results/v7/tta_summary.json`  
Script: `pytorch/eval_mase_tta.py`

**Lesson:** Largest single step after v4 toward 90%. Still **~0.18pp short** of 90%.

---

## Accuracy ladder (full data)

| Step | Test | Δ vs previous meaningful base |
|------|------|-------------------------------|
| v4 (historical best train) | ≈89.58% | — |
| v5 | ~89.16% | −0.42pp vs v4 (failed to beat) |
| v6 Phase 1 | 88.95% | base for TP-AHF |
| v6 Phase 2 | ~89.01% | +0.06pp vs P1 |
| v6 Phase 2.5 | 89.07% | +0.12pp vs P1 |
| v7 | 89.06% | ≈ Phase 2.5 |
| **Phase 2.5 / v7 + TTA** | **89.82%** | **+0.74pp** vs train |

```text
88.95% ──P2──► 89.01% ──P2.5──► 89.07% ──v7──► 89.06%
                                              │
                                         TTA  ▼
                                           89.82%  ··· goal 90%
```

---

## What worked / what did not

| Worked | Did not work (for +0.4pp+) |
|--------|---------------------------|
| Frozen uniform v4 as strong base | v5 single-phase learnable beating v4 |
| Constrained w (no v3-style collapse) | Phase 2 (w-only) alone |
| Phase 2.5 small train gain + paper story | Expecting global w to leave 0.25×4 |
| **TTA +0.74pp** | v7 LR retune vs Phase 2.5 |

---

## Next options (if still targeting 90%)

1. **3-seed ensemble** (seed 42/43/44 Phase-1 or v4-style) ± TTA  
2. **ID-Gate** — sample-dependent `w(x)` (model-side; paper innovation)  
3. Do **not** expect another Phase2.5 LR grid to cross 90% alone  

Report train and TTA as **separate rows** in any paper table.

---

## File index

| Artifact | Path |
|----------|------|
| Phase 1 results | `results/v6/phase1_results.json` |
| Phase 2.5 results | `results/v6/phase25_results.json` |
| Phase 2.5 TTA | `results/v6/phase25_tta_summary.json` |
| v7 results | `results/v7/results.json` |
| v7 TTA | `results/v7/tta_summary.json` |
| Head contribution (10%) | `results/head_contrib_10pct/` |
| Train flags | `pytorch/train_mase_lite.py` (`--v5`, `--v6-phase*`, `--v7`) |
| TTA eval | `pytorch/eval_mase_tta.py` |
