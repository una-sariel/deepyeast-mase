# Full-data learnable ensemble without collapse

**Context:** MaSE Lite v3 (fused-CE + learnable `ensemble_logits`) reached **87.75%** full-data test vs v2 **89.1%**, with weights collapsing to ~`[0.06, 0.06, 0.78, 0.10]`.  
**v4** freezes `w=0.25` (accuracy candidate). This note is the **next line**: keep **learnable** mixture on full data **without** one-head takeover.

---

## Why collapse happens

Fused NLL rewards putting mass on the currently best head. On full data that head pulls ahead early → softmax concentrates → other heads get little gradient through the mixture → collapse reinforces.  
`min_w=0.05` still allows max ≈ 0.85.

---

## Design options (prefer stack, not only one)

### Tier A — constraints (cheap, try first on 5% then full)

| Idea | Mechanism | Suggested start |
|------|-----------|-----------------|
| **Stronger floor** | `min_ensemble_weight ∈ [0.12, 0.18]` | **0.15** (max ≤ 0.55) |
| **Entropy bonus** | `L -= λ H(w)` | **λ=0.01** (0.02 hurt 5% before) |
| **Slow ensemble LR** | `ensemble_lr = 0.25–1.0 × backbone_lr` | **0.5 × backbone** (not 5×) |
| **Temperature > 1** | softer softmax on logits | **T=1.5–2.0** early, anneal → 1 |

Hard rule for “not collapsed”: report run invalid if `max(w) > 0.55` or `min(w) < 0.10` at best.pt.

### Tier B — schedule (training protocol)

**Two-phase (recommended for full data):**

1. **Phase 1 (e.g. 20–30 ep or until val plateaus):** `--freeze-ensemble` (v4). Train fused + heads with uniform mix.  
2. **Phase 2 (short, small ens LR):** unfreeze `ensemble_logits`, `min_w=0.15`, `ent_w=0.01`, `ensemble_lr ≤ 0.5× backbone`, patience short.  

Only accept Phase 2 checkpoint if val ≥ Phase 1 best **and** weight health OK.

### Tier C — loss shaping

| Idea | Note |
|------|------|
| Keep **aux ≥ 0.5** | forces every head competent even if w small |
| Optional **head diversity** | e.g. encourage disagreement on softmax (careful; can hurt) |
| **EMA of w** for inference | train live w, evaluate with EMA → stabler reporting |

### Tier D — architectural (only if A–C fail)

- Mixture of logits instead of probs (different collapse dynamics)  
- Gating network conditioned on features (input-dependent w) — larger change  

Not first priority.

---

## Proposed experiment IDs

| ID | Recipe | When |
|----|--------|------|
| **v5a** | learnable, `min_w=0.15`, `ens_lr=0.5×bb`, `ent_w=0.01` | after v4 5%/full known |
| **v5b** | Phase1 v4 → Phase2 v5a | full-data main bet for learnable |
| **v5c** | v5a + `ensemble_temperature=1.5` | if still peaky |

Success criteria (full):

1. `test > 0.891` **or** (≥ v4 and clearly better calibration / non-uniform but healthy w)  
2. At best.pt: `min(w) ≥ 0.10`, `max(w) ≤ 0.55`  
3. Mask still ~0.625  

---

## Relation to current work (2026-07-21/22)

- **Accuracy path now:** 5% auto-loop on **v4** (frozen); full-data v4 with professor.  
- **Learnable path:** do **not** ship unconstrained v3 again; use **v5a/v5b** after v4 baseline is measured.  
- Code already has `min_ensemble_weight`, `ensemble_entropy_weight`, `ensemble_lr` — v5 is mostly **recipe + two-phase trainer**, not a new backbone.
