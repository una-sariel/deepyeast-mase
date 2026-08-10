# MaSE-Net Lite v8 — ID-Gate fine-tune (single model)

**ID-Gate** = **I**nput-**D**ependent head fusion: a small MLP maps `concat(v,r,d)` → per-image mixture weights `w(x)`.

| | Phase 2.5 / v7 | **v8** |
|--|----------------|--------|
| Fusion | global `w` (4 scalars) | **sample-wise `w(x)`** |
| Trainable | heads + `ensemble_logits` | **heads + `id_gate`** |
| Frozen | selector + branches | same |
| Checkpoint | `mase_lite_full_v6_phase25` / `v7` | **`mase_lite_full_v8`** |

Still **one model** (not multi-seed ensemble).

**Repo:** https://github.com/una-sariel/deepyeast-mase  
See also [FINE_TUNING_V4_TO_V7.md](FINE_TUNING_V4_TO_V7.md), [V7_RUN.md](V7_RUN.md).

---

## Related papers (method origin)

**“ID-Gate” is our project name**, not a paper title. The idea is standard **soft Mixture-of-Experts (MoE) gating**: a network maps the input to softmax weights over experts; the final prediction is a weighted mix of expert outputs. In MaSE v8 the “experts” are the **four progressive heads**, and the gate is a small MLP on `concat(v,r,d)`.

| Paper | Role for v8 |
|-------|-------------|
| Jacobs, Jordan, Nowlan, Hinton. **Adaptive Mixtures of Local Experts.** *Neural Computation*, 3(1):79–87, 1991. [doi](https://doi.org/10.1162/neco.1991.3.1.79) | **Primary citation.** Introduces experts + **gating network** with input-dependent mixing weights. |
| Jordan & Jacobs. **Hierarchical Mixtures of Experts and the EM Algorithm.** *Neural Computation*, 6(2):181–214, 1994. [doi](https://doi.org/10.1162/neco.1994.6.2.181) | Hierarchical MoE; soft gating / EM training lineage. |
| Shazeer et al. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.** ICLR 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | Modern MoE layer; recalls dense softmax gating \(G(x)=\mathrm{softmax}(W_g x)\) before sparse Top‑k variants. |

### How v8 relates (for paper writing)

> We adopt a soft, input-dependent gating network in the sense of Jacobs et al. (1991): given branch features \(x=\mathrm{concat}(v,r,d)\), a lightweight MLP produces logits over the four progressive heads; mixture weights \(w(x)=\mathrm{softmax}(g(x)/T)\) combine head probabilities. Unlike sparse MoE (Shazeer et al., 2017), all four heads remain active (dense soft gating). Unlike our Phase‑2 global logits, \(w\) varies per image. We fine-tune the gate and heads from a Phase‑1 checkpoint with a minimum-weight floor and entropy regularization to avoid single-head collapse.

---

## What to send back

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v8\results.json
2) <deepyeast_full>\checkpoints\mase_lite_full_v8\tta_eval\summary.json   (optional)
```

---

## Paths (v10 machine)

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v10\deepyeast-mase"
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v10\deepyeast_full"
cd $REPO
git pull
.venv\Scripts\activate
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('v8 OK' if '--v8' in t else 'git pull again')"
```

Need Phase-1 weights:

```powershell
dir "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt"
```

---

## Train v8 (~15–45 min)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v8 `
  --seed 42
```

Progress bar: **`MaSELiteV8`**

### Defaults

| Knob | Value |
|------|-------|
| Resume | Phase-1 `best.pt` |
| Trainable | `id_gate` + 4 heads |
| Frozen | selector + PLCNN branches |
| `id_gate_hidden` | 128 |
| `id_gate_lr` | **1e-3** |
| `phase25_head_lr` | **1e-5** |
| `min_ensemble_weight` | **0.15** |
| `ensemble_entropy_weight` | **0.02** |
| Temperature anneal | 1.5 → 1.0 |
| epochs / patience | 15 / 5 |
| `max_ensemble_weight` (mean-w health) | 0.55 |

Gate last layer is **zero-initialized** → starts near uniform `w`, then learns sample-wise routing.

---

## Optional TTA after train

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v8\best.pt" `
  --split both `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v8\tta_eval"
```

Or: `--v8 --with-tta`

---

## Check log

```text
v8 ID-Gate: training id_gate (...) + heads (...); selector+branches frozen
MaSELiteV8
  ep02  val=0.89x  wh=ok  w=[...,...,...,...]   # mean w over last batch
Test:     0.89xx
```

`w=[...]` is the **batch-mean** of sample-wise weights (for logging only).

---

## Success criteria

| Metric | Target |
|--------|--------|
| `test.accuracy` | **> Phase-1** (~0.8895); aim beat Phase 2.5 **0.8907** |
| `weight_health.ok` | true (on mean w) |
| With TTA | compare to Phase 2.5 + TTA **89.82%** |

Report **train** and **TTA** as separate rows.
