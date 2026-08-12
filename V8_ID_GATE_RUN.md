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

---

## 「ID-Gate」是什么（中文说明）

**ID-Gate** = **I**nput-**D**ependent gate（输入相关门控）。这是项目里的叫法，不是某篇论文的正式方法名。背后对应的是软 **Mixture-of-Experts（MoE）门控**（Jacobs et al., 1991）：一个小网络根据**当前这张图**的特征，决定四个 progressive head 各占多少权重。

### v8 之前（Phase 2.5 / v7）

- 融合用的是**全局**一套权重 `w = [w_V, w_R, w_D, w_F]`，**所有图像共用**。
- 这四个数学完（或冻结）后，不随输入变化。

### v8（ID-Gate）

- 融合变成**逐样本**权重 `w(x)`。
- 小 MLP（`id_gate`）读取三支路特征  
  \(x = \mathrm{concat}(v, r, d)\)  
  （VGG / ResNet / DenseNet 分支嵌入），输出 4 维 logits → Softmax → `w(x)`。
- 最终预测：用 `w(x)` 对四个 head 的概率做加权混合（稠密软门控——四个头都参与；不是稀疏 Top‑k MoE）。

| | 全局融合 | **ID-Gate** |
|--|----------|-------------|
| 权重 | 所有图同一套 `w` | 每张图不同的 `w(x)` |
| 谁决定 | 固定/学到的标量 | 由该图的 `v,r,d` 条件化的 MLP |
| 直觉 | 「永远用同一种方式混四个头」 | 「按这张图把流量分给更合适的头」 |

### 为什么这样微调

- 从已训好的 Phase‑1 checkpoint 出发（selector + PLCNN 分支已学好）。
- **冻结** selector + 分支；**只训** `id_gate` + 四个 head（并加最小权重 / 熵正则，避免单头塌缩）。
- Gate 最后一层 **零初始化**，一开始 `w(x)` 接近均匀，再逐渐学到按样本分流。

### 一句话

> **ID-Gate = 学「随输入变化」的四头软混合，而不是全数据集共用一套全局混合权重。**

---

## What “ID-Gate” means (English)

**ID-Gate** = **I**nput-**D**ependent gate. It is a **project nickname**, not a published method name. The underlying idea is soft **Mixture-of-Experts (MoE) gating** (Jacobs et al., 1991): a small network looks at the **current image’s** features and decides how much to trust each of the four progressive heads.

### Before v8 (Phase 2.5 / v7)

- Fusion uses **one global** weight vector `w = [w_V, w_R, w_D, w_F]` shared by **every** image.
- Those four numbers are learned once (or frozen) and do not change with the input.

### After v8 (ID-Gate)

- Fusion uses **per-image** weights `w(x)`.
- A small MLP (`id_gate`) reads branch features  
  \(x = \mathrm{concat}(v, r, d)\)  
  (VGG / ResNet / DenseNet branch embeddings) and outputs 4 logits → Softmax → `w(x)`.
- Final prediction: mix the four head probability vectors with `w(x)` (dense soft gating — all heads stay active; not sparse Top‑k MoE).

| | Global fusion | **ID-Gate** |
|--|---------------|-------------|
| Weights | same `w` for all images | different `w(x)` per image |
| Who decides | fixed / learned scalars | MLP conditioned on this image’s `v,r,d` |
| Intuition | “always blend heads the same way” | “route this image to the heads that fit it” |

### Why fine-tune this way

- Start from a strong Phase‑1 checkpoint (selector + PLCNN branches already trained).
- **Freeze** selector + branches; **train** only `id_gate` + the four heads (plus min-weight / entropy regularizers so one head does not collapse).
- Gate last layer is **zero-init**, so early `w(x)` is near uniform, then learns sample-wise routing.

### One-line takeaway

> **ID-Gate = learn an input-dependent soft mix of the four MaSE heads, instead of one global mix for the whole dataset.**

---

## Full-data result (seed=42, professor run)

Sources: [`results/v8/results.json`](results/v8/results.json), [`results/v8/tta_summary.json`](results/v8/tta_summary.json).

| Metric | Value |
|--------|-------|
| Resume | Phase-1 `best.pt` |
| Best epoch | **1** (val 88.83%) |
| **Test (train)** | **89.05%** |
| Mean w (logged) | `[0.190, 0.218, 0.237, 0.356]` — `weight_health.ok` |
| **Test + TTA** (`flip_rot`) | **89.69%** (+0.66pp vs v8 baseline in TTA script) |
| Elapsed | ~9 min |

| Compare | Acc | Δ vs v8 train |
|---------|-----|---------------|
| Phase 1 | 88.95% | **+0.10pp** (meets “> Phase-1”) |
| Phase 2.5 / v7 | 89.07% / 89.06% | ≈ tie (−0.02pp) |
| v4 train | ≈89.58% | −0.53pp |
| Phase 2.5 / v7 + TTA | **89.82%** | v8+TTA **89.69%** (−0.13pp) |

**Readout:** ID-Gate is healthy and beats Phase-1 slightly, but **does not beat Phase 2.5 train or the 89.82% TTA SOTA**. Best @ ep1 suggests the gate/heads adapt immediately then overfit; mean w is more peaked than global Phase 2.5 (head4 ≈0.36). Keep v8 as the sample-wise fusion story; accuracy table still leads with **v4 train** / **P2.5+TTA**.
