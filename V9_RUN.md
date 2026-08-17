# MaSE-Net Lite **v9** — Random Spatial Bagging (RSB)

**完善老师「随机遮盖 + 森林」想法后的可跑版本。**

| | 老师草图（弱） | **v9（本版）** |
|--|----------------|----------------|
| Mask | 全体图像**共用**同一随机区域 | **每张图独立**随机 top-k patches |
| 角色 | 像数据增强/固定遮挡 | 像随机森林的 **feature bagging（空间版）** |
| 推理 | 单次遮盖 | 对 **R** 次随机视野 Softmax **平均**（RSB） |
| 训练 | — | 从 Phase‑1 微调 heads + `w`；selector/分支冻结 |

**仍是单模型**（不是多 seed 重训）。全量结果见下方；主对照是 learned mask。

**Repo:** https://github.com/una-sariel/deepyeast-mase

---

## Idea（中英）

### 中文

随机森林靠「每棵树看不同特征子集」降低相关误差。v9 把这个思想搬到 **64 个 patch**：  
训练/推理时对**每张图**随机保留 top‑k=40 个 patch（不是全数据集共用一块遮盖）。  
推理时再抽 R 次、平均概率 → **Random Spatial Bagging (RSB)**。

### English

RF-style **spatial feature bagging**: each image draws an independent random top‑k patch mask.  
At test time, average Softmax over **R** independent random masks (RSB).  
Do **not** use one shared region for all images (that correlates errors).

---

## Files to send back

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v9\results.json
2) <deepyeast_full>\checkpoints\mase_lite_full_v9\rsb_eval\summary.json
```

In `rsb_eval/summary.json` look for `splits.test`:

- `rsb.accuracy` ← **primary v9 number** (R-mask average)
- `random_single.accuracy` ← single random mask
- `learned_single.accuracy` ← same weights + learned selector (reference)

---

## Paths

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v10\deepyeast-mase"
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v10\deepyeast_full"
cd $REPO
git pull
.venv\Scripts\activate
python -c "t=open('pytorch/train_mase_lite.py',encoding='utf-8').read(); print('v9 OK' if '--v9' in t else 'git pull again')"
```

Need Phase-1 weights:

```powershell
dir "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt"
```

No Phase-1? Pass an existing strong checkpoint:

```powershell
--resume "$DATA\checkpoints\mase_lite_full_v4\best.pt"
```

---

## Train v9 (~15–45 min on GPU)

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v9 `
  --seed 42
```

Progress bar: **`MaSELiteV9`**

`--v9` auto-runs RSB eval (`--with-rsb`, default `R=16`).

### Defaults

| Knob | Value |
|------|-------|
| Resume | Phase-1 `best.pt` |
| Trainable | 4 heads + `ensemble_logits` |
| Frozen | selector + PLCNN branches |
| `mask_mode` | **random** (per-image top-k) |
| `rsb_samples` | **16** |
| `min_ensemble_weight` | 0.15 |
| `ensemble_entropy_weight` | 0.02 |
| epochs / patience | 15 / 5 |

### Optional knobs

```powershell
python pytorch\train_mase_lite.py --data-dir $DATA --v9 --rsb-samples 32 --seed 42
```

---

## Eval only (no retrain)

```powershell
python pytorch\eval_mase_rsb.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v9\best.pt" `
  --rsb-samples 16 `
  --split both `
  --out-dir "$DATA\checkpoints\mase_lite_full_v9\rsb_eval"
```

---

## Success criteria

| Metric | Target |
|--------|--------|
| `rsb_eval` test `rsb.accuracy` | report vs Phase-1 (~0.8895) and Phase2.5+TTA (**0.8982**) |
| `weight_health.ok` | true |
| vs `learned_single` | if RSB ≪ learned → random bagging is weaker than selector (expected possible) |

Report **train test** and **RSB test** as separate rows.

---

## 「ID-Gate / RSB」不是什么

- **不是**「全体图像遮同一块」——那是草图弱版，已做成 **[v10 SFRM](V10_RUN.md)**（5% 未超过 v4）。  
- **不是**多棵独立随机森林树重训；是 **一次微调 + 推理时多次随机视野平均**。  
- Learned MaSE mask 仍是主线；v9 是空间 Bagging **对照 / 增强实验**。

---

## Full-data result (seed=42, professor run, `deepyeast-mase-main_v12`)

Sources: [`results/v9/results.json`](results/v9/results.json), [`results/v9/rsb_summary.json`](results/v9/rsb_summary.json).

Resume Phase-1. Trainable: heads + `w`. `mask_mode=random`. `weight_health.ok` (`w≈[0.208, 0.221, 0.241, 0.330]`).

No epoch beat the Phase-1 val gate (0.8875); random-mask val peaked **85.08%** @ ep4. Trainer test is the last-epoch random-mask forward.

| Metric | Test |
|--------|------|
| Random mask (trainer) | 84.87% |
| Random single (`rsb_eval`) | 84.50% |
| **RSB (R=16)** | **86.28%** |
| Same weights + learned selector | **88.96%** |

| Compare | Acc | vs v9 RSB |
|---------|-----|-----------|
| Phase 1 | 88.95% | −2.67pp |
| v4 train | ≈89.58% | −3.30pp |
| Phase 2.5 / v7 + TTA | **89.82%** | −3.54pp |
| learned_single (this ckpt) | 88.96% | −2.68pp |

**Readout:** RSB **hurts** vs the learned selector on the same weights. Per-image random top-k is a method story, not an accuracy path. Do not promote v9 over Phase-1 / v4 / P2.5+TTA.
