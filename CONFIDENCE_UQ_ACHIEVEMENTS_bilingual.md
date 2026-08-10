# Confidence & UQ Achievements (MaSE / DeepYeast)

**Date:** 2026-07-31  
**Code (training / eval):** https://github.com/una-sariel/deepyeast-mase  
**This workspace:** result snapshots, recipes, and concept notes  

**How to read:** Full English (**Part I**), then full Chinese (**Part II**).

---

# Part I — English

---

## 1. What we mean by “confidence”

| Term | Meaning in our pipeline |
|------|-------------------------|
| **Softmax confidence** | \(\max_c p(c\mid x)\) from a single forward pass — **not** the primary scientific claim |
| **Predictive entropy (PE)** | Entropy of the **MC-averaged** class distribution; lower PE ≈ sharper / more “confident” prediction |
| **AUROC(PE) = UAUC** | How well PE ranks **incorrect** predictions above correct ones — **primary UQ quality metric** |
| **PE mean** | Average PE on a split — describes sharpness only; **lower ≠ better model** by itself |

**Rule we keep:** report **AUROC(PE)** as the main UQ number. Treat `UAUC_via_1_minus_maxprob` as a Softmax baseline only.

---

## 2. Tooling built (architecture unchanged)

We evaluate confidence **without changing** the MaSE Lite architecture. Training recipes may change dropout / label smoothing; inference UQ is MC Dropout.

| Artifact | Role |
|----------|------|
| `eval_mase_uq.py` | MC Dropout → PE, UAUC, τ sweep (UAcc/USen/USpe/UPre) |
| `eval_deepyeast_uq.py` | Same UQ protocol on official DeepYeastNet |
| `train_deepyeast_baseline.py` | Train paper baseline for fair Acc/UQ comparison |
| `temp_sweep_mase.py` | Post-hoc temperature \(T\) on logits/probs (no retrain) |
| Train flags `--with-uq` / `--v5` | Optional UQ summary written into `results.json` after training |

Outputs per run: `uq_mc_dropout/summary.json`, `per_image_{val,test}.csv`.

Concept background (separate notes):

- `docs/CNN_UQ_concepts_bilingual.md`
- `docs/Native_Multiclass_concepts_bilingual.md`

---

## 3. Main empirical result — 10% data (seed=42, \(T=30\))

**Split:** `deepyeast_10pct` (train 6500 / val 1250 / test 1250).  
**Detail doc:** `docs/UQ_10PCT_BASELINE_VS_MASE.md`  
**JSON:** `docs/mase_results/uq_10pct/`

| Model | Test Acc | AUROC(PE) | AUROC(1−maxp) | PE mean |
|-------|----------|-----------|---------------|---------|
| DeepYeastNet baseline | 81.3% | **0.883** | 0.890 | 0.58 |
| MaSE Lite v2 | **86.0%** | 0.872 | 0.876 | 0.92 |
| MaSE Lite **confident** | 85.2% | 0.877 | 0.876 | **0.39** |

### Takeaways

1. **Accuracy:** MaSE v2 / confident clearly beat the official baseline (+4.7 / +3.9 pp).
2. **UQ ranking quality:** baseline has a slight edge on AUROC(PE); MaSE confident nearly matches it (0.877 vs 0.883) while staying far more accurate than baseline.
3. **Sharpness:** default MaSE v2 is **under-confident** (PE mean 0.92 ≫ baseline 0.58). The **confident recipe** flips this: PE mean **0.39**, sharper than baseline, with only a small Acc cost vs v2 (−0.8 pp).

---

## 4. Confident training recipe (10% pilot)

**Goal:** raise confidence (↓ PE / sharper Softmax) vs `triple_fusion_lite_10pct_v2` **without** changing architecture.  
**Doc:** `docs/CONFIDENT_RECIPE_10PCT.md`

| Knob | v2 | confident |
|------|----|-----------|
| `label_smoothing` | 0.1 | **0** |
| `branch_dropout` / `head_dropout` | 0.4 / 0.5 | **0.25 / 0.3** |
| Init | PLCNN + masked | **resume v2 `best.pt`** |
| Epochs | 60 | **20** (patience 8) |
| Backbone LR | 1e-3 | **3e-4** |
| Loss | `--legacy-v2-loss` | same family |

Architecture flags (top-k, soft mask, etc.) stay the same. Older v2 checkpoints remain valid; confident is a **new** checkpoint name.

---

## 5. Temperature scaling (no retrain)

On **MaSE confident**, apply \(p \propto \exp(\log p / T)\). Deterministic Acc stays flat; PE falls as \(T\) decreases; UAUC stays ≈0.88.

| \(T\) | det Acc | det PE | mc PE | mc UAUC |
|-------|---------|--------|-------|---------|
| 1.0 | 85.12% | 0.369 | 0.388 | 0.879 |
| 0.9 | 85.12% | 0.324 | 0.343 | 0.881 |
| 0.8 | 85.12% | 0.281 | 0.301 | 0.880 |
| 0.7 | 85.12% | 0.241 | 0.260 | 0.878 |
| 0.5 | 85.12% | 0.165 | 0.186 | 0.878 |

**Practical deploy tip:** \(T \approx 0.7\)–\(0.8\) for higher confidence without Acc drop.  
Full table: `docs/mase_results/uq_10pct/mase_confident/temp_sweep.json`.

---

## 6. 5% pilot — MaSE Lite v4 UQ

Same MC protocol (\(T=30\)) on local 5% v4 (`mase_lite_5pct_v4_default`).  
JSON: `docs/mase_results/uq_5pct/mase_v4/uq_summary.json`

| Split metric (test) | Value |
|---------------------|-------|
| Accuracy | **85.87%** |
| AUROC(PE) = UAUC | **0.873** |
| AUROC(1−maxp) | 0.882 |
| PE mean | 1.04 |

So on 5%, v4 already has usable error-ranking UQ (~0.87 UAUC), with relatively high PE mean (less sharp than the 10% confident recipe — expected under heavier regularization / less data).

---



### 5% v4 — higher AUROC without Acc drop (2026-07-31)

Classifier frozen (same MC argmax, test Acc **85.71%**). Better uncertainty scores:

| Score | Test AUROC |
|-------|------------|
| PE (MC) | 0.875 |
| 1−maxprob | 0.885 |
| Mahalanobis (to pred class, correct-fit) | 0.892 |
| **z(1−maxp)+z(Mahalanobis)** | **0.901** |

Δ vs PE ≈ **+0.026**. JSON: `docs/mase_results/uq_5pct/mase_v4/mahalanobis_summary.json`.  
Scripts in MaSE: `eval_mase_uq_mahalanobis.py`, `eval_mase_uq_feature_probe.py`.

## 7. Full-data track (accuracy SOTA vs UQ still pending)

| Item | Status |
|------|--------|
| Full Acc **v4** | ≈**89.58%** (professor run; formal `results.json` archive when available) |
| Full Acc **v2** | 89.1% (reference) |
| Full **UQ** on v4 / v5 | Protocol ready (`V4_RUN.md` Step 3 / `V5_RUN.md`); awaiting professor `uq_mc_dropout/summary.json` |
| Version policy | **Never overwrite** older recipes: v2 / v3 / v4 flags and docs stay; v5+ only adds |

Primary Acc recommendation for full data remains **v4** (`--freeze-ensemble`) until a clean full v5 beats it **and** keeps healthy mixture weights.

---

## 8. How achievements stack (confidence track)

```text
Concepts (PE, MC Dropout, Softmax ≠ confidence)
    → eval scripts + shared metrics (UAUC = AUROC(PE))
        → 10% baseline vs MaSE table
            → confident finetune (↓ PE, Acc still >> baseline)
                → temperature sweep (deploy sharpening, Acc flat)
                    → 5% v4 UQ snapshot
                        → full-data UQ (next: professor JSON)
```

**Success criteria we use**

| Goal | Criterion |
|------|-----------|
| Fair UQ compare | Same split, same \(T\), primary = AUROC(PE) |
| “More confident” MaSE | PE mean ↓ vs prior MaSE, Acc still competitive vs baseline |
| Deploy sharpening | Temperature \(T<1\) with Acc unchanged and UAUC stable |
| Full-data paper UQ | Archive `summary.json` for the chosen Acc checkpoint (likely v4) |

---

## 9. Pointers

| Path | Content |
|------|---------|
| `docs/UQ_10PCT_BASELINE_VS_MASE.md` | 10% numbers + interpretation |
| `docs/BASELINE_UQ_COMPARE.md` | Reproduce baseline vs MaSE UQ |
| `docs/CONFIDENT_RECIPE_10PCT.md` | Confident finetune commands |
| `docs/UQ_MC_DROPOUT.md` | Eval usage |
| `docs/mase_results/uq_10pct/` | Raw 10% JSON |
| `docs/mase_results/uq_5pct/mase_v4/` | Raw 5% v4 UQ JSON |
| MaSE repo `V4_RUN.md` / `V5_RUN.md` | Professor full-data + UQ send-back list |

---

# Part II — 中文

---

## 1. 我们说的「自信度」指什么

| 术语 | 在本流水线中的含义 |
|------|-------------------|
| **Softmax 自信度** | 单次前向的 \(\max_c p(c\mid x)\) —— **不是**主要科学结论 |
| **预测熵 PE** | 对 **MC 平均**后的类分布算熵；PE 越低 ≈ 分布越尖 / 越「自信」 |
| **AUROC(PE) = UAUC** | PE 把**错误**样本排在正确样本前面的能力 —— **主指标** |
| **PE mean** | 某划分上 PE 的平均值 —— 只描述「尖不尖」；**单独越低并不等于模型更好** |

**固定原则：** 主报 **AUROC(PE)**；`UAUC_via_1_minus_maxprob` 仅作 Softmax 基线对照。

---

## 2. 已建成的工具（架构不变）

自信度评估**不改** MaSE Lite 网络结构。训练配方可调 dropout / label smoothing；推理端 UQ 用 MC Dropout。

| 产物 | 作用 |
|------|------|
| `eval_mase_uq.py` | MC Dropout → PE、UAUC、τ 扫描 |
| `eval_deepyeast_uq.py` | 官方 DeepYeastNet 同一套 UQ |
| `train_deepyeast_baseline.py` | 训练论文基线，便于公平对比 |
| `temp_sweep_mase.py` | 后处理温度 \(T\)（无需重训） |
| `--with-uq` / `--v5` | 训练结束后可选写入 `results.json` 的 UQ 摘要 |

概念笔记：

- `docs/CNN_UQ_concepts_bilingual.md`
- `docs/Native_Multiclass_concepts_bilingual.md`

---

## 3. 主要实证结果 — 10% 数据（seed=42，\(T=30\)）

**划分：** `deepyeast_10pct`（train 6500 / val 1250 / test 1250）。  
**详表：** `docs/UQ_10PCT_BASELINE_VS_MASE.md`  
**原始 JSON：** `docs/mase_results/uq_10pct/`

| 模型 | Test Acc | AUROC(PE) | AUROC(1−maxp) | PE mean |
|------|----------|-----------|---------------|---------|
| DeepYeastNet 基线 | 81.3% | **0.883** | 0.890 | 0.58 |
| MaSE Lite v2 | **86.0%** | 0.872 | 0.876 | 0.92 |
| MaSE Lite **confident** | 85.2% | 0.877 | 0.876 | **0.39** |

### 结论要点

1. **准确率：** MaSE v2 / confident 明显超过官方基线（+4.7 / +3.9 个百分点）。
2. **UQ 排序质量：** 基线 AUROC(PE) 略高；confident 已非常接近（0.877 vs 0.883），同时准确率远高于基线。
3. **尖锐程度：** 默认 MaSE v2 **偏不自信**（PE mean 0.92 ≫ 基线 0.58）。**confident 配方**把 PE mean 压到 **0.39**（比基线更尖），相对 v2 仅小幅掉点（−0.8 pp）。

---

## 4. Confident 训练配方（10% 试点）

**目标：** 相对 `triple_fusion_lite_10pct_v2` **提高自信度**（↓ PE），**不改架构**。  
**文档：** `docs/CONFIDENT_RECIPE_10PCT.md`

| 旋钮 | v2 | confident |
|------|----|-----------|
| `label_smoothing` | 0.1 | **0** |
| `branch_dropout` / `head_dropout` | 0.4 / 0.5 | **0.25 / 0.3** |
| 初始化 | PLCNN + masked | **从 v2 `best.pt` 续训** |
| 轮数 | 60 | **20**（patience 8） |
| Backbone LR | 1e-3 | **3e-4** |
| 损失 | `--legacy-v2-loss` | 同一族 |

旧版 v2 checkpoint 与文档保留；confident 使用**新** checkpoint 名（不覆盖原则）。

---

## 5. 温度缩放（无需重训）

在 **MaSE confident** 上对概率做 \(p \propto \exp(\log p / T)\)。确定性 Acc 不变；\(T\) 下降则 PE 下降；UAUC 稳定在约 0.88。

| \(T\) | det Acc | det PE | mc PE | mc UAUC |
|-------|---------|--------|-------|---------|
| 1.0 | 85.12% | 0.369 | 0.388 | 0.879 |
| 0.9 | 85.12% | 0.324 | 0.343 | 0.881 |
| 0.8 | 85.12% | 0.281 | 0.301 | 0.880 |
| 0.7 | 85.12% | 0.241 | 0.260 | 0.878 |
| 0.5 | 85.12% | 0.165 | 0.186 | 0.878 |

**部署建议：** \(T \approx 0.7\)–\(0.8\)，在不掉 Acc 的前提下提高自信度。  
完整表：`docs/mase_results/uq_10pct/mase_confident/temp_sweep.json`。

---

## 6. 5% 试点 — MaSE Lite v4 的 UQ

同一 MC 协议（\(T=30\)），本地 5% v4。  
JSON：`docs/mase_results/uq_5pct/mase_v4/uq_summary.json`

| 测试集指标 | 数值 |
|------------|------|
| Accuracy | **85.87%** |
| AUROC(PE) = UAUC | **0.873** |
| AUROC(1−maxp) | 0.882 |
| PE mean | 1.04 |

说明：5% 上 v4 已具备可用的错误排序能力（UAUC≈0.87）；PE mean 偏高（不如 10% confident 尖锐），与数据更少、正则更强一致。

---



### 5% v4 — 不掉 Acc 抬高 AUROC（2026-07-31）

分类器冻结（同一套 MC argmax，test Acc **85.71%**）。更好的不确定度分数：

| 分数 | Test AUROC |
|------|------------|
| PE (MC) | 0.875 |
| 1−maxprob | 0.885 |
| Mahalanobis（预测类，correct-fit） | 0.892 |
| **z(1−maxp)+z(Mahalanobis)** | **0.901** |

相对 PE 约 **+0.026**。JSON：`docs/mase_results/uq_5pct/mase_v4/mahalanobis_summary.json`。  
脚本（MaSE 仓库）：`eval_mase_uq_mahalanobis.py`、`eval_mase_uq_feature_probe.py`。

## 7. 全量数据轨道（准确率 SOTA vs UQ 待归档）

| 事项 | 状态 |
|------|------|
| 全量 Acc **v4** | ≈**89.58%**（老师机结果；正式 `results.json` 待归档） |
| 全量 Acc **v2** | 89.1%（对照） |
| 全量 **UQ**（v4 / v5） | 流程已写好；等老师回传 `uq_mc_dropout/summary.json` |
| 版本策略 | **永不覆盖**旧配方：v2/v3/v4 开关与文档保留；只新增 v5+ |

全量准确率主推仍是 **v4**（`--freeze-ensemble`），直到干净的全量 v5 超过它且混合权重健康。

---

## 8. 自信度轨道成果链条

```text
概念（PE、MC Dropout、Softmax ≠ 真实自信）
    → 评估脚本 + 统一主指标 UAUC=AUROC(PE)
        → 10% 基线 vs MaSE 对照表
            → confident 微调（↓ PE，Acc 仍远超基线）
                → 温度扫描（部署端变尖，Acc 不变）
                    → 5% v4 UQ 快照
                        → 全量 UQ（下一步：老师 JSON）
```

**成功判据**

| 目标 | 标准 |
|------|------|
| 公平 UQ 对比 | 同划分、同 \(T\)，主指标 = AUROC(PE) |
| 「更自信」的 MaSE | 相对旧 MaSE PE mean↓，且 Acc 仍显著优于基线 |
| 部署变尖 | \(T<1\) 且 Acc 不变、UAUC 稳定 |
| 全量论文 UQ | 为选定 Acc 模型（多半是 v4）归档 `summary.json` |

---

## 9. 相关路径

| 路径 | 内容 |
|------|------|
| `docs/UQ_10PCT_BASELINE_VS_MASE.md` | 10% 数字与解读 |
| `docs/BASELINE_UQ_COMPARE.md` | 复现基线 vs MaSE UQ |
| `docs/CONFIDENT_RECIPE_10PCT.md` | Confident 训练命令 |
| `docs/UQ_MC_DROPOUT.md` | 评估用法 |
| `docs/mase_results/uq_10pct/` | 10% 原始 JSON |
| `docs/mase_results/uq_5pct/mase_v4/` | 5% v4 UQ JSON |
| MaSE 仓库 `V4_RUN.md` / `V5_RUN.md` | 老师全量 + 回传清单 |
