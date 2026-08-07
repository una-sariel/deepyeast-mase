# v7 — 你刚跑完 Phase 2.5，按这个做

**适用：** 全量 Phase 2.5 已跑完（test ≈ **89.07%**），机器上已有 Phase 1 `best.pt`。

**你的路径（v9）：**

```powershell
$REPO = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast-mase"
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v9\deepyeast_full"
```

**Repo：** https://github.com/una-sariel/deepyeast-mase

---

## 0. 更新代码（1 分钟）

```powershell
cd $REPO
git pull
.venv\Scripts\activate
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('v7 OK' if '--v7' in t else 'git pull again')"
```

---

## 1. 确认本地文件（30 秒）

```powershell
dir "$DATA\checkpoints\mase_lite_full_v6_phase1\best.pt"
dir "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt"
dir "$DATA\checkpoints\mase_lite_full_v6_phase25\results.json"
```

三个都在 → **不用重训 Phase 1**，不用重新下数据。

---

## 2. Phase 2.5 上跑 TTA（~15 min，不重训）

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v6_phase25\best.pt" `
  --split test `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6_phase25\tta_eval"
```

看最后一行，例如：

```text
test: baseline=0.8907  tta=0.894x  gain=+0.3xpp
```

---

## 3. 训 v7（~10–45 min）

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v7 `
  --seed 42
```

进度条：**`MaSELiteV7`**  
输出：`$DATA\checkpoints\mase_lite_full_v7\`

---

## 4. v7 上跑 TTA（~15 min）

```powershell
python pytorch\eval_mase_tta.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v7\best.pt" `
  --split both `
  --tta-mode flip_rot `
  --out-dir "$DATA\checkpoints\mase_lite_full_v7\tta_eval"
```

---

## 5. 发回这 3 个文件

```text
checkpoints\mase_lite_full_v6_phase25\tta_eval\summary.json
checkpoints\mase_lite_full_v7\results.json
checkpoints\mase_lite_full_v7\tta_eval\summary.json
```

| 表格行 | 读哪个字段 |
|--------|------------|
| Phase 2.5 | phase25 `results.json` → `test.accuracy` |
| Phase 2.5 + TTA | phase25 `tta_eval/summary.json` → `test_tta_accuracy` |
| v7 + TTA | v7 `tta_eval/summary.json` → `test_tta_accuracy` |

---

## 顺序

```text
git pull  →  TTA(phase25)  →  训 v7  →  TTA(v7)  →  发 3 个 json
```
