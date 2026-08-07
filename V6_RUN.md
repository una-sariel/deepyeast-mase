# MaSE-Net Lite v6 — TP-AHF full-data run (from zero)

**TP-AHF** = **T**wo-**P**hase **A**daptive **H**ead **F**usion under simplex constraints.

| Phase | Idea | Checkpoint folder |
|-------|------|-------------------|
| **Phase 1** | Same as v4: fused-CE, **frozen** w=0.25×4 | `mase_lite_full_v6_phase1` |
| **Phase 2** | Resume Phase-1; **learn only** `ensemble_logits` with floor + entropy reg | `mase_lite_full_v6` |
| **Phase 2.5** | Resume Phase-1; learn **`ensemble_logits` + heads** (branches frozen) | `mase_lite_full_v6_phase25` |

**Goal:** beat v4 test **89.58%** with **learnable but constrained** head weights (paper innovation). **Phase 2.5** is the recommended accuracy push after Phase 2 — see [V6_PHASE25_RUN.md](V6_PHASE25_RUN.md).

**Repo:** https://github.com/una-sariel/deepyeast-mase  

**Does not replace v4/v5 docs** — see also [V4_RUN.md](V4_RUN.md), [FULL_DATA_QUICKSTART.md](FULL_DATA_QUICKSTART.md).

---

## What to send back (after all steps)

```text
1) <deepyeast_full>\checkpoints\mase_lite_full_v6_phase1\results.json
2) <deepyeast_full>\checkpoints\mase_lite_full_v6\results.json
3) <deepyeast_full>\checkpoints\mase_lite_full_v6_phase25\results.json   (recommended after Phase 2)
4) <deepyeast_full>\checkpoints\mase_lite_full_v6\uq_mc_dropout\summary.json   (optional Step 4)
```

---

## Step 0 — Clone, LFS, venv (one time)

```powershell
cd "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v7"
git pull
git lfs install
git lfs pull

dir artifacts\checkpoints_5pct\plcnn_triple\best.pt
dir artifacts\checkpoints_5pct\masked_v3k60_pytorch\best.pt
```

Each `.pt` should be **~28 MB** and **~44 MB**, not ~1 KB.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

Verify v6 flags exist:

```powershell
python -c "import pathlib; t=pathlib.Path('pytorch/train_mase_lite.py').read_text(encoding='utf-8'); print('v6 OK' if '--v6-phase1' in t else 'git pull for v6')"
```

---

## Step 1 — Download **full** data (`deepyeast_full`)

**Important:** default `prepare_deepyeast_subset.py` without flags → **5% only** → folder `deepyeast_5pct` (~4,499 images).  
For v6 you need **`--fraction 1.0`**.

From repo root:

```powershell
python prepare_deepyeast_subset.py --fraction 1.0 --out-dir deepyeast_full --seed 42
```

| Step | What happens |
|------|----------------|
| 1 | Download manifests (cached in `%USERPROFILE%\.deepyeast\cache\`) |
| 2 | Stratified sample: train 65000, val 12500, test 12500 |
| 3 | Download `main.tar.gz` (~398 MB, cached) |
| 4 | Extract PNGs → `deepyeast_full/train|val|test/<class>/` |

**Expected totals:** **90,000** images, **12** classes.

```powershell
dir deepyeast_full\labels.csv
dir deepyeast_full\train
```

If the folder already exists, the script **deletes and recreates** it.

Set your data path (edit for your machine):

```powershell
$DATA = "D:\UG Research\DeepYeast\Qiwu\deepyeast-mase-main_v7\deepyeast_full"
```

---

## Step 2 — Phase 1 (uniform fusion, ~60 epochs)

Same recipe as v4; progress bar **`MaSELiteV6P1`**.

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase1 `
  --epochs 60 `
  --patience 15 `
  --batch-size 64 `
  --top-k 40 `
  --soft-alpha 0.5 `
  --mask-sparsity-weight 0.05 `
  --label-smoothing 0.1 `
  --aux-head-weight 0.5 `
  --distill-weight 0.1 `
  --seed 42
```

Default checkpoint: `$DATA\checkpoints\mase_lite_full_v6_phase1\`

### Check log

- Every epoch: **`w=[0.25,0.25,0.25,0.25]`**
- `method` in results → `mase_lite_v6_phase1`
- Target: test **≥ ~0.895** (similar to v4 **89.58%**)

Wall time: ~**2 h** on GPU.

---

## Step 3 — Phase 2 (learn head weights, ~15 epochs)

**Auto-resumes** Phase-1 `best.pt` from Step 2.  
Trains **only** `ensemble_logits` (backbone + heads frozen).

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase2 `
  --seed 42
```

Default checkpoint: `$DATA\checkpoints\mase_lite_full_v6\`

### Phase 2 defaults (built into `--v6-phase2`)

| Knob | Value |
|------|-------|
| `min_ensemble_weight` | **0.20** (each w ∈ [0.20, 0.40]) |
| `ensemble_entropy_weight` | **0.02** |
| `ensemble_lr` | **0.25 × backbone_lr** |
| Temperature | **1.5 → 1.0** linear anneal |
| epochs / patience | **15 / 5** |
| aux / distill / sparsity | **off** (ensemble-only fine-tune) |

### Check log

- Progress bar: **`MaSELiteV6P2`**
- Lines show **`wh=ok`** or **`wh=BAD`**
- **`wh=ok`** means `min(w)≥0.10` and `max(w)≤0.40`
- **`best.pt` saved only if** val ≥ Phase-1 best val **and** `wh=ok`

### Success criteria

| Metric | Target |
|--------|--------|
| `test.accuracy` | **> Phase-1 test** and ideally **> 0.8958** (v4) |
| `weight_health.ok` | **true** |
| `max(w)` | **≤ 0.40** (not 0.44 like unconstrained v5) |
| learned `w` | non-uniform but balanced, e.g. ~[0.22, 0.22, 0.30, 0.26] |

If Phase 2 test **≤ Phase 1**, report Phase-1 numbers for accuracy; Phase 2 still documents the adaptive-fusion ablation.

Wall time: ~**20–30 min** on GPU.

---

## Step 4 — Optional UQ on final v6 checkpoint

```powershell
python pytorch\eval_mase_uq.py `
  --data-dir $DATA `
  --checkpoint "$DATA\checkpoints\mase_lite_full_v6\best.pt" `
  --mc-samples 30 `
  --split both `
  --out-dir "$DATA\checkpoints\mase_lite_full_v6\uq_mc_dropout"
```

See [UQ_MC_DROPOUT.md](UQ_MC_DROPOUT.md). Report **UAUC** (PE) from `summary.json`.

---

## Override paths (if Phase 1 already exists as v4)

If you already have v4 `best.pt` and want to skip re-training Phase 1:

```powershell
python pytorch\train_mase_lite.py `
  --data-dir $DATA `
  --v6-phase2 `
  --resume "$DATA\checkpoints\mase_lite_full_v4\best.pt" `
  --phase1-checkpoint-name mase_lite_full_v4 `
  --phase1-val-acc 0.8947 `
  --seed 42
```

(`--phase1-val-acc` = v4 best val from your log; else read from that folder's `results.json`.)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No `deepyeast_full` after prepare | You ran default prepare → got `deepyeast_5pct`. Re-run with `--fraction 1.0 --out-dir deepyeast_full` |
| `--v6-phase2 needs Phase-1 best.pt` | Run Step 2 first, or pass `--resume` to an existing v4/v6_phase1 `best.pt` |
| `w` stuck at 0.25 in Phase 2 | Need `--v6-phase2` (not `--freeze-ensemble`) |
| All epochs `wh=BAD` | Weights too peaky; try `--min-ensemble-weight 0.22` or lower `--ensemble-lr-ratio 0.15` |
| Tiny `.pt` / load error | `git lfs pull` |
| Phase 2 test below Phase 1 | Use Phase-1 checkpoint for accuracy table; Phase-2 as negative/ ablation result |

---

## Paper one-liner (TP-AHF)

> Phase 1 trains MaSE's four progressive heads with uniform fusion (v4). Phase 2 calibrates mixture weights on a simplex with minimum mass and entropy regularization, learning data-driven head importance without single-head collapse.
