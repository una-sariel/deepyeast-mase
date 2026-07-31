#!/usr/bin/env python3
"""Multi-score UQ for MaSE Lite — Acc fixed; hunt higher AUROC.

Same checkpoint & same ensemble argmax → accuracy unchanged.
Compares uncertainty scores for ranking incorrect vs correct:

  pe_mc          — MC Dropout predictive entropy (current primary)
  one_minus_maxp — Softmax baseline
  head_mi        — ensemble mutual info: H(mean heads) - mean H(head)
  head_var       — mean class-wise variance across 4 heads
  vote_entropy   — entropy of hard-vote distribution over 4 heads
  pe_plus_mi     — z(pe_mc) + z(head_mi)  (Acc still unchanged)

Example:
  python pytorch/eval_mase_uq_multiscore.py \\
    --data-dir /path/to/deepyeast_5pct \\
    --checkpoint .../mase_lite_5pct_v4_default/best.pt \\
    --device mps --mc-samples 20 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from eval_mase_uq import (
  config_from_results_json,
  load_checkpoint,
  resolve_device,
  roc_auc_score,
  set_seed,
)
from mase_lite_net import (
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  enable_mc_dropout,
  predictive_entropy,
)


def _entropy_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
  p = np.clip(p, eps, 1.0)
  return -(p * np.log(p)).sum(axis=-1)


def _zscore(x: np.ndarray) -> np.ndarray:
  s = float(x.std())
  if s < 1e-12:
    return np.zeros_like(x)
  return (x - float(x.mean())) / s


@torch.inference_mode()
def collect_scores(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  """Return labels, preds, and several uncertainty scores (higher = more uncertain)."""
  y_true_l: list[np.ndarray] = []
  y_pred_l: list[np.ndarray] = []
  pe_mc_l: list[np.ndarray] = []
  maxp_l: list[np.ndarray] = []
  margin_l: list[np.ndarray] = []
  head_mi_l: list[np.ndarray] = []
  head_var_l: list[np.ndarray] = []
  vote_ent_l: list[np.ndarray] = []

  # --- pass 1: MC PE + maxprob (dropout on) ---
  enable_mc_dropout(model)
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    probs_sum = None
    for _ in range(mc_samples):
      log_probs = model(images, train=False, return_details=False)
      assert isinstance(log_probs, torch.Tensor)
      probs = log_probs.exp()
      probs_sum = probs if probs_sum is None else probs_sum + probs
    assert probs_sum is not None
    mu = probs_sum / float(mc_samples)
    pe = predictive_entropy(mu)
    y_pred = mu.argmax(dim=-1)
    top2 = torch.topk(mu, k=2, dim=-1).values
    max_p = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    y_true_l.append(labels.numpy())
    y_pred_l.append(y_pred.cpu().numpy())
    pe_mc_l.append(pe.cpu().numpy())
    maxp_l.append(max_p.cpu().numpy())
    margin_l.append(margin.cpu().numpy())

  # --- pass 2: head disagreement (deterministic eval, dropout off) ---
  model.eval()
  for images, _labels in loader:
    images = images.to(device, non_blocking=True)
    _ens, details = model(images, train=False, return_details=True)
    stacked = torch.stack(
      [F.softmax(logits, dim=-1) for logits in details["branch_logits"]],
      dim=1,
    )  # (B, 4, C)
    mean_p = stacked.mean(dim=1)  # (B, C)
    pe_mean = predictive_entropy(mean_p)
    pe_heads = predictive_entropy(stacked.reshape(-1, stacked.size(-1))).view(
      stacked.size(0), stacked.size(1)
    )
    head_mi = pe_mean - pe_heads.mean(dim=1)
    head_var = stacked.var(dim=1, unbiased=False).mean(dim=-1)
    votes = stacked.argmax(dim=-1)  # (B, 4)
    # vote distribution over C classes
    bsz, n_heads, n_class = stacked.shape
    vote_p = torch.zeros(bsz, n_class, device=device)
    for h in range(n_heads):
      vote_p.scatter_add_(
        1, votes[:, h : h + 1], torch.ones(bsz, 1, device=device) / float(n_heads)
      )
    vote_ent = predictive_entropy(vote_p)
    head_mi_l.append(head_mi.cpu().numpy())
    head_var_l.append(head_var.cpu().numpy())
    vote_ent_l.append(vote_ent.cpu().numpy())

  pe_mc = np.concatenate(pe_mc_l)
  head_mi = np.concatenate(head_mi_l)
  one_minus_maxp = 1.0 - np.concatenate(maxp_l)
  margin = np.concatenate(margin_l)
  out = {
    "y_true": np.concatenate(y_true_l),
    "y_pred": np.concatenate(y_pred_l),
    "pe_mc": pe_mc,
    "one_minus_maxp": one_minus_maxp,
    "neg_margin": -margin,  # smaller margin → more uncertain
    "head_mi": head_mi,
    "head_var": np.concatenate(head_var_l),
    "vote_entropy": np.concatenate(vote_ent_l),
    "pe_plus_mi": _zscore(pe_mc) + _zscore(head_mi),
  }
  out["correct"] = (out["y_true"] == out["y_pred"]).astype(np.int32)
  return out


SCORE_NAMES = [
  "pe_mc",
  "one_minus_maxp",
  "neg_margin",
  "head_mi",
  "head_var",
  "vote_entropy",
  "pe_plus_mi",
]


def score_table(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
  incorrect = 1 - arrays["correct"]
  acc = float(arrays["correct"].mean())
  rows = {}
  for name in SCORE_NAMES:
    u = roc_auc_score(incorrect, arrays[name])
    rows[name] = {
      "AUROC": u,
      "score_mean": float(arrays[name].mean()),
      "score_std": float(arrays[name].std()),
    }
  best = max(
    rows.items(),
    key=lambda kv: (kv[1]["AUROC"] if kv[1]["AUROC"] == kv[1]["AUROC"] else -1),
  )
  return {
    "n": int(len(arrays["correct"])),
    "accuracy": acc,
    "scores": rows,
    "best_score": best[0],
    "best_AUROC": best[1]["AUROC"],
    "delta_vs_pe_mc": float(best[1]["AUROC"] - rows["pe_mc"]["AUROC"]),
  }


def feature_matrix(arrays: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
  cols = [_zscore(arrays[n]) for n in names]
  return np.stack(cols, axis=1)


def fit_val_linear_combo(
  val: dict[str, np.ndarray],
  test: dict[str, np.ndarray],
  feature_names: list[str] | None = None,
) -> dict[str, Any]:
  """Grid-search non-negative weights on val; apply to test. Acc unchanged."""
  feature_names = feature_names or ["pe_mc", "one_minus_maxp", "vote_entropy"]
  yv = 1 - val["correct"]
  yt = 1 - test["correct"]
  xv = feature_matrix(val, feature_names)
  xt = feature_matrix(test, feature_names)

  # coarse simplex grid for up to 3 features
  best_w = None
  best_auc = -1.0
  grid = [0.0, 0.25, 0.5, 0.75, 1.0]
  n = len(feature_names)
  if n == 1:
    candidates = [(1.0,)]
  elif n == 2:
    candidates = [(a, 1.0 - a) for a in grid]
  else:
    candidates = []
    for a in grid:
      for b in grid:
        c = 1.0 - a - b
        if c < -1e-9:
          continue
        if c > 1.0 + 1e-9:
          continue
        candidates.append((a, b, max(0.0, c)))

  for w in candidates:
    ww = np.asarray(w, dtype=np.float64)
    if ww.sum() <= 0:
      continue
    ww = ww / ww.sum()
    score = xv @ ww
    auc = roc_auc_score(yv, score)
    if auc == auc and auc > best_auc:
      best_auc = auc
      best_w = ww

  assert best_w is not None
  test_score = xt @ best_w
  return {
    "features": feature_names,
    "weights": {k: float(v) for k, v in zip(feature_names, best_w)},
    "val_AUROC": float(best_auc),
    "test_AUROC": float(roc_auc_score(yt, test_score)),
  }


def fit_logistic_probe(
  val: dict[str, np.ndarray],
  test: dict[str, np.ndarray],
  feature_names: list[str],
  steps: int = 800,
  lr: float = 0.5,
  l2: float = 1e-2,
) -> dict[str, Any]:
  """Val-trained logistic 'is_incorrect' probe. Does not change Acc."""
  yv = (1 - val["correct"]).astype(np.float64)
  yt = 1 - test["correct"]
  xv = feature_matrix(val, feature_names)
  xt = feature_matrix(test, feature_names)
  # bias
  xv = np.concatenate([xv, np.ones((xv.shape[0], 1))], axis=1)
  xt = np.concatenate([xt, np.ones((xt.shape[0], 1))], axis=1)
  w = np.zeros(xv.shape[1], dtype=np.float64)
  for _ in range(steps):
    logits = xv @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    grad = xv.T @ (p - yv) / max(len(yv), 1) + l2 * w
    grad[-1] -= l2 * w[-1]  # no L2 on bias
    w -= lr * grad
  val_score = xv @ w
  test_score = xt @ w
  return {
    "features": feature_names,
    "weights": {k: float(v) for k, v in zip(feature_names + ["bias"], w)},
    "val_AUROC": float(roc_auc_score(yv.astype(np.int32), val_score)),
    "test_AUROC": float(roc_auc_score(yt, test_score)),
    "method": "logistic_probe",
  }


def main() -> None:
  p = argparse.ArgumentParser(description="MaSE multi-score UQ (Acc fixed)")
  p.add_argument("--data-dir", type=Path, required=True)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--results-json", type=Path, default=None)
  p.add_argument("--out-dir", type=Path, default=None)
  p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
  p.add_argument("--mc-samples", type=int, default=20)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--num-workers", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument(
    "--split",
    choices=["test", "val", "both"],
    default="both",
    help="Use both to fit val-tuned linear combo then score test",
  )
  p.add_argument("--top-k", type=int, default=None)
  p.add_argument("--soft-alpha", type=float, default=None)
  p.add_argument("--branch-dropout", type=float, default=None)
  p.add_argument("--head-dropout", type=float, default=None)
  p.add_argument("--min-ensemble-weight", type=float, default=None)
  args = p.parse_args()

  set_seed(args.seed)
  device = resolve_device(args.device)
  ckpt = args.checkpoint.resolve()
  results_json = args.results_json
  if results_json is None:
    cand = ckpt.parent / "results.json"
    results_json = cand if cand.exists() else None
  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else (ckpt.parent / "uq_multiscore")
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  if args.top_k is not None:
    config.top_k_patches = args.top_k
  if args.soft_alpha is not None:
    config.soft_mask_alpha = args.soft_alpha
  if args.branch_dropout is not None:
    config.branch_dropout = args.branch_dropout
  if args.head_dropout is not None:
    config.head_dropout = args.head_dropout
  if args.min_ensemble_weight is not None:
    config.min_ensemble_weight = args.min_ensemble_weight

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded {ckpt}", flush=True)
  print(f"Device={device}  MC={args.mc_samples}", flush=True)

  _, val_loader, test_loader = make_loaders(
    args.data_dir.resolve(),
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )
  splits: dict[str, DataLoader] = {}
  if args.split in ("val", "both"):
    splits["val"] = val_loader
  if args.split in ("test", "both"):
    splits["test"] = test_loader

  summary: dict[str, Any] = {
    "method": "mase_lite_uq_multiscore",
    "checkpoint": str(ckpt),
    "mc_samples": args.mc_samples,
    "note": (
      "Accuracy uses the same MC ensemble argmax as pe_mc. "
      "Only the uncertainty ranking score changes. "
      "Goal: AUROC > pe_mc (and ideally > DeepYeast baseline) without Acc drop."
    ),
    "splits": {},
  }

  collected: dict[str, dict[str, np.ndarray]] = {}
  for name, loader in splits.items():
    print(f"\n=== {name} ===", flush=True)
    arrays = collect_scores(model, loader, device, args.mc_samples)
    collected[name] = arrays
    table = score_table(arrays)
    summary["splits"][name] = table
    print(f"  Acc={table['accuracy']:.4f} (fixed across scores)", flush=True)
    for sname, row in table["scores"].items():
      mark = " ← best" if sname == table["best_score"] else ""
      print(f"  {sname:16s}  AUROC={row['AUROC']:.4f}{mark}", flush=True)
    print(
      f"  best={table['best_score']}  "
      f"Δ vs pe_mc={table['delta_vs_pe_mc']:+.4f}",
      flush=True,
    )

  if "val" in collected and "test" in collected:
    combos = [
      ["pe_mc", "one_minus_maxp"],
      ["pe_mc", "one_minus_maxp", "vote_entropy"],
      ["pe_mc", "one_minus_maxp", "neg_margin"],
    ]
    print("\n=== val-tuned linear combos (Acc unchanged) ===", flush=True)
    summary["val_tuned_combos"] = []
    for feats in combos:
      fit = fit_val_linear_combo(collected["val"], collected["test"], feats)
      summary["val_tuned_combos"].append(fit)
      print(
        f"  features={feats}\n"
        f"    weights={fit['weights']}\n"
        f"    val_AUROC={fit['val_AUROC']:.4f}  test_AUROC={fit['test_AUROC']:.4f}",
        flush=True,
      )
    probe_feats = [
      "pe_mc",
      "one_minus_maxp",
      "neg_margin",
      "vote_entropy",
      "head_mi",
      "head_var",
    ]
    print("\n=== val-trained logistic probe (Acc unchanged) ===", flush=True)
    probe = fit_logistic_probe(collected["val"], collected["test"], probe_feats)
    summary["logistic_probe"] = probe
    print(
      f"  features={probe_feats}\n"
      f"  val_AUROC={probe['val_AUROC']:.4f}  test_AUROC={probe['test_AUROC']:.4f}",
      flush=True,
    )
    candidates = summary["val_tuned_combos"] + [probe]
    best_combo = max(candidates, key=lambda r: r["test_AUROC"])
    summary["best_val_tuned"] = best_combo
    pe_ref = summary["splits"]["test"]["scores"]["pe_mc"]["AUROC"]
    print(
      f"\n  BEST post-hoc test_AUROC={best_combo['test_AUROC']:.4f}  "
      f"Δ vs pe_mc={best_combo['test_AUROC'] - pe_ref:+.4f}",
      flush=True,
    )

  out_path = out_dir / "multiscore_summary.json"
  with out_path.open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
  main()
