#!/usr/bin/env python3
"""Mahalanobis + MC mutual-information UQ (Acc fixed).

Classifier predictions stay the MC-ensemble argmax.
Uncertainty scores:
  - pe_mc
  - one_minus_maxp
  - mc_mi        = H(mean_t p_t) - mean_t H(p_t)   (epistemic)
  - mahal_pred   = Mahalanobis distance to predicted-class Gaussian
                   (Gaussians fit on val features with true labels)
  - pe_plus_mi / pe_plus_mahal / logistic blends on val

Refs: Lee et al. 2018 (Mahalanobis OOD); Gal & Ghahramani MC Dropout MI.
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
from eval_mase_uq_feature_probe import apply_pca, fit_pca, logistic_fit, logistic_score
from mase_lite_net import (
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  enable_mc_dropout,
  predictive_entropy,
)


def _zscore(x: np.ndarray) -> np.ndarray:
  s = float(x.std())
  if s < 1e-12:
    return np.zeros_like(x)
  return (x - float(x.mean())) / s


@torch.inference_mode()
def extract(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  enable_mc_dropout(model)
  y_true_l, y_pred_l = [], []
  pe_l, maxp_l, mi_l = [], [], []
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    probs_stack = []
    for _ in range(mc_samples):
      log_probs = model(images, train=False, return_details=False)
      assert isinstance(log_probs, torch.Tensor)
      probs_stack.append(log_probs.exp())
    stacked = torch.stack(probs_stack, dim=0)  # (T,B,C)
    mu = stacked.mean(dim=0)
    pe_mean = predictive_entropy(mu)
    pe_each = predictive_entropy(stacked.reshape(-1, stacked.size(-1))).view(
      mc_samples, -1
    )
    mi = pe_mean - pe_each.mean(dim=0)
    y_true_l.append(labels.numpy())
    y_pred_l.append(mu.argmax(dim=-1).cpu().numpy())
    pe_l.append(pe_mean.cpu().numpy())
    maxp_l.append(mu.max(dim=-1).values.cpu().numpy())
    mi_l.append(mi.cpu().numpy())

  model.eval()
  feat_l = []
  for images, _ in loader:
    images = images.to(device, non_blocking=True)
    _ens, details = model(images, train=False, return_details=True)
    v, r, d = details["branch_feats"]
    feat = F.normalize(torch.cat([v, r, d], dim=1), p=2, dim=1)
    feat_l.append(feat.cpu().numpy())

  y_true = np.concatenate(y_true_l)
  y_pred = np.concatenate(y_pred_l)
  return {
    "y_true": y_true,
    "y_pred": y_pred,
    "correct": (y_true == y_pred).astype(np.int32),
    "feat": np.concatenate(feat_l).astype(np.float64),
    "pe_mc": np.concatenate(pe_l).astype(np.float64),
    "one_minus_maxp": (1.0 - np.concatenate(maxp_l)).astype(np.float64),
    "mc_mi": np.concatenate(mi_l).astype(np.float64),
  }


def fit_class_gaussians(
  feat: np.ndarray,
  labels: np.ndarray,
  n_class: int,
  reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
  """Tied covariance Mahalanobis (Lee et al.). Returns means (C,D), precision (D,D)."""
  d = feat.shape[1]
  means = np.zeros((n_class, d), dtype=np.float64)
  cov = np.zeros((d, d), dtype=np.float64)
  n_tot = 0
  for c in range(n_class):
    xc = feat[labels == c]
    if len(xc) == 0:
      means[c] = feat.mean(axis=0)
      continue
    means[c] = xc.mean(axis=0)
    cov += np.cov(xc, rowvar=False) * max(len(xc) - 1, 1)
    n_tot += max(len(xc) - 1, 1)
  if n_tot <= 0:
    cov = np.eye(d)
  else:
    cov = cov / n_tot
  cov = cov + reg * np.eye(d)
  prec = np.linalg.pinv(cov)
  return means, prec


def mahalanobis_to_pred(
  feat: np.ndarray,
  y_pred: np.ndarray,
  means: np.ndarray,
  prec: np.ndarray,
) -> np.ndarray:
  out = np.zeros(len(feat), dtype=np.float64)
  for i in range(len(feat)):
    c = int(y_pred[i])
    delta = feat[i] - means[c]
    out[i] = float(delta @ prec @ delta)
  return out


def mahalanobis_min(
  feat: np.ndarray,
  means: np.ndarray,
  prec: np.ndarray,
) -> np.ndarray:
  # distance to nearest class center
  out = np.full(len(feat), np.inf, dtype=np.float64)
  for c in range(means.shape[0]):
    delta = feat - means[c]
    dist = np.einsum("nd,dd,nd->n", delta, prec, delta)
    out = np.minimum(out, dist)
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--data-dir", type=Path, required=True)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--results-json", type=Path, default=None)
  p.add_argument("--out-dir", type=Path, default=None)
  p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  p.add_argument("--mc-samples", type=int, default=20)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--num-workers", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--pca-dim", type=int, default=64)
  p.add_argument("--top-k", type=int, default=None)
  p.add_argument("--soft-alpha", type=float, default=None)
  p.add_argument("--branch-dropout", type=float, default=None)
  p.add_argument("--head-dropout", type=float, default=None)
  args = p.parse_args()

  set_seed(args.seed)
  device = resolve_device(args.device)
  ckpt = args.checkpoint.resolve()
  results_json = args.results_json or (
    (ckpt.parent / "results.json") if (ckpt.parent / "results.json").exists() else None
  )
  out_dir = args.out_dir.resolve() if args.out_dir else (ckpt.parent / "uq_mahalanobis")
  out_dir.mkdir(parents=True, exist_ok=True)

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  for attr, val in [
    ("top_k_patches", args.top_k),
    ("soft_mask_alpha", args.soft_alpha),
    ("branch_dropout", args.branch_dropout),
    ("head_dropout", args.head_dropout),
  ]:
    if val is not None:
      setattr(config, attr, val)

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded {ckpt} device={device}", flush=True)

  _, val_loader, test_loader = make_loaders(
    args.data_dir.resolve(),
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )
  print("Extract val...", flush=True)
  val = extract(model, val_loader, device, args.mc_samples)
  print("Extract test...", flush=True)
  test = extract(model, test_loader, device, args.mc_samples)

  acc = float(test["correct"].mean())
  yt = 1 - test["correct"]
  yv = 1 - val["correct"]
  print(f"Frozen test Acc={acc:.4f}", flush=True)

  mu, comp, ev = fit_pca(val["feat"], args.pca_dim)
  val_pca = apply_pca(val["feat"], mu, comp)
  test_pca = apply_pca(test["feat"], mu, comp)
  means, prec = fit_class_gaussians(val_pca, val["y_true"], config.num_classes, reg=1e-2)
  val["mahal_pred"] = mahalanobis_to_pred(val_pca, val["y_pred"], means, prec)
  test["mahal_pred"] = mahalanobis_to_pred(test_pca, test["y_pred"], means, prec)
  val["mahal_min"] = mahalanobis_min(val_pca, means, prec)
  test["mahal_min"] = mahalanobis_min(test_pca, means, prec)

  # also fit only on correctly classified val (sometimes better)
  mask = val["correct"].astype(bool)
  if mask.sum() > config.num_classes * 3:
    means_c, prec_c = fit_class_gaussians(
      val_pca[mask], val["y_true"][mask], config.num_classes, reg=1e-2
    )
    test["mahal_pred_correctfit"] = mahalanobis_to_pred(
      test_pca, test["y_pred"], means_c, prec_c
    )
  else:
    test["mahal_pred_correctfit"] = test["mahal_pred"]

  score_names = [
    "pe_mc",
    "one_minus_maxp",
    "mc_mi",
    "mahal_pred",
    "mahal_min",
    "mahal_pred_correctfit",
  ]
  rows = {}
  print("\n=== single scores ===", flush=True)
  for name in score_names:
    auc = roc_auc_score(yt, test[name])
    rows[name] = float(auc)
    print(f"  {name:28s} AUROC={auc:.4f}", flush=True)

  # combos
  combos = {
    "z(pe)+z(mi)": _zscore(test["pe_mc"]) + _zscore(test["mc_mi"]),
    "z(pe)+z(mahal_pred)": _zscore(test["pe_mc"]) + _zscore(test["mahal_pred"]),
    "z(1-maxp)+z(mahal_pred)": _zscore(test["one_minus_maxp"])
    + _zscore(test["mahal_pred"]),
    "z(pe)+z(mi)+z(mahal)": _zscore(test["pe_mc"])
    + _zscore(test["mc_mi"])
    + _zscore(test["mahal_pred"]),
  }
  print("\n=== fixed combos ===", flush=True)
  for name, score in combos.items():
    auc = roc_auc_score(yt, score)
    rows[name] = float(auc)
    print(f"  {name:28s} AUROC={auc:.4f}", flush=True)

  # val-tuned logistic on [pe, 1-maxp, mi, mahal]
  feats = ["pe_mc", "one_minus_maxp", "mc_mi", "mahal_pred"]
  xv = np.stack([_zscore(val[f]) for f in feats], axis=1)
  xt = np.stack([_zscore(test[f]) for f in feats], axis=1)
  # use train stats: zscore each with val stats for fairness
  def z_by_val(v, t):
    mu = v.mean()
    sd = v.std() if v.std() > 1e-12 else 1.0
    return (v - mu) / sd, (t - mu) / sd

  xv_cols, xt_cols = [], []
  for f in feats:
    a, b = z_by_val(val[f], test[f])
    xv_cols.append(a)
    xt_cols.append(b)
  xv = np.stack(xv_cols, axis=1)
  xt = np.stack(xt_cols, axis=1)
  w = logistic_fit(xv, yv.astype(np.float64), l2=1e-2, steps=2000)
  log_auc = roc_auc_score(yt, logistic_score(xt, w))
  rows["logistic_pe_maxp_mi_mahal"] = float(log_auc)
  print(f"\n  logistic_pe_maxp_mi_mahal     AUROC={log_auc:.4f}", flush=True)

  best_name = max(rows, key=lambda k: rows[k])
  pe_ref = rows["pe_mc"]
  print(
    f"\nBEST {best_name}={rows[best_name]:.4f}  "
    f"Δpe={rows[best_name] - pe_ref:+.4f}  Acc={acc:.4f} (fixed)",
    flush=True,
  )

  summary = {
    "method": "mase_lite_uq_mahalanobis_mi",
    "checkpoint": str(ckpt),
    "mc_samples": args.mc_samples,
    "pca_dim": int(comp.shape[0]),
    "pca_explained": ev,
    "accuracy_test": acc,
    "scores": rows,
    "best": {"name": best_name, "AUROC": rows[best_name]},
    "delta_vs_pe_mc": rows[best_name] - pe_ref,
  }
  out = out_dir / "mahalanobis_summary.json"
  with out.open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
  main()
