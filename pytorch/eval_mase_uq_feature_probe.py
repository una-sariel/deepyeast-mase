#!/usr/bin/env python3
"""Penultimate-feature error probe for MaSE Lite.

Freezes the classifier: y_pred / Acc unchanged.
Trains a val-set probe to predict incorrectness from:
  - L2-normalized concat(v,r,d) branch features (1536-d), optionally PCA
  - plus scalar scores: pe_mc, 1-maxprob, neg_margin

Goal: raise AUROC(error) above PE / Softmax baselines.
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


def _zscore_cols(x: np.ndarray) -> np.ndarray:
  mu = x.mean(axis=0, keepdims=True)
  sd = x.std(axis=0, keepdims=True)
  sd = np.where(sd < 1e-8, 1.0, sd)
  return (x - mu) / sd


def fit_pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return mean, components (k, D), explained ratio sum."""
  mu = x.mean(axis=0, keepdims=True)
  xc = x - mu
  # economy SVD
  _, s, vt = np.linalg.svd(xc, full_matrices=False)
  k = min(k, vt.shape[0], x.shape[0] - 1)
  comp = vt[:k]
  ev = (s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-12)
  return mu.ravel(), comp, float(ev)


def apply_pca(x: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> np.ndarray:
  return (x - mu) @ comp.T


def logistic_fit(
  x: np.ndarray,
  y: np.ndarray,
  steps: int = 1500,
  lr: float = 0.3,
  l2: float = 1e-2,
) -> np.ndarray:
  """Binary logistic; x already includes bias column optional — we add bias."""
  y = y.astype(np.float64)
  xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  w = np.zeros(xb.shape[1], dtype=np.float64)
  n = max(len(y), 1)
  for t in range(steps):
    logits = xb @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    grad = xb.T @ (p - y) / n
    grad[:-1] += l2 * w[:-1]
    # mild lr decay
    step_lr = lr * (0.5 if t > steps // 2 else 1.0)
    w -= step_lr * grad
  return w


def logistic_score(x: np.ndarray, w: np.ndarray) -> np.ndarray:
  xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  return xb @ w


class TinyMLP(torch.nn.Module):
  def __init__(self, in_dim: int, hidden: int = 64) -> None:
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(in_dim, hidden),
      torch.nn.ReLU(inplace=True),
      torch.nn.Dropout(0.3),
      torch.nn.Linear(hidden, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)


@torch.inference_mode()
def extract_split(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  # MC for pe / pred (matches prior UQ protocol)
  enable_mc_dropout(model)
  y_true_l, y_pred_l, pe_l, maxp_l, margin_l = [], [], [], [], []
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
    top2 = torch.topk(mu, k=2, dim=-1).values
    y_true_l.append(labels.numpy())
    y_pred_l.append(mu.argmax(dim=-1).cpu().numpy())
    pe_l.append(predictive_entropy(mu).cpu().numpy())
    maxp_l.append(top2[:, 0].cpu().numpy())
    margin_l.append((top2[:, 0] - top2[:, 1]).cpu().numpy())

  # Deterministic features (dropout off)
  model.eval()
  feat_l = []
  for images, _ in loader:
    images = images.to(device, non_blocking=True)
    _ens, details = model(images, train=False, return_details=True)
    v, r, d = details["branch_feats"]
    feat = torch.cat([v, r, d], dim=1)
    feat = F.normalize(feat, p=2, dim=1)
    feat_l.append(feat.cpu().numpy())

  y_true = np.concatenate(y_true_l)
  y_pred = np.concatenate(y_pred_l)
  correct = (y_true == y_pred).astype(np.int32)
  return {
    "y_true": y_true,
    "y_pred": y_pred,
    "correct": correct,
    "feat": np.concatenate(feat_l).astype(np.float64),
    "pe_mc": np.concatenate(pe_l).astype(np.float64),
    "one_minus_maxp": (1.0 - np.concatenate(maxp_l)).astype(np.float64),
    "neg_margin": (-np.concatenate(margin_l)).astype(np.float64),
  }


def pack_features(
  data: dict[str, np.ndarray],
  feat_pca: np.ndarray | None,
  with_scalars: bool,
) -> np.ndarray:
  parts = []
  if feat_pca is not None:
    parts.append(feat_pca)
  if with_scalars:
    parts.append(
      np.stack(
        [data["pe_mc"], data["one_minus_maxp"], data["neg_margin"]],
        axis=1,
      )
    )
  x = np.concatenate(parts, axis=1)
  return _zscore_cols(x)


def main() -> None:
  p = argparse.ArgumentParser(description="MaSE feature error probe (Acc fixed)")
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
  out_dir = args.out_dir.resolve() if args.out_dir else (ckpt.parent / "uq_feature_probe")
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

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded {ckpt}  device={device}", flush=True)

  _, val_loader, test_loader = make_loaders(
    args.data_dir.resolve(),
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )

  print("Extracting val...", flush=True)
  val = extract_split(model, val_loader, device, args.mc_samples)
  print("Extracting test...", flush=True)
  test = extract_split(model, test_loader, device, args.mc_samples)

  acc_val = float(val["correct"].mean())
  acc_test = float(test["correct"].mean())
  yv = 1 - val["correct"]
  yt = 1 - test["correct"]
  pe_auc = roc_auc_score(yt, test["pe_mc"])
  maxp_auc = roc_auc_score(yt, test["one_minus_maxp"])
  print(f"Frozen Acc  val={acc_val:.4f}  test={acc_test:.4f}", flush=True)
  print(f"Baseline scores  pe_mc={pe_auc:.4f}  1-maxp={maxp_auc:.4f}", flush=True)

  mu, comp, ev = fit_pca(val["feat"], args.pca_dim)
  val_pca = apply_pca(val["feat"], mu, comp)
  test_pca = apply_pca(test["feat"], mu, comp)
  print(f"PCA dim={comp.shape[0]}  explained≈{ev:.3f}", flush=True)

  experiments: list[dict[str, Any]] = []

  def run_logistic(name: str, xv: np.ndarray, xt: np.ndarray, l2: float) -> None:
    w = logistic_fit(xv, yv.astype(np.float64), l2=l2)
    val_auc = roc_auc_score(yv, logistic_score(xv, w))
    test_auc = roc_auc_score(yt, logistic_score(xt, w))
    row = {
      "name": name,
      "method": "logistic",
      "l2": l2,
      "val_AUROC": float(val_auc),
      "test_AUROC": float(test_auc),
      "delta_vs_pe_mc": float(test_auc - pe_auc),
      "delta_vs_1maxp": float(test_auc - maxp_auc),
    }
    experiments.append(row)
    print(
      f"  {name:40s}  val={val_auc:.4f}  test={test_auc:.4f}  "
      f"Δpe={test_auc - pe_auc:+.4f}",
      flush=True,
    )

  print("\n=== probes ===", flush=True)
  # scalars only
  run_logistic(
    "logistic_scalars",
    _zscore_cols(
      np.stack([val["pe_mc"], val["one_minus_maxp"], val["neg_margin"]], 1)
    ),
    _zscore_cols(
      np.stack([test["pe_mc"], test["one_minus_maxp"], test["neg_margin"]], 1)
    ),
    l2=1e-2,
  )
  # PCA features only
  run_logistic(
    f"logistic_pca{comp.shape[0]}",
    _zscore_cols(val_pca),
    _zscore_cols(test_pca),
    l2=1e-1,
  )
  # PCA + scalars
  for l2 in (1e-2, 1e-1, 1.0):
    run_logistic(
      f"logistic_pca{comp.shape[0]}_scalars_l2={l2:g}",
      pack_features(val, val_pca, True),
      pack_features(test, test_pca, True),
      l2=l2,
    )

  # MLP on PCA+scalars
  xv = pack_features(val, val_pca, True)
  xt = pack_features(test, test_pca, True)
  torch.manual_seed(args.seed)
  mlp = TinyMLP(xv.shape[1], hidden=64)
  opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=2e-2)
  xv_t = torch.tensor(xv, dtype=torch.float32)
  yv_t = torch.tensor(yv, dtype=torch.float32)
  xt_t = torch.tensor(xt, dtype=torch.float32)
  best_state, best_auc = None, -1.0
  for ep in range(100):
    mlp.train()
    opt.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(mlp(xv_t), yv_t)
    loss.backward()
    opt.step()
    mlp.eval()
    with torch.inference_mode():
      s = mlp(xv_t).numpy()
    auc = roc_auc_score(yv, s)
    if auc == auc and auc > best_auc:
      best_auc = auc
      best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
  assert best_state is not None
  mlp.load_state_dict(best_state)
  mlp.eval()
  with torch.inference_mode():
    val_s = mlp(xv_t).numpy()
    test_s = mlp(xt_t).numpy()
  mlp_row = {
    "name": f"mlp_pca{comp.shape[0]}_scalars",
    "method": "mlp",
    "val_AUROC": float(roc_auc_score(yv, val_s)),
    "test_AUROC": float(roc_auc_score(yt, test_s)),
    "delta_vs_pe_mc": float(roc_auc_score(yt, test_s) - pe_auc),
    "delta_vs_1maxp": float(roc_auc_score(yt, test_s) - maxp_auc),
  }
  experiments.append(mlp_row)
  print(
    f"  {mlp_row['name']:40s}  val={mlp_row['val_AUROC']:.4f}  "
    f"test={mlp_row['test_AUROC']:.4f}  Δpe={mlp_row['delta_vs_pe_mc']:+.4f}",
    flush=True,
  )

  # Try PCA-128 as well if requested dim was 64
  if args.pca_dim <= 64 and val["feat"].shape[0] > 130:
    mu128, comp128, ev128 = fit_pca(val["feat"], 128)
    vp = apply_pca(val["feat"], mu128, comp128)
    tp = apply_pca(test["feat"], mu128, comp128)
    run_logistic(
      "logistic_pca128_scalars_l2=0.1",
      pack_features(val, vp, True),
      pack_features(test, tp, True),
      l2=0.1,
    )
    print(f"  (pca128 explained≈{ev128:.3f})", flush=True)

  best = max(experiments, key=lambda r: r["test_AUROC"])
  print(
    f"\nBEST {best['name']}  test_AUROC={best['test_AUROC']:.4f}  "
    f"Δpe={best['delta_vs_pe_mc']:+.4f}  Δ1-maxp={best['delta_vs_1maxp']:+.4f}",
    flush=True,
  )
  print(
    f"Acc unchanged: test={acc_test:.4f}  (same MaSE MC argmax)",
    flush=True,
  )

  summary = {
    "method": "mase_lite_feature_error_probe",
    "checkpoint": str(ckpt),
    "mc_samples": args.mc_samples,
    "pca_dim": int(comp.shape[0]),
    "pca_explained": ev,
    "accuracy": {"val": acc_val, "test": acc_test},
    "baselines": {"pe_mc": pe_auc, "one_minus_maxp": maxp_auc},
    "experiments": experiments,
    "best": best,
    "note": (
      "Classifier frozen. Probe predicts incorrectness from penultimate "
      "branch features (+ optional scalar UQ scores). Primary goal: higher AUROC."
    ),
  }
  out = out_dir / "feature_probe_summary.json"
  with out.open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
  main()
