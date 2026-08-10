"""Temperature sweep on a fixed MaSE Lite checkpoint (no retrain).

Softmax(logits / T): lower T → sharper probs → lower PE.
Argmax is invariant to T>0 for a single forward, so deterministic Acc is flat;
MC-averaged Acc can shift slightly with T.
"""

from __future__ import annotations

import argparse
import json
import math
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
from eval_mase_uq import load_checkpoint, roc_auc_score, set_seed
from mase_lite_net import (
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  enable_mc_dropout,
  predictive_entropy,
)


def _temps(s: str) -> list[float]:
  return [float(x) for x in s.split(",") if x.strip()]


@torch.inference_mode()
def eval_temperature(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  temperature: float,
  mc_samples: int,
) -> dict[str, float]:
  """Returns Acc/PE for deterministic and optional MC paths."""
  t = max(float(temperature), 1e-6)

  # --- Deterministic (Dropout off) ---
  model.eval()
  y_true: list[np.ndarray] = []
  det_correct = 0
  det_pe: list[np.ndarray] = []
  det_maxp: list[np.ndarray] = []
  n = 0

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    log_probs = model(images, train=False, return_details=False)
    assert isinstance(log_probs, torch.Tensor)
    # model returns log_softmax already for MaSE — recover logits approx via log
    # Safer: use exp then convert; temperature on log-prob space:
    # p_T ∝ p^(1/T) ⇔ log p_T = log p / T - logZ
    log_p = log_probs / t
    log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
    probs = log_p.exp()
    pred = probs.argmax(dim=-1)
    det_correct += int((pred == labels).sum().item())
    n += int(labels.numel())
    det_pe.append(predictive_entropy(probs).cpu().numpy())
    det_maxp.append(probs.max(dim=-1).values.cpu().numpy())
    y_true.append(labels.cpu().numpy())

  det_pe_a = np.concatenate(det_pe)
  det_maxp_a = np.concatenate(det_maxp)
  out: dict[str, float] = {
    "temperature": t,
    "n": float(n),
    "det_acc": det_correct / max(n, 1),
    "det_pe_mean": float(det_pe_a.mean()),
    "det_max_prob_mean": float(det_maxp_a.mean()),
  }

  if mc_samples <= 0:
    return out

  # --- MC Dropout + temperature on each sample's log-probs ---
  enable_mc_dropout(model)
  mc_pe: list[np.ndarray] = []
  mc_maxp: list[np.ndarray] = []
  mc_correct = 0
  mc_n = 0
  incorrect_flags: list[np.ndarray] = []
  pe_for_auc: list[np.ndarray] = []

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    probs_sum: torch.Tensor | None = None
    for _ in range(mc_samples):
      log_probs = model(images, train=False, return_details=False)
      assert isinstance(log_probs, torch.Tensor)
      log_p = log_probs / t
      log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
      probs = log_p.exp()
      probs_sum = probs if probs_sum is None else probs_sum + probs
    assert probs_sum is not None
    mu = probs_sum / float(mc_samples)
    pe = predictive_entropy(mu)
    pred = mu.argmax(dim=-1)
    correct = pred == labels
    mc_correct += int(correct.sum().item())
    mc_n += int(labels.numel())
    mc_pe.append(pe.cpu().numpy())
    mc_maxp.append(mu.max(dim=-1).values.cpu().numpy())
    incorrect_flags.append((~correct).cpu().numpy().astype(np.int32))
    pe_for_auc.append(pe.cpu().numpy())

  pe_a = np.concatenate(pe_for_auc)
  inc_a = np.concatenate(incorrect_flags)
  out["mc_acc"] = mc_correct / max(mc_n, 1)
  out["mc_pe_mean"] = float(pe_a.mean())
  out["mc_max_prob_mean"] = float(np.concatenate(mc_maxp).mean())
  out["mc_UAUC"] = float(roc_auc_score(inc_a, pe_a))
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description="Temperature sweep for MaSE Lite")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out-json", type=Path, required=True)
  parser.add_argument(
    "--temperatures",
    type=str,
    default="1.0,0.9,0.8,0.7,0.5",
  )
  parser.add_argument("--mc-samples", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--split", choices=["val", "test", "both"], default="both")
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--soft-alpha", type=float, default=0.5)
  parser.add_argument("--branch-dropout", type=float, default=0.25)
  parser.add_argument("--head-dropout", type=float, default=0.3)
  parser.add_argument("--min-ensemble-weight", type=float, default=0.0)
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  temps = _temps(args.temperatures)

  config = MaSELiteConfig(
    **{
      **config_to_dict(MASE_LITE_DEFAULT),
      "top_k_patches": args.top_k,
      "soft_mask_alpha": args.soft_alpha,
      "branch_dropout": args.branch_dropout,
      "head_dropout": args.head_dropout,
      "min_ensemble_weight": args.min_ensemble_weight,
      "learnable_ensemble": True,
    }
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, args.checkpoint.resolve())
  print(f"Loaded {args.checkpoint}  device={device}", flush=True)

  _, val_loader, test_loader = make_loaders(
    args.data_dir.resolve(),
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=0,
  )
  splits: dict[str, DataLoader] = {}
  if args.split in ("val", "both"):
    splits["val"] = val_loader
  if args.split in ("test", "both"):
    splits["test"] = test_loader

  summary: dict[str, Any] = {
    "checkpoint": str(args.checkpoint.resolve()),
    "data_dir": str(args.data_dir.resolve()),
    "mc_samples": args.mc_samples,
    "note": (
      "Temperature applied as log_p/T then renormalize. "
      "Deterministic Acc is invariant to T; MC Acc may shift slightly. "
      "Lower T → lower PE / higher max_prob."
    ),
    "splits": {},
  }

  for split_name, loader in splits.items():
    rows = []
    print(f"\n=== {split_name} ===", flush=True)
    print(
      f"{'T':>5}  {'det_acc':>8}  {'det_PE':>8}  {'det_maxp':>8}  "
      f"{'mc_acc':>8}  {'mc_PE':>8}  {'mc_UAUC':>8}",
      flush=True,
    )
    for t in temps:
      row = eval_temperature(model, loader, device, t, args.mc_samples)
      rows.append(row)
      print(
        f"{t:5.2f}  {row['det_acc']:8.4f}  {row['det_pe_mean']:8.4f}  "
        f"{row['det_max_prob_mean']:8.4f}  "
        f"{row.get('mc_acc', float('nan')):8.4f}  "
        f"{row.get('mc_pe_mean', float('nan')):8.4f}  "
        f"{row.get('mc_UAUC', float('nan')):8.4f}",
        flush=True,
      )
    summary["splits"][split_name] = rows

  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  with args.out_json.open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nWrote {args.out_json}", flush=True)


if __name__ == "__main__":
  main()
