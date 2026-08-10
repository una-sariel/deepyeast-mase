"""MC Dropout UQ for official DeepYeastNet — same metrics as eval_mase_uq.py.

DeepYeastNet uses F.dropout(..., training=train). For MC sampling we keep the
model in eval() (BatchNorm frozen) but pass train=True so Dropout stays on.
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
from deepyeast_net import DeepYeastNet
from eval_mase_uq import (
  set_seed,
  summarize_split,
  write_per_image_csv,
)
from mase_lite_net import predictive_entropy


def load_checkpoint(model: DeepYeastNet, path: Path) -> None:
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(path)
  try:
    state = torch.load(path, map_location="cpu", weights_only=True)
  except TypeError:
    state = torch.load(path, map_location="cpu")
  if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
  missing, unexpected = model.load_state_dict(state, strict=True)
  if missing or unexpected:
    print(f"Warning missing={missing} unexpected={unexpected}", flush=True)


def mc_predict_loader(
  model: DeepYeastNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  """BN in eval; Dropout on via train=True in forward."""
  model.eval()
  y_true_all: list[np.ndarray] = []
  y_pred_all: list[np.ndarray] = []
  pe_all: list[np.ndarray] = []
  max_prob_all: list[np.ndarray] = []

  with torch.inference_mode():
    for images, labels in loader:
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      probs_sum: torch.Tensor | None = None
      for _ in range(mc_samples):
        logits = model(images, train=True)  # MC Dropout
        probs = F.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
      assert probs_sum is not None
      mu = probs_sum / float(mc_samples)
      pe = predictive_entropy(mu)
      y_pred = mu.argmax(dim=-1)
      max_p = mu.max(dim=-1).values

      y_true_all.append(labels.cpu().numpy())
      y_pred_all.append(y_pred.cpu().numpy())
      pe_all.append(pe.cpu().numpy())
      max_prob_all.append(max_p.cpu().numpy())

  y_true = np.concatenate(y_true_all)
  y_pred = np.concatenate(y_pred_all)
  pe = np.concatenate(pe_all)
  max_prob = np.concatenate(max_prob_all)
  correct = (y_true == y_pred).astype(np.int32)
  return {
    "y_true": y_true,
    "y_pred": y_pred,
    "pe": pe,
    "max_prob": max_prob,
    "correct": correct,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
    description="MC Dropout UQ for DeepYeastNet baseline (same metrics as MaSE)"
  )
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--mc-samples", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--split", choices=["test", "val", "both"], default="both")
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--num-classes", type=int, default=12)
  parser.add_argument("--n-thresholds", type=int, default=21)
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = args.data_dir.resolve()
  ckpt = args.checkpoint.resolve()
  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else (ckpt.parent / "uq_mc_dropout")
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  model = DeepYeastNet(
    num_classes=args.num_classes, dropout_rate=args.dropout
  ).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded {ckpt}", flush=True)
  print(f"Device={device}  MC samples={args.mc_samples}", flush=True)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )

  pe_max = math.log(args.num_classes)
  thresholds = [float(x) for x in np.linspace(0.0, pe_max, args.n_thresholds)]

  splits: dict[str, DataLoader] = {}
  if args.split in ("val", "both"):
    splits["val"] = val_loader
  if args.split in ("test", "both"):
    splits["test"] = test_loader

  summary: dict[str, Any] = {
    "method": "deepyeast_net_mc_dropout",
    "checkpoint": str(ckpt),
    "data_dir": str(data_dir),
    "mc_samples": args.mc_samples,
    "seed": args.seed,
    "dropout": args.dropout,
    "num_classes": args.num_classes,
    "device": str(device),
    "note": (
      "Same UQ metrics as eval_mase_uq.py. "
      "PE = predictive entropy of mean MC softmax. "
      "UAUC = AUROC of PE ranking incorrect predictions. "
      "BN stays in eval(); Dropout via forward(train=True)."
    ),
    "splits": {},
  }

  for split_name, loader in splits.items():
    print(f"\n=== DeepYeast MC Dropout on {split_name} (T={args.mc_samples}) ===", flush=True)
    arrays = mc_predict_loader(model, loader, device, args.mc_samples)
    write_per_image_csv(out_dir / f"per_image_{split_name}.csv", arrays)
    split_summary = summarize_split(arrays, thresholds, args.num_classes)
    summary["splits"][split_name] = split_summary
    print(
      f"  n={split_summary['n']}  acc={split_summary['accuracy']:.4f}  "
      f"UAUC(PE)={split_summary['UAUC']:.4f}  "
      f"UAUC(1-maxp)={split_summary['UAUC_via_1_minus_maxprob']:.4f}  "
      f"PE mean={split_summary['pe_mean']:.4f}",
      flush=True,
    )

  with (out_dir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nWrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
  main()
