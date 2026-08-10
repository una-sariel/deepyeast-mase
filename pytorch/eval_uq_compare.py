#!/usr/bin/env python3
"""Compare UQ metrics across MaSE checkpoints (+ optional DeepYeast baseline).

Outputs summary.json and comparison_table.md with UAUC, AURC, Acc@coverage.

Example (professor full-data):
  python pytorch/eval_uq_compare.py \\
    --data-dir D:/.../deepyeast_full \\
    --mc-samples 30 --split test \\
    --tag phase1 --checkpoint .../mase_lite_full_v6_phase1/best.pt \\
    --tag phase2 --checkpoint .../mase_lite_full_v6/best.pt \\
    --tag baseline --deepyeast-checkpoint .../deepyeast_baseline/best.pt \\
    --out-dir .../uq_compare_full
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
  config_from_results_json,
  load_checkpoint,
  mc_predict_loader as mase_mc_predict,
  resolve_device,
  roc_auc_score,
  set_seed,
  summarize_split,
)
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict
from uq_metrics import selective_metrics


def mc_predict_baseline(
  model: DeepYeastNet,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
) -> dict[str, np.ndarray]:
  from mase_lite_net import predictive_entropy

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
        logits = model(images, train=True)
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


def eval_mase_entry(
  tag: str,
  ckpt: Path,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
  num_classes: int,
) -> dict[str, Any]:
  results_json = ckpt.parent / "results.json"
  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  arrays = mase_mc_predict(model, loader, device, mc_samples)
  pe_max = math.log(num_classes)
  thresholds = [float(x) for x in np.linspace(0.0, pe_max, 21)]
  split_sum = summarize_split(arrays, thresholds, num_classes)
  sel = selective_metrics(arrays["pe"], arrays["correct"])
  return {
    "tag": tag,
    "type": "mase",
    "checkpoint": str(ckpt),
    "accuracy": split_sum["accuracy"],
    "uauc_pe": split_sum["UAUC"],
    **sel,
  }


def eval_baseline_entry(
  tag: str,
  ckpt: Path,
  loader: DataLoader,
  device: torch.device,
  mc_samples: int,
  num_classes: int,
) -> dict[str, Any]:
  model = DeepYeastNet().to(device)
  try:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
  except TypeError:
    state = torch.load(ckpt, map_location="cpu")
  if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
  model.load_state_dict(state, strict=True)
  arrays = mc_predict_baseline(model, loader, device, mc_samples)
  pe_max = math.log(num_classes)
  thresholds = [float(x) for x in np.linspace(0.0, pe_max, 21)]
  split_sum = summarize_split(arrays, thresholds, num_classes)
  sel = selective_metrics(arrays["pe"], arrays["correct"])
  return {
    "tag": tag,
    "type": "deepyeast_baseline",
    "checkpoint": str(ckpt),
    "accuracy": split_sum["accuracy"],
    "uauc_pe": split_sum["UAUC"],
    **sel,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="UQ comparison table for multiple models")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--split", choices=["test", "val"], default="test")
  parser.add_argument("--mc-samples", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  parser.add_argument(
    "--tag",
    action="append",
    default=[],
    help="Model label (repeat with --checkpoint or --deepyeast-checkpoint)",
  )
  parser.add_argument("--checkpoint", action="append", default=[], help="MaSE best.pt")
  parser.add_argument(
    "--deepyeast-checkpoint",
    action="append",
    default=[],
    help="DeepYeastNet baseline best.pt",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  set_seed(args.seed)
  device = resolve_device(args.device)
  data_dir = args.data_dir.resolve()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  if len(args.tag) != len(args.checkpoint) + len(args.deepyeast_checkpoint):
    raise SystemExit(
      "Need one --tag per model: "
      f"{len(args.tag)} tags vs {len(args.checkpoint)} MaSE + "
      f"{len(args.deepyeast_checkpoint)} baseline checkpoints"
    )

  mase_pairs = list(zip(args.tag[: len(args.checkpoint)], args.checkpoint))
  baseline_pairs = list(
    zip(args.tag[len(args.checkpoint) :], args.deepyeast_checkpoint)
  )

  if not mase_pairs and not baseline_pairs:
    raise SystemExit("Pass at least one --checkpoint or --deepyeast-checkpoint")

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=False,
    num_workers=args.num_workers,
  )
  loader = test_loader if args.split == "test" else val_loader
  num_classes = 12

  rows: list[dict[str, Any]] = []
  for tag, ckpt in mase_pairs:
    print(f"\n=== UQ: {tag} (MaSE) ===", flush=True)
    row = eval_mase_entry(tag, Path(ckpt), loader, device, args.mc_samples, num_classes)
    rows.append(row)
    print(
      f"  acc={row['accuracy']:.4f}  UAUC={row['uauc_pe']:.4f}  "
      f"AURC={row['aurc']:.4f}  Acc@80%={row['acc_at_80pct']:.4f}",
      flush=True,
    )

  for tag, ckpt in baseline_pairs:
    print(f"\n=== UQ: {tag} (baseline) ===", flush=True)
    row = eval_baseline_entry(
      tag, Path(ckpt), loader, device, args.mc_samples, num_classes
    )
    rows.append(row)
    print(
      f"  acc={row['accuracy']:.4f}  UAUC={row['uauc_pe']:.4f}  "
      f"AURC={row['aurc']:.4f}  Acc@80%={row['acc_at_80pct']:.4f}",
      flush=True,
    )

  payload = {
    "data_dir": str(data_dir),
    "split": args.split,
    "mc_samples": args.mc_samples,
    "models": rows,
  }
  json_path = out_dir / "comparison.json"
  with json_path.open("w") as f:
    json.dump(payload, f, indent=2)

  md_lines = [
    f"# UQ comparison ({args.split}, MC T={args.mc_samples})",
    "",
    "| model | acc | UAUC (PE) | AURC ↓ | Acc@60% | Acc@80% | Acc@90% |",
    "|-------|-----|-----------|--------|---------|---------|---------|",
  ]
  for r in rows:
    md_lines.append(
      f"| {r['tag']} | {r['accuracy']:.4f} | {r['uauc_pe']:.4f} | "
      f"{r['aurc']:.4f} | {r['acc_at_60pct']:.4f} | "
      f"{r['acc_at_80pct']:.4f} | {r['acc_at_90pct']:.4f} |"
    )
  md_path = out_dir / "comparison_table.md"
  md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
  print(f"\nWrote {json_path}", flush=True)
  print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
  main()
