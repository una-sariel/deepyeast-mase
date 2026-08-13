#!/usr/bin/env python3
"""Random Spatial Bagging (RSB) eval for MaSE Lite v9.

For each image, average Softmax probs over R independent per-image random
top-k masks (RF-style spatial bagging). Also reports a single-pass random
baseline and (if the checkpoint has a selector) learned-mask accuracy.

Example:
  python pytorch/eval_mase_rsb.py \\
    --data-dir /path/to/deepyeast_full \\
    --checkpoint /path/to/mase_lite_full_v9/best.pt \\
    --rsb-samples 16 --split both \\
    --out-dir /path/to/mase_lite_full_v9/rsb_eval
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
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict


@torch.inference_mode()
def eval_single_pass(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
) -> dict[str, float]:
  model.eval()
  correct, total = 0, 0
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    log_probs = model(images, train=False)
    pred = log_probs.argmax(dim=-1)
    correct += int((pred == labels).sum().item())
    total += int(labels.numel())
  return {"accuracy": float(correct / max(total, 1)), "n": total}


@torch.inference_mode()
def eval_rsb(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  rsb_samples: int,
) -> dict[str, float]:
  """Average R random-mask forwards per batch (mask_mode must be random)."""
  if model.config.mask_mode != "random":
    raise ValueError("eval_rsb requires config.mask_mode == 'random'")
  model.eval()
  correct, total = 0, 0
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    probs_sum = None
    for _ in range(rsb_samples):
      log_probs = model(images, train=False)
      probs = F.softmax(log_probs, dim=-1)
      probs_sum = probs if probs_sum is None else probs_sum + probs
    assert probs_sum is not None
    mu = probs_sum / float(rsb_samples)
    pred = mu.argmax(dim=-1)
    correct += int((pred == labels).sum().item())
    total += int(labels.numel())
  return {
    "accuracy": float(correct / max(total, 1)),
    "n": total,
    "rsb_samples": rsb_samples,
  }


def run_rsb_eval(
  data_dir: Path,
  checkpoint: Path,
  results_json: Path | None,
  split: str,
  rsb_samples: int,
  batch_size: int,
  device_pref: str,
  seed: int,
  out_dir: Path,
) -> dict[str, Any]:
  torch.manual_seed(seed)
  np.random.seed(seed)
  device = resolve_device(device_pref)
  cfg = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  # Always evaluate RSB under random masks
  cfg.mask_mode = "random"
  model = MaSELiteNet(cfg).to(device)
  load_checkpoint(model, checkpoint)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=batch_size,
    augment=False,
    strong_augment=False,
    num_workers=0,
  )
  loaders: dict[str, DataLoader] = {}
  if split in ("val", "both"):
    loaders["val"] = val_loader
  if split in ("test", "both"):
    loaders["test"] = test_loader

  payload: dict[str, Any] = {
    "method": "mase_lite_v9_rsb",
    "checkpoint": str(checkpoint.resolve()),
    "data_dir": str(data_dir.resolve()),
    "rsb_samples": rsb_samples,
    "seed": seed,
    "note": (
      "RSB = average Softmax over R per-image random top-k masks "
      "(not one shared region for all images)."
    ),
    "splits": {},
  }

  for name, loader in loaders.items():
    print(f"\n=== RSB on {name} (R={rsb_samples}) ===", flush=True)
    single = eval_single_pass(model, loader, device)
    bagged = eval_rsb(model, loader, device, rsb_samples)
    # Optional learned-mask reference (selector in checkpoint)
    learned_acc = None
    try:
      model.config.mask_mode = "learned"
      learned = eval_single_pass(model, loader, device)
      learned_acc = learned["accuracy"]
    except Exception as exc:  # noqa: BLE001
      print(f"  learned-mask skip: {exc}", flush=True)
    finally:
      model.config.mask_mode = "random"

    split_out = {
      "random_single": single,
      "rsb": bagged,
      "learned_single": (
        {"accuracy": learned_acc} if learned_acc is not None else None
      ),
    }
    payload["splits"][name] = split_out
    print(
      f"  random_single={single['accuracy']:.4f}  "
      f"rsb(R={rsb_samples})={bagged['accuracy']:.4f}"
      + (
        f"  learned={learned_acc:.4f}"
        if learned_acc is not None
        else ""
      ),
      flush=True,
    )

  out_dir.mkdir(parents=True, exist_ok=True)
  with (out_dir / "summary.json").open("w") as f:
    json.dump(payload, f, indent=2)
  print(f"\nWrote {out_dir / 'summary.json'}", flush=True)
  return payload


def main() -> None:
  p = argparse.ArgumentParser(description="MaSE Lite v9 Random Spatial Bagging eval")
  p.add_argument("--data-dir", type=Path, required=True)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--results-json", type=Path, default=None)
  p.add_argument("--out-dir", type=Path, default=None)
  p.add_argument("--split", choices=["test", "val", "both"], default="both")
  p.add_argument("--rsb-samples", type=int, default=16)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--device", type=str, default="auto")
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()

  ckpt = args.checkpoint.resolve()
  results_json = args.results_json
  if results_json is None:
    cand = ckpt.parent / "results.json"
    results_json = cand if cand.exists() else None
  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else (ckpt.parent / "rsb_eval")
  )
  run_rsb_eval(
    data_dir=args.data_dir.resolve(),
    checkpoint=ckpt,
    results_json=results_json,
    split=args.split,
    rsb_samples=args.rsb_samples,
    batch_size=args.batch_size,
    device_pref=args.device,
    seed=args.seed,
    out_dir=out_dir,
  )


if __name__ == "__main__":
  main()
