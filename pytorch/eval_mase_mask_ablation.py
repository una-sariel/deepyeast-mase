#!/usr/bin/env python3
"""Mask ablation on a fixed MaSE checkpoint: learned vs none vs random.

Does not retrain. Patches ``_mask_input`` for the duration of each eval pass.

Example:
  python pytorch/eval_mase_mask_ablation.py \\
    --data-dir /path/to/deepyeast_full \\
    --checkpoint /path/to/mase_lite_full_v6_phase1/best.pt \\
    --split test --out-dir /path/to/mask_ablation_v6_p1
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device
from masked_net import apply_spatial_mask, patch_mask_to_pixels
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict


@contextmanager
def mask_mode(model: MaSELiteNet, mode: str) -> Iterator[None]:
  original = model._mask_input

  def _mask_input(x: torch.Tensor, train: bool = True):
    cfg = model.config
    b, _, h, w = x.shape
    if mode == "learned":
      return original(x, train=train)

    if mode == "none":
      pixel_mask = torch.ones(b, h, w, device=x.device, dtype=x.dtype)
      grid = cfg.image_size // cfg.patch_size
      patch_probs = torch.ones(b, grid * grid, device=x.device, dtype=x.dtype)
      masked_x = apply_spatial_mask(
        x,
        pixel_mask,
        mask_gfp_only=cfg.mask_gfp_only,
        soft_alpha=cfg.soft_mask_alpha,
      )
      return masked_x, pixel_mask, patch_probs

    if mode == "random":
      grid = cfg.image_size // cfg.patch_size
      k = cfg.top_k_patches
      if k is None:
        raise ValueError("random ablation requires top_k_patches in config")
      patch_flat = torch.zeros(b, grid * grid, device=x.device, dtype=x.dtype)
      for i in range(b):
        perm = torch.randperm(grid * grid, device=x.device)[:k]
        patch_flat[i, perm] = 1.0
      patch_mask = patch_flat.view(b, grid, grid)
      pixel_mask = patch_mask_to_pixels(patch_mask, cfg.patch_size)
      patch_probs = patch_flat
      masked_x = apply_spatial_mask(
        x,
        pixel_mask,
        mask_gfp_only=cfg.mask_gfp_only,
        soft_alpha=cfg.soft_mask_alpha,
      )
      return masked_x, pixel_mask, patch_probs

    raise ValueError(f"Unknown mask mode: {mode}")

  model._mask_input = _mask_input  # type: ignore[method-assign]
  try:
    yield
  finally:
    model._mask_input = original  # type: ignore[method-assign]


@torch.inference_mode()
def eval_split(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
) -> dict[str, float]:
  model.eval()
  correct, total = 0, 0
  coverages: list[float] = []
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    log_probs, details = model(images, train=False, return_details=True)
    pred = log_probs.argmax(dim=-1)
    correct += int((pred == labels).sum().item())
    total += int(labels.numel())
    coverages.append(float(details["mask"].mean().item()))
  acc = correct / max(total, 1)
  return {
    "accuracy": float(acc),
    "n": total,
    "mean_mask_coverage": float(np.mean(coverages)) if coverages else float("nan"),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="MaSE mask ablation (same checkpoint)")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--split", choices=["test", "val", "both"], default="test")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  device = resolve_device(args.device)
  data_dir = args.data_dir.resolve()
  ckpt = args.checkpoint.resolve()
  results_json = args.results_json or (ckpt.parent / "results.json")
  out_dir = args.out_dir or (ckpt.parent / "mask_ablation")
  out_dir.mkdir(parents=True, exist_ok=True)

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=False,
    num_workers=args.num_workers,
  )
  loaders = {}
  if args.split in ("val", "both"):
    loaders["val"] = val_loader
  if args.split in ("test", "both"):
    loaders["test"] = test_loader

  modes = ("learned", "none", "random")
  report: dict[str, object] = {
    "checkpoint": str(ckpt),
    "data_dir": str(data_dir),
    "config": config_to_dict(config),
    "modes": {},
  }

  for split_name, loader in loaders.items():
    report["modes"][split_name] = {}
    print(f"\n=== Mask ablation ({split_name}) ===", flush=True)
    for mode in modes:
      with mask_mode(model, mode):
        metrics = eval_split(model, loader, device)
      report["modes"][split_name][mode] = metrics
      print(
        f"  {mode:8s}  acc={metrics['accuracy']:.4f}  "
        f"mask_cov={metrics['mean_mask_coverage']:.3f}",
        flush=True,
      )

  json_path = out_dir / "ablation.json"
  with json_path.open("w") as f:
    json.dump(report, f, indent=2)

  md_path = out_dir / "ablation_table.md"
  lines = [
    "# Mask ablation",
    "",
    f"Checkpoint: `{ckpt}`",
    "",
    "| split | mode | accuracy | mean mask cov |",
    "|-------|------|----------|---------------|",
  ]
  for split_name, modes_dict in report["modes"].items():
    for mode, m in modes_dict.items():
      lines.append(
        f"| {split_name} | {mode} | {m['accuracy']:.4f} | "
        f"{m['mean_mask_coverage']:.3f} |"
      )
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"\nWrote {json_path}", flush=True)
  print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
  main()
