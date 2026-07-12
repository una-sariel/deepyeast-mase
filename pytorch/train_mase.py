#!/usr/bin/env python3
"""Train MaSE-Net (Masked + MSMM + PLCNN) on DeepYeast subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "pytorch") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "pytorch"))

from dataset import make_loaders  # noqa: E402
from mase_net import (  # noqa: E402
  MASE_DEFAULT,
  MaSEConfig,
  MaSENet,
  config_to_dict,
  mase_loss,
)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def run_epoch(
  model: MaSENet,
  loader: DataLoader,
  device: torch.device,
  optimizer: torch.optim.Optimizer | None = None,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
) -> dict[str, Any]:
  train = optimizer is not None
  model.train(train)
  losses, accs, coverages = [], [], []

  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if train:
      optimizer.zero_grad(set_to_none=True)
      fused_logits, details = model(images, train=True, return_details=True)
      loss = mase_loss(
        details,
        labels,
        mask_sparsity_weight=mask_sparsity_weight,
        target_mask_fraction=target_mask_fraction,
      )
      loss.backward()
      optimizer.step()
    else:
      with torch.inference_mode():
        fused_logits, details = model(images, train=False, return_details=True)
        loss = mase_loss(
          details,
          labels,
          mask_sparsity_weight=mask_sparsity_weight,
          target_mask_fraction=target_mask_fraction,
        )

    acc = (fused_logits.argmax(dim=-1) == labels).float().mean().item()
    losses.append(float(loss.item()))
    accs.append(acc)
    coverages.append(float(details["mask"].mean().item()))

  return {
    "loss": float(np.mean(losses)),
    "accuracy": float(np.mean(accs)),
    "mask_coverage": float(np.mean(coverages)),
  }


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train MaSE-Net: Masked + MSMM + PLCNN"
  )
  parser.add_argument("--data-dir", type=Path, default=Path("../deepyeast_10pct"))
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--selector-lr", type=float, default=3e-3)
  parser.add_argument("--backbone-lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--patience", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--top-k", type=int, default=38)
  parser.add_argument("--soft-alpha", type=float, default=0.5)
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--mask-sparsity-weight", type=float, default=0.0)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--no-augment", action="store_true")
  parser.add_argument(
    "--checkpoint-name",
    type=str,
    default="mase_10pct",
  )
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = args.data_dir.resolve()
  ckpt_dir = data_dir / "checkpoints" / args.checkpoint_name

  config = MaSEConfig(
    **{
      **config_to_dict(MASE_DEFAULT),
      "top_k_patches": args.top_k,
      "soft_mask_alpha": args.soft_alpha,
      "plcnn_dropout": args.dropout,
    }
  )
  model = MaSENet(config).to(device)

  optimizer = torch.optim.Adam(
    [
      {"params": model.selector.parameters(), "lr": args.selector_lr},
      {"params": model.msmm.parameters(), "lr": args.backbone_lr},
      {"params": model.plcnn.parameters(), "lr": args.backbone_lr},
    ],
    weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=4, min_lr=1e-6
  )

  train_loader, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=not args.no_augment,
    num_workers=args.num_workers,
  )

  expected_cov = args.top_k / 64.0
  n_params = sum(p.numel() for p in model.parameters())
  print(
    f"MaSE-Net | device={device} | params={n_params:,} | "
    f"top_k={args.top_k} (mask~{expected_cov:.3f}) | "
    f"epochs={args.epochs} | data={data_dir}"
  )
  print(
    f"  train={len(train_loader.dataset)} "
    f"val={len(val_loader.dataset)} "
    f"test={len(test_loader.dataset)}"
  )

  history: list[dict[str, Any]] = []
  best_val = -1.0
  best_epoch = 0
  patience_counter = 0
  best_state: dict[str, torch.Tensor] | None = None
  t0 = time.time()

  for epoch in tqdm(range(args.epochs), desc="MaSE-Net"):
    train_m = run_epoch(
      model,
      train_loader,
      device,
      optimizer=optimizer,
      mask_sparsity_weight=args.mask_sparsity_weight,
      target_mask_fraction=expected_cov,
    )
    val_m = run_epoch(
      model,
      val_loader,
      device,
      optimizer=None,
      mask_sparsity_weight=args.mask_sparsity_weight,
      target_mask_fraction=expected_cov,
    )
    scheduler.step(val_m["accuracy"])

    row = {
      "epoch": epoch + 1,
      "train": train_m,
      "val": val_m,
      "lr": float(optimizer.param_groups[1]["lr"]),
    }
    history.append(row)
    print(
      f"  ep{epoch + 1:02d}  "
      f"train={train_m['accuracy']:.3f}  "
      f"val={val_m['accuracy']:.3f}  "
      f"mask={val_m['mask_coverage']:.3f}  "
      f"loss={val_m['loss']:.4f}"
    )

    if val_m["accuracy"] > best_val:
      best_val = val_m["accuracy"]
      best_epoch = epoch + 1
      patience_counter = 0
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      ckpt_dir.mkdir(parents=True, exist_ok=True)
      torch.save(best_state, ckpt_dir / "best.pt")
    else:
      patience_counter += 1
      if patience_counter >= args.patience:
        print(f"Early stop @ epoch {epoch + 1} (best val={best_val:.4f} @ {best_epoch})")
        break

  if best_state is not None:
    model.load_state_dict(best_state)

  test_m = run_epoch(
    model,
    test_loader,
    device,
    optimizer=None,
    mask_sparsity_weight=args.mask_sparsity_weight,
    target_mask_fraction=expected_cov,
  )

  elapsed = time.time() - t0
  results = {
    "method": "mase_full",
    "framework": "pytorch",
    "config": config_to_dict(config),
    "params": n_params,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val,
    "test": test_m,
    "elapsed_sec": elapsed,
    "history": history,
    "data_dir": str(data_dir),
    "seed": args.seed,
  }
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  with (ckpt_dir / "results.json").open("w") as f:
    json.dump(results, f, indent=2)
  with (ckpt_dir / "meta.json").open("w") as f:
    json.dump(
      {
        "method": results["method"],
        "components": ["PatchRegionSelector", "MSMMResNet34", "PLCNNTripleNet"],
        "fusion": "mean(softmax(MSMM), softmax(PLCNN))",
        "config": config_to_dict(config),
      },
      f,
      indent=2,
    )

  print("\n=== MaSE-Net done ===")
  print(f"Best val: {best_val:.4f} @ epoch {best_epoch}")
  print(
    f"Test:     {test_m['accuracy']:.4f}  "
    f"(mask={test_m['mask_coverage']:.3f})"
  )
  print(f"Elapsed:  {elapsed / 60:.1f} min")
  print(f"Saved:    {ckpt_dir}")


if __name__ == "__main__":
  main()
