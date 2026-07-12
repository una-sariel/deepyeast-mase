#!/usr/bin/env python3
"""Train Lite Triple Fusion v2 (longer + stronger reg + init + learnable ensemble)."""

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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_PYTORCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PYTORCH_DIR.parent
if str(_PYTORCH_DIR) not in sys.path:
  sys.path.insert(0, str(_PYTORCH_DIR))

from dataset import make_loaders  # noqa: E402
from triple_fusion_lite_net import (  # noqa: E402
  TRIPLE_FUSION_LITE_DEFAULT,
  TripleFusionLiteConfig,
  TripleFusionLiteNet,
  config_to_dict,
  lite_fusion_loss,
  load_partial_state,
)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def run_epoch(
  model: TripleFusionLiteNet,
  loader: DataLoader,
  device: torch.device,
  optimizer: torch.optim.Optimizer | None = None,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
  label_smoothing: float = 0.0,
) -> dict[str, Any]:
  train = optimizer is not None
  model.train(train)
  losses, accs, coverages = [], [], []

  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if train:
      optimizer.zero_grad(set_to_none=True)
      ensemble, details = model(images, train=True, return_details=True)
      loss = lite_fusion_loss(
        details,
        labels,
        mask_sparsity_weight=mask_sparsity_weight,
        target_mask_fraction=target_mask_fraction,
        label_smoothing=label_smoothing,
      )
      loss.backward()
      optimizer.step()
    else:
      with torch.inference_mode():
        ensemble, details = model(images, train=False, return_details=True)
        loss = lite_fusion_loss(
          details,
          labels,
          mask_sparsity_weight=mask_sparsity_weight,
          target_mask_fraction=target_mask_fraction,
          label_smoothing=label_smoothing,
        )

    acc = (ensemble.argmax(dim=-1) == labels).float().mean().item()
    losses.append(float(loss.item()))
    accs.append(acc)
    coverages.append(float(details["mask"].mean().item()))

  out: dict[str, Any] = {
    "loss": float(np.mean(losses)),
    "accuracy": float(np.mean(accs)),
    "mask_coverage": float(np.mean(coverages)),
  }
  if "ensemble_weights" in details:
    out["ensemble_weights"] = [
      float(w) for w in details["ensemble_weights"].cpu().tolist()
    ]
  return out


def resolve_default_init(name: str) -> Path | None:
  candidates = [
    _REPO_ROOT / "artifacts" / "checkpoints_5pct" / name / "best.pt",
    Path("artifacts") / "checkpoints_5pct" / name / "best.pt",
  ]
  for path in candidates:
    if path.exists():
      return path.resolve()
  return None


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train Lite Triple Fusion v2 (optimized recipe)"
  )
  parser.add_argument("--data-dir", type=Path, default=Path("../deepyeast_full"))
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--selector-lr", type=float, default=3e-3)
  parser.add_argument("--backbone-lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--patience", type=int, default=15)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--soft-alpha", type=float, default=0.5)
  parser.add_argument("--branch-dropout", type=float, default=0.4)
  parser.add_argument("--head-dropout", type=float, default=0.5)
  parser.add_argument("--mask-sparsity-weight", type=float, default=0.05)
  parser.add_argument("--label-smoothing", type=float, default=0.1)
  parser.add_argument("--ensemble-temperature", type=float, default=1.0)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--no-augment", action="store_true")
  parser.add_argument("--no-strong-augment", action="store_true")
  parser.add_argument("--no-learnable-ensemble", action="store_true")
  parser.add_argument(
    "--init-plcnn",
    type=Path,
    default=None,
    help="PLCNN best.pt for branch init (default: artifacts plcnn_triple)",
  )
  parser.add_argument(
    "--init-selector",
    type=Path,
    default=None,
    help="Masked best.pt for selector init (default: artifacts masked_v3k60)",
  )
  parser.add_argument("--no-init", action="store_true")
  parser.add_argument(
    "--checkpoint-name",
    type=str,
    default="triple_fusion_lite_full_v2",
  )
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = args.data_dir.resolve()
  ckpt_dir = data_dir / "checkpoints" / args.checkpoint_name

  config = TripleFusionLiteConfig(
    **{
      **config_to_dict(TRIPLE_FUSION_LITE_DEFAULT),
      "top_k_patches": args.top_k,
      "soft_mask_alpha": args.soft_alpha,
      "branch_dropout": args.branch_dropout,
      "head_dropout": args.head_dropout,
      "learnable_ensemble": not args.no_learnable_ensemble,
      "ensemble_temperature": args.ensemble_temperature,
    }
  )
  model = TripleFusionLiteNet(config).to(device)

  init_info: dict[str, Any] = {}
  if not args.no_init:
    plcnn_path = args.init_plcnn or resolve_default_init("plcnn_triple")
    selector_path = args.init_selector or resolve_default_init(
      "masked_v3k60_pytorch"
    )
    if plcnn_path is not None:
      loaded = load_partial_state(
        model,
        plcnn_path,
        prefixes=("branch_vgg.", "branch_resnet.", "branch_densenet."),
      )
      init_info["plcnn"] = {"path": str(plcnn_path), "n_tensors": len(loaded)}
      print(f"Init PLCNN branches: {len(loaded)} tensors from {plcnn_path}", flush=True)
    else:
      print("Init PLCNN: skipped (checkpoint not found)", flush=True)
    if selector_path is not None:
      loaded = load_partial_state(model, selector_path, prefixes=("selector.",))
      init_info["selector"] = {
        "path": str(selector_path),
        "n_tensors": len(loaded),
      }
      print(
        f"Init selector: {len(loaded)} tensors from {selector_path}", flush=True
      )
    else:
      print("Init selector: skipped (checkpoint not found)", flush=True)

  optimizer = torch.optim.Adam(
    [
      {"params": model.selector.parameters(), "lr": args.selector_lr},
      {
        "params": list(model.branch_vgg.parameters())
        + list(model.branch_resnet.parameters())
        + list(model.branch_densenet.parameters())
        + list(model.heads.parameters())
        + (
          [model.ensemble_logits]
          if isinstance(model.ensemble_logits, nn.Parameter)
          else []
        ),
        "lr": args.backbone_lr,
      },
    ],
    weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
  )

  train_loader, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=not args.no_augment,
    strong_augment=(not args.no_augment) and (not args.no_strong_augment),
    num_workers=args.num_workers,
  )

  expected_cov = args.top_k / 64.0
  n_params = sum(p.numel() for p in model.parameters())
  print(
    f"Triple Fusion Lite v2 | device={device} | params={n_params:,} | "
    f"top_k={args.top_k} (mask~{expected_cov:.3f}) | "
    f"epochs={args.epochs} patience={args.patience} | "
    f"ls={args.label_smoothing} sparse_w={args.mask_sparsity_weight}",
    flush=True,
  )
  print(
    f"  train={len(train_loader.dataset)} "
    f"val={len(val_loader.dataset)} "
    f"test={len(test_loader.dataset)} | data={data_dir}",
    flush=True,
  )

  history: list[dict[str, Any]] = []
  best_val = -1.0
  best_epoch = 0
  patience_counter = 0
  best_state: dict[str, torch.Tensor] | None = None
  t0 = time.time()

  for epoch in tqdm(range(args.epochs), desc="TripleFusionLiteV2", mininterval=5):
    train_m = run_epoch(
      model,
      train_loader,
      device,
      optimizer=optimizer,
      mask_sparsity_weight=args.mask_sparsity_weight,
      target_mask_fraction=expected_cov,
      label_smoothing=args.label_smoothing,
    )
    val_m = run_epoch(
      model,
      val_loader,
      device,
      optimizer=None,
      mask_sparsity_weight=args.mask_sparsity_weight,
      target_mask_fraction=expected_cov,
      label_smoothing=0.0,
    )
    scheduler.step(val_m["accuracy"])

    row = {
      "epoch": epoch + 1,
      "train": train_m,
      "val": val_m,
      "lr": float(optimizer.param_groups[1]["lr"]),
    }
    history.append(row)
    ew = val_m.get("ensemble_weights")
    ew_str = (
      " w=[" + ",".join(f"{x:.2f}" for x in ew) + "]" if ew is not None else ""
    )
    print(
      f"  ep{epoch + 1:02d}  "
      f"train={train_m['accuracy']:.3f}  "
      f"val={val_m['accuracy']:.3f}  "
      f"mask={val_m['mask_coverage']:.3f}  "
      f"loss={val_m['loss']:.4f}{ew_str}",
      flush=True,
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
        print(
          f"Early stop @ epoch {epoch + 1} "
          f"(best val={best_val:.4f} @ {best_epoch})",
          flush=True,
        )
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
    label_smoothing=0.0,
  )

  elapsed = time.time() - t0
  results = {
    "method": "triple_fusion_lite_v2",
    "framework": "pytorch",
    "config": config_to_dict(config),
    "optimizations": {
      "epochs": args.epochs,
      "patience": args.patience,
      "label_smoothing": args.label_smoothing,
      "mask_sparsity_weight": args.mask_sparsity_weight,
      "top_k": args.top_k,
      "strong_augment": (not args.no_augment) and (not args.no_strong_augment),
      "learnable_ensemble": not args.no_learnable_ensemble,
      "branch_dropout": args.branch_dropout,
      "head_dropout": args.head_dropout,
      "init": init_info,
    },
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
        "components": [
          "PatchRegionSelector",
          "PLCNN VGG/ResNet/DenseNet",
          "MSMM progressive 4-head + learnable weights",
        ],
        "fusion": "sum(w_i * softmax(head_i))",
        "config": config_to_dict(config),
        "optimizations": results["optimizations"],
      },
      f,
      indent=2,
    )

  print("\n=== Triple Fusion Lite v2 done ===", flush=True)
  print(f"Best val: {best_val:.4f} @ epoch {best_epoch}", flush=True)
  print(
    f"Test:     {test_m['accuracy']:.4f}  "
    f"(mask={test_m['mask_coverage']:.3f})",
    flush=True,
  )
  print(f"Elapsed:  {elapsed / 60:.1f} min", flush=True)
  print(f"Saved:    {ckpt_dir}", flush=True)


if __name__ == "__main__":
  main()
