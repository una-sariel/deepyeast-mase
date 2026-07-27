#!/usr/bin/env python3
"""Train MaSE-Net Lite (v3 fused-CE default; --legacy-v2-loss for 89.1% recipe)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
from mase_lite_net import (  # noqa: E402
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  mase_lite_legacy_loss,
  mase_lite_loss,
  load_partial_state,
)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def run_epoch(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  loss_fn: Callable[..., torch.Tensor],
  optimizer: torch.optim.Optimizer | None = None,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
  label_smoothing: float = 0.0,
  aux_head_weight: float = 0.5,
  distill_weight: float = 0.1,
  ensemble_entropy_weight: float = 0.0,
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
      loss = loss_fn(
        details,
        labels,
        mask_sparsity_weight=mask_sparsity_weight,
        target_mask_fraction=target_mask_fraction,
        label_smoothing=label_smoothing,
        aux_head_weight=aux_head_weight,
        distill_weight=distill_weight,
        ensemble_entropy_weight=ensemble_entropy_weight,
      )
      loss.backward()
      optimizer.step()
    else:
      with torch.inference_mode():
        ensemble, details = model(images, train=False, return_details=True)
        loss = loss_fn(
          details,
          labels,
          mask_sparsity_weight=mask_sparsity_weight,
          target_mask_fraction=target_mask_fraction,
          label_smoothing=label_smoothing,
          aux_head_weight=aux_head_weight,
          distill_weight=distill_weight,
          ensemble_entropy_weight=ensemble_entropy_weight,
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
    description="Train MaSE-Net Lite (v3 fused-CE default)"
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
  parser.add_argument(
    "--aux-head-weight",
    type=float,
    default=0.5,
    help="Weight for mean per-head CE (v3 only; 0 = fused CE only)",
  )
  parser.add_argument(
    "--distill-weight",
    type=float,
    default=0.1,
    help="KL weight: heads ← fused teacher (v3 only; 0 = off)",
  )
  parser.add_argument(
    "--min-ensemble-weight",
    type=float,
    default=0.05,
    help="Floor each mixture weight before renormalize (v3 anti-collapse)",
  )
  parser.add_argument(
    "--ensemble-entropy-weight",
    type=float,
    default=0.0,
    help="Encourage high-entropy mixture weights (optional)",
  )
  parser.add_argument(
    "--ensemble-lr",
    type=float,
    default=None,
    help="LR for ensemble_logits (default: 5x backbone-lr)",
  )
  parser.add_argument(
    "--legacy-v2-loss",
    action="store_true",
    help="Use v2 head-mean CE loss (reproduces 89.1%% full-data run)",
  )
  parser.add_argument(
    "--freeze-ensemble",
    action="store_true",
    help=(
      "MaSE Lite v4: fused-CE with mixture weights frozen at uniform 0.25 "
      "(full-data ~89.6%%; avoids learnable-weight collapse)"
    ),
  )
  parser.add_argument(
    "--v5",
    action="store_true",
    help=(
      "MaSE Lite v5: fused-CE + learnable mixture with anti-collapse "
      "(min_w=0.15, ent_w=0.01, ens_lr=0.5x backbone). Goal: beat v4 accuracy."
    ),
  )
  parser.add_argument(
    "--with-uq",
    action="store_true",
    help="After test: MC Dropout PE → AUROC(PE)=UAUC; write uq into results + uq_mc_dropout/",
  )
  parser.add_argument(
    "--no-uq",
    action="store_true",
    help="Disable UQ even for --v5 (v5 enables --with-uq by default)",
  )
  parser.add_argument(
    "--uq-mc-samples",
    type=int,
    default=30,
    help="MC Dropout forward samples for --with-uq (default 30)",
  )
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
    default="mase_lite_full_v3",
  )
  args = parser.parse_args()

  exclusive = [
    name
    for name, flag in [
      ("--legacy-v2-loss", args.legacy_v2_loss),
      ("--freeze-ensemble", args.freeze_ensemble),
      ("--v5", args.v5),
    ]
    if flag
  ]
  if len(exclusive) > 1:
    raise SystemExit(f"Use only one of: {', '.join(exclusive)}")

  if args.v5:
    # Anti-collapse learnable mixture
    args.no_learnable_ensemble = False
    if "--min-ensemble-weight" not in sys.argv:
      args.min_ensemble_weight = 0.15
    if "--ensemble-entropy-weight" not in sys.argv:
      args.ensemble_entropy_weight = 0.01
    if "--ensemble-lr" not in sys.argv:
      args.ensemble_lr = args.backbone_lr * 0.5
    if not args.no_uq:
      args.with_uq = True
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v5"

  if args.freeze_ensemble:
    args.no_learnable_ensemble = True
    args.min_ensemble_weight = 0.0
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v4"

  if args.legacy_v2_loss:
    args.min_ensemble_weight = 0.0
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v2"

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = args.data_dir.resolve()
  ckpt_dir = data_dir / "checkpoints" / args.checkpoint_name

  config = MaSELiteConfig(
    **{
      **config_to_dict(MASE_LITE_DEFAULT),
      "top_k_patches": args.top_k,
      "soft_mask_alpha": args.soft_alpha,
      "branch_dropout": args.branch_dropout,
      "head_dropout": args.head_dropout,
      "learnable_ensemble": not args.no_learnable_ensemble,
      "ensemble_temperature": args.ensemble_temperature,
      "min_ensemble_weight": args.min_ensemble_weight,
    }
  )
  model = MaSELiteNet(config).to(device)

  init_info: dict[str, Any] = {}
  if not args.no_init:
    plcnn_path = args.init_plcnn or resolve_default_init("plcnn_triple")
    selector_path = args.init_selector or resolve_default_init(
      "masked_v3k60_pytorch"
    )
    try:
      if plcnn_path is not None:
        loaded = load_partial_state(
          model,
          plcnn_path,
          prefixes=("branch_vgg.", "branch_resnet.", "branch_densenet."),
        )
        init_info["plcnn"] = {"path": str(plcnn_path), "n_tensors": len(loaded)}
        print(
          f"Init PLCNN branches: {len(loaded)} tensors from {plcnn_path}",
          flush=True,
        )
      else:
        print("Init PLCNN: skipped (checkpoint not found)", flush=True)
      if selector_path is not None:
        loaded = load_partial_state(model, selector_path, prefixes=("selector.",))
        init_info["selector"] = {
          "path": str(selector_path),
          "n_tensors": len(loaded),
        }
        print(
          f"Init selector: {len(loaded)} tensors from {selector_path}",
          flush=True,
        )
      else:
        print("Init selector: skipped (checkpoint not found)", flush=True)
    except RuntimeError as exc:
      print(f"ERROR: {exc}", flush=True)
      raise SystemExit(1) from exc

  ens_params = (
    [model.ensemble_logits]
    if isinstance(model.ensemble_logits, nn.Parameter)
    else []
  )
  ensemble_lr = (
    args.ensemble_lr if args.ensemble_lr is not None else args.backbone_lr * 5.0
  )
  param_groups: list[dict[str, Any]] = [
    {"params": model.selector.parameters(), "lr": args.selector_lr},
    {
      "params": list(model.branch_vgg.parameters())
      + list(model.branch_resnet.parameters())
      + list(model.branch_densenet.parameters())
      + list(model.heads.parameters()),
      "lr": args.backbone_lr,
    },
  ]
  if ens_params and not args.legacy_v2_loss:
    param_groups.append({"params": ens_params, "lr": ensemble_lr})
  optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
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

  if args.legacy_v2_loss:

    def loss_fn(
      details: dict,
      labels: torch.Tensor,
      mask_sparsity_weight: float = 0.0,
      target_mask_fraction: float = 0.6,
      label_smoothing: float = 0.0,
      **kwargs: Any,
    ) -> torch.Tensor:
      return mase_lite_legacy_loss(
        details,
        labels,
        mask_sparsity_weight=mask_sparsity_weight,
        target_mask_fraction=target_mask_fraction,
        label_smoothing=label_smoothing,
      )

    method = "mase_lite_v2"
    loss_desc = "mean_head_ce + sparsity"
    recipe = "v2"
    tqdm_desc = "MaSELiteV2"
  else:

    def loss_fn(
      details: dict,
      labels: torch.Tensor,
      mask_sparsity_weight: float = 0.0,
      target_mask_fraction: float = 0.6,
      label_smoothing: float = 0.0,
      aux_head_weight: float = 0.5,
      distill_weight: float = 0.1,
      ensemble_entropy_weight: float = 0.0,
    ) -> torch.Tensor:
      return mase_lite_loss(
        details,
        labels,
        mask_sparsity_weight=mask_sparsity_weight,
        target_mask_fraction=target_mask_fraction,
        label_smoothing=label_smoothing,
        aux_head_weight=aux_head_weight,
        distill_weight=distill_weight,
        ensemble_entropy_weight=ensemble_entropy_weight,
      )

    method = "mase_lite_v3"
    loss_desc = "fused_nll + aux_head_ce + kl_distill"
    recipe = "v3"
    tqdm_desc = "MaSELiteV3"
    if args.v5:
      method = "mase_lite_v5"
      recipe = "v5"
      loss_desc = (
        "fused_nll + aux + kl | learnable anti-collapse "
        f"min_w={args.min_ensemble_weight} ent_w={args.ensemble_entropy_weight}"
      )
      tqdm_desc = "MaSELiteV5"
    elif args.freeze_ensemble:
      method = "mase_lite_v4"
      recipe = "v4"
      loss_desc = "fused_nll + aux + kl | frozen uniform w=0.25"
      tqdm_desc = "MaSELiteV4"
    elif args.no_learnable_ensemble:
      # Same math as v4; keep explicit --freeze-ensemble for reporting
      method = "mase_lite_v4"
      recipe = "v4"
      loss_desc = "fused_nll + aux + kl | frozen uniform w=0.25"
      tqdm_desc = "MaSELiteV4"

  expected_cov = args.top_k / 64.0
  n_params = sum(p.numel() for p in model.parameters())
  print(
    f"MaSE-Net Lite {recipe} | device={device} | params={n_params:,} | "
    f"top_k={args.top_k} (mask~{expected_cov:.3f}) | "
    f"epochs={args.epochs} patience={args.patience} | "
    f"ls={args.label_smoothing} sparse_w={args.mask_sparsity_weight}",
    flush=True,
  )
  if not args.legacy_v2_loss:
    if args.v5:
      ens_mode = (
        f"v5 learnable anti-collapse ens_lr={ensemble_lr:g} "
        f"min_w={args.min_ensemble_weight} ent_w={args.ensemble_entropy_weight}"
      )
    elif args.freeze_ensemble or args.no_learnable_ensemble:
      ens_mode = "v4 frozen_uniform w=0.25"
    else:
      ens_mode = (
        f"v3 learnable ens_lr={ensemble_lr:g} "
        f"min_w={args.min_ensemble_weight}"
      )
    print(
      f"  fused-CE aux={args.aux_head_weight} distill={args.distill_weight} "
      f"ent_w={args.ensemble_entropy_weight} | {ens_mode}",
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

  epoch_kw = dict(
    mask_sparsity_weight=args.mask_sparsity_weight,
    target_mask_fraction=expected_cov,
    label_smoothing=args.label_smoothing,
    aux_head_weight=args.aux_head_weight,
    distill_weight=args.distill_weight,
    ensemble_entropy_weight=args.ensemble_entropy_weight,
  )

  for epoch in tqdm(range(args.epochs), desc=tqdm_desc, mininterval=5):
    train_m = run_epoch(
      model,
      train_loader,
      device,
      loss_fn,
      optimizer=optimizer,
      **epoch_kw,
    )
    val_m = run_epoch(
      model,
      val_loader,
      device,
      loss_fn,
      optimizer=None,
      **{**epoch_kw, "label_smoothing": 0.0},
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
    loss_fn,
    optimizer=None,
    **{**epoch_kw, "label_smoothing": 0.0},
  )

  elapsed = time.time() - t0

  # Weight health (anti-collapse check for learnable recipes)
  ew = test_m.get("ensemble_weights")
  weight_health: dict[str, Any] | None = None
  if ew is not None:
    min_w = float(min(ew))
    max_w = float(max(ew))
    weight_health = {
      "min_w": min_w,
      "max_w": max_w,
      "ok": bool(min_w >= 0.10 - 1e-6 and max_w <= 0.55 + 1e-6),
      "rule": "ok if min_w>=0.10 and max_w<=0.55",
    }
    if args.v5 and not weight_health["ok"]:
      print(
        f"WARNING: v5 weight health FAIL min={min_w:.3f} max={max_w:.3f} "
        f"(want min>=0.10 max<=0.55)",
        flush=True,
      )

  results = {
    "method": method,
    "framework": "pytorch",
    "config": config_to_dict(config),
    "optimizations": {
      "recipe": recipe,
      "epochs": args.epochs,
      "patience": args.patience,
      "label_smoothing": args.label_smoothing,
      "mask_sparsity_weight": args.mask_sparsity_weight,
      "top_k": args.top_k,
      "strong_augment": (not args.no_augment) and (not args.no_strong_augment),
      "learnable_ensemble": not args.no_learnable_ensemble,
      "legacy_v2_loss": args.legacy_v2_loss,
      "freeze_ensemble": args.freeze_ensemble,
      "v5": args.v5,
      "aux_head_weight": args.aux_head_weight,
      "distill_weight": args.distill_weight,
      "min_ensemble_weight": args.min_ensemble_weight,
      "ensemble_entropy_weight": args.ensemble_entropy_weight,
      "ensemble_lr": ensemble_lr if not args.legacy_v2_loss else None,
      "loss": loss_desc,
      "branch_dropout": args.branch_dropout,
      "head_dropout": args.head_dropout,
      "init": init_info,
      "with_uq": bool(args.with_uq),
      "uq_mc_samples": args.uq_mc_samples if args.with_uq else None,
    },
    "params": n_params,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val,
    "test": test_m,
    "weight_health": weight_health,
    "elapsed_sec": elapsed,
    "history": history,
    "data_dir": str(data_dir),
    "seed": args.seed,
  }

  # PE → AUROC (primary UQ metric = UAUC)
  if args.with_uq:
    from eval_mase_uq import (  # noqa: E402
      mc_predict_loader,
      summarize_split,
      write_per_image_csv,
    )
    import math

    print(
      f"\n=== UQ: MC Dropout T={args.uq_mc_samples} (PE → AUROC) ===",
      flush=True,
    )
    uq_dir = ckpt_dir / "uq_mc_dropout"
    uq_dir.mkdir(parents=True, exist_ok=True)
    pe_max = math.log(config.num_classes)
    thresholds = [float(x) for x in np.linspace(0.0, pe_max, 21)]
    uq_splits: dict[str, Any] = {}
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
      arrays = mc_predict_loader(model, loader, device, args.uq_mc_samples)
      write_per_image_csv(uq_dir / f"per_image_{split_name}.csv", arrays)
      split_sum = summarize_split(arrays, thresholds, config.num_classes)
      uq_splits[split_name] = {
        "n": split_sum["n"],
        "accuracy_mc": split_sum["accuracy"],
        "AUROC_PE": split_sum["UAUC"],  # canonical name
        "UAUC": split_sum["UAUC"],  # alias
        "UAUC_via_1_minus_maxprob": split_sum["UAUC_via_1_minus_maxprob"],
        "pe_mean": split_sum["pe_mean"],
        "pe_std": split_sum["pe_std"],
        "suggested_tau_max_U_F1": split_sum.get("suggested_tau_max_U_F1"),
        "threshold_sweep": split_sum["threshold_sweep"],
      }
      print(
        f"  {split_name}: AUROC(PE)={split_sum['UAUC']:.4f}  "
        f"mc_acc={split_sum['accuracy']:.4f}  "
        f"pe_mean={split_sum['pe_mean']:.4f}",
        flush=True,
      )
    uq_payload = {
      "note": (
        "Primary metric is AUROC(PE)=UAUC: ROC AUC using predictive entropy "
        "to rank incorrect predictions. "
        "UAUC_via_1_minus_maxprob is Softmax baseline only — not primary."
      ),
      "mc_samples": args.uq_mc_samples,
      "splits": uq_splits,
    }
    results["uq"] = {
      "mc_samples": args.uq_mc_samples,
      "test_AUROC_PE": uq_splits["test"]["AUROC_PE"],
      "val_AUROC_PE": uq_splits["val"]["AUROC_PE"],
      "test_UAUC": uq_splits["test"]["UAUC"],
      "val_UAUC": uq_splits["val"]["UAUC"],
      "note": uq_payload["note"],
    }
    with (uq_dir / "summary.json").open("w") as f:
      json.dump(uq_payload, f, indent=2)
    print(f"UQ saved: {uq_dir / 'summary.json'}", flush=True)

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

  print(f"\n=== MaSE-Net Lite {recipe} done ===", flush=True)
  print(f"Best val: {best_val:.4f} @ epoch {best_epoch}", flush=True)
  print(
    f"Test:     {test_m['accuracy']:.4f}  "
    f"(mask={test_m['mask_coverage']:.3f})",
    flush=True,
  )
  if ew := test_m.get("ensemble_weights"):
    print(f"Weights:  [{', '.join(f'{x:.3f}' for x in ew)}]", flush=True)
  if weight_health is not None:
    print(
      f"Weight health: min={weight_health['min_w']:.3f} "
      f"max={weight_health['max_w']:.3f} ok={weight_health['ok']}",
      flush=True,
    )
  if args.with_uq and "uq" in results:
    print(
      f"AUROC(PE) test={results['uq']['test_AUROC_PE']:.4f}  "
      f"val={results['uq']['val_AUROC_PE']:.4f}",
      flush=True,
    )
  print(f"Elapsed:  {elapsed / 60:.1f} min", flush=True)
  print(f"Saved:    {ckpt_dir}", flush=True)


if __name__ == "__main__":
  main()
