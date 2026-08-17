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
from eval_mase_uq import roc_auc_score  # noqa: E402
from mase_lite_net import (  # noqa: E402
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  mase_lite_legacy_loss,
  mase_lite_loss,
  load_partial_state,
  predictive_entropy,
)
from sfrm import apply_region_zero, sample_window  # noqa: E402


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
  """Prefer CUDA, then Apple MPS, then CPU. Use --device to force."""
  pref = (pref or "auto").lower()
  if pref == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda")
  if pref == "mps":
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
      raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device("mps")
  if pref == "cpu":
    return torch.device("cpu")
  if pref != "auto":
    raise ValueError(f"Unknown --device {pref!r} (use auto|cuda|mps|cpu)")
  if torch.cuda.is_available():
    return torch.device("cuda")
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


@torch.inference_mode()
def eval_val_uauc_det(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
) -> float:
  """Deterministic PE UAUC on a loader (cheap val monitor; no MC)."""
  model.eval()
  pe_all: list[np.ndarray] = []
  incorrect_all: list[np.ndarray] = []
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    log_probs = model(images, train=False, return_details=False)
    assert isinstance(log_probs, torch.Tensor)
    probs = log_probs.exp()
    pred = probs.argmax(dim=-1)
    pe_all.append(predictive_entropy(probs).cpu().numpy())
    incorrect_all.append((pred != labels).cpu().numpy().astype(np.int32))
  return float(roc_auc_score(np.concatenate(incorrect_all), np.concatenate(pe_all)))


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
  sfrm_window: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
  train = optimizer is not None
  model.train(train)
  losses, accs, coverages = [], [], []

  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if train and sfrm_window is not None:
      r, c, s = sfrm_window
      images = apply_region_zero(images, r, c, s)

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


def weight_health(
  weights: list[float] | None,
  *,
  min_w: float = 0.10,
  max_w: float = 0.40,
) -> dict[str, Any]:
  if not weights:
    return {"min_w": float("nan"), "max_w": float("nan"), "ok": False}
  lo = float(min(weights))
  hi = float(max(weights))
  return {"min_w": lo, "max_w": hi, "ok": lo >= min_w and hi <= max_w}


def load_phase1_val_acc(path: Path) -> float | None:
  if not path.exists():
    return None
  with path.open() as f:
    data = json.load(f)
  val = data.get("best_val_accuracy")
  return float(val) if val is not None else None


def set_phase2_ensemble_only(model: MaSELiteNet) -> int:
  """Freeze all parameters except ensemble_logits; return trainable count."""
  n_trainable = 0
  for name, param in model.named_parameters():
    if name == "ensemble_logits":
      param.requires_grad = True
      n_trainable += param.numel()
    else:
      param.requires_grad = False
  return n_trainable


def set_phase25_ensemble_and_heads(model: MaSELiteNet) -> tuple[int, int]:
  """Freeze selector + branches; train ensemble_logits + progressive heads."""
  n_ens = 0
  n_head = 0
  for name, param in model.named_parameters():
    if name == "ensemble_logits" or name.startswith("heads."):
      param.requires_grad = True
      if name == "ensemble_logits":
        n_ens += param.numel()
      else:
        n_head += param.numel()
    else:
      param.requires_grad = False
  return n_ens, n_head


def set_v8_id_gate_and_heads(model: MaSELiteNet) -> tuple[int, int]:
  """Freeze selector + branches; train id_gate + progressive heads."""
  n_gate = 0
  n_head = 0
  for name, param in model.named_parameters():
    if name.startswith("id_gate.") or name.startswith("heads."):
      param.requires_grad = True
      if name.startswith("id_gate."):
        n_gate += param.numel()
      else:
        n_head += param.numel()
    else:
      param.requires_grad = False
  return n_gate, n_head


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
  parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="auto = CUDA > MPS > CPU (Apple Silicon uses MPS when available)",
  )
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
    "--resume",
    type=Path,
    default=None,
    help="Full MaSE Lite best.pt to continue from (implies --no-init)",
  )
  parser.add_argument(
    "--checkpoint-name",
    type=str,
    default="mase_lite_full_v3",
  )
  parser.add_argument(
    "--track-val-uauc",
    action="store_true",
    help="After each epoch, log deterministic PE UAUC on val",
  )
  parser.add_argument(
    "--select-by",
    choices=["acc", "uauc", "acc_uauc"],
    default="acc",
    help="Criterion for best.pt (acc_uauc = 0.5*acc + 0.5*uauc; needs --track-val-uauc)",
  )
  parser.add_argument(
    "--v6-phase1",
    action="store_true",
    help=(
      "TP-AHF Phase 1 (= v4 recipe): fused-CE, frozen uniform w=0.25. "
      "Default checkpoint: mase_lite_full_v6_phase1"
    ),
  )
  parser.add_argument(
    "--v6-phase2",
    action="store_true",
    help=(
      "TP-AHF Phase 2: resume Phase-1 best.pt, learn mixture weights under "
      "simplex constraints (ensemble-only by default). "
      "Default checkpoint: mase_lite_full_v6"
    ),
  )
  parser.add_argument(
    "--v6-phase25",
    action="store_true",
    help=(
      "TP-AHF Phase 2.5: resume Phase-1 best.pt; learn ensemble_logits + "
      "4 progressive heads (selector + branches frozen). "
      "Default checkpoint: mase_lite_full_v6_phase25"
    ),
  )
  parser.add_argument(
    "--v7",
    action="store_true",
    help=(
      "MaSE Lite v7: tuned Phase 2.5 (head_lr=1e-5, ens_lr_ratio=0.15, "
      "patience=3). Default checkpoint: mase_lite_full_v7. "
      "Run eval_mase_tta.py after (or pass --with-tta)."
    ),
  )
  parser.add_argument(
    "--v8",
    action="store_true",
    help=(
      "MaSE Lite v8 ID-Gate: resume Phase-1; train sample-wise fusion gate "
      "+ heads (selector/branches frozen). "
      "Default checkpoint: mase_lite_full_v8"
    ),
  )
  parser.add_argument(
    "--v9",
    action="store_true",
    help=(
      "MaSE Lite v9 Random Spatial Bagging: resume Phase-1; train heads+w "
      "with per-image random top-k masks (selector/branches frozen). "
      "Default checkpoint: mase_lite_full_v9"
    ),
  )
  parser.add_argument(
    "--rsb-samples",
    type=int,
    default=16,
    help="v9: number of random masks to average at RSB eval (default: 16)",
  )
  parser.add_argument(
    "--with-tta",
    action="store_true",
    help="After test: TTA eval on best.pt → tta_eval/summary.json",
  )
  parser.add_argument(
    "--with-rsb",
    action="store_true",
    help="After test: RSB eval on best.pt → rsb_eval/summary.json (auto with --v9)",
  )
  parser.add_argument(
    "--sfrm",
    action="store_true",
    help=(
      "Shared-fixed-region mask: one random SxS window per train epoch, "
      "applied to all training images. Val/test stay unmasked. "
      "Default: bypass learned selector (v4 frozen-w recipe)."
    ),
  )
  parser.add_argument(
    "--sfrm-size",
    type=int,
    default=24,
    help="SFRM window side length in pixels (default 24)",
  )
  parser.add_argument(
    "--sfrm-keep-selector",
    action="store_true",
    help="Keep learned selector together with SFRM (default: bypass selector)",
  )
  parser.add_argument(
    "--sfrm-windows",
    type=int,
    default=7,
    help="After test: multi-window vote views (plus full image). 0 = skip",
  )
  parser.add_argument(
    "--v10",
    action="store_true",
    help=(
      "MaSE Lite v10 Shared-Fixed-Region Mask (SFRM): alias for --sfrm. "
      "One random SxS window per train epoch on all images. "
      "Default checkpoint: mase_lite_5pct_v10"
    ),
  )
  parser.add_argument(
    "--phase25-head-lr",
    type=float,
    default=None,
    help="LR for progressive heads in --v6-phase25 / --v8 (default: 2e-5)",
  )
  parser.add_argument(
    "--id-gate-lr",
    type=float,
    default=None,
    help="LR for ID-Gate MLP in --v8 (default: 1e-3)",
  )
  parser.add_argument(
    "--id-gate-hidden",
    type=int,
    default=128,
    help="Hidden size of ID-Gate MLP (default: 128)",
  )
  parser.add_argument(
    "--phase2-ensemble-only",
    action="store_true",
    help="Train only ensemble_logits (auto-enabled with --v6-phase2)",
  )
  parser.add_argument(
    "--ensemble-lr-ratio",
    type=float,
    default=None,
    help="ensemble_lr = backbone_lr * ratio (v6 Phase 2 default: 0.25)",
  )
  parser.add_argument(
    "--ensemble-temp-start",
    type=float,
    default=None,
    help="Anneal ensemble temperature from this value (v6 Phase 2 default: 1.5)",
  )
  parser.add_argument(
    "--ensemble-temp-end",
    type=float,
    default=None,
    help="Anneal ensemble temperature to this value (v6 Phase 2 default: 1.0)",
  )
  parser.add_argument(
    "--phase1-checkpoint-name",
    type=str,
    default="mase_lite_full_v6_phase1",
    help="Phase-1 folder under checkpoints/ (for Phase-2 resume + val gate)",
  )
  parser.add_argument(
    "--phase1-val-acc",
    type=float,
    default=None,
    help="Override Phase-1 val acc gate for Phase-2 best.pt (else read results.json)",
  )
  parser.add_argument(
    "--max-ensemble-weight",
    type=float,
    default=0.40,
    help="Phase-2 weight health: reject best if max(w) exceeds this",
  )
  parser.add_argument(
    "--min-weight-health",
    type=float,
    default=0.10,
    help="Phase-2 weight health: reject best if min(w) below this",
  )
  args = parser.parse_args()
  if args.v10:
    args.sfrm = True
  if args.select_by in ("uauc", "acc_uauc"):
    args.track_val_uauc = True
  if args.resume is not None:
    args.no_init = True

  if args.v7:
    if args.v6_phase1 or args.v6_phase2 or args.v6_phase25 or args.v8 or args.v9:
      raise SystemExit("--v7 cannot combine with --v6-phase1/2/25 or --v8/--v9")
    args.v6_phase25 = True

  tp_phase_flags = [
    args.v6_phase1,
    args.v6_phase2,
    args.v6_phase25,
    args.v8,
    args.v9,
  ]
  if sum(bool(x) for x in tp_phase_flags) > 1:
    raise SystemExit(
      "Use only one of --v6-phase1, --v6-phase2, --v6-phase25, --v8, --v9 per run"
    )
  if args.v6_phase2 and args.freeze_ensemble:
    raise SystemExit("--v6-phase2 conflicts with --freeze-ensemble")
  if args.v6_phase25 and args.freeze_ensemble:
    raise SystemExit("--v6-phase25 conflicts with --freeze-ensemble")
  if args.v8 and args.freeze_ensemble:
    raise SystemExit("--v8 conflicts with --freeze-ensemble")
  if args.v9 and args.freeze_ensemble:
    raise SystemExit("--v9 conflicts with --freeze-ensemble")
  if args.v6_phase2 and args.legacy_v2_loss:
    raise SystemExit("--v6-phase2 requires fused-CE (do not use --legacy-v2-loss)")
  if args.v6_phase25 and args.legacy_v2_loss:
    raise SystemExit("--v6-phase25 requires fused-CE (do not use --legacy-v2-loss)")
  if args.v8 and args.legacy_v2_loss:
    raise SystemExit("--v8 requires fused-CE (do not use --legacy-v2-loss)")
  if args.v9 and args.legacy_v2_loss:
    raise SystemExit("--v9 requires fused-CE (do not use --legacy-v2-loss)")
  if args.v9 and args.sfrm:
    raise SystemExit("--v9 conflicts with --v10/--sfrm (RSB vs shared-region)")

  if args.v6_phase1:
    args.freeze_ensemble = True
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v6_phase1"

  if args.v6_phase2:
    args.phase2_ensemble_only = True
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v6"
    if args.epochs == 60:
      args.epochs = 15
    if args.patience == 15:
      args.patience = 5
    if args.min_ensemble_weight == 0.05:
      args.min_ensemble_weight = 0.20
    if args.ensemble_entropy_weight == 0.0:
      args.ensemble_entropy_weight = 0.02
    if args.aux_head_weight == 0.5:
      args.aux_head_weight = 0.0
    if args.distill_weight == 0.1:
      args.distill_weight = 0.0
    if args.mask_sparsity_weight == 0.05:
      args.mask_sparsity_weight = 0.0
    if args.ensemble_lr_ratio is None:
      args.ensemble_lr_ratio = 0.25
    if args.ensemble_temp_start is None:
      args.ensemble_temp_start = 1.5
    if args.ensemble_temp_end is None:
      args.ensemble_temp_end = 1.0
    if args.ensemble_temperature == 1.0 and args.ensemble_temp_start is not None:
      args.ensemble_temperature = args.ensemble_temp_start
    args.no_learnable_ensemble = False
    if args.resume is None:
      phase1_best = (
        args.data_dir.resolve()
        / "checkpoints"
        / args.phase1_checkpoint_name
        / "best.pt"
      )
      if not phase1_best.exists():
        raise SystemExit(
          f"--v6-phase2 needs Phase-1 best.pt at {phase1_best}\n"
          "Run --v6-phase1 first, or pass --resume <path/to/phase1/best.pt>"
        )
      args.resume = phase1_best
      args.no_init = True

  if args.v6_phase25:
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = (
        "mase_lite_full_v7" if args.v7 else "mase_lite_full_v6_phase25"
      )
    if args.epochs == 60:
      args.epochs = 15
    if args.patience == 15:
      args.patience = 5
    if args.min_ensemble_weight == 0.05:
      args.min_ensemble_weight = 0.20
    if args.ensemble_entropy_weight == 0.0:
      args.ensemble_entropy_weight = 0.02
    if args.aux_head_weight == 0.5:
      args.aux_head_weight = 0.0
    if args.distill_weight == 0.1:
      args.distill_weight = 0.0
    if args.mask_sparsity_weight == 0.05:
      args.mask_sparsity_weight = 0.0
    if args.ensemble_lr_ratio is None:
      args.ensemble_lr_ratio = 0.25
    if args.ensemble_temp_start is None:
      args.ensemble_temp_start = 1.5
    if args.ensemble_temp_end is None:
      args.ensemble_temp_end = 1.0
    if args.ensemble_temperature == 1.0 and args.ensemble_temp_start is not None:
      args.ensemble_temperature = args.ensemble_temp_start
    if args.phase25_head_lr is None:
      args.phase25_head_lr = 2e-5
    args.no_learnable_ensemble = False
    if args.resume is None:
      phase1_best = (
        args.data_dir.resolve()
        / "checkpoints"
        / args.phase1_checkpoint_name
        / "best.pt"
      )
      if not phase1_best.exists():
        raise SystemExit(
          f"--v6-phase25 needs Phase-1 best.pt at {phase1_best}\n"
          "Run --v6-phase1 first, or pass --resume <path/to/phase1/best.pt>"
        )
      args.resume = phase1_best
      args.no_init = True

  if args.v7:
    if "--phase25-head-lr" not in sys.argv:
      args.phase25_head_lr = 1e-5
    if "--ensemble-lr-ratio" not in sys.argv:
      args.ensemble_lr_ratio = 0.15
    if args.patience == 5 and "--patience" not in sys.argv:
      args.patience = 3

  if args.v8:
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v8"
    if args.epochs == 60:
      args.epochs = 15
    if args.patience == 15:
      args.patience = 5
    if args.min_ensemble_weight == 0.05:
      args.min_ensemble_weight = 0.15
    if args.ensemble_entropy_weight == 0.0:
      args.ensemble_entropy_weight = 0.02
    if args.aux_head_weight == 0.5:
      args.aux_head_weight = 0.0
    if args.distill_weight == 0.1:
      args.distill_weight = 0.0
    if args.mask_sparsity_weight == 0.05:
      args.mask_sparsity_weight = 0.0
    if args.ensemble_temp_start is None:
      args.ensemble_temp_start = 1.5
    if args.ensemble_temp_end is None:
      args.ensemble_temp_end = 1.0
    if args.ensemble_temperature == 1.0 and args.ensemble_temp_start is not None:
      args.ensemble_temperature = args.ensemble_temp_start
    if args.phase25_head_lr is None:
      args.phase25_head_lr = 1e-5
    if args.id_gate_lr is None:
      args.id_gate_lr = 1e-3
    # Mean-over-batch weight health is softer; allow slightly peakier means
    if "--max-ensemble-weight" not in sys.argv:
      args.max_ensemble_weight = 0.55
    args.no_learnable_ensemble = False
    if args.resume is None:
      phase1_best = (
        args.data_dir.resolve()
        / "checkpoints"
        / args.phase1_checkpoint_name
        / "best.pt"
      )
      if not phase1_best.exists():
        raise SystemExit(
          f"--v8 needs Phase-1 best.pt at {phase1_best}\n"
          "Run --v6-phase1 first, or pass --resume <path/to/phase1/best.pt>"
        )
      args.resume = phase1_best
      args.no_init = True

  if args.v9:
    # Random Spatial Bagging fine-tune (RF-style per-image random masks)
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_full_v9"
    if args.epochs == 60:
      args.epochs = 15
    if args.patience == 15:
      args.patience = 5
    if args.min_ensemble_weight == 0.05:
      args.min_ensemble_weight = 0.15
    if args.ensemble_entropy_weight == 0.0:
      args.ensemble_entropy_weight = 0.02
    if args.aux_head_weight == 0.5:
      args.aux_head_weight = 0.0
    if args.distill_weight == 0.1:
      args.distill_weight = 0.0
    if args.mask_sparsity_weight == 0.05:
      args.mask_sparsity_weight = 0.0
    if args.ensemble_temp_start is None:
      args.ensemble_temp_start = 1.5
    if args.ensemble_temp_end is None:
      args.ensemble_temp_end = 1.0
    if args.ensemble_temperature == 1.0 and args.ensemble_temp_start is not None:
      args.ensemble_temperature = args.ensemble_temp_start
    if args.phase25_head_lr is None:
      args.phase25_head_lr = 1e-5
    if args.ensemble_lr_ratio is None:
      args.ensemble_lr_ratio = 0.25
    if "--max-ensemble-weight" not in sys.argv:
      args.max_ensemble_weight = 0.55
    args.no_learnable_ensemble = False
    args.with_rsb = True
    if args.resume is None:
      phase1_best = (
        args.data_dir.resolve()
        / "checkpoints"
        / args.phase1_checkpoint_name
        / "best.pt"
      )
      if not phase1_best.exists():
        raise SystemExit(
          f"--v9 needs Phase-1 best.pt at {phase1_best}\n"
          "Run --v6-phase1 first, or pass --resume <path/to/phase1/best.pt>"
        )
      args.resume = phase1_best
      args.no_init = True

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

  if args.sfrm:
    if not (
      args.v5
      or args.v6_phase1
      or args.v6_phase2
      or args.v6_phase25
      or args.v7
      or args.v8
      or args.v9
      or args.legacy_v2_loss
    ):
      args.freeze_ensemble = True
    if not args.sfrm_keep_selector and "--mask-sparsity-weight" not in sys.argv:
      args.mask_sparsity_weight = 0.0
    if args.checkpoint_name == "mase_lite_full_v3":
      args.checkpoint_name = "mase_lite_5pct_v10"

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
  device = resolve_device(args.device)
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
      "id_gate": bool(args.v8),
      "id_gate_hidden": int(args.id_gate_hidden),
      "mask_mode": "random" if args.v9 else "learned",
      "bypass_selector": bool(args.sfrm and not args.sfrm_keep_selector),
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
      if config.bypass_selector:
        print("Init selector: skipped (SFRM bypass_selector)", flush=True)
      elif selector_path is not None:
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

  if args.resume is not None:
    resume_path = args.resume.resolve()
    if not resume_path.exists():
      raise SystemExit(f"--resume not found: {resume_path}")
    try:
      state = torch.load(resume_path, map_location=device, weights_only=True)
    except TypeError:
      state = torch.load(resume_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
      state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Resume from {resume_path}", flush=True)
    if missing:
      print(f"  missing keys: {len(missing)}", flush=True)
    if unexpected:
      print(f"  unexpected keys: {len(unexpected)}", flush=True)
    init_info["resume"] = str(resume_path)

  phase1_val_gate = args.phase1_val_acc
  if (
    args.v6_phase2 or args.v6_phase25 or args.v7 or args.v8 or args.v9
  ) and phase1_val_gate is None:
    phase1_results = (
      data_dir / "checkpoints" / args.phase1_checkpoint_name / "results.json"
    )
    phase1_val_gate = load_phase1_val_acc(phase1_results)
    if phase1_val_gate is None:
      print(
        f"Warning: could not read Phase-1 val from {phase1_results}; "
        "Phase-2 gate disabled",
        flush=True,
      )

  if args.v8:
    if model.id_gate is None:
      raise SystemExit("--v8 requires id_gate=True in config")
    n_gate, n_head = set_v8_id_gate_and_heads(model)
    print(
      f"v8 ID-Gate: training id_gate ({n_gate:,}) + "
      f"heads ({n_head:,}); selector+branches frozen",
      flush=True,
    )
  elif args.v9:
    if not isinstance(model.ensemble_logits, nn.Parameter):
      raise SystemExit(
        "v9 requires learnable ensemble_logits "
        "(check --no-learnable-ensemble / --freeze-ensemble)"
      )
    n_ens, n_head = set_phase25_ensemble_and_heads(model)
    print(
      f"v9 RSB: training ensemble_logits ({n_ens:,}) + "
      f"heads ({n_head:,}); selector+branches frozen; "
      f"mask_mode=random (per-image top-k)",
      flush=True,
    )
  elif args.v6_phase25:
    if not isinstance(model.ensemble_logits, nn.Parameter):
      raise SystemExit(
        "v6-phase25 requires learnable ensemble_logits "
        "(check --no-learnable-ensemble / --freeze-ensemble)"
      )
    n_ens, n_head = set_phase25_ensemble_and_heads(model)
    print(
      f"Phase-2.5: training ensemble_logits ({n_ens:,}) + "
      f"heads ({n_head:,}); selector+branches frozen",
      flush=True,
    )
  elif args.phase2_ensemble_only:
    if not isinstance(model.ensemble_logits, nn.Parameter):
      raise SystemExit(
        "phase2-ensemble-only requires learnable ensemble_logits "
        "(check --no-learnable-ensemble / --freeze-ensemble)"
      )
    n_train = set_phase2_ensemble_only(model)
    print(
      f"Phase-2 ensemble-only: training {n_train} params in ensemble_logits",
      flush=True,
    )

  ens_params = (
    [model.ensemble_logits]
    if isinstance(model.ensemble_logits, nn.Parameter)
    else []
  )
  if args.ensemble_lr is not None:
    ensemble_lr = args.ensemble_lr
  elif args.ensemble_lr_ratio is not None:
    ensemble_lr = args.backbone_lr * args.ensemble_lr_ratio
  else:
    ensemble_lr = args.backbone_lr * 5.0

  if args.v8:
    assert model.id_gate is not None
    optimizer = torch.optim.Adam(
      [
        {"params": model.heads.parameters(), "lr": args.phase25_head_lr},
        {"params": model.id_gate.parameters(), "lr": args.id_gate_lr},
      ],
      weight_decay=args.weight_decay,
    )
  elif args.v9:
    optimizer = torch.optim.Adam(
      [
        {"params": model.heads.parameters(), "lr": args.phase25_head_lr},
        {"params": ens_params, "lr": ensemble_lr},
      ],
      weight_decay=args.weight_decay,
    )
  elif args.v6_phase25:
    optimizer = torch.optim.Adam(
      [
        {"params": model.heads.parameters(), "lr": args.phase25_head_lr},
        {"params": ens_params, "lr": ensemble_lr},
      ],
      weight_decay=args.weight_decay,
    )
  elif args.phase2_ensemble_only:
    optimizer = torch.optim.Adam(
      ens_params,
      lr=ensemble_lr,
      weight_decay=args.weight_decay,
    )
  else:
    param_groups: list[dict[str, Any]] = []
    if not config.bypass_selector:
      param_groups.append(
        {"params": model.selector.parameters(), "lr": args.selector_lr}
      )
    else:
      for param in model.selector.parameters():
        param.requires_grad = False
    param_groups.append(
      {
        "params": list(model.branch_vgg.parameters())
        + list(model.branch_resnet.parameters())
        + list(model.branch_densenet.parameters())
        + list(model.heads.parameters()),
        "lr": args.backbone_lr,
      }
    )
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
    if args.v6_phase1:
      method = "mase_lite_v6_phase1"
      recipe = "v6_phase1"
      loss_desc = "TP-AHF Phase1 (= v4): fused_nll + aux + kl | frozen w=0.25"
      tqdm_desc = "MaSELiteV6P1"
    elif args.v6_phase2:
      method = "mase_lite_v6"
      recipe = "v6_phase2"
      loss_desc = (
        "TP-AHF Phase2: fused_nll + ent(w) | simplex min_w + ensemble-only"
      )
      tqdm_desc = "MaSELiteV6P2"
    elif args.v7:
      method = "mase_lite_v7"
      recipe = "v7"
      loss_desc = (
        "v7: tuned Phase2.5 fused_nll + ent(w) | head_lr=1e-5 ens_ratio=0.15"
      )
      tqdm_desc = "MaSELiteV7"
    elif args.v8:
      method = "mase_lite_v8"
      recipe = "v8"
      loss_desc = (
        "v8 ID-Gate: fused_nll + ent(w) | sample-wise w(x) + heads fine-tune"
      )
      tqdm_desc = "MaSELiteV8"
    elif args.v9:
      method = "mase_lite_v9"
      recipe = "v9"
      loss_desc = (
        "v9 RSB: fused_nll + ent(w) | per-image random top-k + heads fine-tune"
      )
      tqdm_desc = "MaSELiteV9"
    elif args.v6_phase25:
      method = "mase_lite_v6_phase25"
      recipe = "v6_phase25"
      loss_desc = (
        "TP-AHF Phase2.5: fused_nll + ent(w) | min_w + heads+w fine-tune"
      )
      tqdm_desc = "MaSELiteV6P25"
    elif args.v5:
      method = "mase_lite_v5"
      recipe = "v5"
      loss_desc = (
        "fused_nll + aux + kl | learnable anti-collapse "
        f"min_w={args.min_ensemble_weight} ent_w={args.ensemble_entropy_weight}"
      )
      tqdm_desc = "MaSELiteV5"
    elif args.freeze_ensemble:
      method = "mase_lite_v10" if args.sfrm else "mase_lite_v4"
      recipe = "v10" if args.sfrm else "v4"
      if args.sfrm:
        loss_desc = (
          f"v10 SFRM S={args.sfrm_size} + fused_nll + aux + kl | frozen w=0.25 | "
          f"selector={'on' if args.sfrm_keep_selector else 'bypass'}"
        )
        tqdm_desc = "MaSELiteV10"
      else:
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
    elif args.v8:
      ens_mode = (
        f"v8 ID-Gate hidden={args.id_gate_hidden} "
        f"gate_lr={args.id_gate_lr:g} heads_lr={args.phase25_head_lr:g} "
        f"min_w={args.min_ensemble_weight} ent_w={args.ensemble_entropy_weight}"
      )
    elif args.v9:
      ens_mode = (
        f"v9 RSB mask=random heads_lr={args.phase25_head_lr:g} "
        f"ens_lr={ensemble_lr:g} rsb_R={args.rsb_samples} "
        f"min_w={args.min_ensemble_weight} ent_w={args.ensemble_entropy_weight}"
      )
    elif args.v6_phase25 or args.v7:
      label = "v7" if args.v7 else "Phase2.5"
      ens_mode = (
        f"{label} heads_lr={args.phase25_head_lr:g} "
        f"ens_lr={ensemble_lr:g} min_w={args.min_ensemble_weight} "
        f"ent_w={args.ensemble_entropy_weight}"
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
  if args.sfrm:
    print(
      f"  SFRM: size={args.sfrm_size}  "
      f"selector={'keep' if args.sfrm_keep_selector else 'bypass'}  "
      f"vote_windows={args.sfrm_windows}  "
      f"(train only; val/test unmasked)",
      flush=True,
    )

  history: list[dict[str, Any]] = []
  best_val = -1.0
  best_epoch = 0
  patience_counter = 0
  best_state: dict[str, torch.Tensor] | None = None
  best_score = -1.0
  best_val_uauc = float("nan")
  t0 = time.time()

  epoch_kw = dict(
    mask_sparsity_weight=args.mask_sparsity_weight,
    target_mask_fraction=expected_cov,
    label_smoothing=args.label_smoothing,
    aux_head_weight=args.aux_head_weight,
    distill_weight=args.distill_weight,
    ensemble_entropy_weight=args.ensemble_entropy_weight,
  )

  temp_start = args.ensemble_temp_start
  temp_end = args.ensemble_temp_end
  anneal_temps = (
    temp_start is not None
    and temp_end is not None
    and abs(temp_start - temp_end) > 1e-9
  )

  for epoch in tqdm(range(args.epochs), desc=tqdm_desc, mininterval=5):
    if anneal_temps:
      denom = max(args.epochs - 1, 1)
      frac = epoch / denom
      t_cur = temp_start + (temp_end - temp_start) * frac
      model.ensemble_temperature = max(float(t_cur), 1e-3)
    sfrm_window = None
    if args.sfrm:
      rng = np.random.default_rng(args.seed + 10007 * (epoch + 1))
      sfrm_window = sample_window(config.image_size, args.sfrm_size, rng)
    train_m = run_epoch(
      model,
      train_loader,
      device,
      loss_fn,
      optimizer=optimizer,
      sfrm_window=sfrm_window,
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

    val_uauc = float("nan")
    if args.track_val_uauc:
      val_uauc = eval_val_uauc_det(model, val_loader, device)
      val_m["uauc_pe_det"] = val_uauc

    if args.select_by == "uauc":
      select_score = val_uauc
    elif args.select_by == "acc_uauc":
      select_score = 0.5 * float(val_m["accuracy"]) + 0.5 * val_uauc
    else:
      select_score = float(val_m["accuracy"])

    row = {
      "epoch": epoch + 1,
      "train": train_m,
      "val": val_m,
      "select_score": select_score,
      "lr": float(optimizer.param_groups[-1]["lr"]),
    }
    if sfrm_window is not None:
      row["sfrm_window"] = list(sfrm_window)
    history.append(row)
    ew = val_m.get("ensemble_weights")
    wh = weight_health(
      ew,
      min_w=args.min_weight_health,
      max_w=args.max_ensemble_weight,
    )
    ew_str = (
      " w=[" + ",".join(f"{x:.2f}" for x in ew) + "]" if ew is not None else ""
    )
    health_str = ""
    if ew is not None and (
      args.v6_phase2 or args.v6_phase25 or args.v7 or args.v8 or args.v9
    ):
      health_str = f"  wh={'ok' if wh['ok'] else 'BAD'}"
    uauc_str = f"  uauc={val_uauc:.3f}" if args.track_val_uauc else ""
    temp_str = ""
    if anneal_temps:
      temp_str = f"  T={model.ensemble_temperature:.2f}"
    print(
      f"  ep{epoch + 1:02d}  "
      f"train={train_m['accuracy']:.3f}  "
      f"val={val_m['accuracy']:.3f}  "
      f"mask={val_m['mask_coverage']:.3f}  "
      f"loss={val_m['loss']:.4f}{uauc_str}{temp_str}{ew_str}{health_str}"
      + (
        f"  sfrm=({sfrm_window[0]},{sfrm_window[1]},{sfrm_window[2]})"
        if sfrm_window is not None
        else ""
      ),
      flush=True,
    )

    phase_gate_eligible = True
    if args.v6_phase2 or args.v6_phase25 or args.v7 or args.v8 or args.v9:
      phase_gate_eligible = wh["ok"]
      if phase1_val_gate is not None:
        phase_gate_eligible = phase_gate_eligible and (
          float(val_m["accuracy"]) >= float(phase1_val_gate)
        )

    if select_score > best_score and phase_gate_eligible:
      best_score = select_score
      best_val = float(val_m["accuracy"])
      best_val_uauc = val_uauc
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
          f"(best val_acc={best_val:.4f} val_uauc={best_val_uauc:.4f} "
          f"score={best_score:.4f} @ {best_epoch})",
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
  final_ew = test_m.get("ensemble_weights")
  wh_max = 0.55 if args.v5 else args.max_ensemble_weight
  wh_min = args.min_weight_health
  final_wh = weight_health(final_ew, min_w=wh_min, max_w=wh_max)
  if args.v5:
    final_wh["rule"] = "ok if min_w>=0.10 and max_w<=0.55"
  elif args.v6_phase2 or args.v6_phase25 or args.v7 or args.v8 or args.v9:
    final_wh["rule"] = (
      f"ok if min_w>={wh_min} and max_w<={args.max_ensemble_weight}"
    )
  if args.v5 and final_ew is not None and not final_wh["ok"]:
    print(
      f"WARNING: v5 weight health FAIL min={final_wh['min_w']:.3f} "
      f"max={final_wh['max_w']:.3f} (want min>=0.10 max<=0.55)",
      flush=True,
    )
  tp_ahf: dict[str, Any] | None = None
  if (
    args.v6_phase1
    or args.v6_phase2
    or args.v6_phase25
    or args.v7
    or args.v8
    or args.v9
  ):
    tp_ahf = {
      "phase": recipe,
      "phase1_checkpoint_name": args.phase1_checkpoint_name,
      "phase1_val_gate": phase1_val_gate,
      "phase2_ensemble_only": bool(args.phase2_ensemble_only),
      "phase25_unfreeze_heads": bool(args.v6_phase25 and not args.v6_phase2),
      "phase25_head_lr": (
        args.phase25_head_lr
        if (args.v6_phase25 or args.v8 or args.v9)
        else None
      ),
      "v7": bool(args.v7),
      "v8_id_gate": bool(args.v8),
      "v9_rsb": bool(args.v9),
      "rsb_samples": args.rsb_samples if args.v9 else None,
      "mask_mode": "random" if args.v9 else "learned",
      "id_gate_lr": args.id_gate_lr if args.v8 else None,
      "id_gate_hidden": args.id_gate_hidden if args.v8 else None,
      "ensemble_lr_ratio": args.ensemble_lr_ratio,
      "ensemble_temp_start": temp_start,
      "ensemble_temp_end": temp_end,
      "max_ensemble_weight": args.max_ensemble_weight,
      "min_weight_health": args.min_weight_health,
    }
  results = {
    "method": method,
    "framework": "pytorch",
    "config": config_to_dict(config),
    "weight_health": final_wh if final_ew is not None else None,
    "tp_ahf": tp_ahf,
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
      "track_val_uauc": args.track_val_uauc,
      "select_by": args.select_by,
      "phase2_ensemble_only": bool(args.phase2_ensemble_only),
      "v6_phase25": bool(args.v6_phase25 and not args.v7),
      "v7": bool(args.v7),
      "v8": bool(args.v8),
      "v9": bool(args.v9),
      "phase25_head_lr": (
        args.phase25_head_lr
        if (args.v6_phase25 or args.v8 or args.v9)
        else None
      ),
      "id_gate_lr": args.id_gate_lr if args.v8 else None,
      "id_gate_hidden": args.id_gate_hidden if args.v8 else None,
      "rsb_samples": args.rsb_samples if args.v9 else None,
      "mask_mode": "random" if args.v9 else "learned",
      "ensemble_lr_ratio": args.ensemble_lr_ratio,
      "ensemble_temp_start": temp_start,
      "ensemble_temp_end": temp_end,
      "init": init_info,
      "with_uq": bool(args.with_uq),
      "with_tta": bool(args.with_tta),
      "with_rsb": bool(args.with_rsb),
      "sfrm": bool(args.sfrm),
      "sfrm_size": args.sfrm_size if args.sfrm else None,
      "sfrm_keep_selector": bool(args.sfrm_keep_selector) if args.sfrm else None,
      "sfrm_windows": args.sfrm_windows if args.sfrm else None,
      "uq_mc_samples": args.uq_mc_samples if args.with_uq else None,
    },
    "params": n_params,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val,
    "best_val_uauc_pe_det": best_val_uauc,
    "best_select_score": best_score,
    "test": test_m,
    "elapsed_sec": elapsed,
    "history": history,
    "data_dir": str(data_dir),
    "seed": args.seed,
  }

  if args.with_tta:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "results.json").open("w") as f:
      json.dump(results, f, indent=2)
    from eval_mase_tta import run_tta_eval  # noqa: E402

    print("\n=== TTA eval (flip + rot90, no retrain) ===", flush=True)
    tta_payload = run_tta_eval(
      data_dir=data_dir,
      checkpoint=ckpt_dir / "best.pt",
      results_json=ckpt_dir / "results.json",
      split="both",
      tta_mode="flip_rot",
      batch_size=args.batch_size,
      device_pref=args.device,
      seed=args.seed,
      out_dir=ckpt_dir / "tta_eval",
    )
    results["tta"] = {
      "tta_mode": tta_payload.get("tta_mode"),
      "n_views": tta_payload.get("n_views"),
      "test_baseline_accuracy": tta_payload.get("test_baseline_accuracy"),
      "test_tta_accuracy": tta_payload.get("test_tta_accuracy"),
      "test_tta_gain_pp": tta_payload.get("test_tta_gain_pp"),
      "val_baseline_accuracy": tta_payload.get("splits", {})
      .get("val", {})
      .get("baseline_accuracy"),
      "val_tta_accuracy": tta_payload.get("splits", {})
      .get("val", {})
      .get("tta_accuracy"),
    }
    if results["tta"].get("test_tta_accuracy") is not None:
      print(
        f"  test: baseline={results['tta']['test_baseline_accuracy']:.4f}  "
        f"tta={results['tta']['test_tta_accuracy']:.4f}  "
        f"gain={results['tta']['test_tta_gain_pp']:+.2f}pp",
        flush=True,
      )
    print(f"TTA saved: {ckpt_dir / 'tta_eval' / 'summary.json'}", flush=True)

  if args.with_rsb:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "results.json").open("w") as f:
      json.dump(results, f, indent=2)
    from eval_mase_rsb import run_rsb_eval  # noqa: E402

    print(
      f"\n=== RSB eval (R={args.rsb_samples} random masks / image) ===",
      flush=True,
    )
    rsb_payload = run_rsb_eval(
      data_dir=data_dir,
      checkpoint=ckpt_dir / "best.pt",
      results_json=ckpt_dir / "results.json",
      split="both",
      rsb_samples=args.rsb_samples,
      batch_size=args.batch_size,
      device_pref=args.device,
      seed=args.seed,
      out_dir=ckpt_dir / "rsb_eval",
    )
    test_rsb = rsb_payload.get("splits", {}).get("test", {})
    results["rsb"] = {
      "rsb_samples": args.rsb_samples,
      "test_random_single": (test_rsb.get("random_single") or {}).get(
        "accuracy"
      ),
      "test_rsb": (test_rsb.get("rsb") or {}).get("accuracy"),
      "test_learned_single": (test_rsb.get("learned_single") or {}).get(
        "accuracy"
      ),
    }
    if results["rsb"].get("test_rsb") is not None:
      print(
        f"  test: random1={results['rsb']['test_random_single']:.4f}  "
        f"rsb={results['rsb']['test_rsb']:.4f}"
        + (
          f"  learned={results['rsb']['test_learned_single']:.4f}"
          if results["rsb"].get("test_learned_single") is not None
          else ""
        ),
        flush=True,
      )
    print(f"RSB saved: {ckpt_dir / 'rsb_eval' / 'summary.json'}", flush=True)

  if args.sfrm and args.sfrm_windows > 0:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "results.json").open("w") as f:
      json.dump(results, f, indent=2)
    from eval_mase_sfrm import run_sfrm_vote_eval  # noqa: E402

    print(
      f"\n=== SFRM multi-window vote (S={args.sfrm_size}, "
      f"M={args.sfrm_windows} + full) ===",
      flush=True,
    )
    vote_payload = run_sfrm_vote_eval(
      data_dir=data_dir,
      checkpoint=ckpt_dir / "best.pt",
      results_json=ckpt_dir / "results.json",
      split="both",
      sfrm_size=args.sfrm_size,
      sfrm_windows=args.sfrm_windows,
      batch_size=args.batch_size,
      device_pref=args.device,
      seed=args.seed,
      out_dir=ckpt_dir / "sfrm_vote",
    )
    results["sfrm_vote"] = {
      "sfrm_size": args.sfrm_size,
      "n_windows": args.sfrm_windows,
      "test_baseline_accuracy": vote_payload.get("test_baseline_accuracy"),
      "test_sfrm_vote_accuracy": vote_payload.get("test_sfrm_vote_accuracy"),
      "test_sfrm_vote_gain_pp": vote_payload.get("test_sfrm_vote_gain_pp"),
    }
    if results["sfrm_vote"].get("test_sfrm_vote_accuracy") is not None:
      print(
        f"  test: baseline={results['sfrm_vote']['test_baseline_accuracy']:.4f}  "
        f"vote={results['sfrm_vote']['test_sfrm_vote_accuracy']:.4f}  "
        f"gain={results['sfrm_vote']['test_sfrm_vote_gain_pp']:+.2f}pp",
        flush=True,
      )
    print(f"SFRM vote saved: {ckpt_dir / 'sfrm_vote' / 'summary.json'}", flush=True)

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
  print(
    f"Best val: acc={best_val:.4f} uauc={best_val_uauc:.4f} "
    f"score={best_score:.4f} @ epoch {best_epoch} (select_by={args.select_by})",
    flush=True,
  )
  print(
    f"Test:     {test_m['accuracy']:.4f}  "
    f"(mask={test_m['mask_coverage']:.3f})",
    flush=True,
  )
  if ew := test_m.get("ensemble_weights"):
    print(f"Weights:  [{', '.join(f'{x:.3f}' for x in ew)}]", flush=True)
  if final_ew is not None and (
    args.v5
    or args.v6_phase2
    or args.v6_phase25
    or args.v7
    or args.v8
    or args.v9
  ):
    print(
      f"Weight health: min={final_wh['min_w']:.3f} "
      f"max={final_wh['max_w']:.3f} ok={final_wh['ok']}"
      + (" (mean over batch)" if args.v8 else ""),
      flush=True,
    )
  if args.v9 and results.get("rsb", {}).get("test_rsb") is not None:
    print(
      f"RSB test:     {results['rsb']['test_rsb']:.4f} "
      f"(R={args.rsb_samples})",
      flush=True,
    )
  if (
    args.v6_phase2 or args.v6_phase25 or args.v7 or args.v8 or args.v9
  ) and phase1_val_gate is not None:
    beat = float(test_m["accuracy"]) >= float(phase1_val_gate)
    print(
      f"vs Phase-1 val gate ({phase1_val_gate:.4f}): "
      f"test={'PASS' if beat else 'below'}",
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
