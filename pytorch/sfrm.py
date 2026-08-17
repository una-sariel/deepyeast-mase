"""Shared-Fixed-Region Mask (SFRM): RF-style spatial bagging for images.

Train: one random S×S window per epoch, applied to every training image.
Val/test (main protocol): no occlusion.
Optional eval: average softmax over the full image plus M random windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mase_lite_net import MaSELiteNet


def sample_window(
  image_size: int,
  size: int,
  rng: np.random.Generator,
) -> tuple[int, int, int]:
  if size <= 0 or size > image_size:
    raise ValueError(f"sfrm size must be in 1..{image_size}, got {size}")
  max_r = image_size - size
  r = int(rng.integers(0, max_r + 1))
  c = int(rng.integers(0, max_r + 1))
  return r, c, size


def apply_region_zero(
  x: torch.Tensor,
  r: int,
  c: int,
  size: int,
) -> torch.Tensor:
  """Zero an axis-aligned window on NCHW (or CHW) tensors."""
  out = x.clone()
  if out.dim() == 3:
    out[:, r : r + size, c : c + size] = 0
  else:
    out[:, :, r : r + size, c : c + size] = 0
  return out


def eval_windows(
  n: int,
  image_size: int,
  size: int,
  seed: int,
) -> list[tuple[int, int, int]]:
  rng = np.random.default_rng(seed)
  return [sample_window(image_size, size, rng) for _ in range(n)]


@torch.inference_mode()
def forward_sfrm_vote(
  model: MaSELiteNet,
  images: torch.Tensor,
  windows: list[tuple[int, int, int]],
  *,
  include_full: bool = True,
) -> torch.Tensor:
  views: list[torch.Tensor] = []
  if include_full:
    views.append(images)
  for r, c, s in windows:
    views.append(apply_region_zero(images, r, c, s))
  probs_sum: torch.Tensor | None = None
  for view in views:
    log_probs = model(view, train=False, return_details=False)
    assert isinstance(log_probs, torch.Tensor)
    probs = log_probs.exp()
    probs_sum = probs if probs_sum is None else probs_sum + probs
  assert probs_sum is not None
  return probs_sum / float(len(views))


@torch.inference_mode()
def eval_sfrm_vote_split(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  windows: list[tuple[int, int, int]],
) -> dict[str, Any]:
  model.eval()
  correct_base = correct_vote = total = 0
  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    log_probs = model(images, train=False, return_details=False)
    assert isinstance(log_probs, torch.Tensor)
    correct_base += int((log_probs.argmax(dim=-1) == labels).sum().item())
    probs = forward_sfrm_vote(model, images, windows, include_full=True)
    correct_vote += int((probs.argmax(dim=-1) == labels).sum().item())
    total += int(labels.numel())
  n = max(total, 1)
  base = correct_base / n
  vote = correct_vote / n
  return {
    "n": total,
    "baseline_accuracy": base,
    "sfrm_vote_accuracy": vote,
    "sfrm_vote_gain_pp": 100.0 * (vote - base),
    "n_windows": len(windows),
    "n_views": len(windows) + 1,
  }
