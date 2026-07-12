"""Lite Triple Fusion for CPU / 10% pilots (v2: learnable ensemble + stronger reg).

  Masked  — ViT patch selector, hard top-k + soft α + sparsity regularizer
  PLCNN   — VGG / ResNet / DenseNet diversity + SE + higher dropout
  MSMM    — progressive multi-head ensemble with learnable weights / temperature
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_net import (
  MaskedModelConfig,
  PatchRegionSelector,
  apply_spatial_mask,
  mcherry_cell_mask,
  mask_sparsity_loss,
)
from plcnn_triple_net import (
  BRANCH_DIM,
  DenseNetBranch,
  ResNetBranch,
  VGGBranch,
)


@dataclass
class TripleFusionLiteConfig(MaskedModelConfig):
  branch_dropout: float = 0.4
  head_dropout: float = 0.5
  learnable_ensemble: bool = True
  ensemble_temperature: float = 1.0


TRIPLE_FUSION_LITE_DEFAULT = TripleFusionLiteConfig(
  top_k_patches=40,
  soft_mask_alpha=0.5,
  selector_layers=2,
  branch_dropout=0.4,
  head_dropout=0.5,
  learnable_ensemble=True,
  ensemble_temperature=1.0,
)


class TripleFusionLiteNet(nn.Module):
  """Masked PLCNN branches + MSMM-style multi-head ensemble."""

  def __init__(self, config: TripleFusionLiteConfig | None = None) -> None:
    super().__init__()
    self.config = config or TRIPLE_FUSION_LITE_DEFAULT
    cfg = self.config
    self.selector = PatchRegionSelector(cfg)
    self.branch_vgg = VGGBranch(dropout=cfg.branch_dropout)
    self.branch_resnet = ResNetBranch(dropout=cfg.branch_dropout)
    self.branch_densenet = DenseNetBranch(dropout=cfg.branch_dropout)

    self.heads = nn.ModuleList(
      [
        nn.Linear(BRANCH_DIM, cfg.num_classes),
        nn.Linear(BRANCH_DIM * 2, cfg.num_classes),
        nn.Linear(BRANCH_DIM * 3, cfg.num_classes),
        nn.Sequential(
          nn.Linear(BRANCH_DIM * 3, 512),
          nn.ReLU(inplace=True),
          nn.Dropout(cfg.head_dropout),
          nn.Linear(512, cfg.num_classes),
        ),
      ]
    )
    # Learnable log-weights over the 4 heads (softmax → mixture)
    if cfg.learnable_ensemble:
      self.ensemble_logits = nn.Parameter(torch.zeros(4))
    else:
      self.register_buffer("ensemble_logits", torch.zeros(4), persistent=False)
    self.ensemble_temperature = max(float(cfg.ensemble_temperature), 1e-3)

  def _mask_input(
    self, x: torch.Tensor, train: bool
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = self.config
    mask, patch_probs = self.selector(x, train=train)
    mcherry_prior = None
    if cfg.mcherry_guided:
      mcherry_prior = mcherry_cell_mask(x, cfg.mcherry_threshold)
    masked_x = apply_spatial_mask(
      x,
      mask,
      mask_gfp_only=cfg.mask_gfp_only,
      soft_alpha=cfg.soft_mask_alpha,
      mcherry_prior=mcherry_prior,
    )
    return masked_x, mask, patch_probs

  def _ensemble_probs(self, branch_logits: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(
      [F.softmax(logits, dim=-1) for logits in branch_logits], dim=1
    )  # (B, 4, C)
    weights = F.softmax(self.ensemble_logits / self.ensemble_temperature, dim=0)
    return (stacked * weights.view(1, 4, 1)).sum(dim=1)

  def forward(
    self,
    x: torch.Tensor,
    train: bool = True,
    return_details: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, dict]:
    masked_x, mask, patch_probs = self._mask_input(x, train=train)
    v = self.branch_vgg(masked_x)
    r = self.branch_resnet(masked_x)
    d = self.branch_densenet(masked_x)

    feats = [
      v,
      torch.cat([v, r], dim=1),
      torch.cat([v, r, d], dim=1),
      torch.cat([v, r, d], dim=1),
    ]
    branch_logits = [head(feat) for head, feat in zip(self.heads, feats)]
    probs = self._ensemble_probs(branch_logits)
    ensemble = torch.log(probs.clamp_min(1e-8))

    if return_details:
      return ensemble, {
        "mask": mask,
        "patch_probs": patch_probs,
        "branch_logits": branch_logits,
        "branch_feats": (v, r, d),
        "ensemble_weights": F.softmax(
          self.ensemble_logits / self.ensemble_temperature, dim=0
        ).detach(),
      }
    return ensemble


def lite_fusion_loss(
  details: dict,
  labels: torch.Tensor,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
  label_smoothing: float = 0.0,
) -> torch.Tensor:
  losses = [
    F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    for logits in details["branch_logits"]
  ]
  loss = torch.stack(losses).mean()
  if mask_sparsity_weight > 0.0:
    loss = loss + mask_sparsity_weight * mask_sparsity_loss(
      details["patch_probs"], target_mask_fraction
    )
  return loss


def load_partial_state(
  model: TripleFusionLiteNet,
  checkpoint_path: Path,
  prefixes: tuple[str, ...] | None = None,
) -> list[str]:
  """Load matching keys from a checkpoint; optional key-prefix filter."""
  state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
  model_state = model.state_dict()
  loaded = []
  for key, value in state.items():
    if prefixes is not None and not any(key.startswith(p) for p in prefixes):
      continue
    if key in model_state and model_state[key].shape == value.shape:
      model_state[key] = value
      loaded.append(key)
  model.load_state_dict(model_state)
  return loaded


def config_to_dict(config: TripleFusionLiteConfig) -> dict:
  return asdict(config)
