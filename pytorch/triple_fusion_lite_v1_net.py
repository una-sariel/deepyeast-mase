"""Lite Triple Fusion v1 (pre-optimization baseline).

  Masked  — ViT patch selector, hard top-k + soft α
  PLCNN   — VGG / ResNet / DenseNet diversity + SE vector heads + dropout
  MSMM    — progressive multi-head softmax ensemble (uniform mean)

This is the architecture that reached ~84.4% test on 10% data
(checkpoint: triple_fusion_lite_10pct). For the optimized recipe see
triple_fusion_lite_net.py + train_triple_fusion_lite.py (v2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
class TripleFusionLiteV1Config(MaskedModelConfig):
  branch_dropout: float = 0.3
  head_dropout: float = 0.5


TRIPLE_FUSION_LITE_V1_DEFAULT = TripleFusionLiteV1Config(
  top_k_patches=38,
  soft_mask_alpha=0.5,
  selector_layers=2,
  branch_dropout=0.3,
  head_dropout=0.5,
)


class TripleFusionLiteV1Net(nn.Module):
  """Masked PLCNN branches + uniform MSMM-style 4-head ensemble (v1)."""

  def __init__(self, config: TripleFusionLiteV1Config | None = None) -> None:
    super().__init__()
    self.config = config or TRIPLE_FUSION_LITE_V1_DEFAULT
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
    probs = torch.stack(
      [F.softmax(logits, dim=-1) for logits in branch_logits], dim=0
    ).mean(dim=0)
    ensemble = torch.log(probs.clamp_min(1e-8))

    if return_details:
      return ensemble, {
        "mask": mask,
        "patch_probs": patch_probs,
        "branch_logits": branch_logits,
        "branch_feats": (v, r, d),
      }
    return ensemble


def lite_fusion_v1_loss(
  details: dict,
  labels: torch.Tensor,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
) -> torch.Tensor:
  losses = [
    F.cross_entropy(logits, labels) for logits in details["branch_logits"]
  ]
  loss = torch.stack(losses).mean()
  if mask_sparsity_weight > 0.0:
    loss = loss + mask_sparsity_weight * mask_sparsity_loss(
      details["patch_probs"], target_mask_fraction
    )
  return loss


def config_to_dict(config: TripleFusionLiteV1Config) -> dict:
  return asdict(config)
