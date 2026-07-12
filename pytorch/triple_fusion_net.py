"""Triple fusion: Masked selector + MSMM multi-scale ensemble + PLCNN branches.

Design (advantages from each paper line):
  Masked  — ViT patch selector, hard top-k + soft α blend (focus on informative regions)
  MSMM    — no-downsample ResNet-34 + SE(r=4) + 4-head multi-scale softmax ensemble
  PLCNN   — VGG / ResNet / DenseNet diversity + SE vector heads + dropout

Pipeline:
  x → PatchRegionSelector → soft/hard spatial mask → masked_x
    ├─ MSMMResNet34 → 4 branch logits → MSMM ensemble probs
    └─ PLCNNTripleNet → single logits → PLCNN probs
  final = mean(MSMM_probs, PLCNN_probs)  → log-probs for CE
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
)
from msmm_net import MSMMResNet34, branch_mean_loss, ensemble_logits_from_branches
from plcnn_triple_net import PLCNNTripleNet


@dataclass
class TripleFusionConfig(MaskedModelConfig):
  """Mask hyper-params; backbones use their own defaults."""

  plcnn_dropout: float = 0.5


# Default for 10% pilot: ~60% patch keep, soft blend, Keras-style mask recipe
TRIPLE_FUSION_DEFAULT = TripleFusionConfig(
  top_k_patches=38,
  soft_mask_alpha=0.5,
  selector_layers=2,
  plcnn_dropout=0.5,
)


class TripleFusionNet(nn.Module):
  """Masked input → MSMM ensemble ⊕ PLCNN triple → probability average."""

  def __init__(self, config: TripleFusionConfig | None = None) -> None:
    super().__init__()
    self.config = config or TRIPLE_FUSION_DEFAULT
    self.selector = PatchRegionSelector(self.config)
    self.msmm = MSMMResNet34(num_classes=self.config.num_classes)
    self.plcnn = PLCNNTripleNet(
      num_classes=self.config.num_classes,
      dropout=self.config.plcnn_dropout,
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

    msmm_ensemble, msmm_branches = self.msmm(
      masked_x, return_branch_logits=True
    )
    plcnn_logits = self.plcnn(masked_x)

    msmm_probs = F.softmax(msmm_ensemble, dim=-1)
    plcnn_probs = F.softmax(plcnn_logits, dim=-1)
    fused_probs = 0.5 * (msmm_probs + plcnn_probs)
    fused_logits = torch.log(fused_probs.clamp_min(1e-8))

    if return_details:
      return fused_logits, {
        "mask": mask,
        "patch_probs": patch_probs,
        "msmm_ensemble": msmm_ensemble,
        "msmm_branches": msmm_branches,
        "plcnn_logits": plcnn_logits,
        "fused_probs": fused_probs,
      }
    return fused_logits


def triple_fusion_loss(
  details: dict,
  labels: torch.Tensor,
  mask_sparsity_weight: float = 0.0,
  target_mask_fraction: float = 0.6,
) -> torch.Tensor:
  """Supervise MSMM branches + PLCNN; optional mask coverage regularizer."""
  from masked_net import mask_sparsity_loss

  msmm_ce = branch_mean_loss(details["msmm_branches"], labels)
  plcnn_ce = F.cross_entropy(details["plcnn_logits"], labels)
  loss = 0.5 * (msmm_ce + plcnn_ce)
  if mask_sparsity_weight > 0.0:
    loss = loss + mask_sparsity_weight * mask_sparsity_loss(
      details["patch_probs"], target_mask_fraction
    )
  return loss


def config_to_dict(config: TripleFusionConfig) -> dict:
  return asdict(config)
