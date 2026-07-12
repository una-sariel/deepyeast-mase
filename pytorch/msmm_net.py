"""Ding et al. (2023) MSMM — PyTorch (shared with deepyeast-msmm)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
  def __init__(self, channels: int, reduction: int = 4) -> None:
    super().__init__()
    hidden = max(channels // reduction, 4)
    self.fc1 = nn.Linear(channels, hidden)
    self.fc2 = nn.Linear(hidden, channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, _, _, _ = x.shape
    squeeze = x.mean(dim=(2, 3))
    excite = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
    return x * excite.view(b, -1, 1, 1)


class MSMMResidualBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.se = SEBlock(out_channels, reduction=4)
    if in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
      )
    else:
      self.shortcut = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self.shortcut(x)
    y = F.relu(self.bn1(self.conv1(x)))
    y = self.se(self.bn2(self.conv2(y)))
    return F.relu(residual + y)


class MSMMResNet34(nn.Module):
  def __init__(self, num_classes: int = 12) -> None:
    super().__init__()
    self.conv_input = nn.Conv2d(2, 32, 3, padding=1, bias=False)
    self.bn_input = nn.BatchNorm2d(32)

    stage_depths = (3, 4, 6, 3)
    stage_channels = (32, 64, 128, 256)
    blocks = []
    in_ch = 32
    for depth, out_ch in zip(stage_depths, stage_channels):
      stage = []
      for _ in range(depth):
        stage.append(MSMMResidualBlock(in_ch, out_ch))
        in_ch = out_ch
      blocks.append(nn.Sequential(*stage))
    self.stages = nn.ModuleList(blocks)

    feat_dims = [256, 256 + 128, 256 + 128 + 64, 256 + 128 + 64 + 32]
    self.heads = nn.ModuleList(
      [nn.Linear(dim, num_classes) for dim in feat_dims]
    )

  def forward(
    self, x: torch.Tensor, return_branch_logits: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    x = F.relu(self.bn_input(self.conv_input(x)))
    gap_vectors: list[torch.Tensor] = []
    for stage in self.stages:
      x = stage(x)
      gap_vectors.append(x.mean(dim=(2, 3)))

    gap1, gap2, gap3, gap4 = gap_vectors
    model_features = [
      gap4,
      torch.cat([gap4, gap3], dim=1),
      torch.cat([gap4, gap3, gap2], dim=1),
      torch.cat([gap4, gap3, gap2, gap1], dim=1),
    ]
    branch_logits = [head(feat) for head, feat in zip(self.heads, model_features)]
    ensemble = ensemble_logits_from_branches(branch_logits)
    if return_branch_logits:
      return ensemble, branch_logits
    return ensemble


def ensemble_logits_from_branches(
  branch_logits: list[torch.Tensor],
) -> torch.Tensor:
  probs = torch.stack([F.softmax(l, dim=-1) for l in branch_logits], dim=0).mean(dim=0)
  return torch.log(probs.clamp_min(1e-8))


def branch_mean_loss(
  branch_logits: list[torch.Tensor], labels: torch.Tensor
) -> torch.Tensor:
  losses = [F.cross_entropy(logits, labels) for logits in branch_logits]
  return torch.stack(losses).mean()
