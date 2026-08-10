#!/usr/bin/env python3
"""Per-head / mixture contribution analysis for MaSE-Net Lite (no UQ).

Reports:
  - global mixture weights w
  - single-head test accuracy (H0..H3) vs fused vs oracle
  - per-class accuracy heatmap (classes x heads)
  - per-class head vote share on fused-correct samples

Example:
  python pytorch/eval_mase_head_contrib.py \\
    --data-dir C:/Users/unaliuqw/deepyeast_10pct \\
    --checkpoint .../triple_fusion_lite_10pct_v2/best.pt \\
    --out-dir .../triple_fusion_lite_10pct_v2/head_contrib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import load_metadata, make_loaders
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device, set_seed
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict

HEAD_NAMES = (
  "H0_VGG",
  "H1_VGG+Res",
  "H2_VGG+Res+Dense",
  "H3_VGG+Res+Dense+MLP",
)


def load_class_names(data_dir: Path, num_classes: int) -> list[str]:
  path = data_dir / "class_map.json"
  if not path.exists():
    return [f"class_{i}" for i in range(num_classes)]
  with path.open() as f:
    data = json.load(f)
  idx_map = data.get("index_to_name") or data
  return [str(idx_map.get(str(i), f"class_{i}")) for i in range(num_classes)]


@torch.inference_mode()
def eval_head_contrib(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
  num_classes: int,
) -> dict:
  n = 0
  correct_fused = 0
  correct_head = np.zeros(4, dtype=np.int64)
  oracle_correct = 0

  # per-class: count and correct per head + fused
  cls_n = np.zeros(num_classes, dtype=np.int64)
  cls_correct_fused = np.zeros(num_classes, dtype=np.int64)
  cls_correct_head = np.zeros((num_classes, 4), dtype=np.int64)

  # on fused-correct samples: how often each head also voted the true label
  fused_ok_head_vote = np.zeros(4, dtype=np.int64)
  fused_ok_n = 0

  disagree_sum = 0.0

  for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    b = labels.size(0)

    log_probs, details = model(images, train=False, return_details=True)
    stacked = torch.stack(
      [F.softmax(logits, dim=-1) for logits in details["branch_logits"]],
      dim=1,
    )  # (B, 4, C)
    head_preds = stacked.argmax(dim=-1)  # (B, 4)
    fused_pred = log_probs.argmax(dim=-1)
    y = labels

    n += b
    correct_fused += int((fused_pred == y).sum().item())

    for h in range(4):
      correct_head[h] += int((head_preds[:, h] == y).sum().item())

    oracle = (head_preds == y.unsqueeze(1)).any(dim=1)
    oracle_correct += int(oracle.sum().item())

    mean_pred = stacked.mean(dim=1).argmax(dim=-1)
    disagree_sum += float((head_preds != mean_pred.unsqueeze(1)).float().mean().item()) * b

    fused_ok = fused_pred == y
    if fused_ok.any():
      fo = fused_ok
      fused_ok_n += int(fo.sum().item())
      for h in range(4):
        fused_ok_head_vote[h] += int((head_preds[fo, h] == y[fo]).sum().item())

    for c in range(num_classes):
      mask = y == c
      cnt = int(mask.sum().item())
      if cnt == 0:
        continue
      cls_n[c] += cnt
      cls_correct_fused[c] += int((fused_pred[mask] == c).sum().item())
      for h in range(4):
        cls_correct_head[c, h] += int((head_preds[mask, h] == c).sum().item())

  weights = model.mixture_weights().detach().cpu().tolist()
  acc_fused = correct_fused / max(n, 1)
  acc_heads = [correct_head[h] / max(n, 1) for h in range(4)]
  acc_oracle = oracle_correct / max(n, 1)

  per_class_acc_fused = np.divide(
    cls_correct_fused,
    np.maximum(cls_n, 1),
    where=cls_n > 0,
    out=np.full(num_classes, np.nan),
  )
  per_class_acc_head = np.divide(
    cls_correct_head,
    cls_n[:, None],
    where=cls_n[:, None] > 0,
    out=np.full((num_classes, 4), np.nan),
  )

  vote_share = [
    fused_ok_head_vote[h] / max(fused_ok_n, 1) for h in range(4)
  ]

  return {
    "n": n,
    "ensemble_weights": [float(w) for w in weights],
    "head_names": list(HEAD_NAMES),
    "accuracy": {
      "fused": float(acc_fused),
      "heads": {HEAD_NAMES[h]: float(acc_heads[h]) for h in range(4)},
      "oracle_any_head": float(acc_oracle),
      "fusion_gain_vs_best_head": float(
        acc_fused - max(acc_heads) if acc_heads else acc_fused
      ),
      "oracle_gap": float(acc_oracle - acc_fused),
    },
    "mean_head_disagree_frac": float(disagree_sum / max(n, 1)),
    "fused_correct_head_vote_share": {
      HEAD_NAMES[h]: float(vote_share[h]) for h in range(4)
    },
    "fused_correct_n": fused_ok_n,
    "per_class_n": cls_n.tolist(),
    "per_class_acc_fused": [
      float(x) if x == x else None for x in per_class_acc_fused.tolist()
    ],
    "per_class_acc_head": {
      HEAD_NAMES[h]: [
        float(per_class_acc_head[c, h]) if cls_n[c] > 0 else None
        for c in range(num_classes)
      ]
      for h in range(4)
    },
  }


def save_heatmap(
  path: Path,
  matrix: np.ndarray,
  row_labels: list[str],
  col_labels: list[str],
  title: str,
) -> None:
  fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(row_labels))))
  im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGn")
  ax.set_xticks(range(len(col_labels)))
  ax.set_xticklabels(col_labels, rotation=30, ha="right")
  ax.set_yticks(range(len(row_labels)))
  ax.set_yticklabels(row_labels, fontsize=8)
  ax.set_title(title)
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      if not np.isnan(matrix[i, j]):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
  fig.colorbar(im, ax=ax, fraction=0.03)
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=140, bbox_inches="tight")
  plt.close(fig)


def write_markdown(path: Path, report: dict, class_names: list[str]) -> None:
  acc = report["accuracy"]
  w = report["ensemble_weights"]
  lines = [
    "# MaSE head contribution",
    "",
    f"**N (test):** {report['n']}",
    "",
    "## Global mixture weights",
    "",
    "| Head | w |",
    "|------|---|",
  ]
  for name, wi in zip(HEAD_NAMES, w):
    lines.append(f"| {name} | {wi:.3f} |")

  lines += [
    "",
    "## Test accuracy",
    "",
    "| Predictor | Acc |",
    "|-----------|-----|",
  ]
  for name in HEAD_NAMES:
    lines.append(f"| {name} alone | {acc['heads'][name]:.4f} |")
  lines += [
    f"| **Fused (MaSE)** | **{acc['fused']:.4f}** |",
    f"| Oracle (any head correct) | {acc['oracle_any_head']:.4f} |",
    f"| Fusion gain vs best single head | {acc['fusion_gain_vs_best_head']:+.4f} |",
    f"| Oracle gap (room to improve) | {acc['oracle_gap']:.4f} |",
    "",
    "## When fused is correct: head also voted true label",
    "",
    "| Head | Share |",
    "|------|-------|",
  ]
  for name in HEAD_NAMES:
    share = report["fused_correct_head_vote_share"][name]
    lines.append(f"| {name} | {share:.3f} |")

  lines += [
    "",
    f"Mean head disagree fraction: {report['mean_head_disagree_frac']:.3f}",
    "",
    "## Per-class accuracy (fused vs heads)",
    "",
    "| Class | n | Fused | " + " | ".join(HEAD_NAMES) + " |",
    "|-------|---|-------|" + "|".join(["---"] * 4) + "|",
  ]
  for c, name in enumerate(class_names):
    nc = report["per_class_n"][c]
    if nc == 0:
      continue
    row = [
      name[:20],
      str(nc),
      f"{report['per_class_acc_fused'][c]:.3f}",
    ]
    for hname in HEAD_NAMES:
      v = report["per_class_acc_head"][hname][c]
      row.append(f"{v:.3f}" if v is not None else "-")
    lines.append("| " + " | ".join(row) + " |")

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description="MaSE head / branch contribution")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--split", choices=["test", "val"], default="test")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  args = parser.parse_args()

  set_seed(args.seed)
  device = resolve_device(args.device)
  data_dir = args.data_dir.resolve()
  ckpt = args.checkpoint.resolve()
  out_dir = args.out_dir or (ckpt.parent / "head_contrib")
  out_dir.mkdir(parents=True, exist_ok=True)

  results_json = args.results_json or (ckpt.parent / "results.json")
  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  num_classes = config.num_classes
  class_names = load_class_names(data_dir, num_classes)

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  model.eval()

  _, val_loader, test_loader = make_loaders(
    data_dir, batch_size=args.batch_size, augment=False
  )
  loader = test_loader if args.split == "test" else val_loader

  report = eval_head_contrib(model, loader, device, num_classes)
  report["checkpoint"] = str(ckpt)
  report["data_dir"] = str(data_dir)
  report["split"] = args.split
  report["class_names"] = class_names

  json_path = out_dir / "head_contrib.json"
  with json_path.open("w") as f:
    json.dump(report, f, indent=2)

  md_path = out_dir / "head_contrib_table.md"
  write_markdown(md_path, report, class_names)

  # heatmap: classes with n>0
  rows, mat = [], []
  for c, name in enumerate(class_names):
    if report["per_class_n"][c] == 0:
      continue
    rows.append(name[:24])
    mat.append([report["per_class_acc_head"][hn][c] for hn in HEAD_NAMES])
  if mat:
    save_heatmap(
      out_dir / "per_class_head_acc_heatmap.png",
      np.array(mat, dtype=np.float64),
      rows,
      list(HEAD_NAMES),
      f"Per-class accuracy by head ({args.split})",
    )

  w = report["ensemble_weights"]
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.bar(range(4), w, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
  ax.set_xticks(range(4))
  ax.set_xticklabels([n.replace("_", "\n") for n in HEAD_NAMES], fontsize=8)
  ax.set_ylim(0, max(w) * 1.25 if w else 0.3)
  ax.set_ylabel("mixture weight w")
  ax.set_title("Global ensemble weights")
  for i, wi in enumerate(w):
    ax.text(i, wi + 0.01, f"{wi:.3f}", ha="center", fontsize=9)
  fig.tight_layout()
  fig.savefig(out_dir / "ensemble_weights_bar.png", dpi=140)
  plt.close(fig)

  acc = report["accuracy"]
  print(f"\n=== Head contribution ({args.split}, n={report['n']}) ===", flush=True)
  print(f"Weights: {[round(x, 3) for x in w]}", flush=True)
  for name in HEAD_NAMES:
    print(f"  {name:22s} acc={acc['heads'][name]:.4f}", flush=True)
  print(f"  {'Fused':22s} acc={acc['fused']:.4f}", flush=True)
  print(f"  {'Oracle':22s} acc={acc['oracle_any_head']:.4f}", flush=True)
  print(f"  fusion gain vs best head: {acc['fusion_gain_vs_best_head']:+.4f}", flush=True)
  print(f"\nWrote {json_path}", flush=True)
  print(f"Wrote {md_path}", flush=True)
  print(f"Wrote figures under {out_dir}", flush=True)


if __name__ == "__main__":
  main()
