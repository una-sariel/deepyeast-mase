#!/usr/bin/env python3
"""Export MaSE interpretability figures: GFP + mask overlay + pred/label.

Samples up to N test images (stratified by class), saves PNGs under:
  <out-dir>/<class_name>/*.png

Example:
  python pytorch/visualize_mase_interpret.py \\
    --data-dir /path/to/deepyeast_full \\
    --checkpoint /path/to/checkpoints/mase_lite_full_v6_phase1/best.pt \\
    --n-samples 100 --out-dir /path/to/interpret_v6_p1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import DeepYeastTorchDataset, load_metadata
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict


def load_class_names(data_dir: Path) -> dict[int, str]:
  path = data_dir / "class_map.json"
  if not path.exists():
    return {i: f"class_{i}" for i in range(12)}
  with path.open() as f:
    data = json.load(f)
  raw = data.get("index_to_name") or data
  return {int(k): str(v) for k, v in raw.items()}


def sanitize_name(name: str) -> str:
  return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def stratified_indices(
  labels: np.ndarray,
  n_total: int,
  seed: int,
) -> np.ndarray:
  rng = np.random.default_rng(seed)
  n_total = min(n_total, len(labels))
  classes = np.unique(labels)
  per_class = max(1, n_total // len(classes))
  picked: list[int] = []
  for c in classes:
    idx = np.where(labels == c)[0]
    k = min(per_class, len(idx))
    choice = rng.choice(idx, size=k, replace=False)
    picked.extend(choice.tolist())
  if len(picked) < n_total:
    rest = np.setdiff1d(np.arange(len(labels)), picked, assume_unique=True)
    extra = rng.choice(rest, size=min(n_total - len(picked), len(rest)), replace=False)
    picked.extend(extra.tolist())
  return np.array(picked[:n_total], dtype=np.int64)


def tensor_to_display(ch: torch.Tensor) -> np.ndarray:
  """Single channel NCHW slice in [-1,1] → [0,1] H×W."""
  x = ch.detach().cpu().float().numpy()
  return np.clip(x * 0.5 + 0.5, 0.0, 1.0)


def save_panel(
  out_path: Path,
  *,
  mcherry: np.ndarray,
  gfp: np.ndarray,
  mask: np.ndarray,
  title: str,
) -> None:
  fig, axes = plt.subplots(1, 3, figsize=(9, 3))
  axes[0].imshow(mcherry, cmap="magma")
  axes[0].set_title("mCherry")
  axes[1].imshow(gfp, cmap="Greens")
  axes[1].set_title("GFP")
  axes[2].imshow(gfp, cmap="gray")
  axes[2].imshow(mask, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
  axes[2].set_title("GFP + mask")
  fig.suptitle(title, fontsize=10)
  for ax in axes:
    ax.axis("off")
  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=120, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description="MaSE mask / pred interpretability export")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--split", choices=["test", "val"], default="test")
  parser.add_argument("--n-samples", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  args = parser.parse_args()

  device = resolve_device(args.device)
  data_dir = args.data_dir.resolve()
  ckpt = args.checkpoint.resolve()
  results_json = args.results_json or (ckpt.parent / "results.json")
  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )

  class_names = load_class_names(data_dir)
  meta = load_metadata(data_dir)
  ds = DeepYeastTorchDataset(meta, args.split, augment=False)
  labels = ds.rows["label"].to_numpy(dtype=np.int64)
  indices = stratified_indices(labels, args.n_samples, args.seed)

  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  model.eval()

  loader = DataLoader(
    torch.utils.data.Subset(ds, indices.tolist()),
    batch_size=1,
    shuffle=False,
  )

  manifest: list[dict] = []
  for local_i, (images, label_t) in enumerate(loader):
    global_i = int(indices[local_i])
    images = images.to(device)
    label = int(label_t.item())
    with torch.inference_mode():
      log_probs, details = model(images, train=False, return_details=True)
    pred = int(log_probs.argmax(dim=-1).item())
    mask = details["mask"][0, 0].cpu().numpy()
    mch = tensor_to_display(images[0, 0])
    gfp = tensor_to_display(images[0, 1])
    true_name = class_names.get(label, f"class_{label}")
    pred_name = class_names.get(pred, f"class_{pred}")
    ok = pred == label
    fname = (
      f"{global_i:05d}_true-{sanitize_name(true_name)}"
      f"_pred-{sanitize_name(pred_name)}_{'ok' if ok else 'err'}.png"
    )
    out_path = args.out_dir / sanitize_name(true_name) / fname
    title = f"true={true_name}  pred={pred_name}  {'OK' if ok else 'ERR'}"
    save_panel(out_path, mcherry=mch, gfp=gfp, mask=mask, title=title)
    manifest.append(
      {
        "index": global_i,
        "true_label": label,
        "true_name": true_name,
        "pred_label": pred,
        "pred_name": pred_name,
        "correct": ok,
        "path": str(out_path),
      }
    )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  with (args.out_dir / "manifest.json").open("w") as f:
    json.dump(
      {
        "checkpoint": str(ckpt),
        "data_dir": str(data_dir),
        "split": args.split,
        "n_samples": len(manifest),
        "samples": manifest,
      },
      f,
      indent=2,
    )
  print(f"Wrote {len(manifest)} figures under {args.out_dir}", flush=True)
  print(f"Manifest: {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
  main()
