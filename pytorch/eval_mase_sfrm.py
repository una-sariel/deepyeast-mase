#!/usr/bin/env python3
"""SFRM multi-window vote eval for MaSE-Net Lite — no retraining.

Averages softmax over the full image plus N shared-size random occlusions.

Example:
  python pytorch/eval_mase_sfrm.py ^
    --data-dir C:\\Users\\unaliuqw\\deepyeast_5pct ^
    --checkpoint ...\\mase_lite_5pct_sfrm\\best.pt ^
    --sfrm-size 24 --sfrm-windows 7 --split both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from eval_mase_uq import config_from_results_json, load_checkpoint, resolve_device, set_seed
from mase_lite_net import MASE_LITE_DEFAULT, MaSELiteConfig, MaSELiteNet, config_to_dict
from sfrm import eval_sfrm_vote_split, eval_windows


def run_sfrm_vote_eval(
  *,
  data_dir: Path,
  checkpoint: Path,
  results_json: Path | None = None,
  split: str = "test",
  sfrm_size: int = 24,
  sfrm_windows: int = 7,
  batch_size: int = 64,
  device_pref: str = "auto",
  seed: int = 42,
  out_dir: Path | None = None,
) -> dict[str, Any]:
  set_seed(seed)
  device = resolve_device(device_pref)
  data_dir = data_dir.resolve()
  checkpoint = checkpoint.resolve()

  if results_json is None:
    candidate = checkpoint.parent / "results.json"
    results_json = candidate if candidate.exists() else None

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, checkpoint)

  _, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=batch_size,
    augment=False,
    strong_augment=False,
    num_workers=0,
  )
  loaders = {"val": val_loader, "test": test_loader}
  windows = eval_windows(sfrm_windows, config.image_size, sfrm_size, seed)

  splits: dict[str, Any] = {}
  for name in ("val", "test"):
    if split not in (name, "both"):
      continue
    splits[name] = eval_sfrm_vote_split(model, loaders[name], device, windows)

  payload: dict[str, Any] = {
    "method": "mase_sfrm_vote_eval",
    "checkpoint": str(checkpoint),
    "data_dir": str(data_dir),
    "sfrm_size": sfrm_size,
    "sfrm_windows": windows,
    "seed": seed,
    "splits": splits,
  }
  if splits.get("test"):
    payload["test_baseline_accuracy"] = splits["test"]["baseline_accuracy"]
    payload["test_sfrm_vote_accuracy"] = splits["test"]["sfrm_vote_accuracy"]
    payload["test_sfrm_vote_gain_pp"] = splits["test"]["sfrm_vote_gain_pp"]

  if out_dir is not None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
      json.dump(payload, f, indent=2)
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description="MaSE SFRM multi-window vote eval")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--split", choices=["test", "val", "both"], default="both")
  parser.add_argument("--sfrm-size", type=int, default=24)
  parser.add_argument("--sfrm-windows", type=int, default=7)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  out_dir = args.out_dir or args.checkpoint.resolve().parent / "sfrm_vote"
  payload = run_sfrm_vote_eval(
    data_dir=args.data_dir,
    checkpoint=args.checkpoint,
    results_json=args.results_json,
    split=args.split,
    sfrm_size=args.sfrm_size,
    sfrm_windows=args.sfrm_windows,
    batch_size=args.batch_size,
    device_pref=args.device,
    seed=args.seed,
    out_dir=out_dir,
  )
  print(f"SFRM vote saved: {out_dir / 'summary.json'}", flush=True)
  for name, split in payload["splits"].items():
    print(
      f"  {name}: baseline={split['baseline_accuracy']:.4f}  "
      f"vote={split['sfrm_vote_accuracy']:.4f}  "
      f"gain={split['sfrm_vote_gain_pp']:+.2f}pp",
      flush=True,
    )


if __name__ == "__main__":
  main()
