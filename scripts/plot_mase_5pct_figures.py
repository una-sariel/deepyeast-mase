#!/usr/bin/env python3
"""Plot 5% MaSE accuracy / weight curves and optional PE histograms.

Examples:
  python scripts/plot_mase_5pct_figures.py \\
    --results /path/to/mase_lite_5pct_v4_default/results.json \\
    --out-dir /path/to/figs

  python scripts/plot_mase_5pct_figures.py \\
    --results .../v5/results.json \\
    --uq-csv .../v5/uq_mc_dropout/per_image_test.csv \\
    --out-dir .../figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
  with path.open() as f:
    return json.load(f)


def plot_acc(history: list[dict], out: Path, title: str) -> None:
  epochs = [h["epoch"] for h in history]
  train = [h["train"]["accuracy"] for h in history]
  val = [h["val"]["accuracy"] for h in history]
  fig, ax = plt.subplots(figsize=(6.2, 3.6))
  ax.plot(epochs, train, label="train", linewidth=1.8)
  ax.plot(epochs, val, label="val", linewidth=1.8)
  ax.set_xlabel("epoch")
  ax.set_ylabel("accuracy")
  ax.set_title(title)
  ax.legend(frameon=False)
  ax.set_ylim(0, 1)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  plt.close(fig)


def plot_weights(history: list[dict], out: Path, title: str) -> None:
  epochs = [h["epoch"] for h in history]
  ws = np.array([h["train"].get("ensemble_weights") or h["val"].get("ensemble_weights") for h in history], dtype=float)
  if ws.ndim != 2 or ws.shape[1] != 4:
    # fallback: last known
    return
  fig, ax = plt.subplots(figsize=(6.2, 3.6))
  for i in range(4):
    ax.plot(epochs, ws[:, i], label=f"w{i}", linewidth=1.6)
  ax.axhline(0.25, color="0.5", linestyle="--", linewidth=1, label="uniform")
  ax.set_xlabel("epoch")
  ax.set_ylabel("ensemble weight")
  ax.set_title(title)
  ax.set_ylim(0, 1)
  ax.legend(frameon=False, ncol=3)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  plt.close(fig)


def plot_pe(csv_path: Path, out: Path, title: str) -> None:
  import csv

  pe_ok: list[float] = []
  pe_bad: list[float] = []
  with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      pe = float(row.get("pe") or row.get("predictive_entropy") or row["PE"])
      correct = row.get("correct") or row.get("is_correct")
      if correct is None and "y_true" in row and "y_pred" in row:
        ok = int(row["y_true"]) == int(row["y_pred"])
      else:
        ok = str(correct).lower() in {"1", "true", "yes"}
      (pe_ok if ok else pe_bad).append(pe)
  fig, ax = plt.subplots(figsize=(6.2, 3.6))
  bins = np.linspace(0, max(pe_ok + pe_bad + [1e-6]), 30)
  ax.hist(pe_ok, bins=bins, alpha=0.55, label=f"correct (n={len(pe_ok)})", density=True)
  ax.hist(pe_bad, bins=bins, alpha=0.55, label=f"incorrect (n={len(pe_bad)})", density=True)
  ax.set_xlabel("predictive entropy (PE)")
  ax.set_ylabel("density")
  ax.set_title(title)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  plt.close(fig)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--results", type=Path, required=True)
  p.add_argument("--out-dir", type=Path, required=True)
  p.add_argument("--uq-csv", type=Path, default=None)
  p.add_argument("--tag", type=str, default=None)
  args = p.parse_args()

  data = load_results(args.results)
  tag = args.tag or data.get("method") or args.results.parent.name
  out = args.out_dir
  out.mkdir(parents=True, exist_ok=True)
  history = data.get("history") or []
  if history:
    plot_acc(history, out / f"{tag}_acc.png", f"{tag} accuracy")
    plot_weights(history, out / f"{tag}_weights.png", f"{tag} ensemble weights")
  test = data.get("test") or {}
  print(f"{tag}: test_acc={test.get('accuracy')} weights={test.get('ensemble_weights')}")
  if args.uq_csv and args.uq_csv.exists():
    plot_pe(args.uq_csv, out / f"{tag}_pe.png", f"{tag} PE (test)")
  print(f"wrote figures under {out}")


if __name__ == "__main__":
  main()
