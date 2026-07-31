#!/usr/bin/env python3
"""Plot UAUC ROC curves from per_image_*.csv (PE ranking incorrect vs correct).

Example:
  python scripts/plot_uauc_roc.py \\
    --csv path/to/uq_mc_dropout/per_image_test.csv \\
    --label "MaSE v4 5%" \\
    --out figs/uauc_roc_v4.png

  python scripts/plot_uauc_roc.py \\
    --csv a/per_image_test.csv --label "v4" \\
    --csv b/per_image_test.csv --label "v5" \\
    --out figs/uauc_roc_compare.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> dict[str, np.ndarray]:
  pe, maxp, correct = [], [], []
  with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      pe.append(float(row["pe"]))
      if "max_prob" in row and row["max_prob"] != "":
        maxp.append(float(row["max_prob"]))
      correct.append(int(row["correct"]))
  out = {
    "pe": np.asarray(pe, dtype=np.float64),
    "correct": np.asarray(correct, dtype=np.int32),
  }
  if maxp:
    out["max_prob"] = np.asarray(maxp, dtype=np.float64)
  return out


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
  """y_true: 1 = positive (incorrect). Higher score → more positive."""
  y_true = np.asarray(y_true).astype(np.int32)
  y_score = np.asarray(y_score).astype(np.float64)
  n_pos = int((y_true == 1).sum())
  n_neg = int((y_true == 0).sum())
  if n_pos == 0 or n_neg == 0:
    return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
  order = np.argsort(-y_score, kind="mergesort")
  y_true = y_true[order]
  tps = np.cumsum(y_true == 1)
  fps = np.cumsum(y_true == 0)
  tpr = np.concatenate([[0.0], tps / n_pos])
  fpr = np.concatenate([[0.0], fps / n_neg])
  try:
    auc = float(np.trapezoid(tpr, fpr))
  except AttributeError:
    auc = float(np.trapz(tpr, fpr))
  return fpr, tpr, auc


def main() -> None:
  p = argparse.ArgumentParser(description="Plot UAUC ROC from per_image CSV")
  p.add_argument("--csv", type=Path, action="append", required=True, help="per_image_*.csv")
  p.add_argument("--label", type=str, action="append", default=None)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--title", type=str, default="UAUC ROC (PE ranks incorrect)")
  p.add_argument("--also-maxprob", action="store_true", help="Also plot 1-maxprob ROC")
  args = p.parse_args()

  labels = args.label or [f"run{i+1}" for i in range(len(args.csv))]
  if len(labels) != len(args.csv):
    raise SystemExit("--label count must match --csv count")

  fig, ax = plt.subplots(figsize=(5.5, 5.2))
  ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1, label="chance")

  for path, label in zip(args.csv, labels):
    data = load_csv(path)
    incorrect = 1 - data["correct"]
    fpr, tpr, auc = roc_curve(incorrect, data["pe"])
    ax.plot(fpr, tpr, linewidth=2.0, label=f"{label} PE  UAUC={auc:.3f}")
    if args.also_maxprob and "max_prob" in data:
      fpr2, tpr2, auc2 = roc_curve(incorrect, 1.0 - data["max_prob"])
      ax.plot(
        fpr2,
        tpr2,
        linewidth=1.6,
        linestyle=":",
        label=f"{label} 1−maxp  AUC={auc2:.3f}",
      )

  ax.set_xlabel("False positive rate\n(correct called uncertain)")
  ax.set_ylabel("True positive rate\n(incorrect called uncertain)")
  ax.set_title(args.title)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect("equal", adjustable="box")
  ax.legend(frameon=False, fontsize=8, loc="lower right")
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=180)
  plt.close(fig)
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
