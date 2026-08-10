"""Shared UQ helpers: risk-coverage and selective prediction metrics."""

from __future__ import annotations

import numpy as np


def acc_at_coverage(
  pe: np.ndarray,
  correct: np.ndarray,
  coverage: float,
) -> float:
  """Accuracy on the lowest-PE (most confident) fraction of samples."""
  pe = np.asarray(pe, dtype=np.float64)
  correct = np.asarray(correct, dtype=np.float64)
  n = len(pe)
  if n == 0:
    return float("nan")
  k = max(1, int(round(n * coverage)))
  order = np.argsort(pe, kind="mergesort")
  keep = order[:k]
  return float(correct[keep].mean())


def aurc(pe: np.ndarray, correct: np.ndarray) -> float:
  """Area under risk-coverage curve (lower is better).

  Risk = error rate on the retained (low-PE) set at each coverage level.
  """
  pe = np.asarray(pe, dtype=np.float64)
  correct = np.asarray(correct, dtype=np.float64)
  n = len(pe)
  if n == 0:
    return float("nan")
  order = np.argsort(pe, kind="mergesort")
  sorted_corr = correct[order]
  coverages = np.arange(1, n + 1, dtype=np.float64) / n
  risks = 1.0 - np.cumsum(sorted_corr) / np.arange(1, n + 1, dtype=np.float64)
  try:
    return float(np.trapezoid(risks, coverages))
  except AttributeError:
    return float(np.trapz(risks, coverages))


def selective_metrics(
  pe: np.ndarray,
  correct: np.ndarray,
  coverages: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9),
) -> dict[str, float]:
  out: dict[str, float] = {
    "aurc": aurc(pe, correct),
    "acc_full": float(np.mean(correct)),
  }
  for cov in coverages:
    key = f"acc_at_{int(cov * 100)}pct"
    out[key] = acc_at_coverage(pe, correct, cov)
  return out
