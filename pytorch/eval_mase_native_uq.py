"""MaSE-native UQ: mask entropy + head disagreement + selective deferral.

Compared to vanilla MC-Dropout PE (DeepYeastNet / MaSE), this script extracts
signals that only MaSE has, then evaluates:

  - UAUC  = AUROC(incorrect, uncertainty)   [threshold-free ranking]
  - AURC  = area under risk-coverage curve  [selective prediction]
  - Acc@coverage / Risk@coverage            [same-coverage comparison]

Example:
  python pytorch/eval_mase_native_uq.py \\
    --data-dir C:/Users/unaliuqw/deepyeast_10pct \\
    --mase-checkpoint .../mase_lite_10pct_confident/best.pt \\
    --baseline-uq-dir .../deepyeast_baseline_10pct/uq_mc_dropout \\
    --out-dir .../mase_lite_10pct_confident/uq_native
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from eval_mase_uq import (
  config_from_results_json,
  load_checkpoint,
  roc_auc_score,
  set_seed,
)
from mase_lite_net import (
  MASE_LITE_DEFAULT,
  MaSELiteConfig,
  MaSELiteNet,
  config_to_dict,
  predictive_entropy,
)


def _safe_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  probs = probs.clamp_min(eps)
  probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
  return -(probs * probs.log()).sum(dim=-1)


def mask_entropy(patch_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """Entropy of normalized patch selection probs — diffuse mask → higher U."""
  flat = patch_probs.flatten(1).clamp_min(eps)
  return _safe_entropy(flat, eps=eps)


def head_disagreement(branch_logits: list[torch.Tensor]) -> dict[str, torch.Tensor]:
  """Ensemble-style epistemic signals from the 4 MaSE heads."""
  stacked = torch.stack(
    [F.softmax(logits, dim=-1) for logits in branch_logits], dim=1
  )  # (B, 4, C)
  mean_p = stacked.mean(dim=1)
  pe_mean = predictive_entropy(mean_p)
  pe_heads = predictive_entropy(stacked)  # (B, 4)
  # BALD / mutual information proxy
  head_mi = pe_mean - pe_heads.mean(dim=1)
  # Total variation of head probs
  head_var = stacked.var(dim=1, unbiased=False).sum(dim=-1)
  # Fraction of heads disagreeing with argmax of mean
  pred = mean_p.argmax(dim=-1)
  head_preds = stacked.argmax(dim=-1)
  disagree_frac = (head_preds != pred.unsqueeze(1)).float().mean(dim=1)
  return {
    "head_mi": head_mi,
    "head_var": head_var,
    "head_disagree_frac": disagree_frac,
    "ensemble_pe": pe_mean,
  }


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
  mu = float(np.mean(x))
  sd = float(np.std(x))
  return (x - mu) / (sd + eps)


def risk_coverage_curve(
  correct: np.ndarray,
  uncertainty: np.ndarray,
  n_points: int = 51,
) -> dict[str, Any]:
  """Keep lowest-uncertainty fraction = coverage; risk = error rate on kept set."""
  correct = np.asarray(correct).astype(np.int32)
  u = np.asarray(uncertainty).astype(np.float64)
  n = len(correct)
  order = np.argsort(u, kind="mergesort")  # most certain first
  correct_sorted = correct[order]
  coverages = np.linspace(1.0 / n, 1.0, n_points)
  risks: list[float] = []
  accs: list[float] = []
  for c in coverages:
    k = max(1, int(round(c * n)))
    kept = correct_sorted[:k]
    acc = float(kept.mean())
    accs.append(acc)
    risks.append(1.0 - acc)
  coverages_f = coverages.astype(float)
  risks_a = np.asarray(risks, dtype=float)
  try:
    aurc = float(np.trapezoid(risks_a, coverages_f))
  except AttributeError:
    aurc = float(np.trapz(risks_a, coverages_f))
  # Full-coverage risk (no abstention)
  full_risk = float(1.0 - correct.mean())
  return {
    "coverage": coverages_f.tolist(),
    "risk": risks,
    "accuracy": accs,
    "AURC": aurc,
    "full_risk": full_risk,
    "acc_at_coverage": {
      "0.50": accs[int(0.50 * (n_points - 1))],
      "0.70": accs[int(0.70 * (n_points - 1))],
      "0.80": accs[int(0.80 * (n_points - 1))],
      "0.90": accs[int(0.90 * (n_points - 1))],
      "1.00": accs[-1],
    },
    "risk_at_coverage": {
      "0.50": risks[int(0.50 * (n_points - 1))],
      "0.70": risks[int(0.70 * (n_points - 1))],
      "0.80": risks[int(0.80 * (n_points - 1))],
      "0.90": risks[int(0.90 * (n_points - 1))],
      "1.00": risks[-1],
    },
  }


def score_metrics(correct: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
  incorrect = 1 - correct.astype(np.int32)
  uauc = roc_auc_score(incorrect, uncertainty)
  rc = risk_coverage_curve(correct, uncertainty)
  return {
    "UAUC": uauc,
    "AURC": rc["AURC"],
    "full_accuracy": float(correct.mean()),
    "acc_at_coverage": rc["acc_at_coverage"],
    "risk_at_coverage": rc["risk_at_coverage"],
    "risk_coverage": {
      "coverage": rc["coverage"],
      "risk": rc["risk"],
      "accuracy": rc["accuracy"],
    },
  }


def fuse_scores_val_tuned(
  val_scores: dict[str, np.ndarray],
  val_correct: np.ndarray,
  test_scores: dict[str, np.ndarray],
  keys: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
  """Equal z-score average; report per-key val UAUC for transparency."""
  val_z = [zscore(val_scores[k]) for k in keys]
  test_z = [zscore(test_scores[k]) for k in keys]
  # Use val mean/std for test z-score (no leakage of test stats)
  test_z_aligned = []
  for k in keys:
    mu = float(np.mean(val_scores[k]))
    sd = float(np.std(val_scores[k])) + 1e-8
    test_z_aligned.append((test_scores[k] - mu) / sd)
  val_fused = np.mean(np.stack(val_z, axis=0), axis=0)
  test_fused = np.mean(np.stack(test_z_aligned, axis=0), axis=0)
  per_key_uauc = {
    k: roc_auc_score(1 - val_correct.astype(np.int32), val_scores[k]) for k in keys
  }
  return val_fused, test_fused, per_key_uauc


def mase_native_predict(
  model: MaSELiteNet,
  loader: DataLoader,
  device: torch.device,
) -> dict[str, np.ndarray]:
  """Single deterministic forward; Dropout/BN in eval (no MC)."""
  model.eval()
  buckets: dict[str, list[np.ndarray]] = {
    "y_true": [],
    "y_pred": [],
    "pe": [],
    "max_prob": [],
    "u_1mmax": [],
    "mask_ent": [],
    "head_mi": [],
    "head_var": [],
    "head_disagree_frac": [],
  }

  with torch.inference_mode():
    for images, labels in loader:
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      log_probs, details = model(images, train=False, return_details=True)
      assert isinstance(log_probs, torch.Tensor)
      probs = log_probs.exp()
      pe = predictive_entropy(probs)
      max_p = probs.max(dim=-1).values
      y_pred = probs.argmax(dim=-1)
      m_ent = mask_entropy(details["patch_probs"])
      hd = head_disagreement(details["branch_logits"])

      buckets["y_true"].append(labels.cpu().numpy())
      buckets["y_pred"].append(y_pred.cpu().numpy())
      buckets["pe"].append(pe.cpu().numpy())
      buckets["max_prob"].append(max_p.cpu().numpy())
      buckets["u_1mmax"].append((1.0 - max_p).cpu().numpy())
      buckets["mask_ent"].append(m_ent.cpu().numpy())
      buckets["head_mi"].append(hd["head_mi"].cpu().numpy())
      buckets["head_var"].append(hd["head_var"].cpu().numpy())
      buckets["head_disagree_frac"].append(hd["head_disagree_frac"].cpu().numpy())

  out = {k: np.concatenate(v) for k, v in buckets.items()}
  out["correct"] = (out["y_true"] == out["y_pred"]).astype(np.int32)
  return out


def load_baseline_csv(uq_dir: Path, split: str) -> dict[str, np.ndarray]:
  path = uq_dir / f"per_image_{split}.csv"
  if not path.exists():
    raise FileNotFoundError(path)
  y_true: list[int] = []
  y_pred: list[int] = []
  correct: list[int] = []
  pe: list[float] = []
  max_prob: list[float] = []
  with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      y_true.append(int(row["y_true"]))
      y_pred.append(int(row["y_pred"]))
      correct.append(int(row["correct"]))
      pe.append(float(row["pe"]))
      max_prob.append(float(row["max_prob"]))
  return {
    "y_true": np.asarray(y_true),
    "y_pred": np.asarray(y_pred),
    "correct": np.asarray(correct, dtype=np.int32),
    "pe": np.asarray(pe, dtype=np.float64),
    "u_1mmax": 1.0 - np.asarray(max_prob, dtype=np.float64),
  }


def write_scores_csv(path: Path, arrays: dict[str, np.ndarray], score_keys: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fields = ["index", "y_true", "y_pred", "correct"] + score_keys
  with path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(fields)
    n = len(arrays["y_true"])
    for i in range(n):
      row = [
        i,
        int(arrays["y_true"][i]),
        int(arrays["y_pred"][i]),
        int(arrays["correct"][i]),
      ]
      row.extend(float(arrays[k][i]) for k in score_keys)
      w.writerow(row)


def summarize_model_scores(
  arrays: dict[str, np.ndarray],
  score_keys: list[str],
) -> dict[str, Any]:
  out: dict[str, Any] = {
    "n": int(len(arrays["correct"])),
    "accuracy": float(arrays["correct"].mean()),
    "scores": {},
  }
  for k in score_keys:
    out["scores"][k] = score_metrics(arrays["correct"], arrays[k])
  return out


def main() -> None:
  parser = argparse.ArgumentParser(
    description="MaSE-native UQ (mask / heads) vs DeepYeastNet PE"
  )
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--mase-checkpoint", type=Path, required=True)
  parser.add_argument(
    "--baseline-uq-dir",
    type=Path,
    required=True,
    help="Dir with per_image_{val,test}.csv from eval_deepyeast_uq.py",
  )
  parser.add_argument("--results-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--split", choices=["test", "val", "both"], default="both")
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = args.mase_checkpoint.resolve()
  results_json = args.results_json
  if results_json is None:
    cand = ckpt.parent / "results.json"
    results_json = cand if cand.exists() else None

  out_dir = (
    args.out_dir.resolve()
    if args.out_dir is not None
    else (ckpt.parent / "uq_native")
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  config = config_from_results_json(results_json) or MaSELiteConfig(
    **config_to_dict(MASE_LITE_DEFAULT)
  )
  model = MaSELiteNet(config).to(device)
  load_checkpoint(model, ckpt)
  print(f"Loaded MaSE {ckpt}", flush=True)
  print(f"Device={device}  (deterministic native UQ, no MC)", flush=True)

  _, val_loader, test_loader = make_loaders(
    args.data_dir.resolve(),
    batch_size=args.batch_size,
    augment=False,
    strong_augment=False,
    num_workers=args.num_workers,
  )

  loaders: dict[str, DataLoader] = {}
  if args.split in ("val", "both"):
    loaders["val"] = val_loader
  if args.split in ("test", "both"):
    loaders["test"] = test_loader

  mase_arrays: dict[str, dict[str, np.ndarray]] = {}
  score_keys = [
    "pe",
    "u_1mmax",
    "mask_ent",
    "head_mi",
    "head_var",
    "head_disagree_frac",
  ]

  for split_name, loader in loaders.items():
    print(f"\n=== MaSE native UQ on {split_name} ===", flush=True)
    arr = mase_native_predict(model, loader, device)
    mase_arrays[split_name] = arr
    write_scores_csv(out_dir / f"mase_scores_{split_name}.csv", arr, score_keys)
    print(
      f"  n={len(arr['correct'])}  acc={arr['correct'].mean():.4f}",
      flush=True,
    )

  # Fuse MaSE-native signals (val-normalized z-average).
  # mask_ent alone is weak on this data (~chance UAUC); keep it reported but
  # fuse PE + head disagreement (MaSE-only epistemic signal).
  fuse_keys = ["pe", "head_mi"]
  if "val" in mase_arrays and "test" in mase_arrays:
    val_f, test_f, per_key = fuse_scores_val_tuned(
      mase_arrays["val"],
      mase_arrays["val"]["correct"],
      mase_arrays["test"],
      fuse_keys,
    )
    mase_arrays["val"]["fused_z"] = val_f
    mase_arrays["test"]["fused_z"] = test_f
    score_keys = score_keys + ["fused_z"]
    print(f"\nFused keys={fuse_keys}  val UAUC per key: {per_key}", flush=True)
  elif "val" in mase_arrays:
    mase_arrays["val"]["fused_z"] = np.mean(
      np.stack([zscore(mase_arrays["val"][k]) for k in fuse_keys], axis=0),
      axis=0,
    )
    score_keys = score_keys + ["fused_z"]

  summary: dict[str, Any] = {
    "method": "mase_native_uq",
    "mase_checkpoint": str(ckpt),
    "baseline_uq_dir": str(args.baseline_uq_dir.resolve()),
    "data_dir": str(args.data_dir.resolve()),
    "device": str(device),
    "fuse_keys": fuse_keys,
    "note": (
      "Native scores from one deterministic forward. "
      "mask_ent = entropy of patch_probs; head_mi = PE(mean heads)-mean PE(heads); "
      "fused_z = mean of val-z-scored (pe, mask_ent, head_mi). "
      "Baseline uses existing MC-Dropout PE CSVs. "
      "Primary claims: UAUC (ranking) and AURC / Acc@coverage (selective deferral)."
    ),
    "mase": {},
    "baseline": {},
    "comparison": {},
  }

  for split_name, arr in mase_arrays.items():
    keys_here = [k for k in score_keys if k in arr]
    summary["mase"][split_name] = summarize_model_scores(arr, keys_here)
    best_key = max(
      keys_here,
      key=lambda k: summary["mase"][split_name]["scores"][k]["UAUC"]
      if summary["mase"][split_name]["scores"][k]["UAUC"]
      == summary["mase"][split_name]["scores"][k]["UAUC"]
      else -1.0,
    )
    summary["mase"][split_name]["best_uauc_score"] = best_key
    s = summary["mase"][split_name]["scores"]
    print(
      f"\n[MaSE {split_name}] best UAUC score={best_key} "
      f"UAUC={s[best_key]['UAUC']:.4f} AURC={s[best_key]['AURC']:.4f}",
      flush=True,
    )
    for k in keys_here:
      print(
        f"  {k:22s}  UAUC={s[k]['UAUC']:.4f}  AURC={s[k]['AURC']:.4f}  "
        f"acc@0.8={s[k]['acc_at_coverage']['0.80']:.4f}",
        flush=True,
      )

  # Baseline from precomputed CSVs
  for split_name in loaders:
    base = load_baseline_csv(args.baseline_uq_dir.resolve(), split_name)
    base_keys = ["pe", "u_1mmax"]
    summary["baseline"][split_name] = summarize_model_scores(base, base_keys)
    bs = summary["baseline"][split_name]["scores"]
    print(
      f"\n[Baseline {split_name}] PE  UAUC={bs['pe']['UAUC']:.4f}  "
      f"AURC={bs['pe']['AURC']:.4f}  acc@0.8={bs['pe']['acc_at_coverage']['0.80']:.4f}",
      flush=True,
    )

  # Head-to-head: MaSE fused/best vs baseline PE at same coverage
  for split_name in loaders:
    if split_name not in summary["mase"] or split_name not in summary["baseline"]:
      continue
    mase_scores = summary["mase"][split_name]["scores"]
    base_pe = summary["baseline"][split_name]["scores"]["pe"]
    # Primary = lowest AURC among pe / fused_z / head_mi (selective-prediction claim)
    candidates = [
      k
      for k in ("pe", "fused_z", "head_mi", "head_var")
      if k in mase_scores and mase_scores[k]["AURC"] == mase_scores[k]["AURC"]
    ]
    prefer = min(candidates, key=lambda k: mase_scores[k]["AURC"]) if candidates else (
      summary["mase"][split_name]["best_uauc_score"]
    )
    # Also always report head_mi and mask_ent
    cmp: dict[str, Any] = {
      "baseline_score": "pe",
      "mase_primary_score": prefer,
      "delta_UAUC": mase_scores[prefer]["UAUC"] - base_pe["UAUC"],
      "delta_AURC": mase_scores[prefer]["AURC"] - base_pe["AURC"],
      "delta_acc_at_coverage": {
        c: mase_scores[prefer]["acc_at_coverage"][c] - base_pe["acc_at_coverage"][c]
        for c in ("0.50", "0.70", "0.80", "0.90", "1.00")
      },
      "mase_full_acc": summary["mase"][split_name]["accuracy"],
      "baseline_full_acc": summary["baseline"][split_name]["accuracy"],
    }
    summary["comparison"][split_name] = cmp
    print(
      f"\n=== Comparison {split_name}: MaSE[{prefer}] vs Baseline[PE] ===",
      flush=True,
    )
    print(
      f"  dUAUC={cmp['delta_UAUC']:+.4f}  dAURC={cmp['delta_AURC']:+.4f} "
      f"(lower AURC better)",
      flush=True,
    )
    for c in ("0.70", "0.80", "0.90"):
      print(
        f"  Acc@{c}: MaSE={mase_scores[prefer]['acc_at_coverage'][c]:.4f}  "
        f"Base={base_pe['acc_at_coverage'][c]:.4f}  "
        f"d={cmp['delta_acc_at_coverage'][c]:+.4f}",
        flush=True,
      )

    # Also compare MaSE PE (strongest single score) for selective deferral claim
    if prefer != "pe" and "pe" in mase_scores:
      cmp_pe = {
        "delta_UAUC": mase_scores["pe"]["UAUC"] - base_pe["UAUC"],
        "delta_AURC": mase_scores["pe"]["AURC"] - base_pe["AURC"],
        "delta_acc_at_coverage": {
          c: mase_scores["pe"]["acc_at_coverage"][c] - base_pe["acc_at_coverage"][c]
          for c in ("0.50", "0.70", "0.80", "0.90", "1.00")
        },
      }
      summary["comparison"][split_name]["mase_pe_vs_baseline"] = cmp_pe
      print(
        f"  [MaSE pe vs Base pe] dUAUC={cmp_pe['delta_UAUC']:+.4f}  "
        f"dAURC={cmp_pe['delta_AURC']:+.4f}  "
        f"dAcc@0.8={cmp_pe['delta_acc_at_coverage']['0.80']:+.4f}",
        flush=True,
      )

  out_path = out_dir / "summary_native.json"
  with out_path.open("w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
  main()
