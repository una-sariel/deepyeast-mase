"""Train official DeepYeastNet (Pärnamaa & Parts) on a local DeepYeast split.

Same data loaders as MaSE. Produces best.pt + results.json for eval_deepyeast_uq.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from dataset import make_loaders
from deepyeast_net import DeepYeastNet


def set_seed(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
  model: DeepYeastNet,
  loader: DataLoader,
  device: torch.device,
) -> float:
  model.eval()
  correct = 0
  total = 0
  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images, train=False)
    pred = logits.argmax(dim=-1)
    correct += int((pred == labels).sum().item())
    total += int(labels.numel())
  return correct / max(total, 1)


def main() -> None:
  parser = argparse.ArgumentParser(description="Train DeepYeastNet baseline")
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--label-smoothing", type=float, default=0.0)
  parser.add_argument("--patience", type=int, default=12)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--num-classes", type=int, default=12)
  args = parser.parse_args()

  set_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = args.data_dir.resolve()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  train_loader, val_loader, test_loader = make_loaders(
    data_dir,
    batch_size=args.batch_size,
    augment=True,
    strong_augment=True,
    num_workers=args.num_workers,
  )

  model = DeepYeastNet(
    num_classes=args.num_classes, dropout_rate=args.dropout
  ).to(device)
  opt = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )

  history: list[dict] = []
  best_val = -1.0
  best_epoch = -1
  stale = 0
  t0 = time.time()

  for epoch in range(1, args.epochs + 1):
    model.train()
    loss_sum = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
    for images, labels in pbar:
      images = images.to(device)
      labels = labels.to(device)
      opt.zero_grad(set_to_none=True)
      logits = model(images, train=True)
      loss = F.cross_entropy(
        logits, labels, label_smoothing=args.label_smoothing
      )
      loss.backward()
      opt.step()
      loss_sum += float(loss.item())
      n_batches += 1
      pbar.set_postfix(loss=f"{loss.item():.3f}")

    train_loss = loss_sum / max(n_batches, 1)
    val_acc = evaluate(model, val_loader, device)
    row = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
    history.append(row)
    print(
      f"epoch {epoch:03d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}",
      flush=True,
    )

    if val_acc > best_val + 1e-6:
      best_val = val_acc
      best_epoch = epoch
      stale = 0
      torch.save(model.state_dict(), out_dir / "best.pt")
    else:
      stale += 1
      if stale >= args.patience:
        print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
        break

  # Reload best and test
  state = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
  model.load_state_dict(state)
  test_acc = evaluate(model, test_loader, device)
  elapsed = time.time() - t0

  results = {
    "method": "deepyeast_net_baseline",
    "paper": "Pärnamaa & Parts 2017 (official DeepYeastNet)",
    "data_dir": str(data_dir),
    "seed": args.seed,
    "device": str(device),
    "best_epoch": best_epoch,
    "best_val_acc": best_val,
    "test_acc": test_acc,
    "elapsed_sec": elapsed,
    "config": {
      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "lr": args.lr,
      "weight_decay": args.weight_decay,
      "dropout": args.dropout,
      "label_smoothing": args.label_smoothing,
      "patience": args.patience,
      "num_classes": args.num_classes,
    },
    "history": history,
  }
  with (out_dir / "results.json").open("w") as f:
    json.dump(results, f, indent=2)
  print(
    f"\nDone. best_val={best_val:.4f} @ epoch {best_epoch}  "
    f"test={test_acc:.4f}\nWrote {out_dir / 'best.pt'}",
    flush=True,
  )


if __name__ == "__main__":
  main()
