"""Plot utilities for training curves (initial version).

This minimal script reads metrics.jsonl produced by train.py and saves a
single plot for training/validation loss. A follow-up commit will add ROUGE plots.

Usage:
    python plots.py --run_dir runs/project13 --out_dir runs/project13/figures
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_loss(rows: List[Dict], out_dir: Path) -> None:
    """Save a simple train/val loss curve to loss.png."""
    if not rows:
        print("No rows found in metrics.jsonl; skipping plots.")
        return

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
    train_loss = [r.get("train_avg_loss") for r in rows]
    val_loss = [r.get("val_loss") for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_avg_loss", marker="o")
    plt.plot(epochs, val_loss, label="val_loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Directory containing metrics.jsonl")
    ap.add_argument("--out_dir", type=str, default=None, help="Directory to save figures (default: <run_dir>/figures)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
    _ensure_out(out_dir)

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    plot_loss(rows, out_dir)
    print(f"Saved figures to: {out_dir}")
