"""Plot utilities for training curves.

Reads a metrics.jsonl produced by train.py and saves plots for:
- training/validation loss
- ROUGE metrics (rouge1, rouge2, rougeL, rougeLsum)

Usage:
    python plots.py --run_dir runs/<project> --out_dir runs/<project>/figures
Examples:
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

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    train_loss = [float(r.get("train_avg_loss", float("nan"))) for r in rows]
    val_loss = [float(r.get("val_loss", float("nan"))) for r in rows]

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


def plot_rouge(rows: List[Dict], out_dir: Path) -> None:
    """Save ROUGE plots: rougeL, and combined rouge1/rouge2/rougeLsum."""
    if not rows:
        return
    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    rougeL = [float(r.get("rougeL", float("nan"))) for r in rows]
    rouge1 = [float(r.get("rouge1", float("nan"))) for r in rows]
    rouge2 = [float(r.get("rouge2", float("nan"))) for r in rows]
    rougeLsum = [float(r.get("rougeLsum", float("nan"))) for r in rows]

    if any(v is not None for v in rougeL):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rougeL, label="rougeL", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("ROUGE-L")
        plt.title("ROUGE-L over epochs")
        plt.tight_layout()
        plt.savefig(out_dir / "rougeL.png", dpi=150)
        plt.close()

    if any(v is not None for v in (rouge1 + rouge2 + rougeLsum)):
        plt.figure(figsize=(6, 4))
        if any(v is not None for v in rouge1):
            plt.plot(epochs, rouge1, label="rouge1", marker="o")
        if any(v is not None for v in rouge2):
            plt.plot(epochs, rouge2, label="rouge2", marker="o")
        if any(v is not None for v in rougeLsum):
            plt.plot(epochs, rougeLsum, label="rougeLsum", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("ROUGE")
        plt.title("ROUGE metrics over epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rouge_all.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate training curves (loss and ROUGE) from a run directory.")
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory containing metrics.jsonl")
    ap.add_argument("--out_dir", type=str, default=None, help="Directory to save figures (default: <run_dir>/figures)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
    _ensure_out(out_dir)

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    plot_loss(rows, out_dir)
    plot_rouge(rows, out_dir)
    print(f"Saved figures to: {out_dir}")
