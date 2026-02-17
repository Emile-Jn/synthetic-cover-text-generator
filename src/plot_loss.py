"""Plot loss vs epoch from test.json and save figure.

Generates two series on the same figure:
 - raw points (scatter) for each logged step
 - per-epoch mean loss (line)

Saves result to ./plots/loss_vs_epoch.png
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import statistics

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = ROOT / "test.json"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "loss_vs_epoch.png"


def load_entries(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting a list of dicts with keys 'epoch' and 'loss'
    entries = []
    for i, item in enumerate(data):
        try:
            epoch = float(item.get("epoch"))
            loss = float(item.get("loss"))
            entries.append((epoch, loss))
        except Exception:
            # skip malformed entries but warn
            print(f"skipping entry {i}: {item}")
    return entries


def aggregate_by_epoch(entries):
    by_epoch = defaultdict(list)
    for epoch, loss in entries:
        by_epoch[epoch].append(loss)
    epochs = sorted(by_epoch.keys())
    means = [statistics.mean(by_epoch[e]) for e in epochs]
    medians = [statistics.median(by_epoch[e]) for e in epochs]
    counts = [len(by_epoch[e]) for e in epochs]
    return epochs, means, medians, counts


def plot(entries, out_path: Path):
    if not entries:
        raise SystemExit("no valid entries found in test.json")
    # raw scatter
    epochs_raw = [e for e, _ in entries]
    losses_raw = [l for _, l in entries]

    epochs, means, medians, counts = aggregate_by_epoch(entries)

    plt.figure(figsize=(10, 6))
    plt.scatter(epochs_raw, losses_raw, alpha=0.4, s=20, label="raw")
    plt.plot(epochs, means, color="red", marker="o", linewidth=2, label="mean per-epoch")
    plt.plot(epochs, medians, color="orange", marker="x", linewidth=1, linestyle="--", label="median per-epoch")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss vs Epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    entries = load_entries(TEST_JSON)
    plot(entries, OUT_PNG)

