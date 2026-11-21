"""Plot train/val loss curves from every log_*.csv written by gpt.py.

Usage:  python3 plot_losses.py      (needs: pip install matplotlib)
Output: loss_curves.png
"""
import csv
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
logs = sorted(glob.glob(os.path.join(HERE, "log_*.csv")))
if not logs:
    raise SystemExit("No log_*.csv found. Run gpt.py first.")

runs = []
plt.figure(figsize=(8, 5))
for path in logs:
    name = os.path.basename(path)[len("log_"):-len(".csv")]
    steps, train, val, bpc = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train"]))
            val.append(float(row["val"]))
            # older logs have no val_bpc column: those runs are character-level
            bpc.append(float(row["val_bpc"]) if "val_bpc" in row else float(row["val"]) / math.log(2))
    runs.append((name, steps, bpc))
    line, = plt.plot(steps, val, label=f"{name} val")
    plt.plot(steps, train, "--", color=line.get_color(), alpha=0.6, label=f"{name} train")

plt.axhline(2.47, color="gray", linestyle=":", label="bigram baseline (~2.47)")
plt.xlabel("training step")
plt.ylabel("cross-entropy loss")
plt.title("minigpt: train (dashed) vs val (solid) loss")
plt.ylim(0.5, 3.0)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(HERE, "loss_curves.png")
plt.savefig(out, dpi=150)
print("Saved", out)

# Second figure: validation bits-per-character, the fair way to compare different tokenizers
plt.figure(figsize=(8, 5))
for name, steps, bpc in runs:
    plt.plot(steps, bpc, label=f"{name} (best {min(bpc):.3f})")
plt.xlabel("training step")
plt.ylabel("validation bits per character (lower is better)")
plt.title("minigpt: validation bits per character")
plt.ylim(1.6, 3.5)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(HERE, "bpc_curves.png")
plt.savefig(out2, dpi=150)
print("Saved", out2)