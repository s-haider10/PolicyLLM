#!/usr/bin/env python3
"""Generate a chart of Policy Recall only, by method (from ACL_metrics_comparison.csv)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "ACL_metrics_comparison.csv"
OUT_PATH = SCRIPT_DIR / "ACL_recall_vs_metrics.png"


def load_data():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"].strip()
            recall = float(row["Policy Recall"])
            rows.append({"method": method, "recall": recall})
    return rows


def main():
    data = load_data()
    data = sorted(data, key=lambda x: x["recall"], reverse=True)

    methods = [d["method"] for d in data]
    recalls = [d["recall"] for d in data]
    n = len(methods)

    # One color per bar: highlight PolicyLLM (Ours), rest gradient
    colors = []
    for d in data:
        if "PolicyLLM" in d["method"] or "(Ours)" in d["method"]:
            colors.append("#C62828")
        else:
            colors.append("#5C6BC0")
    bar_edge = ["#8B0000" if c == "#C62828" else "#3F51B5" for c in colors]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, recalls, height=0.6, color=colors, edgecolor=bar_edge, linewidth=1.2)

    # Recall value labels on the bars
    for i, (bar, v) in enumerate(zip(bars, recalls)):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", fontsize=10, va="center", fontweight="bold", color="#212121")

    # Highlight PolicyLLM row
    try:
        ours_idx = next(i for i, d in enumerate(data) if "PolicyLLM" in d["method"] or "(Ours)" in d["method"])
        ax.axhspan(ours_idx - 0.52, ours_idx + 0.52, facecolor="#FFEBEE", zorder=0, alpha=0.8)
    except StopIteration:
        pass

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=10, color="#212121")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Policy Recall", fontsize=12, color="#212121")
    ax.set_title("Policy Recall by Method", fontsize=14, fontweight="bold", color="#212121", pad=12)
    ax.axvline(0, color="#CFD8DC", linewidth=0.8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CFD8DC", linestyle="-", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
