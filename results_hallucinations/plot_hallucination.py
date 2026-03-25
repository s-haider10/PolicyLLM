"""Publication-quality hallucination benchmark visualizations with CIs.

Produces:
  1. Grouped bar: Faithfulness + Q² + AIS  (higher is better)
  2. Hallucination rate bar               (lower is better)
  3. Maynez taxonomy: Intrinsic vs Extrinsic (lower is better)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "hallucination_metrics.csv"
OUT_FAITH = SCRIPT_DIR / "hallucination_faithfulness_comparison.png"
OUT_RATE  = SCRIPT_DIR / "hallucination_rate_comparison.png"
OUT_TAXONOMY = SCRIPT_DIR / "hallucination_taxonomy.png"


def _parse_ci(ci_str: str):
    """Parse '[0.123-0.456]' -> (0.123, 0.456)."""
    ci_str = ci_str.strip("[]")
    parts = ci_str.split("-", 1)
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return 0.0, 0.0


def load_data():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            faith_lo, faith_hi = _parse_ci(r.get("Faith CI", "[0-0]"))
            q2_lo, q2_hi = _parse_ci(r.get("Q² CI", "[0-0]"))
            halluc_lo, halluc_hi = _parse_ci(r.get("Halluc CI", "[0-0]"))
            intr_lo, intr_hi = _parse_ci(r.get("Intr CI", "[0-0]"))
            extr_lo, extr_hi = _parse_ci(r.get("Extr CI", "[0-0]"))
            ais_lo, ais_hi = _parse_ci(r.get("AIS CI", "[0-0]"))
            rows.append({
                "method": r["Method"],
                "faithfulness": float(r["Faithfulness"]),
                "faith_err": (float(r["Faithfulness"]) - faith_lo, faith_hi - float(r["Faithfulness"])),
                "q2": float(r["Q² Score"]),
                "q2_err": (float(r["Q² Score"]) - q2_lo, q2_hi - float(r["Q² Score"])),
                "halluc_rate": float(r["Halluc Rate"]),
                "halluc_err": (float(r["Halluc Rate"]) - halluc_lo, halluc_hi - float(r["Halluc Rate"])),
                "intrinsic": float(r["Intrinsic"]),
                "intr_err": (float(r["Intrinsic"]) - intr_lo, intr_hi - float(r["Intrinsic"])),
                "extrinsic": float(r["Extrinsic"]),
                "extr_err": (float(r["Extrinsic"]) - extr_lo, extr_hi - float(r["Extrinsic"])),
                "ais": float(r["AIS Score"]),
                "ais_err": (float(r["AIS Score"]) - ais_lo, ais_hi - float(r["AIS Score"])),
            })
    return rows


def _is_ours(m: str) -> bool:
    return "PolicyLLM" in m or "(Ours)" in m


def plot_faithfulness(data):
    """Grouped horizontal bars: Faithfulness, Q², AIS (higher is better)."""
    data = sorted(data, key=lambda x: x["faithfulness"])
    methods = [d["method"] for d in data]
    n = len(methods)

    metrics_cfg = [
        ("Faithfulness (NLI)", "faithfulness", "faith_err", "#1565C0"),
        ("Q² Consistency", "q2", "q2_err", "#2E7D32"),
        ("AIS Attribution", "ais", "ais_err", "#6A1B9A"),
    ]
    bar_h = 0.22
    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.6)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    y = np.arange(n)
    for i, (label, key, err_key, color) in enumerate(metrics_cfg):
        vals = [d[key] for d in data]
        errs_lo = [max(0, d[err_key][0]) for d in data]
        errs_hi = [max(0, d[err_key][1]) for d in data]
        offset = (i - 1) * bar_h
        bars = ax.barh(y + offset, vals, height=bar_h, color=color, label=label,
                       edgecolor="white", linewidth=0.5, alpha=0.88)
        ax.errorbar(vals, y + offset, xerr=[errs_lo, errs_hi],
                    fmt="none", ecolor="#555", elinewidth=0.8, capsize=2)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", fontsize=7.5, va="center", color="#333")

    for i, d in enumerate(data):
        if _is_ours(d["method"]):
            ax.axhspan(i - 0.45, i + 0.45, facecolor="#E8F5E9", zorder=0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9.5, color="#212121")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score (higher is better)", fontsize=11, color="#333")
    ax.set_title("Hallucination Benchmark — Faithfulness Metrics (with 95% CI)",
                 fontsize=13, fontweight="bold", color="#212121", pad=14)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#CCC")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E0E0E0", linestyle="-", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#CFD8DC")
    plt.tight_layout()
    plt.savefig(OUT_FAITH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT_FAITH}")


def plot_halluc_rate(data):
    """Horizontal bars: Hallucination Rate (lower is better) with error bars."""
    data = sorted(data, key=lambda x: x["halluc_rate"], reverse=True)
    methods = [d["method"] for d in data]
    rates = [d["halluc_rate"] for d in data]
    errs_lo = [max(0, d["halluc_err"][0]) for d in data]
    errs_hi = [max(0, d["halluc_err"][1]) for d in data]
    n = len(methods)

    colors = ["#C62828" if _is_ours(d["method"]) else "#5C6BC0" for d in data]
    edges = ["#8B0000" if c == "#C62828" else "#3F51B5" for c in colors]

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.5)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    y = np.arange(n)
    bars = ax.barh(y, rates, height=0.6, color=colors, edgecolor=edges, linewidth=1.2)
    ax.errorbar(rates, y, xerr=[errs_lo, errs_hi],
                fmt="none", ecolor="#333", elinewidth=0.9, capsize=3)

    for bar, v in zip(bars, rates):
        ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", fontsize=10, va="center", fontweight="bold", color="#212121")

    for i, d in enumerate(data):
        if _is_ours(d["method"]):
            ax.axhspan(i - 0.52, i + 0.52, facecolor="#E8F5E9", zorder=0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10, color="#212121")
    ax.set_xlim(0, max(rates) * 1.25 + 0.08)
    ax.set_xlabel("Hallucination Rate (lower is better)", fontsize=12, color="#212121")
    ax.set_title("Hallucination Rate by Method (with 95% CI)",
                 fontsize=14, fontweight="bold", color="#212121", pad=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CFD8DC", linestyle="-", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#CFD8DC")
    plt.tight_layout()
    plt.savefig(OUT_RATE, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT_RATE}")


def plot_taxonomy(data):
    """Grouped bars: Intrinsic vs Extrinsic hallucination (Maynez taxonomy)."""
    data = sorted(data, key=lambda x: x["intrinsic"] + d["extrinsic"], reverse=True)
    methods = [d["method"] for d in data]
    n = len(methods)

    bar_h = 0.3
    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.5)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    y = np.arange(n)
    intr_vals = [d["intrinsic"] for d in data]
    extr_vals = [d["extrinsic"] for d in data]

    ax.barh(y - bar_h / 2, intr_vals, height=bar_h, color="#D32F2F", label="Intrinsic (contradicts source)",
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.barh(y + bar_h / 2, extr_vals, height=bar_h, color="#FF8F00", label="Extrinsic (unverifiable)",
            edgecolor="white", linewidth=0.5, alpha=0.85)

    for i, d in enumerate(data):
        if _is_ours(d["method"]):
            ax.axhspan(i - 0.45, i + 0.45, facecolor="#E8F5E9", zorder=0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10, color="#212121")
    ax.set_xlabel("Rate (lower is better)", fontsize=11, color="#333")
    ax.set_title("Hallucination Taxonomy — Intrinsic vs Extrinsic (Maynez et al., 2020)",
                 fontsize=13, fontweight="bold", color="#212121", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#CCC")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E0E0E0", linestyle="-", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color("#CFD8DC")
    plt.tight_layout()
    plt.savefig(OUT_TAXONOMY, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {OUT_TAXONOMY}")


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run: python -m hallucination.runner")
        sys.exit(1)
    data = load_data()
    plot_faithfulness(data)
    plot_halluc_rate(data)
    try:
        plot_taxonomy(data)
    except Exception as e:
        print(f"Taxonomy plot skipped: {e}")
    print("All plots generated.")


if __name__ == "__main__":
    main()
