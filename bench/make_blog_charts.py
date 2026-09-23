#!/usr/bin/env python3
"""Render summary chart PNGs from results/summary.csv into results/."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "results"
TASKS = [("triage", "Ticket triage"), ("intent_oos", "Intent +\nout-of-scope"),
         ("toxicity", "Toxicity\nguardrail"), ("tool_select", "Tool selection")]
MODELS = [("jev", "Jev (API)", "#2a78d6"), ("openjev", "Open-Jev-2B (local)", "#eb6834"),
          ("needle", "Needle 3 (local)", "#1baf7a")]
INK, MUTED, GRID = "#15181c", "#4b5157", "#e2e4e6"


def load():
    rows = {}
    for r in csv.DictReader(open(ROOT / "results" / "summary.csv")):
        rows[(r["model"], r["task"])] = r
    return rows


def panel(ax, rows, key, title, subtitle):
    h, gap = 0.24, 0.06
    for mi, (mid, mname, color) in enumerate(MODELS):
        for ti, (tid, _) in enumerate(TASKS):
            v = float(rows[(mid, tid)][key]) * 100
            y = ti - (len(MODELS) - 1) / 2 * (h + gap) + mi * (h + gap)
            ax.barh(y, v, height=h, color=color, zorder=3,
                    label=mname if ti == 0 else None)
            ax.text(v + 1.5, y, f"{v:.1f}%", va="center", ha="left",
                    fontsize=9.5, color=MUTED, zorder=4)
    ax.set_yticks(range(len(TASKS)), [t[1] for t in TASKS], fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=9.5, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


def main():
    OUT.mkdir(exist_ok=True)
    rows = load()
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
    panel(axes[0], rows, "accuracy", "Accuracy", "share of ~200 cases answered correctly")
    panel(axes[1], rows, "auto@95", "Handled without a human - or an LLM",
          "share of cases answered above the confidence threshold that keeps accuracy at 95%")
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(OUT / "results.png", dpi=170, facecolor="white")
    print(OUT / "results.png")


if __name__ == "__main__":
    main()
