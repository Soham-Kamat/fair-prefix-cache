"""Headline figures from results/*.csv.

    python -m analysis.plots            # writes results/fig_*.png
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def hit_rate_vs_size(rows, out):
    wls = sorted({r["workload"] for r in rows})
    fig, axes = plt.subplots(1, len(wls), figsize=(4.2 * len(wls), 3.4), squeeze=False)
    for ax, wl in zip(axes[0], wls):
        series = defaultdict(list)
        for r in rows:
            if r["workload"] == wl:
                series[r["policy"]].append((float(r["cache_frac"]), float(r["hit_rate"])))
        for pol, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], marker="o", label=pol)
        ax.set_title(wl)
        ax.set_xlabel("cache size (fraction of unique tokens)")
        ax.set_ylabel("prompt-token hit rate (%)")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_hit_rate_vs_size.png"), dpi=150)


def fairness_frontier(rows, out):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for pol in sorted({r["policy"] for r in rows}):
        pts = [(100 * float(r["hit_rate"]), float(r["jain"])) for r in rows if r["policy"] == pol]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=pol, alpha=0.8)
    ax.set_xlabel("hit rate (%)")
    ax.set_ylabel("Jain fairness index (1 = equal)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_fairness_vs_hit_rate.png"), dpi=150)


def _grouped(ax, groups, series, value, ylabel, title):
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        ax.bar([j + i * w for j in range(len(groups))], [value(g, s) for g in groups], width=w, label=s)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def cheater_bars(rows, out):
    """Left: keep-alive pings - cheater's hit-rate gain under each eviction policy.
    Right: late arrival - cheater's bill saving under writer-pays vs Shapley (LRU)."""
    wls = sorted({r["workload"] for r in rows})
    pols = ["lru", "lfu", "quota", "shapley"]
    idx = {(r["workload"], r["attack"], r["policy"]): r for r in rows}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _grouped(a1, wls, pols,
             lambda g, s: float(idx[(g, "keepalive", s)]["cheater_hit_rate_gain_pp"]),
             "cheater hit-rate gain (pp)", "Keep-alive pings, by eviction policy")
    rules = {"writer pays": "bill_saving_pct[pay_what_you_compute]", "Shapley": "bill_saving_pct[shapley]"}
    _grouped(a2, wls, list(rules),
             lambda g, s: float(idx[(g, "late_arrival", "lru")][rules[s]]),
             "cheater bill saving (%)", "Late arrival, by billing rule (LRU)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cheater_gain.png"), dpi=150)


def main(out="results"):
    hit_rate_vs_size(read(os.path.join(out, "sweep.csv")), out)
    fairness_frontier(read(os.path.join(out, "sweep.csv")), out)
    cheater_bars(read(os.path.join(out, "cheaters.csv")), out)
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
