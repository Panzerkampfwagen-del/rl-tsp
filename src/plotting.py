"""Visualization: tour plots, training curves, distribution-shift bar chart."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional


def plot_tour(cities_np: np.ndarray, tour_np: np.ndarray,
              title: str = "TSP Tour", save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ordered = cities_np[tour_np]
    loop = np.vstack([ordered, ordered[:1]])
    ax.plot(loop[:, 0], loop[:, 1], "b-", lw=1)
    ax.scatter(cities_np[:, 0], cities_np[:, 1], c="red", s=30, zorder=3)
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_csv: str, save_path: str) -> None:
    import csv
    epochs, losses, lengths = [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))
            lengths.append(float(row["avg_tour_len"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, losses, lw=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("REINFORCE Loss"); ax1.set_title("Training Loss")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.plot(epochs, lengths, lw=1.5, color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean Tour Length"); ax2.set_title("Tour Length")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shift_bar_chart(
    results: Dict[str, Dict],
    method: str,
    save_path: str,
    n_cities: int = 20,
) -> None:
    """
    results: {dist_name: {"gap_mean": float, "gap_std": float}}
    method: label for the baseline, e.g. '2-opt'
    """
    dists = list(results.keys())
    means = [results[d]["gap_mean"] for d in dists]
    stds  = [results[d]["gap_std"]  for d in dists]

    colors = ["tab:blue" if d == "uniform" else "tab:orange" for d in dists]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dists))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(dists, fontsize=11)
    ax.set_ylabel(f"Optimality Gap vs {method} (%)", fontsize=11)
    ax.set_title(f"Distribution-Shift Robustness (TSP-{n_cities})", fontsize=13)
    ax.axhline(0, color="black", lw=0.8, ls="--")

    patches = [
        mpatches.Patch(color="tab:blue",   label="in-distribution (uniform)"),
        mpatches.Patch(color="tab:orange", label="out-of-distribution"),
    ]
    ax.legend(handles=patches, fontsize=9)

    for bar, m, s in zip(bars, means, stds):
        if m >= 0:
            y_pos, va = bar.get_height() + s + 0.2, "bottom"
        else:
            y_pos, va = bar.get_height() - s - 0.2, "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{m:.1f}%", ha="center", va=va, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_distributions(n_per_dist: int = 50, n_cities: int = 20,
                            save_path: Optional[str] = None) -> None:
    """Visual overview of all 5 city distributions."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.env import sample_instances, DISTRIBUTIONS
    import numpy as np

    fig, axes = plt.subplots(1, len(DISTRIBUTIONS), figsize=(15, 3))
    rng = np.random.default_rng(42)
    for ax, dist in zip(axes, DISTRIBUTIONS):
        cities = sample_instances(dist, 1, n_cities, rng=rng)[0].numpy()
        ax.scatter(cities[:, 0], cities[:, 1], s=20, c="steelblue")
        ax.set_title(dist, fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("City Distributions", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
