"""E1 - Where does each kind of uncertainty live?  (2D toy classification)

Two overlapping Gaussian classes: inside the overlap the label is genuinely random
(aleatoric); far away from the data nobody has told the model anything (epistemic).
We fit three ensembles and map U(x), AU(x), EU(x) over the input plane.

    python experiments/e1_toy_classification.py
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from common import FIG_DIR, save_json

from ensemble_uq import LaplaceRandomForest, MCEnsemble, ParallelMLPEnsemble, decompose_classification, member_table
from ensemble_uq.data import grid_2d, make_toy_2d
from ensemble_uq.models import MLP
from ensemble_uq.plotting import INK2, member_boundaries, plot_points, set_style, style_map_axis, uncertainty_map

LIM = 6.0
N_MEMBERS = 10
MLP_FIT = dict(epochs=300, lr=1e-2, weight_decay=1e-3)
QUERIES = {"A": (0.0, 0.0), "B": (0.0, 5.0)}  # A: inside the class overlap, B: far from all data


def fit_methods(X, y, n_members: int = N_MEMBERS, seed: int = 0):
    """Train the three ensembles compared in the slides."""
    dropconnect = lambda: MLP(2, 2, hidden=(64, 64), dropconnect=0.2)  # noqa: E731
    return {
        "Deep Ensemble": ParallelMLPEnsemble(2, 2, n_members=n_members, hidden=(32, 32), seed=seed).fit(X, y, **MLP_FIT),
        "Random Forest": LaplaceRandomForest(n_estimators=100, min_samples_leaf=5, seed=seed).fit(X, y),
        # one weight mask per forward pass, shared by the whole grid -> every pass is one coherent member
        "MC DropConnect": MCEnsemble(dropconnect, n_samples=50, seed=seed, batch_size=100_000).fit(
            X, y, epochs=500, lr=1e-2, weight_decay=1e-4),
    }


def mark_queries(ax):
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    for q, (qx, qy) in QUERIES.items():
        ax.scatter([qx], [qy], s=150, marker="*", color="#eda100", edgecolors="#0b0b0b", linewidths=0.8, zorder=5)
        ax.annotate(q, (qx, qy), xytext=(8, 5), textcoords="offset points", fontsize=13, fontweight="bold",
                    path_effects=halo, zorder=6)


def decomposition_row(axes, xx, yy, dec, X, y, label=None):
    for ax, kind in zip(axes, ("total", "aleatoric", "epistemic")):
        uncertainty_map(ax, xx, yy, getattr(dec, kind), kind, X, y)
    if label:
        axes[0].set_ylabel(label, color=INK2, fontsize=12, fontweight="bold")


def run():
    set_style()
    t0 = time.time()
    X, y = make_toy_2d("blobs", n=400, noise=1.0, seed=0)
    xx, yy, grid = grid_2d(LIM, 220)
    methods = fit_methods(X, y)
    member_p = {name: m.member_probs(grid) for name, m in methods.items()}
    decs = {name: decompose_classification(p) for name, p in member_p.items()}
    single = decompose_classification(member_p["Deep Ensemble"][:1])  # member 1 on its own

    # --- figure: one network vs the ensemble ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    uncertainty_map(axes[0], xx, yy, single.total, "total", X, y, title="Single network: entropy")
    uncertainty_map(axes[1], xx, yy, decs["Deep Ensemble"].total, "total", X, y,
                    title=f"Deep ensemble (M = {N_MEMBERS}): entropy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_single_vs_ensemble.png")
    plt.close(fig)

    # --- figure: decomposition per method --------------------------------------------
    for name, dec in decs.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        decomposition_row(axes, xx, yy, dec, X, y)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"e1_decomposition_{name.lower().replace(' ', '_')}.png")
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12))
    for row, (name, dec) in zip(axes, decs.items()):
        decomposition_row(row, xx, yy, dec, X, y, label=name)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_decomposition_grid.png")
    plt.close(fig)

    # --- figure: data alone, and the members' decision boundaries -------------------
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    plot_points(ax, X, y, size=14)
    style_map_axis(ax, LIM)
    fig.savefig(FIG_DIR / "e1_data.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    member_boundaries(ax, xx, yy, member_p["Deep Ensemble"], X, y)
    mark_queries(ax)
    ax.set_title(f"{N_MEMBERS} members, {N_MEMBERS} decision boundaries")
    fig.savefig(FIG_DIR / "e1_member_boundaries.png")
    plt.close(fig)

    # --- figure: single network + the three maps, with the query points (slides) -----
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3))
    uncertainty_map(axes[0], xx, yy, single.total, "total", X, y, title="One network: entropy")
    for ax, kind in zip(axes[1:], ("total", "aleatoric", "epistemic")):
        uncertainty_map(ax, xx, yy, getattr(decs["Deep Ensemble"], kind), kind, X, y,
                        title=f"Ensemble: {kind}" if kind != "total" else "Ensemble: total")
        mark_queries(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_maps_with_queries.png")
    plt.close(fig)

    # --- figure: the data with the two kinds of "not knowing" marked (concept slide) ---
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    ax.add_patch(plt.Rectangle((-0.9, -3.4), 1.8, 6.8, color="#eb6834", alpha=0.16, lw=0, zorder=1))
    ax.add_patch(plt.Rectangle((-LIM, 3.9), 2 * LIM, LIM - 3.9, color="#2a78d6", alpha=0.14, lw=0, zorder=1))
    plot_points(ax, X, y, size=10)
    style_map_axis(ax, LIM)
    ax.text(0, -4.4, "classes overlap", ha="center", va="top", fontsize=12, color="#c24e1c", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=2), zorder=6)
    ax.text(0, 5.0, "no training data", ha="center", va="center", fontsize=12, color="#184f95", fontweight="bold")
    fig.savefig(FIG_DIR / "e1_data_annotated.png")
    plt.close(fig)

    # --- the slide-33 table for two query points --------------------------------------
    queries = {}
    for q, point in QUERIES.items():
        probs = methods["Deep Ensemble"].member_probs(np.array([point], dtype=np.float32))[:, 0, :]
        table, summary = member_table(probs, class_names=["p(red)", "p(black)"])
        queries[q] = {"point": point, "member_probs": probs, "entropies": table["entropy"].to_numpy(),
                      "summary": summary}
        print(f"\nQuery {q} at {point}\n{table.round(3)}\n{summary}")

    # --- summary numbers -----------------------------------------------------------------
    inside = (np.abs(grid[:, 0]) < 1.0) & (np.abs(grid[:, 1]) < 1.0)   # the class overlap
    far = np.hypot(grid[:, 0], grid[:, 1]) > 5.0                       # well outside the data
    summary = {name: {"mean_AU_overlap": float(d.aleatoric[inside].mean()),
                      "mean_EU_overlap": float(d.epistemic[inside].mean()),
                      "mean_AU_far": float(d.aleatoric[far].mean()),
                      "mean_EU_far": float(d.epistemic[far].mean())} for name, d in decs.items()}
    summary["Single network"] = {"mean_entropy_far": float(single.total[far].mean()),
                                 "mean_entropy_overlap": float(single.total[inside].mean())}
    save_json({"n_members": N_MEMBERS, "queries": queries, "regions": summary,
               "n_train": int(len(X))}, "e1_toy_classification.json")
    print(f"\nE1 done in {time.time() - t0:.0f}s")
    return methods, decs


if __name__ == "__main__":
    run()
