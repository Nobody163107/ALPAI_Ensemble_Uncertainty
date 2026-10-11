"""Shared figure style. Colour carries meaning everywhere in this project:

    aleatoric = orange    epistemic = blue    total = grey
"""
from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ink and chrome
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical slots (validated colour-blind-safe order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

EPISTEMIC, ALEATORIC, TOTAL = BLUE, ORANGE, INK2
# methods get their own hues so that orange/blue always mean aleatoric/epistemic
METHOD_COLORS = {
    "Deep Ensemble": VIOLET,
    "Random Forest": AQUA,
    "MC DropConnect": YELLOW,
    "MC Dropout": MAGENTA,
    "Single network": MUTED,
}
SEEN, UNSEEN = "#c3c2b7", VIOLET  # in-distribution vs out-of-distribution inputs
CLASS_COLORS = (RED, INK)  # the lecture's red vs black classes

CMAP_TOTAL = LinearSegmentedColormap.from_list(
    "total", ["#ffffff", "#ecebe7", "#c3c2b7", "#898781", "#52514e", "#2c2c2a"])
CMAP_ALEATORIC = LinearSegmentedColormap.from_list(
    "aleatoric", ["#ffffff", "#fde3d6", "#f7b393", "#f08a5e", "#eb6834", "#c24e1c", "#8a3410"])
CMAP_EPISTEMIC = LinearSegmentedColormap.from_list(
    "epistemic", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
UNCERTAINTY_CMAPS = {"total": CMAP_TOTAL, "aleatoric": CMAP_ALEATORIC, "epistemic": CMAP_EPISTEMIC}
UNCERTAINTY_TITLES = {
    "total": "Total  U(x)",
    "aleatoric": "Aleatoric  AU(x)",
    "epistemic": "Epistemic  EU(x)",
}


def set_style(font_size: float = 11) -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Carlito", "Calibri", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def plot_points(ax, X, y, size: float = 16) -> None:
    """Training data in the lecture's red/black, with a white ring so it reads on any background."""
    for k, colour in enumerate(CLASS_COLORS):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=size, c=colour, edgecolors="white", linewidths=0.6,
                   zorder=3, label=f"class {k}")


def style_map_axis(ax, lim: float) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(AXIS)


def uncertainty_map(ax, xx, yy, values, kind: str, X=None, y=None, vmax: float = 1.0,
                    title: str | None = None, colorbar: bool = True):
    """One heat map of an uncertainty measure over the 2D input space (values in bits)."""
    im = ax.pcolormesh(xx, yy, values.reshape(xx.shape), cmap=UNCERTAINTY_CMAPS[kind],
                       vmin=0.0, vmax=vmax, shading="auto", rasterized=True)
    if X is not None:
        plot_points(ax, X, y, size=10)
    style_map_axis(ax, float(np.abs(xx).max()))
    ax.set_title(title or UNCERTAINTY_TITLES[kind], color=INK)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9, colors=MUTED, labelcolor=INK2)
        cb.set_label("bits", color=INK2, fontsize=9)
    return im


def member_boundaries(ax, xx, yy, member_probs, X=None, y=None, colour: str = BLUE, alpha: float = 0.55):
    """Decision boundary (p = 0.5) of every ensemble member - their spread is the epistemic part."""
    for p in member_probs:
        ax.contour(xx, yy, p[:, 1].reshape(xx.shape), levels=[0.5], colors=[colour],
                   linewidths=1.2, alpha=alpha)
    if X is not None:
        plot_points(ax, X, y, size=10)
    style_map_axis(ax, float(np.abs(xx).max()))


def regression_panels(ax_top, ax_bottom, x_grid, dec, x_train, y_train, truth=None, noise_std=None,
                      member_means=None, support=None):
    """Top: data, members and predictive bands. Bottom: the two standard deviations vs x.

    ``support`` = list of (a, b) intervals that contain training inputs; everything else
    inside the plotted range is shaded and labelled "no data".
    """
    if support is not None:
        edges = [float(x_grid.min())] + [v for ab in sorted(support) for v in ab] + [float(x_grid.max())]
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 1e-6:
                for ax in (ax_top, ax_bottom):
                    ax.axvspan(a, b, color="#f1f0ec", zorder=0, lw=0)
                ax_top.text((a + b) / 2, 0.97, "no data", transform=ax_top.get_xaxis_transform(),
                            ha="center", va="top", fontsize=9, color=MUTED)
    total_sd, alea_sd = dec.total_std, dec.aleatoric_std
    ax_top.fill_between(x_grid, dec.mean - 2 * total_sd, dec.mean + 2 * total_sd, color=EPISTEMIC,
                        alpha=0.18, lw=0, label="±2σ total (aleatoric + epistemic)")
    ax_top.fill_between(x_grid, dec.mean - 2 * alea_sd, dec.mean + 2 * alea_sd, color=ALEATORIC,
                        alpha=0.35, lw=0, label="±2σ aleatoric only")
    if member_means is not None:
        for k, mu in enumerate(member_means):
            ax_top.plot(x_grid, mu, color=EPISTEMIC, lw=1.0, alpha=0.6,
                        label="ensemble members" if k == 0 else None)
    if truth is not None:
        ax_top.plot(x_grid, truth, color=INK, lw=1.2, ls="--", label="true function")
    ax_top.plot(x_grid, dec.mean, color=INK, lw=2.0, label="ensemble mean")
    ax_top.scatter(x_train, y_train, s=9, color=INK2, alpha=0.7, lw=0, zorder=3, label="training data")
    ax_top.set_ylabel("y")

    ax_bottom.plot(x_grid, dec.aleatoric_std, color=ALEATORIC, label="aleatoric σ (avg. predicted noise)")
    ax_bottom.plot(x_grid, dec.epistemic_std, color=EPISTEMIC, label="epistemic σ (member disagreement)")
    if noise_std is not None:
        ax_bottom.plot(x_grid, noise_std, color=ALEATORIC, ls=":", lw=1.5, label="true noise σ")
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel("std. dev.")
