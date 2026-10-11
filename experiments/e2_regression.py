"""E2 - Regression: splitting the error bar, and checking what more data can fix.

Part 1: a Deep Ensemble of mean/variance networks on 1D data with input-dependent
noise and a gap. Law of total variance -> aleatoric band vs epistemic band.

Part 2: constant noise (sigma = 0.25), N = 16 ... 2048 points. Epistemic uncertainty should shrink towards
zero, aleatoric uncertainty should settle at the true noise level - the lecture's
"reducible vs irreducible" distinction, measured.

    python experiments/e2_regression.py
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, save_json

from ensemble_uq import GaussianEnsemble
from ensemble_uq.data import REGRESSION_SUPPORT, make_regression_1d, regression_noise_std, regression_truth
from ensemble_uq.plotting import ALEATORIC, EPISTEMIC, MUTED, regression_panels, set_style

SUPPORT = REGRESSION_SUPPORT  # gap between -2 and 1.5
N_MEMBERS = 5
MORE_DATA_NOISE = 0.25
FIT = dict(epochs=2500, lr=3e-3, warmup_epochs=300)


def run(sizes=(16, 32, 64, 128, 256, 512, 1024, 2048), seeds=(0, 1, 2, 3, 4)):
    set_style()
    t0 = time.time()

    # ---------------- part 1: the picture ---------------------------------------------
    x, y = make_regression_1d(200, seed=0, support=SUPPORT)
    ens = GaussianEnsemble(n_members=N_MEMBERS, seed=0).fit(x, y, **FIT)
    x_grid = np.linspace(-7, 7, 600).astype(np.float32)
    means, variances = ens.member_moments(x_grid)
    dec = ens.predict(x_grid)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.1, 1]})
    regression_panels(ax1, ax2, x_grid, dec, x, y, truth=regression_truth(x_grid),
                      noise_std=regression_noise_std(x_grid), member_means=means, support=SUPPORT)
    ax1.set_ylim(-4.2, 4.2)
    ax1.legend(loc="lower right", fontsize=9, ncol=2, handlelength=1.5, columnspacing=1.0)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper center", fontsize=9, ncol=3, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e2_regression_bands.png")
    plt.close(fig)

    def region_means(xs, d):
        in_data = ((xs > SUPPORT[0][0]) & (xs < SUPPORT[0][1])) | ((xs > SUPPORT[1][0]) & (xs < SUPPORT[1][1]))
        gap = (xs > SUPPORT[0][1]) & (xs < SUPPORT[1][0])
        outside = np.abs(xs) > 4.5
        return {region: {"aleatoric_std": float(d.aleatoric_std[m].mean()),
                         "epistemic_std": float(d.epistemic_std[m].mean())}
                for region, m in (("in_data", in_data), ("gap", gap), ("outside", outside))}

    picture = region_means(x_grid, dec)

    # ---------------- part 2: more data shrinks only the epistemic part ----------------
    # Constant noise (sigma = 0.25) so the irreducible level is known exactly: 0.0625.
    x_eval = np.linspace(-4, 4, 400).astype(np.float32)
    curves = {"n": list(sizes), "noise_std": MORE_DATA_NOISE, "aleatoric_var": [], "epistemic_var": [],
              "true_noise_var": MORE_DATA_NOISE ** 2, "per_seed": {"aleatoric_var": [], "epistemic_var": []}}
    for n in sizes:
        alea, epi = [], []
        for s in seeds:
            xs, ys = make_regression_1d(n, seed=100 + s, support=((-4.0, 4.0),), noise_std=MORE_DATA_NOISE)
            e = GaussianEnsemble(n_members=N_MEMBERS, seed=s).fit(xs, ys, **FIT)
            d = e.predict(x_eval)
            alea.append(d.aleatoric_var.mean())
            epi.append(d.epistemic_var.mean())
        for key, vals in (("aleatoric_var", alea), ("epistemic_var", epi)):
            logs = np.log(vals)  # geometric mean / sd: the natural summary on a log axis
            curves[key].append([float(np.exp(logs.mean())), float(np.exp(logs.std()))])
            curves["per_seed"][key].append([float(v) for v in vals])
        print(f"N={n:5d}  aleatoric var {curves['aleatoric_var'][-1][0]:.4f}  "
              f"epistemic var {curves['epistemic_var'][-1][0]:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, colour, label in (("aleatoric_var", ALEATORIC, "aleatoric (avg. predicted noise)"),
                               ("epistemic_var", EPISTEMIC, "epistemic (member disagreement)")):
        gmean, gsd = np.array(curves[key]).T
        ax.plot(sizes, gmean, marker="o", ms=6, color=colour, label=label)
        ax.fill_between(sizes, gmean / gsd, gmean * gsd, color=colour, alpha=0.15, lw=0)
    ax.axhline(curves["true_noise_var"], color=ALEATORIC, ls=":", lw=1.5, label="true noise variance (0.0625)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("number of training points N")
    ax.set_ylabel("variance (avg. over x in [-4, 4])")
    ax.legend(fontsize=9, loc="lower left")
    ax.tick_params(axis="x", colors=MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e2_more_data.png")
    plt.close(fig)

    save_json({"n_members": N_MEMBERS, "picture_regions": picture, "more_data": curves},
              "e2_regression.json")
    print(f"E2 done in {time.time() - t0:.0f}s")
    return ens, curves


if __name__ == "__main__":
    run()
