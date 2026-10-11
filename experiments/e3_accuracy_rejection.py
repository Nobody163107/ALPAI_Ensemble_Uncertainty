"""E3 - Does the uncertainty know when the model is wrong?  (accuracy-rejection curves)

Re-creates the evaluation on lecture slide 37 (Shaker & Huellermeier, 2020): on the
'diabetes' and 'SPECT' data sets, reject the test points with the highest aleatoric
(or epistemic, or total) uncertainty and measure accuracy on the rest. Averaged over
10 random 70/30 splits.

    python experiments/e3_accuracy_rejection.py
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import FIG_DIR, save_json

from ensemble_uq import LaplaceRandomForest, MCEnsemble, ParallelMLPEnsemble
from ensemble_uq.data import load_tabular
from ensemble_uq.metrics import accuracy_rejection_curve
from ensemble_uq.models import MLP
from ensemble_uq.plotting import METHOD_COLORS, MUTED, set_style

DATASETS = {"diabetes": "Diabetes (Pima)", "spect": "SPECT heart"}
MEASURES = {"total": "total", "aleatoric": "aleatoric", "epistemic": "epistemic"}
RATES = np.linspace(0.0, 0.8, 81)
FIT = dict(epochs=100, lr=5e-3, weight_decay=1e-2)  # light training: these data sets are tiny


def make_methods(n_features: int, seed: int):
    return {
        "Deep Ensemble": ParallelMLPEnsemble(n_features, 2, n_members=10, hidden=(64, 64), seed=seed),
        "Random Forest": LaplaceRandomForest(n_estimators=100, min_samples_leaf=3, seed=seed),
        "MC DropConnect": MCEnsemble(lambda: MLP(n_features, 2, hidden=(64, 64), dropconnect=0.2),
                                     n_samples=50, seed=seed),
        "MC Dropout": MCEnsemble(lambda: MLP(n_features, 2, hidden=(64, 64), dropout=0.2),
                                 n_samples=50, seed=seed),
    }


def run(n_splits: int = 10):
    set_style()
    t0 = time.time()
    results = {}
    for ds in DATASETS:
        X, y = load_tabular(ds)
        curves = {}  # method -> measure -> list of curves
        base_acc = {}
        for split in range(n_splits):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
            for name, method in make_methods(X.shape[1], seed=split).items():
                if isinstance(method, LaplaceRandomForest):
                    method.fit(Xtr, ytr)
                    dec = method.uncertainty(Xte)
                else:
                    method.fit(Xtr_s, ytr, **FIT)
                    dec = method.uncertainty(Xte_s)
                pred = dec.prediction
                base_acc.setdefault(name, []).append(float((pred == yte).mean()))
                for measure in MEASURES:
                    _, acc = accuracy_rejection_curve(yte, pred, getattr(dec, measure), RATES, seed=split)
                    curves.setdefault(name, {}).setdefault(measure, []).append(acc)
            print(f"[{ds}] split {split + 1}/{n_splits} done ({time.time() - t0:.0f}s)")
        results[ds] = {
            name: {measure: {"mean": np.mean(c, axis=0), "std": np.std(c, axis=0),
                             "area": float(np.mean(c))} for measure, c in per_measure.items()}
            for name, per_measure in curves.items()
        }
        for name in results[ds]:
            results[ds][name]["accuracy_no_rejection"] = float(np.mean(base_acc[name]))

    plot(results, measures=("aleatoric", "epistemic"), fname="e3_accuracy_rejection.png")
    plot(results, measures=("total", "aleatoric", "epistemic"), fname="e3_accuracy_rejection_all.png")
    save_json({"rejection_rates": RATES, "n_splits": n_splits, "curves": results}, "e3_accuracy_rejection.json")
    for ds, per in results.items():
        for name, r in per.items():
            print(ds, name, {m: round(r[m]["area"], 4) for m in MEASURES}, "acc@0", round(r["accuracy_no_rejection"], 4))
    print(f"E3 done in {time.time() - t0:.0f}s")
    return results


def plot(results, measures, fname, methods=("Deep Ensemble", "Random Forest", "MC DropConnect", "MC Dropout")):
    fig, axes = plt.subplots(len(measures), len(DATASETS), figsize=(7.8, 2.65 * len(measures)), squeeze=False)
    for col, (ds, title) in enumerate(DATASETS.items()):
        for row, measure in enumerate(measures):
            ax = axes[row, col]
            for name in methods:
                r = results[ds][name][measure]
                pct = RATES * 100
                ax.plot(pct, r["mean"], color=METHOD_COLORS[name], label=name)
                ax.fill_between(pct, r["mean"] - r["std"], r["mean"] + r["std"],
                                color=METHOD_COLORS[name], alpha=0.12, lw=0)
            ax.axhline(results[ds]["Deep Ensemble"]["accuracy_no_rejection"], color=MUTED, ls="--", lw=1.2,
                       label="random rejection")
            ax.set_title(f"{title}: reject by {measure}", fontsize=11)
            ax.set_xlabel("rejection rate (%)" if row == len(measures) - 1 else "")
            ax.set_ylabel("accuracy on kept" if col == 0 else "")
            ax.set_xlim(0, 80)
    axes[0, 0].legend(fontsize=8.5, loc="upper left", handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


if __name__ == "__main__":
    run()
