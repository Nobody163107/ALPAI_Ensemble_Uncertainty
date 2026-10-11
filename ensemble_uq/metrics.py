"""How good is an uncertainty estimate? Metrics used in the experiments."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(axis=-1) == y).mean())


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood (nats) - a proper scoring rule; lower is better."""
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1.0)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between the predicted distribution and the one-hot label."""
    onehot = np.eye(probs.shape[-1])[y]
    return float(((probs - onehot) ** 2).sum(axis=-1).mean())


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Per confidence bin: mean confidence, accuracy and count (for reliability diagrams)."""
    conf = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == y
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    mean_conf, acc, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = which == b
        count[b] = mask.sum()
        if count[b]:
            mean_conf[b], acc[b] = conf[mask].mean(), correct[mask].mean()
    return mean_conf, acc, count


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE (Guo et al., 2017): average |accuracy - confidence| over bins, weighted by bin size."""
    mean_conf, acc, count = reliability_curve(probs, y, n_bins)
    return float((count / count.sum() * np.abs(acc - mean_conf)).sum())


def ood_auroc(score_in: np.ndarray, score_out: np.ndarray) -> float:
    """AUROC for telling out-of-distribution inputs (positives) from in-distribution ones.

    ``score`` is an uncertainty: higher should mean "more likely OOD".
    0.5 = chance, 1.0 = perfect separation.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.r_[np.zeros(len(score_in)), np.ones(len(score_out))]
    return float(roc_auc_score(labels, np.r_[score_in, score_out]))


def accuracy_rejection_curve(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray,
                             rejection_rates: np.ndarray | None = None, seed: int = 0):
    """Accuracy on the inputs we keep after abstaining on the most uncertain ones.

    For each rejection rate r, the fraction r of test points with the highest
    uncertainty is rejected and accuracy is computed on the rest (lecture slides 36-37).
    A useful uncertainty measure gives a curve that rises with r; a random measure
    gives a flat line. Ties are broken at random.
    """
    if rejection_rates is None:
        rejection_rates = np.linspace(0.0, 0.8, 81)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    shuffle = rng.permutation(n)
    order = shuffle[np.argsort(np.asarray(uncertainty)[shuffle], kind="stable")]  # most certain first
    correct = (np.asarray(y_true) == np.asarray(y_pred))[order]
    accs = []
    for r in rejection_rates:
        keep = max(1, int(round((1.0 - r) * n)))
        accs.append(correct[:keep].mean())
    return np.asarray(rejection_rates), np.asarray(accs)
