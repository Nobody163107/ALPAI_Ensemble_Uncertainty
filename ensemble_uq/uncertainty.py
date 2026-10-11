"""Turning an ensemble's predictions into uncertainty estimates.

Classification (information-theoretic decomposition, lecture slides 27-33;
Depeweg et al., 2018). Given M ensemble members h_1..h_M and their predicted
class distributions p(y | x, h_m):

    total      U(x)  = H[ (1/M) * sum_m p(y | x, h_m) ]     entropy of the average
    aleatoric  AU(x) = (1/M) * sum_m H[ p(y | x, h_m) ]     average of the entropies
    epistemic  EU(x) = U(x) - AU(x)                         mutual information I(y; h | x, D)

EU is zero when all members agree and grows with their disagreement; it equals
the Jensen-Shannon divergence between the member distributions.

Regression (law of total variance). Each member predicts a Gaussian
N(mu_m(x), sigma2_m(x)); the ensemble is the equally weighted mixture:

    Var[y | x] = (1/M) * sum_m sigma2_m(x)   +   (1/M) * sum_m (mu_m(x) - mu_bar(x))^2
                 '---- aleatoric (noise) ---'    '------ epistemic (disagreement) ------'
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def entropy(p: np.ndarray, axis: int = -1, base: float = 2.0) -> np.ndarray:
    """Shannon entropy of the distribution(s) in ``p`` along ``axis`` (bits by default)."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis) / np.log(base)


@dataclass
class Decomposition:
    """Result of :func:`decompose_classification`; every array has one entry per input."""

    mean_probs: np.ndarray  # (N, K) ensemble predictive distribution p(y | x)
    total: np.ndarray       # (N,)   U(x)
    aleatoric: np.ndarray   # (N,)   AU(x)
    epistemic: np.ndarray   # (N,)   EU(x)

    @property
    def prediction(self) -> np.ndarray:
        return self.mean_probs.argmax(axis=-1)

    @property
    def confidence(self) -> np.ndarray:
        return self.mean_probs.max(axis=-1)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"total": self.total, "aleatoric": self.aleatoric, "epistemic": self.epistemic}


def decompose_classification(member_probs: np.ndarray, base: float = 2.0) -> Decomposition:
    """Split predictive uncertainty into aleatoric and epistemic parts.

    Parameters
    ----------
    member_probs : array of shape (M, N, K) - or (M, K) for a single input -
        holding p(y = k | x_n, h_m) for every member m, input n and class k.
    base : logarithm base of the entropies (2 -> bits, as in the lecture).
    """
    P = np.asarray(member_probs, dtype=np.float64)
    if P.ndim == 2:
        P = P[:, None, :]
    if P.ndim != 3:
        raise ValueError(f"expected (M, N, K) member probabilities, got shape {P.shape}")

    mean_probs = P.mean(axis=0)                        # p(y | x) = (1/M) sum_m p(y | x, h_m)
    total = entropy(mean_probs, base=base)             # entropy of the average
    aleatoric = entropy(P, base=base).mean(axis=0)     # average of the entropies
    epistemic = np.clip(total - aleatoric, 0.0, None)  # >= 0 by concavity of entropy (Jensen)
    return Decomposition(mean_probs, total, aleatoric, epistemic)


def jensen_shannon(member_probs: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Generalised Jensen-Shannon divergence (1/M) sum_m KL(p_m || p_bar).

    Mathematically identical to the epistemic part above; computed the long way
    round so the identity can be checked (see tests/test_uncertainty.py).
    """
    P = np.clip(np.asarray(member_probs, dtype=np.float64), _EPS, 1.0)
    if P.ndim == 2:
        P = P[:, None, :]
    p_bar = P.mean(axis=0, keepdims=True)
    kl = (P * (np.log(P) - np.log(p_bar))).sum(axis=-1) / np.log(base)
    return kl.mean(axis=0)


def member_table(member_probs_x: np.ndarray, class_names=None, base: float = 2.0):
    """The ensemble table from lecture slide 33 for a single input x.

    Rows are the members h_1..h_M plus the average p(y|x); columns are the class
    probabilities and the entropy of each row. Returns ``(table, summary)`` where
    ``summary`` holds U(x), AU(x) and EU(x).
    """
    import pandas as pd

    P = np.asarray(member_probs_x, dtype=np.float64)
    M, K = P.shape
    names = list(class_names) if class_names is not None else [f"y{k + 1}" for k in range(K)]
    table = pd.DataFrame(P, columns=names, index=[f"h{m + 1}" for m in range(M)])
    table["entropy"] = entropy(P, base=base)
    mean_row = P.mean(axis=0)
    table.loc["average"] = list(mean_row) + [entropy(mean_row, base=base)]
    d = decompose_classification(P, base=base)
    summary = {
        "U (total) = entropy of average": float(d.total[0]),
        "AU (aleatoric) = average of entropies": float(d.aleatoric[0]),
        "EU (epistemic) = U - AU": float(d.epistemic[0]),
    }
    return table, summary


@dataclass
class RegressionDecomposition:
    """Mean and variance split of a Gaussian ensemble (law of total variance)."""

    mean: np.ndarray           # (N,) ensemble mean  mu_bar(x)
    aleatoric_var: np.ndarray  # (N,) average predicted noise variance
    epistemic_var: np.ndarray  # (N,) variance of the member means

    @property
    def total_var(self) -> np.ndarray:
        return self.aleatoric_var + self.epistemic_var

    @property
    def aleatoric_std(self) -> np.ndarray:
        return np.sqrt(self.aleatoric_var)

    @property
    def epistemic_std(self) -> np.ndarray:
        return np.sqrt(self.epistemic_var)

    @property
    def total_std(self) -> np.ndarray:
        return np.sqrt(self.total_var)


def decompose_regression(member_means: np.ndarray, member_vars: np.ndarray) -> RegressionDecomposition:
    """Law of total variance for an equally weighted mixture of M Gaussians.

    ``member_means`` and ``member_vars`` have shape (M, N).
    """
    mu = np.asarray(member_means, dtype=np.float64)
    var = np.asarray(member_vars, dtype=np.float64)
    mean = mu.mean(axis=0)
    aleatoric = var.mean(axis=0)                   # E_m[sigma2_m]  - noise the members expect
    epistemic = ((mu - mean) ** 2).mean(axis=0)    # Var_m[mu_m]    - how much they disagree
    return RegressionDecomposition(mean, aleatoric, epistemic)
