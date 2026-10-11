"""Ways of building an ensemble {h_1, ..., h_M}.

Every classifier exposes ``member_probs(X)`` -> array (M, N, K) holding p(y | x, h_m),
so any of them plugs into :func:`ensemble_uq.uncertainty.decompose_classification`
(the table on lecture slide 33). That is the point of the ensemble approach: the
uncertainty maths does not care *how* the members were obtained.

* :class:`DeepEnsemble`         - M networks of any architecture, each trained from its own
                                  random initialisation (Lakshminarayanan et al., 2017).
                                  A plain loop; used for the CNNs.
* :class:`ParallelMLPEnsemble`  - the same idea for small MLPs, all M members trained at
                                  once with batched matrix products (fast enough for live demos).
* :class:`GaussianEnsemble`     - Deep Ensemble for regression: members predict N(mu, sigma2).
* :class:`MCEnsemble`           - ONE network with dropout / DropConnect sampled M times at test
                                  time: an *implicit* ensemble (Gal & Ghahramani, 2016;
                                  Mobiny et al., 2021; lecture slide 34).
* :class:`LaplaceRandomForest`  - trees with Laplace-corrected leaf frequencies
                                  (Shaker & Huellermeier, 2020; lecture slide 35).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .models import ParallelMLP
from .training import fit_classifier, gaussian_nll
from .uncertainty import Decomposition, RegressionDecomposition, decompose_classification, decompose_regression


def _to_tensor(X) -> torch.Tensor:
    return X.float() if torch.is_tensor(X) else torch.as_tensor(np.asarray(X), dtype=torch.float32)


@torch.no_grad()
def _softmax_batches(model: torch.nn.Module, X: torch.Tensor, batch_size: int) -> np.ndarray:
    return torch.cat([torch.softmax(model(xb), dim=-1) for xb in X.split(batch_size)]).numpy()


class Ensemble:
    """Common interface: subclasses implement ``member_probs``."""

    def member_probs(self, X) -> np.ndarray:  # (M, N, K)
        raise NotImplementedError

    def uncertainty(self, X) -> Decomposition:
        return decompose_classification(self.member_probs(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.member_probs(X).mean(axis=0)


class DeepEnsemble(Ensemble):
    """M independently trained networks.

    Diversity comes from (i) a different random initialisation per member and (ii) a
    different shuffling of the training data. With ``bootstrap=True`` each member also
    trains on its own bootstrap resample (bagging, Breiman 1996).
    """

    def __init__(self, model_fn: Callable[[], torch.nn.Module] | None = None, n_members: int = 5,
                 bootstrap: bool = False, seed: int = 0, members=None):
        self.model_fn = model_fn
        self.n_members = n_members if members is None else len(members)
        self.bootstrap = bootstrap
        self.seed = seed
        self.members = list(members) if members is not None else []

    def fit(self, X, y, **fit_kw) -> "DeepEnsemble":
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            member_seed = 1000 * self.seed + m
            torch.manual_seed(member_seed)          # (i) different random initialisation
            model = self.model_fn()
            if self.bootstrap:                      # bagging: resample the data with replacement
                idx = rng.integers(0, len(X), len(X))
                Xm, ym = X[idx], y[idx]
            else:
                Xm, ym = X, y
            fit_classifier(model, Xm, ym, seed=member_seed, **fit_kw)  # (ii) different data order
            self.members.append(model.eval())
        return self

    def member_probs(self, X, batch_size: int = 4096) -> np.ndarray:
        X = _to_tensor(X)
        return np.stack([_softmax_batches(model.eval(), X, batch_size) for model in self.members])

    def subset(self, k: int) -> "DeepEnsemble":
        """The first ``k`` members as a smaller ensemble (for ensemble-size studies)."""
        return DeepEnsemble(members=self.members[:k])


class ParallelMLPEnsemble(Ensemble):
    """Deep Ensemble of M small MLP classifiers, trained simultaneously.

    Identical in spirit to :class:`DeepEnsemble`: every member has its own random
    initialisation and its own loss; the losses are summed only so that one optimiser
    step updates all members at once (their gradients do not interact).
    """

    def __init__(self, in_dim: int, n_classes: int = 2, n_members: int = 10, hidden=(32, 32),
                 activation: str = "relu", bootstrap: bool = False, seed: int = 0):
        self.in_dim, self.n_classes, self.n_members = in_dim, n_classes, n_members
        self.hidden, self.activation, self.bootstrap, self.seed = hidden, activation, bootstrap, seed
        self.net: ParallelMLP | None = None

    def fit(self, X, y, epochs: int = 300, lr: float = 1e-2, weight_decay: float = 1e-3) -> "ParallelMLPEnsemble":
        torch.manual_seed(self.seed)
        self.net = ParallelMLP(self.n_members, self.in_dim, self.n_classes, self.hidden, self.activation)
        X, y = _to_tensor(X), torch.as_tensor(np.asarray(y), dtype=torch.long)
        M, N = self.n_members, len(X)
        if self.bootstrap:  # bagging via per-member sample counts
            gen = torch.Generator().manual_seed(self.seed)
            idx = torch.randint(0, N, (M, N), generator=gen)
            weights = torch.zeros(M, N).scatter_add_(1, idx, torch.ones(M, N))
        else:
            weights = torch.ones(M, N)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.net.train()
        for _ in range(epochs):
            logits = self.net(X)                                               # (M, N, K)
            per_example = F.cross_entropy(logits.reshape(M * N, -1), y.repeat(M), reduction="none").view(M, N)
            loss = ((per_example * weights).sum(1) / weights.sum(1)).sum()     # sum of M member losses
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def member_probs(self, X) -> np.ndarray:
        return torch.softmax(self.net(_to_tensor(X)), dim=-1).numpy()


class GaussianEnsemble:
    """Deep Ensemble for regression (Lakshminarayanan et al., 2017).

    Each member outputs a mean and a variance, N(mu_m(x), sigma2_m(x)), and is trained by
    minimising the Gaussian negative log-likelihood - a proper scoring rule, so a member
    is rewarded for reporting the noise level honestly. Members are trained in parallel.
    """

    def __init__(self, n_members: int = 5, hidden=(64, 64), activation: str = "silu", seed: int = 0,
                 min_var: float = 1e-4):
        self.n_members, self.hidden, self.activation, self.seed, self.min_var = (
            n_members, hidden, activation, seed, min_var)
        self.net: ParallelMLP | None = None

    def _forward(self, x: torch.Tensor):
        out = self.net(x)                                          # (M, N, 2)
        return out[..., 0], F.softplus(out[..., 1]) + self.min_var  # mean, variance > 0

    def fit(self, x, y, epochs: int = 2500, lr: float = 3e-3, warmup_epochs: int = 300) -> "GaussianEnsemble":
        """Gaussian NLL, after ``warmup_epochs`` of fitting the mean only (MSE) - a common
        trick that stops the variance head from explaining everything away as noise early on."""
        torch.manual_seed(self.seed)
        self.net = ParallelMLP(self.n_members, 1, 2, self.hidden, self.activation)
        x = _to_tensor(x).reshape(-1, 1)
        y = _to_tensor(y).reshape(1, -1)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for epoch in range(epochs):
            mean, var = self._forward(x)
            if epoch < warmup_epochs:
                loss = ((mean - y) ** 2).mean(1).sum()
            else:
                loss = gaussian_nll(mean, var, y, dim=1).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def member_moments(self, x) -> tuple[np.ndarray, np.ndarray]:
        mean, var = self._forward(_to_tensor(x).reshape(-1, 1))
        return mean.numpy(), var.numpy()

    def predict(self, x) -> RegressionDecomposition:
        return decompose_regression(*self.member_moments(x))


class MCEnsemble(Ensemble):
    """Implicit ensemble from a single network with dropout or DropConnect.

    Every stochastic forward pass at test time corresponds to one sampled member h_m.
    Training costs ONE network; prediction costs ``n_samples`` forward passes.
    """

    def __init__(self, model_fn: Callable[[], torch.nn.Module] | None = None, n_samples: int = 20,
                 seed: int = 0, model: torch.nn.Module | None = None, batch_size: int = 4096):
        self.model_fn = model_fn
        self.n_samples = n_samples
        self.seed = seed
        self.model = model
        self.batch_size = batch_size

    def fit(self, X, y, **fit_kw) -> "MCEnsemble":
        torch.manual_seed(1000 * self.seed)
        self.model = self.model_fn()
        fit_classifier(self.model, X, y, seed=1000 * self.seed, **fit_kw)
        return self

    def member_probs(self, X) -> np.ndarray:
        X = _to_tensor(X)
        with torch.random.fork_rng():             # reproducible masks, global RNG untouched
            torch.manual_seed(12345 + self.seed)
            self.model.train()                    # keep the random masks switched ON
            samples = [_softmax_batches(self.model, X, self.batch_size) for _ in range(self.n_samples)]
            self.model.eval()
        return np.stack(samples)


class LaplaceRandomForest(Ensemble):
    """Random forest whose trees predict Laplace-corrected leaf frequencies.

    A tree that sends x to a leaf with class counts n_1..n_K predicts
    p(y = k | x, h) = (n_k + 1) / (n + K). Without the correction a pure leaf says
    "100 % sure", every tree has zero entropy and AU would vanish; with it, small leaves
    stay appropriately unsure (lecture slide 35).
    """

    def __init__(self, n_estimators: int = 100, max_features="sqrt", min_samples_leaf: int = 1,
                 max_depth: int | None = None, seed: int = 0, **rf_kwargs):
        from sklearn.ensemble import RandomForestClassifier

        self.forest = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                             min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                             random_state=seed, **rf_kwargs)
        self._leaf_counts: list[np.ndarray] = []

    def fit(self, X, y) -> "LaplaceRandomForest":
        self.forest.fit(np.asarray(X), np.asarray(y))
        self._leaf_counts = []
        for tree in self.forest.estimators_:
            t = tree.tree_
            value = t.value[:, 0, :]
            fractions = value / np.clip(value.sum(axis=1, keepdims=True), 1e-12, None)
            # weighted_n_node_samples counts bootstrap duplicates, i.e. the tree's own training set
            self._leaf_counts.append(fractions * t.weighted_n_node_samples[:, None])
        return self

    @property
    def n_classes(self) -> int:
        return len(self.forest.classes_)

    def member_probs(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        K = self.n_classes
        out = []
        for tree, counts in zip(self.forest.estimators_, self._leaf_counts):
            n_k = counts[tree.apply(X)]                                      # (N, K) class counts in the leaf
            out.append((n_k + 1.0) / (n_k.sum(axis=1, keepdims=True) + K))   # Laplace correction
        return np.stack(out)
