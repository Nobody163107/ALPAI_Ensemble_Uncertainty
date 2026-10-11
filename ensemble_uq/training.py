"""Training loops. Both losses are proper scoring rules, as Deep Ensembles require."""
from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_tensor(a, dtype) -> torch.Tensor:
    return a.to(dtype) if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dtype)


def fit_classifier(model: torch.nn.Module, X, y, epochs: int = 500, lr: float = 1e-2,
                   weight_decay: float = 1e-4, batch_size: int | None = None, seed: int = 0,
                   log: Callable[[str], None] | None = None) -> torch.nn.Module:
    """Minimise the cross-entropy (negative log-likelihood) with Adam.

    ``batch_size=None`` trains full-batch (small tabular / toy data). With mini-batches,
    the data order is shuffled with ``seed``, so ensemble members also differ in the
    order they see the data - the second source of diversity in Deep Ensembles.
    """
    X, y = _as_tensor(X, torch.float32), _as_tensor(y, torch.long)
    n = len(X)
    bs = n if batch_size is None else batch_size
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n, generator=gen)
        running = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        if log is not None:
            log(f"epoch {epoch + 1}/{epochs}  loss {running / n:.4f}")
    model.eval()
    return model


def gaussian_nll(mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor, dim=None) -> torch.Tensor:
    """Negative log-likelihood of y under N(mean, var), averaged over ``dim`` (all entries if None)."""
    nll = 0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)
    return nll.mean() if dim is None else nll.mean(dim)
