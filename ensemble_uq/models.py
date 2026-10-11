"""Neural network building blocks used as ensemble members."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class DropConnectLinear(nn.Linear):
    """Linear layer that randomly drops individual *weights* (DropConnect, Wan et al., 2013).

    Like dropout, the random mask is only active in ``train()`` mode. Monte-Carlo
    DropConnect (Mobiny et al., 2021) keeps it active at prediction time, so every
    forward pass uses a different thinned network - an implicit ensemble member.
    """

    def __init__(self, in_features: int, out_features: int, p: float = 0.1):
        super().__init__(in_features, out_features)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(self.weight, 1.0 - self.p))
            return F.linear(x, self.weight * mask / (1.0 - self.p), self.bias)
        return super().forward(x)


class MLP(nn.Module):
    """Fully connected network.

    ``dropout`` adds dropout after every hidden layer (MC Dropout);
    ``dropconnect`` replaces every linear layer by :class:`DropConnectLinear`.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden=(64, 64), activation: str = "relu",
                 dropout: float = 0.0, dropconnect: float = 0.0):
        super().__init__()

        def linear(i, o):
            return DropConnectLinear(i, o, dropconnect) if dropconnect > 0 else nn.Linear(i, o)

        layers, d = [], in_dim
        for h in hidden:
            layers += [linear(d, h), _ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelMLP(nn.Module):
    """M independent MLPs stored as stacked weight tensors and run with batched matmuls.

    Mathematically this is just M separate networks (no weights are shared); evaluating
    them together makes a 10-member ensemble of small MLPs train in about a second.
    Input (N, d) -> output (M, N, out_dim).
    """

    def __init__(self, n_members: int, in_dim: int, out_dim: int, hidden=(32, 32), activation: str = "relu"):
        super().__init__()
        self.n_members = n_members
        dims = [in_dim, *hidden, out_dim]
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(fan_in)  # same initialisation as torch.nn.Linear
            self.weights.append(nn.Parameter(torch.empty(n_members, fan_in, fan_out).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n_members, 1, fan_out).uniform_(-bound, bound)))
        self.act = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(0).expand(self.n_members, *x.shape)
        last = len(self.weights) - 1
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, W)  # (M, N, fan_in) @ (M, fan_in, fan_out) + bias
            if k < last:
                h = self.act(h)
        return h


class SmallCNN(nn.Module):
    """Compact LeNet-style CNN for 28x28 grey-scale images (about 106k parameters)."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
