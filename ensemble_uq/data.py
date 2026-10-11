"""Datasets: 2D and 1D toy problems, two small tabular benchmarks, MNIST / Fashion-MNIST."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# --------------------------------------------------------------------------- 2D toy
def make_toy_2d(kind: str = "blobs", n: int = 400, noise: float = 1.0, seed: int = 0,
                hole: tuple[float, float, float] | None = None):
    """Two-class 2D data.

    * ``blobs``  - two overlapping Gaussian clouds: the overlap is pure aleatoric uncertainty.
    * ``moons``, ``circles``, ``xor`` - classic non-linear shapes.

    ``noise`` scales the spread of the classes. ``hole=(cx, cy, r)`` deletes every point
    within radius r of (cx, cy), creating a region with no data (epistemic uncertainty).
    """
    from sklearn.datasets import make_circles, make_moons

    rng = np.random.default_rng(seed)
    if kind == "blobs":
        y = rng.integers(0, 2, n)
        centres = np.array([[-1.25, 0.0], [1.25, 0.0]])
        X = centres[y] + noise * rng.standard_normal((n, 2))
    elif kind == "moons":
        X, y = make_moons(n, noise=0.2 * noise, random_state=seed)
        X = (X - [0.5, 0.25]) * 2.2
    elif kind == "circles":
        X, y = make_circles(n, noise=0.08 * noise, factor=0.45, random_state=seed)
        X = X * 3.0
    elif kind == "xor":
        centres = np.array([[1.6, 1.6], [-1.6, -1.6], [1.6, -1.6], [-1.6, 1.6]])
        c = rng.integers(0, 4, n)
        X = centres[c] + 0.6 * noise * rng.standard_normal((n, 2))
        y = (c >= 2).astype(np.int64)  # same-sign quadrants -> class 0, opposite-sign -> class 1
    else:
        raise ValueError(f"unknown toy dataset {kind!r}")
    if hole is not None:
        cx, cy, r = hole
        keep = (X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2 > r ** 2
        X, y = X[keep], y[keep]
    return X.astype(np.float32), y.astype(np.int64)


def grid_2d(lim: float = 6.0, resolution: int = 200):
    """Evaluation grid over [-lim, lim]^2; returns xx, yy (resolution x resolution) and points (R*R, 2)."""
    axis = np.linspace(-lim, lim, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


# --------------------------------------------------------------------------- 1D regression
def regression_truth(x: np.ndarray) -> np.ndarray:
    """The true mean function f(x)."""
    return np.sin(1.3 * x) + 0.25 * x


def regression_noise_std(x: np.ndarray) -> np.ndarray:
    """The true (heteroscedastic) noise level: almost noiseless, except a noisy bump around x = 2.7."""
    return 0.04 + 0.30 * np.exp(-((x - 2.7) ** 2))


REGRESSION_SUPPORT = ((-4.0, -2.0), (1.5, 4.0))  # training inputs come from here; (-2, 1.5) is a gap


def make_regression_1d(n: int = 200, seed: int = 0, support=REGRESSION_SUPPORT, noise_std: float | None = None):
    """Noisy samples y = f(x) + eps(x) with x drawn uniformly from the ``support`` intervals.

    ``noise_std=None`` uses the heteroscedastic :func:`regression_noise_std`; a number gives
    constant noise of that standard deviation.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array([b - a for a, b in support])
    which = rng.choice(len(support), size=n, p=lengths / lengths.sum())
    lo = np.array([support[i][0] for i in which])
    x = lo + rng.random(n) * lengths[which]
    sd = regression_noise_std(x) if noise_std is None else noise_std
    y = regression_truth(x) + sd * rng.standard_normal(n)
    return x.astype(np.float32), y.astype(np.float32)


# --------------------------------------------------------------------------- tabular
TABULAR = {
    # file, target column - see data/README.md for sources and licences
    "diabetes": ("diabetes.csv", "diabetes"),  # Pima Indians Diabetes (UCI), 768 x 8
    "spect": ("spect.csv", "diagnosis"),       # SPECT Heart (UCI), 267 x 22 binary features
}


def load_tabular(name: str):
    import pandas as pd

    fname, target = TABULAR[name]
    df = pd.read_csv(DATA_DIR / "tabular" / fname)
    y = df.pop(target).to_numpy(dtype=np.int64)
    return df.to_numpy(dtype=np.float32), y


# --------------------------------------------------------------------------- images
_GITHUB_COPIES = {  # identical files (same MD5) used if torchvision's own mirrors are unreachable
    "MNIST": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "FashionMNIST": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
_IDX_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _load_torchvision(cls_name: str, train: bool, root: Path):
    import urllib.request

    import torch
    import torchvision

    dataset = getattr(torchvision.datasets, cls_name)
    try:
        ds = dataset(root=str(root), train=train, download=True)
    except Exception:  # e.g. a campus proxy blocks the default mirrors: fetch the same files from GitHub
        raw = Path(root) / cls_name / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for fname in _IDX_FILES:
            if not (raw / fname).exists():
                urllib.request.urlretrieve(_GITHUB_COPIES[cls_name] + fname, raw / fname)
        ds = dataset(root=str(root), train=train, download=True)  # finds and checks the files, then unpacks
    X = ds.data.float().div(255.0).unsqueeze(1)  # (N, 1, 28, 28) in [0, 1]
    return X, torch.as_tensor(ds.targets, dtype=torch.long)


def normalize(X):
    """Standardise images with the MNIST statistics (also used for OOD images)."""
    return (X - MNIST_MEAN) / MNIST_STD


def load_mnist(train: bool = False, root: Path = DATA_DIR):
    """MNIST as raw [0, 1] tensors (downloaded by torchvision on first use)."""
    return _load_torchvision("MNIST", train, root)


def load_fashion_mnist(train: bool = False, root: Path = DATA_DIR):
    return _load_torchvision("FashionMNIST", train, root)


def rotate(X, angle: float):
    """Rotate a batch of images (N, 1, 28, 28) by ``angle`` degrees."""
    import torchvision.transforms.functional as TF

    return TF.rotate(X, angle)


def load_demo_samples(path: Path = ASSETS_DIR / "demo_samples.npz"):
    """Small offline subset used by the live app: MNIST and Fashion-MNIST test images in [0, 1]."""
    data = np.load(path)
    return {k: data[k] for k in data.files}
