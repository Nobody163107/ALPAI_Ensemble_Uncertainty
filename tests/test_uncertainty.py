"""Sanity checks for the decomposition (run with:  python -m pytest -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_uq.uncertainty import (decompose_classification, decompose_regression, entropy,  # noqa: E402
                                     jensen_shannon)


def test_agreeing_members_have_zero_epistemic():
    # every member says 50/50 -> all uncertainty is aleatoric
    P = np.array([[0.5, 0.5]] * 5)
    d = decompose_classification(P)
    assert np.isclose(d.total[0], 1.0) and np.isclose(d.aleatoric[0], 1.0)
    assert np.isclose(d.epistemic[0], 0.0)


def test_confident_disagreement_is_purely_epistemic():
    # half the members are sure of class 0, half of class 1 -> 1 bit, all epistemic
    P = np.array([[1.0, 0.0], [0.0, 1.0]] * 3)
    d = decompose_classification(P)
    assert np.isclose(d.total[0], 1.0, atol=1e-6)
    assert np.isclose(d.aleatoric[0], 0.0, atol=1e-6)
    assert np.isclose(d.epistemic[0], 1.0, atol=1e-6)


def test_total_equals_aleatoric_plus_epistemic_and_js():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(10), size=(7, 50))  # 7 members, 50 inputs, 10 classes
    d = decompose_classification(P)
    assert np.allclose(d.total, d.aleatoric + d.epistemic)
    assert np.allclose(d.epistemic, jensen_shannon(P), atol=1e-9)
    assert (d.epistemic >= 0).all() and (d.total <= np.log2(10) + 1e-9).all()


def test_entropy_in_bits():
    assert np.isclose(entropy(np.array([0.25] * 4)), 2.0)


def test_law_of_total_variance():
    rng = np.random.default_rng(1)
    mus, variances = rng.normal(size=(5, 100)), rng.uniform(0.1, 1.0, size=(5, 100))
    d = decompose_regression(mus, variances)
    # variance of the Gaussian mixture computed directly: E[sigma^2 + mu^2] - (E[mu])^2
    mixture_var = (variances + mus ** 2).mean(0) - mus.mean(0) ** 2
    assert np.allclose(d.total_var, mixture_var)
