"""ensemble_uq - uncertainty modelling with ensembles (aleatoric vs epistemic).

Group demo for "Advanced Learning Paradigms for AI", IIT Kharagpur, Autumn 2026.
"""
from .ensembles import DeepEnsemble, GaussianEnsemble, LaplaceRandomForest, MCEnsemble, ParallelMLPEnsemble
from .uncertainty import (Decomposition, RegressionDecomposition, decompose_classification,
                          decompose_regression, entropy, jensen_shannon, member_table)

__all__ = [
    "DeepEnsemble", "GaussianEnsemble", "LaplaceRandomForest", "MCEnsemble", "ParallelMLPEnsemble",
    "Decomposition", "RegressionDecomposition", "decompose_classification", "decompose_regression",
    "entropy", "jensen_shannon", "member_table",
]
