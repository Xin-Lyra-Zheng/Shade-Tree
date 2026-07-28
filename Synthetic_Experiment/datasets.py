"""Synthetic data generators specified in the experiment proposal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import expit
from scipy.optimize import brentq


DATASET_NAMES = ("additive", "xor", "logic", "gated", "region")


@dataclass(frozen=True)
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    true_logit: np.ndarray
    true_probability: np.ndarray
    region: np.ndarray
    feature_names: tuple[str, ...]
    dataset: str
    signal_strength: float
    intercept: float


def _additive(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = (
        1.5 * np.sin(np.pi * X[:, 0])
        + 1.2 * (2.0 * X[:, 1] ** 2 - 2.0 / 3.0)
        + np.tanh(3.0 * X[:, 2])
    )
    return f, np.zeros(X.shape[0], dtype=np.int8)


def _xor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xor = (X[:, 0] > 0) != (X[:, 1] > 0)
    f = 2.5 * (2.0 * xor.astype(float) - 1.0)
    return f, xor.astype(np.int8)


def _logic(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0 = (X[:, 0] > 0.2) & (X[:, 1] < -0.3)
    r1 = (X[:, 0] > 0.2) & ~r0
    r2 = (X[:, 0] <= 0.2) & (X[:, 2] > 0.5)
    region = np.select([r0, r1, r2], [0, 1, 2], default=3).astype(np.int8)
    f = np.choose(region, [2.5, -2.0, 1.8, -1.5])
    return f, region


def _gated(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate = (X[:, 0] > 0) & (X[:, 1] > 0)
    f = 0.8 * np.sin(np.pi * X[:, 2])
    f += gate * (2.0 * np.sin(2.0 * np.pi * X[:, 3]) + 1.5 * X[:, 4])
    return f, gate.astype(np.int8)


def _region(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0 = (X[:, 0] > 0) & (X[:, 1] > 0)
    r1 = (X[:, 0] > 0) & ~r0
    r2 = (X[:, 0] <= 0) & (X[:, 3] > 0)
    region = np.select([r0, r1, r2], [0, 1, 2], default=3).astype(np.int8)
    x3 = X[:, 2]
    values = np.vstack(
        [
            2.0 * np.sin(np.pi * x3),
            1.5 * (x3**2 - 1.0 / 3.0),
            -2.0 * x3,
            0.5 * np.tanh(4.0 * x3),
        ]
    )
    f = values[region, np.arange(X.shape[0])]
    return f, region


_FUNCTIONS: dict[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
    "additive": _additive,
    "xor": _xor,
    "logic": _logic,
    "gated": _gated,
    "region": _region,
}


def true_function(dataset: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unscaled true function and true region labels."""
    if dataset not in _FUNCTIONS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_NAMES}.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 5:
        raise ValueError("X must be a two-dimensional array with at least five features.")
    return _FUNCTIONS[dataset](X)


def balanced_intercept(signal: np.ndarray, target_rate: float = 0.5) -> float:
    """Find an intercept whose mean Bernoulli probability equals target_rate."""
    signal = np.asarray(signal, dtype=np.float64)
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must lie strictly between zero and one.")

    def objective(intercept: float) -> float:
        return float(expit(signal + intercept).mean() - target_rate)

    return float(brentq(objective, -50.0, 50.0))


def generate_dataset(
    dataset: str,
    n_samples: int,
    seed: int,
    signal_strength: float = 1.0,
    n_features: int = 10,
    target_rate: float = 0.5,
) -> SyntheticData:
    """Generate one complete dataset without performing a train/test split."""
    if n_samples < 4:
        raise ValueError("n_samples must be at least four.")
    if n_features < 10:
        raise ValueError("The proposal requires at least ten features.")
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    base_function, region = true_function(dataset, X)
    scaled_function = float(signal_strength) * base_function
    intercept = balanced_intercept(scaled_function, target_rate=target_rate)
    true_logit = scaled_function + intercept
    probability = expit(true_logit)
    y = rng.binomial(1, probability).astype(np.int8)
    return SyntheticData(
        X=X,
        y=y,
        true_logit=true_logit,
        true_probability=probability,
        region=region,
        feature_names=tuple(f"X{i}" for i in range(1, n_features + 1)),
        dataset=dataset,
        signal_strength=float(signal_strength),
        intercept=intercept,
    )


def generate_evaluation_set(
    dataset: str,
    n_samples: int,
    seed: int,
    signal_strength: float,
    intercept: float,
    n_features: int = 10,
) -> SyntheticData:
    """Generate a label-free Monte Carlo evaluation set using a fitted intercept."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    base_function, region = true_function(dataset, X)
    true_logit = float(signal_strength) * base_function + float(intercept)
    probability = expit(true_logit)
    y = rng.binomial(1, probability).astype(np.int8)
    return SyntheticData(
        X=X,
        y=y,
        true_logit=true_logit,
        true_probability=probability,
        region=region,
        feature_names=tuple(f"X{i}" for i in range(1, n_features + 1)),
        dataset=dataset,
        signal_strength=float(signal_strength),
        intercept=float(intercept),
    )
