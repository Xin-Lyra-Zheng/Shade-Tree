"""Self-contained CPU implementation of the FastStepFitter API used by V1/V2."""

from __future__ import annotations

import numpy as np


class StepModel:
    def __init__(self, feature_index, edges, coefficients, constant=None):
        self.feature_index = int(feature_index)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.constant = constant

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.constant is not None:
            return np.full(X.shape[0], float(self.constant), dtype=np.float64)
        bins = np.searchsorted(self.edges, X[:, self.feature_index], side="right")
        return self.coefficients[bins]


class FastStepFitter:
    """Fit the best ridge-regularized univariate degree-0 shape function."""

    def __init__(self, X_train_full, min_samples_leaf=50, n_internal_knots=2,
                 alpha=0.1, **kwargs):
        self.X = np.asarray(X_train_full, dtype=np.float64)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_internal_knots = int(n_internal_knots)
        self.alpha = float(alpha)
        self.edges_ = []
        for j in range(self.X.shape[1]):
            lo, hi = np.nanmin(self.X[:, j]), np.nanmax(self.X[:, j])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                self.edges_.append(np.empty(0, dtype=np.float64))
            else:
                self.edges_.append(np.linspace(lo, hi, self.n_internal_knots + 2)[1:-1])

    def _fit_feature(self, idx, residual, weights, feature):
        edges = self.edges_[feature]
        bins = np.searchsorted(edges, self.X[idx, feature], side="right")
        count = edges.size + 1
        sw = np.bincount(bins, weights=weights, minlength=count)
        swr = np.bincount(bins, weights=weights * residual, minlength=count)
        coefficients = swr / np.maximum(sw + self.alpha, 1e-12)
        prediction = coefficients[bins]
        sse = float(np.sum(weights * (residual - prediction) ** 2))
        return sse, StepModel(feature, edges, coefficients)

    def fit(self, leaf_data_indices, full_residuals, full_weights, feature_idx=None):
        idx = np.asarray(leaf_data_indices, dtype=np.int64)
        residual = np.asarray(full_residuals, dtype=np.float64)[idx]
        weights = np.asarray(full_weights, dtype=np.float64)[idx]
        if idx.size == 0:
            return StepModel(-1, [], [], constant=0.0)
        if idx.size < self.min_samples_leaf:
            return StepModel(-1, [], [], constant=np.average(residual, weights=weights))
        features = range(self.X.shape[1]) if feature_idx is None else [int(feature_idx)]
        candidates = [self._fit_feature(idx, residual, weights, j) for j in features]
        return min(candidates, key=lambda item: item[0])[1]


__all__ = ["FastStepFitter", "StepModel"]
