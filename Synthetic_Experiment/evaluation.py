"""Metrics and model-agnostic evaluation helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


@dataclass
class PredictiveMetrics:
    log_loss: float
    brier: float
    auroc: float
    accuracy: float
    calibration_intercept: float
    calibration_slope: float


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    scores = logit(clipped).reshape(-1, 1)
    if np.unique(y).size < 2 or np.std(scores) < 1e-12:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e8, solver="lbfgs")
    model.fit(scores, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def predictive_metrics(y: np.ndarray, probability: np.ndarray) -> PredictiveMetrics:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    intercept, slope = _calibration(y, p)
    auc = roc_auc_score(y, p) if np.unique(y).size == 2 else float("nan")
    return PredictiveMetrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        auroc=float(auc),
        accuracy=float(accuracy_score(y, p >= 0.5)),
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def evaluate_fitted_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_mc: np.ndarray,
    true_logit_mc: np.ndarray,
    true_probability_mc: np.ndarray,
    region_mc: np.ndarray,
) -> dict[str, float]:
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = asdict(predictive_metrics(y_test, p_test))
    p_mc = np.clip(model.predict_proba(X_mc)[:, 1], 1e-12, 1 - 1e-12)
    if hasattr(model, "decision_function"):
        predicted_logit = np.asarray(model.decision_function(X_mc), dtype=float).reshape(-1)
    else:
        predicted_logit = logit(p_mc)
    metrics["rmse_logit"] = float(np.sqrt(np.mean((predicted_logit - true_logit_mc) ** 2)))
    metrics["rmse_probability"] = float(
        np.sqrt(np.mean((p_mc - true_probability_mc) ** 2))
    )
    for region in np.unique(region_mc):
        mask = region_mc == region
        metrics[f"rmse_logit_region_{int(region)}"] = float(
            np.sqrt(np.mean((predicted_logit[mask] - true_logit_mc[mask]) ** 2))
        )
    metrics.update({k: float(v) for k, v in model.complexity(X_mc).items()})
    return metrics


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.seconds = time.perf_counter() - self.started
