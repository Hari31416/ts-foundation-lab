"""Evaluation metrics module for time series forecasting benchmark."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    model_name: str
    mae: float
    rmse: float
    wape: float
    crps: Optional[float]
    coverage_80: Optional[
        float
    ]  # Empirical coverage between 10th and 90th percentile (target = 0.80)
    avg_interval_width: Optional[float]
    inference_time_ms: float


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error."""
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / denom)


def compute_pinball_loss(
    actual: np.ndarray, pred_quantiles: np.ndarray, quantile_levels: List[float]
) -> float:
    """Compute average Pinball / Quantile Loss across all quantile levels."""
    losses = []
    for i, q in enumerate(quantile_levels):
        q_pred = pred_quantiles[:, i]
        err = actual - q_pred
        loss = np.maximum(q * err, (q - 1) * err)
        losses.append(np.mean(loss))
    return float(np.mean(losses))


def compute_crps(
    actual: np.ndarray, pred_quantiles: np.ndarray, quantile_levels: List[float]
) -> float:
    """Approximate Continuous Ranked Probability Score (CRPS) via quantile integration."""
    return 2.0 * compute_pinball_loss(actual, pred_quantiles, quantile_levels)


def compute_interval_coverage(
    actual: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray
) -> tuple[float, float]:
    """Compute empirical coverage percentage and average interval width."""
    in_interval = (actual >= q_lower) & (actual <= q_upper)
    coverage = float(np.mean(in_interval))
    avg_width = float(np.mean(q_upper - q_lower))
    return coverage, avg_width


def evaluate_forecast(
    model_name: str,
    actual: np.ndarray,
    point_pred: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
    quantile_levels: Optional[List[float]] = None,
    inference_time_ms: float = 0.0,
) -> EvaluationMetrics:
    """Calculate full suite of point and probabilistic metrics for a forecast."""
    mae = compute_mae(actual, point_pred)
    rmse = compute_rmse(actual, point_pred)
    wape = compute_wape(actual, point_pred)

    crps = None
    coverage_80 = None
    avg_width = None

    if quantiles is not None and quantile_levels is not None:
        crps = compute_crps(actual, quantiles, quantile_levels)

        # 10th and 90th percentile indices
        if 0.1 in quantile_levels and 0.9 in quantile_levels:
            idx_10 = quantile_levels.index(0.1)
            idx_90 = quantile_levels.index(0.9)
            coverage_80, avg_width = compute_interval_coverage(
                actual, quantiles[:, idx_10], quantiles[:, idx_90]
            )

    return EvaluationMetrics(
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        wape=wape,
        crps=crps,
        coverage_80=coverage_80,
        avg_interval_width=avg_width,
        inference_time_ms=inference_time_ms,
    )
