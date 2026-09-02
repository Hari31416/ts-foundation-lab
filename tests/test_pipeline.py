"""Unit tests for the benchmarking pipeline, models, and evaluation metrics."""

import numpy as np
import pytest

from src.data.dataset import BenchmarkWindow
from src.evaluation.metrics import (
    compute_crps,
    compute_interval_coverage,
    compute_mae,
    compute_pinball_loss,
    compute_rmse,
    compute_wape,
    evaluate_forecast,
)
from src.models.classical_model import ClassicalForecaster
from src.models.deep_model import DeepLearningForecaster
from src.models.tree_model import LightGBMForecaster


@pytest.fixture
def mock_benchmark_window() -> BenchmarkWindow:
    """Fixture providing synthetic benchmark window."""
    rng = np.random.default_rng(42)
    context_len = 512
    horizon_len = 96
    num_past = 4
    num_future = 6

    # Synthesize smooth sinusoidal context + horizon
    t = np.linspace(0, 10 * np.pi, context_len + horizon_len)
    full_target = (
        15.0 + 5.0 * np.sin(t) + rng.normal(0, 0.2, size=len(t)).astype(np.float32)
    )

    context_target = full_target[:context_len]
    horizon_target = full_target[context_len:]

    past_only_context = rng.normal(100.0, 10.0, size=(num_past, context_len)).astype(
        np.float32
    )
    past_future_full = rng.normal(
        0.0, 1.0, size=(num_future, context_len + horizon_len)
    ).astype(np.float32)

    timestamps = pd_date_range = np.arange(context_len + horizon_len)

    return BenchmarkWindow(
        context_target=context_target,
        horizon_target=horizon_target,
        past_only_context=past_only_context,
        past_future_full=past_future_full,
        timestamps_context=timestamps[:context_len],
        timestamps_horizon=timestamps[context_len:],
        target_name="T (degC)",
        past_only_names=[f"past_cov_{i}" for i in range(num_past)],
        past_future_names=[f"future_cov_{i}" for i in range(num_future)],
    )


def test_metric_computations() -> None:
    """Verify precision and correctness of evaluation metrics."""
    actual = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    pred = np.array([12.0, 18.0, 33.0], dtype=np.float32)

    mae = compute_mae(actual, pred)
    rmse = compute_rmse(actual, pred)
    wape = compute_wape(actual, pred)

    assert pytest.approx(mae, rel=1e-3) == (2.0 + 2.0 + 3.0) / 3.0
    assert pytest.approx(rmse, rel=1e-3) == np.sqrt((4.0 + 4.0 + 9.0) / 3.0)
    assert pytest.approx(wape, rel=1e-3) == 7.0 / 60.0


def test_coverage_and_crps() -> None:
    """Verify uncertainty interval coverage and CRPS calculation."""
    actual = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    q10 = np.array([8.0, 15.0, 25.0], dtype=np.float32)
    q90 = np.array([12.0, 25.0, 35.0], dtype=np.float32)

    cov, width = compute_interval_coverage(actual, q10, q90)
    assert cov == 1.0
    assert pytest.approx(width, rel=1e-3) == (4.0 + 10.0 + 10.0) / 3.0

    quantiles = np.column_stack([q10, q90])
    crps = compute_crps(actual, quantiles, [0.1, 0.9])
    assert crps > 0.0


def test_lightgbm_forecaster(mock_benchmark_window: BenchmarkWindow) -> None:
    """Test LightGBM model fitting and multi-step inference."""
    forecaster = LightGBMForecaster(n_estimators=10, learning_rate=0.1)
    result = forecaster.forecast(
        context=mock_benchmark_window.context_target,
        horizon=96,
        past_only_covariates=mock_benchmark_window.past_only_context,
        past_future_covariates=mock_benchmark_window.past_future_full,
    )
    assert result.point_forecast.shape == (96,)
    assert result.quantiles is not None
    assert result.quantiles.shape == (96, 9)
    assert result.inference_time_ms > 0.0


def test_deepar_forecaster(mock_benchmark_window: BenchmarkWindow) -> None:
    """Test DeepAR model fitting and sampling output."""
    forecaster = DeepLearningForecaster(
        hidden_dim=16, num_layers=1, epochs=2, batch_size=16
    )
    result = forecaster.forecast(
        context=mock_benchmark_window.context_target,
        horizon=96,
        past_only_covariates=mock_benchmark_window.past_only_context,
        past_future_covariates=mock_benchmark_window.past_future_full,
    )
    assert result.point_forecast.shape == (96,)
    assert result.quantiles is not None
    assert result.quantiles.shape == (96, 9)


def test_classical_forecaster(mock_benchmark_window: BenchmarkWindow) -> None:
    """Test AutoARIMA baseline forecasting."""
    forecaster = ClassicalForecaster(seasonal=False, max_p=1, max_q=1)
    result = forecaster.forecast(
        context=mock_benchmark_window.context_target,
        horizon=96,
        past_only_covariates=mock_benchmark_window.past_only_context,
        past_future_covariates=mock_benchmark_window.past_future_full,
    )
    assert result.point_forecast.shape == (96,)
    assert result.quantiles is not None
    assert result.quantiles.shape == (96, 9)
