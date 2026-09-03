"""Unit tests for the Universal Forecasting Engine supporting arbitrary user datasets."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ui.engine import UniversalForecastingEngine, get_column_options


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Generate arbitrary synthetic retail dataframe."""
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sales_amount": np.random.uniform(100, 500, size=150).astype(np.float32),
            "traffic_count": np.random.uniform(50, 200, size=150).astype(np.float32),
            "promo_flag": np.random.binomial(1, 0.2, size=150).astype(np.float32),
        }
    )


def test_get_column_options(sample_sales_df: pd.DataFrame) -> None:
    """Test column detection on arbitrary dataframe."""
    dt_cols, num_cols, def_time, def_target = get_column_options(sample_sales_df)
    assert "date" in dt_cols
    assert "sales_amount" in num_cols
    assert "traffic_count" in num_cols
    assert def_time == "date"
    assert def_target in ["sales_amount", "traffic_count", "promo_flag"]


def test_ui_engine_backtest_forecast(sample_sales_df: pd.DataFrame) -> None:
    """Test UniversalForecastingEngine in backtesting mode on arbitrary data."""
    engine = UniversalForecastingEngine()
    fig, metrics_df, metrics_html, pred_df = engine.run_forecasting(
        df=sample_sales_df,
        target_col="sales_amount",
        timestamp_col="date",
        past_only_cols=["traffic_count"],
        past_future_cols=["promo_flag"],
        context_length=64,
        horizon=16,
        selected_models=["Chronos-2 (Zero-Shot)"],
        backtest_mode=True,
    )

    assert fig is not None
    assert isinstance(metrics_html, str)
    assert "<table" in metrics_html
    assert not metrics_df.empty
    assert "MAE" in metrics_df.columns
    assert len(pred_df) == 16
    assert "Actual" in pred_df.columns
    assert "Chronos_2_Zero_Shot_Point" in pred_df.columns


def test_ui_engine_dirty_csv_handling() -> None:
    """Test column option detection and forecasting with dirty strings (e.g. '?0.2', '$100')."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df_dirty = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Temp": ["?12.4", "15.1", "?14.0", "16.8", "13.2"] * 20,
            "Cost": ["$10.5", "$12.0", "$11.2", "$14.1", "$15.0"] * 20,
        }
    )
    dt_cols, num_cols, def_time, def_target = get_column_options(df_dirty)
    assert def_time == "Date"
    assert "Temp" in num_cols
    assert "Cost" in num_cols

    engine = UniversalForecastingEngine()
    fig, metrics_df, metrics_html, pred_df = engine.run_forecasting(
        df=df_dirty,
        target_col="Temp",
        timestamp_col="Date",
        past_only_cols=["Cost"],
        past_future_cols=[],
        context_length=32,
        horizon=8,
        selected_models=["Chronos-2 (Zero-Shot)"],
        backtest_mode=True,
    )
    assert fig is not None
    assert isinstance(metrics_html, str)
    assert "<table" in metrics_html
    assert len(pred_df) == 8
    assert not metrics_df.empty


def test_ui_engine_lightgbm_and_nan_covariates(
    sample_sales_df: pd.DataFrame,
) -> None:
    """Test LightGBM baseline execution and NaN covariate handling in UniversalForecastingEngine."""
    df_with_nan = sample_sales_df.copy()
    # Inject NaNs into numeric covariate
    df_with_nan.loc[5:15, "traffic_count"] = np.nan

    engine = UniversalForecastingEngine()
    fig, metrics_df, metrics_html, pred_df = engine.run_forecasting(
        df=df_with_nan,
        target_col="sales_amount",
        timestamp_col="date",
        past_only_cols=["traffic_count"],
        past_future_cols=["promo_flag"],
        context_length=64,
        horizon=16,
        selected_models=["LightGBM"],
        backtest_mode=True,
    )
    assert fig is not None
    assert not metrics_df.empty
    assert "LightGBM" in metrics_df["Model"].values
    assert len(pred_df) == 16
    assert "LightGBM_Point" in pred_df.columns
