"""Tree-based Gradient Boosting baseline using Gradient Boosted Trees with lag, rolling, and covariate features."""

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class GradientBoostingForecaster:
    """Gradient Boosting multi-step time series forecaster with quantile uncertainty estimation."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
    ) -> None:
        """Initialize Gradient Boosting forecaster.

        Args:
            lags: Target lag steps to extract.
            rolling_windows: Window sizes for rolling mean and std.
            n_estimators: Number of boosting trees.
            learning_rate: Boosting learning rate.
        """
        self.lags = lags or [1, 2, 3, 6, 12, 24, 48]
        self.rolling_windows = rolling_windows or [6, 12, 24]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_lookback = max(self.lags + self.rolling_windows)

    def _build_tabular_dataset(
        self,
        context: np.ndarray,
        past_only_covariates: Optional[np.ndarray],
        past_future_covariates: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert context time series into tabular features and target labels."""
        n = len(context)
        feature_rows = []
        target_vals = []

        po_trans = past_only_covariates.T if past_only_covariates is not None else None
        pf_trans = (
            past_future_covariates.T if past_future_covariates is not None else None
        )

        for t in range(self.max_lookback, n):
            row = []
            # Lag features
            for lag in self.lags:
                row.append(context[t - lag])

            # Rolling statistics
            for w in self.rolling_windows:
                window_slice = context[t - w : t]
                row.append(float(np.mean(window_slice)))
                row.append(float(np.std(window_slice)))

            # Past-only covariates at time t-1
            if po_trans is not None and t - 1 < len(po_trans):
                row.extend(po_trans[t - 1].tolist())

            # Past-future covariates at time t
            if pf_trans is not None and t < len(pf_trans):
                row.extend(pf_trans[t].tolist())

            feature_rows.append(row)
            target_vals.append(context[t])

        return np.asarray(feature_rows, dtype=np.float32), np.asarray(
            target_vals, dtype=np.float32
        )

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Train Gradient Boosting model on context features and recursively forecast 96 steps ahead.

        Args:
            context: 1D context array (context_length,).
            horizon: Forecast horizon (int).
            past_only_covariates: 2D array of shape (num_cov, context_length).
            past_future_covariates: 2D array of shape (num_cov, context_length + horizon).

        Returns:
            ForecastResult containing point predictions, quantiles, and latency.
        """
        start_time = time.perf_counter()
        x_train, y_train = self._build_tabular_dataset(
            context, past_only_covariates, past_future_covariates
        )

        # Train mean regressor using fast histogram gradient boosting
        model_mean = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        model_mean.fit(x_train, y_train)

        # Train 10th and 90th percentile quantile regressors for uncertainty estimation
        model_q10 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.1,
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        model_q10.fit(x_train, y_train)

        model_q90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.9,
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        model_q90.fit(x_train, y_train)

        # Recursive autoregressive forecasting over horizon
        curr_context = list(context)
        context_len = len(context)
        predictions = []
        q10_preds = []
        q90_preds = []

        po_trans = past_only_covariates.T if past_only_covariates is not None else None
        pf_trans = (
            past_future_covariates.T if past_future_covariates is not None else None
        )
        last_po = po_trans[-1].tolist() if po_trans is not None else None

        for step in range(horizon):
            t_curr = len(curr_context)
            row = []
            for lag in self.lags:
                row.append(curr_context[t_curr - lag])

            for w in self.rolling_windows:
                window_slice = curr_context[t_curr - w : t_curr]
                row.append(float(np.mean(window_slice)))
                row.append(float(np.std(window_slice)))

            if last_po is not None:
                row.extend(last_po)

            if pf_trans is not None:
                row.extend(pf_trans[context_len + step].tolist())

            feat_arr = np.asarray([row], dtype=np.float32)
            y_hat = float(model_mean.predict(feat_arr)[0])
            q10_val = float(model_q10.predict(feat_arr)[0])
            q90_val = float(model_q90.predict(feat_arr)[0])

            predictions.append(y_hat)
            q10_preds.append(q10_val)
            q90_preds.append(q90_val)
            curr_context.append(y_hat)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info("Gradient Boosting forecast completed in %.2f ms", elapsed_ms)

        # Construct quantile array across 10th to 90th percentiles
        pred_arr = np.asarray(predictions, dtype=np.float32)
        q10_arr = np.asarray(q10_preds, dtype=np.float32)
        q90_arr = np.asarray(q90_preds, dtype=np.float32)

        quantiles_arr = np.zeros(
            (horizon, len(self.DEFAULT_QUANTILES)), dtype=np.float32
        )
        for i, q in enumerate(self.DEFAULT_QUANTILES):
            if q == 0.5:
                quantiles_arr[:, i] = pred_arr
            elif q < 0.5:
                alpha_ratio = (q - 0.1) / 0.4
                quantiles_arr[:, i] = q10_arr + alpha_ratio * (pred_arr - q10_arr)
            else:
                alpha_ratio = (q - 0.5) / 0.4
                quantiles_arr[:, i] = pred_arr + alpha_ratio * (q90_arr - pred_arr)

        return ForecastResult(
            model_name="Gradient Boosting (Trees)",
            point_forecast=pred_arr,
            quantiles=quantiles_arr,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )


# Alias for LightGBM compatibility
LightGBMForecaster = GradientBoostingForecaster
