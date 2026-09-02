"""Classical statistical forecasting baseline (AutoARIMA / SARIMAX with exogenous covariates)."""

import logging
import time
from typing import Optional

import numpy as np
import pmdarima as pm
from scipy import stats

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class ClassicalForecaster:
    """Classical statistical model baseline using AutoARIMA with exogenous regressors."""

    def __init__(self, seasonal: bool = False, max_p: int = 3, max_q: int = 3) -> None:
        """Initialize AutoARIMA baseline forecaster.

        Args:
            seasonal: Whether to include seasonal components in ARIMA search.
            max_p: Maximum AR order.
            max_q: Maximum MA order.
        """
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_q = max_q
        self.model = None

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Fit AutoARIMA on historical context and forecast over horizon.

        Args:
            context: 1D historical target series (context_length,).
            horizon: Prediction horizon (int).
            past_only_covariates: 2D array of shape (num_covariates, context_length).
            past_future_covariates: 2D array of shape (num_covariates, context_length + horizon).

        Returns:
            ForecastResult with point forecast, confidence intervals, and latency.
        """
        context_arr = np.asarray(context, dtype=np.float64)
        context_len = len(context_arr)

        # Prepare exogenous features for context and horizon
        # For horizon, only past_future covariates are known in advance
        exog_train = None
        exog_future = None

        if past_future_covariates is not None:
            # Shape: (num_cov, total_len) -> Transpose to (total_len, num_cov)
            pf_cov = np.asarray(past_future_covariates, dtype=np.float64).T
            exog_train = pf_cov[:context_len]
            exog_future = pf_cov[context_len : context_len + horizon]

        start_time = time.perf_counter()
        logger.info("Fitting AutoARIMA on context length %d...", context_len)

        try:
            self.model = pm.auto_arima(
                y=context_arr,
                X=exog_train,
                seasonal=self.seasonal,
                max_p=self.max_p,
                max_q=self.max_q,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
            logger.info("AutoARIMA selected order: %s", self.model.order)
            point_pred, conf_int = self.model.predict(
                n_periods=horizon,
                X=exog_future,
                return_conf_int=True,
                alpha=0.2,  # 80% confidence interval (approx 10th to 90th percentile)
            )
        except Exception as exc:
            logger.warning(
                "AutoARIMA fitting failed with exog, falling back to simple ARIMA: %s",
                exc,
            )
            self.model = pm.auto_arima(
                y=context_arr,
                seasonal=False,
                max_p=2,
                max_q=2,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
            point_pred, conf_int = self.model.predict(
                n_periods=horizon,
                return_conf_int=True,
                alpha=0.2,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info("AutoARIMA forecast completed in %.2f ms", elapsed_ms)

        # Estimate quantiles from forecast mean and std
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        std_est = (upper_bound - lower_bound) / (2 * stats.norm.ppf(0.9))
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles_arr = np.zeros((horizon, len(quantile_levels)), dtype=np.float32)
        for i, q in enumerate(quantile_levels):
            quantiles_arr[:, i] = point_pred + stats.norm.ppf(q) * std_est

        return ForecastResult(
            model_name="AutoARIMA",
            point_forecast=np.asarray(point_pred, dtype=np.float32),
            quantiles=quantiles_arr,
            quantile_levels=quantile_levels,
            inference_time_ms=elapsed_ms,
        )
