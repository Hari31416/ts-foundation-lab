"""TimesFM-3 Foundation Model Wrapper for zero-shot time series forecasting."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from timesfm import TimesFM3Forecaster

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Standard container for model forecast results."""

    model_name: str
    point_forecast: np.ndarray  # Shape: (horizon,)
    quantiles: Optional[np.ndarray] = None  # Shape: (horizon, num_quantiles)
    quantile_levels: Optional[list[float]] = None
    inference_time_ms: float = 0.0


class TimesFM3ModelWrapper:
    """Wrapper around Google TimesFM-3 Foundation Model for multivariate zero-shot forecasting."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        pretrained_model_id: str = "google/timesfm-3.0-pytorch",
        device: Optional[str] = None,
    ) -> None:
        """Initialize and load the TimesFM-3 model.

        Args:
            pretrained_model_id: HuggingFace model repository ID.
            device: Computing device ('cuda', 'mps', 'cpu', or None for auto).
        """
        self.model_id = pretrained_model_id
        self.device = device
        logger.info("Initializing TimesFM-3 from checkpoint: %s", self.model_id)
        self.forecaster = TimesFM3Forecaster.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            device=self.device,
        )
        logger.info("TimesFM-3 model loaded successfully.")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Run zero-shot inference for target series with past and future covariates.

        Args:
            context: 1D numpy array of historical target values (context_length,).
            horizon: Forecast horizon (int).
            past_only_covariates: 2D numpy array of shape (num_covariates, context_length).
            past_future_covariates: 2D numpy array of shape (num_covariates, context_length + horizon).

        Returns:
            ForecastResult containing point predictions, 9 quantiles, and latency.
        """
        context_arr = np.asarray(context, dtype=np.float32)
        po_cov = (
            np.asarray(past_only_covariates, dtype=np.float32)
            if past_only_covariates is not None
            else None
        )
        pf_cov = (
            np.asarray(past_future_covariates, dtype=np.float32)
            if past_future_covariates is not None
            else None
        )

        start_time = time.perf_counter()
        output = self.forecaster.predict(
            context=context_arr,
            horizon=horizon,
            past_only_covariates=po_cov,
            past_future_covariates=pf_cov,
            return_quantiles=True,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        point_preds = np.asarray(output.forecast, dtype=np.float32)
        quantiles_arr = (
            np.asarray(output.quantiles, dtype=np.float32)
            if output.quantiles is not None
            else None
        )

        logger.info("TimesFM-3 inference completed in %.2f ms", elapsed_ms)

        return ForecastResult(
            model_name="TimesFM-3 (Zero-Shot)",
            point_forecast=point_preds,
            quantiles=quantiles_arr,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
