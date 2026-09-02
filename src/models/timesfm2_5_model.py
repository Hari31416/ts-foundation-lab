"""TimesFM-2.5 Foundation Model Wrapper for zero-shot time series forecasting (Apache 2.0)."""

import logging
import time
from typing import Optional

import numpy as np
from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class TimesFM2p5ModelWrapper:
    """Wrapper around Google TimesFM-2.5 Foundation Model (Apache 2.0) for zero-shot forecasting."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        pretrained_model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
        max_horizon: int = 256,
    ) -> None:
        """Initialize and load the TimesFM-2.5 model.

        Args:
            pretrained_model_id: HuggingFace model repository ID (default: google/timesfm-2.5-200m-pytorch).
            max_context: Maximum historical lookback context window.
            max_horizon: Maximum forecast prediction horizon.
        """
        self.model_id = pretrained_model_id
        self.max_context = max_context
        self.max_horizon = max_horizon
        logger.info("Initializing TimesFM-2.5 from checkpoint: %s", self.model_id)
        self.model = TimesFM_2p5_200M_torch.from_pretrained(
            self.model_id, torch_compile=False
        )
        self.model.compile(
            ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                per_core_batch_size=1,
            )
        )
        logger.info("TimesFM-2.5 model compiled successfully.")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Run zero-shot inference for target series.

        Args:
            context: 1D numpy array of historical target values (context_length,).
            horizon: Forecast horizon (int).
            past_only_covariates: Optional 2D numpy array of shape (num_covariates, context_length).
            past_future_covariates: Optional 2D numpy array of shape (num_covariates, context_length + horizon).

        Returns:
            ForecastResult containing point predictions, 9 quantiles, and latency.
        """
        context_arr = np.asarray(context, dtype=np.float32)
        if len(context_arr) > self.max_context:
            context_arr = context_arr[-self.max_context :]

        start_time = time.perf_counter()
        point_preds, quantiles_out = self.model.forecast(
            horizon=horizon, inputs=[context_arr]
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        point = np.asarray(point_preds[0], dtype=np.float32)
        # In TimesFM 2.5, quantiles_out has shape (batch, horizon, 10) where index 1..9 are 10th to 90th percentiles
        quantiles = (
            np.asarray(quantiles_out[0, :, 1:10], dtype=np.float32)
            if quantiles_out is not None and quantiles_out.shape[-1] >= 10
            else None
        )

        logger.info("TimesFM-2.5 inference completed in %.2f ms", elapsed_ms)

        return ForecastResult(
            model_name="TimesFM-2.5 (Zero-Shot)",
            point_forecast=point,
            quantiles=quantiles,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
