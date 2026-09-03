"""Chronos-2 Foundation Model Wrapper for zero-shot multivariate time series forecasting."""

import logging
import time
from typing import List, Optional

import numpy as np
import torch
from chronos import Chronos2Pipeline

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class Chronos2ModelWrapper:
    """Wrapper around Amazon Chronos-2 Foundation Model for zero-shot time series forecasting."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        pretrained_model_id: str = "amazon/chronos-2",
        device_map: Optional[str] = None,
    ) -> None:
        """Initialize Chronos-2 pipeline.

        Args:
            pretrained_model_id: HuggingFace model repository ID.
            device_map: Computing device ('cpu', 'cuda', etc. Defaults to 'cuda' if available, else 'cpu').
        """
        self.model_id = pretrained_model_id
        if device_map is None:
            self.device_map = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_map = device_map

        logger.info(
            "Initializing Chronos-2 pipeline from %s on %s...",
            self.model_id,
            self.device_map,
        )
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device_map,
        )
        logger.info("Chronos-2 pipeline loaded successfully.")

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
        ctx_len = len(context_arr)

        past_covs = {}
        if past_only_covariates is not None:
            for idx, row in enumerate(past_only_covariates):
                past_covs[f"past_cov_{idx}"] = np.asarray(row, dtype=np.float32)

        future_covs = {}
        if past_future_covariates is not None:
            for idx, row in enumerate(past_future_covariates):
                row_arr = np.asarray(row, dtype=np.float32)
                past_covs[f"pf_cov_{idx}"] = row_arr[:ctx_len]
                future_covs[f"pf_cov_{idx}"] = row_arr[ctx_len : ctx_len + horizon]

        input_entry = {"target": context_arr}
        if past_covs:
            input_entry["past_covariates"] = past_covs
        if future_covs:
            input_entry["future_covariates"] = future_covs

        start_time = time.perf_counter()
        quantiles_list, mean_list = self.pipeline.predict_quantiles(
            inputs=[input_entry],
            prediction_length=horizon,
            quantile_levels=self.DEFAULT_QUANTILES,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract predictions for first series & first target variate
        # quantiles_list[0] has shape (1, horizon, 9)
        # mean_list[0] has shape (1, horizon)
        point_preds = mean_list[0][0].detach().cpu().numpy().astype(np.float32)
        quantiles_arr = quantiles_list[0][0].detach().cpu().numpy().astype(np.float32)

        logger.info("Chronos-2 inference completed in %.2f ms", elapsed_ms)

        return ForecastResult(
            model_name="Chronos-2 (Zero-Shot)",
            point_forecast=point_preds,
            quantiles=quantiles_arr,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
