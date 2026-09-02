"""Fine-tuned Chronos-2 Model Wrapper for evaluation."""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from chronos import Chronos2Pipeline

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class Chronos2FineTunedWrapper:
    """Wrapper around fine-tuned Amazon Chronos-2 checkpoint for benchmark evaluation."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        checkpoint_path: Path,
        device_map: str = "cpu",
    ) -> None:
        """Initialize fine-tuned Chronos-2 pipeline.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint folder.
            device_map: Computing device ('cpu', 'mps', 'cuda', etc.).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device_map = device_map

        logger.info(
            "Loading fine-tuned Chronos-2 pipeline from %s...", self.checkpoint_path
        )
        self.pipeline = Chronos2Pipeline.from_pretrained(
            str(self.checkpoint_path),
            device_map=self.device_map,
            import_allowlist=["chronos.chronos2.model"],
        )
        logger.info("Fine-tuned Chronos-2 pipeline loaded successfully.")

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Run fine-tuned inference for target series with past and future covariates.

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

        point_preds = mean_list[0][0].detach().cpu().numpy().astype(np.float32)
        quantiles_arr = quantiles_list[0][0].detach().cpu().numpy().astype(np.float32)

        logger.info("Chronos-2 Fine-Tuned inference completed in %.2f ms", elapsed_ms)

        return ForecastResult(
            model_name="Chronos-2 (Fine-Tuned)",
            point_forecast=point_preds,
            quantiles=quantiles_arr,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
