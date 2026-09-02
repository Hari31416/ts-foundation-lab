"""TimesFM-3 Fine-Tuned Model Wrapper for time series forecasting."""

import logging
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch

from src.models.timesfm_model import ForecastResult, TimesFM3ModelWrapper
from timesfm import TimesFM3Forecaster

logger = logging.getLogger(__name__)


class TimesFM3FineTunedWrapper:
    """Wrapper around fine-tuned Google TimesFM-3 checkpoint."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        checkpoint_path: Path,
        pretrained_model_id: str = "google/timesfm-3.0-pytorch",
        device: Optional[str] = None,
    ) -> None:
        """Initialize and load the fine-tuned TimesFM-3 checkpoint.

        Args:
            checkpoint_path: Path to fine-tuned PyTorch state_dict checkpoint.
            pretrained_model_id: HuggingFace model repository ID.
            device: Computing device ('cuda', 'mps', 'cpu', or None for auto).
        """
        self.checkpoint_path = checkpoint_path
        self.model_id = pretrained_model_id
        self.device = device

        logger.info("Initializing base TimesFM-3 model from %s...", self.model_id)
        self.forecaster = TimesFM3Forecaster.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            device=self.device,
        )

        if self.checkpoint_path.exists():
            logger.info(
                "Loading fine-tuned checkpoint weights from %s", self.checkpoint_path
            )
            state_dict = torch.load(
                self.checkpoint_path,
                map_location=self.forecaster.device,
                weights_only=True,
            )
            self.forecaster.model.load_state_dict(state_dict)
            logger.info("Fine-tuned weights successfully loaded.")
        else:
            logger.warning(
                "Fine-tuned checkpoint not found at %s. Running in base mode.",
                self.checkpoint_path,
            )

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Run inference using the fine-tuned model checkpoint."""
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

        logger.info("TimesFM-3 (Fine-Tuned) inference completed in %.2f ms", elapsed_ms)

        return ForecastResult(
            model_name="TimesFM-3 (Fine-Tuned)",
            point_forecast=point_preds,
            quantiles=quantiles_arr,
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
