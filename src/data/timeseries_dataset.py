"""PyTorch Dataset for sliding-window multi-variate time series training."""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WeatherSlidingDataset(Dataset):
    """Sliding-window dataset generating isolated multivariate time series samples."""

    def __init__(
        self,
        targets_df: pd.DataFrame,
        past_only_df: pd.DataFrame,
        past_future_df: pd.DataFrame,
        context_length: int = 512,
        horizon: int = 96,
        stride: int = 64,
    ) -> None:
        """Initialize dataset.

        Args:
            targets_df: Target variable series (N, 1).
            past_only_df: Historical-only covariates (N, num_past).
            past_future_df: Past and future calendar covariates (N, num_future).
            context_length: Length of context history (512).
            horizon: Forecast horizon length (96).
            stride: Step stride between consecutive window samples.
        """
        self.context_length = context_length
        self.horizon = horizon
        self.total_length = context_length + horizon

        self.targets = targets_df.to_numpy(dtype=np.float32)
        self.past_only = past_only_df.to_numpy(dtype=np.float32)
        self.past_future = past_future_df.to_numpy(dtype=np.float32)

        n_samples = len(self.targets)
        self.valid_start_indices = [
            i for i in range(0, n_samples - self.total_length + 1, stride)
        ]

    def __len__(self) -> int:
        """Return total number of sliding window samples."""
        return len(self.valid_start_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch a single training window.

        Returns:
            Tuple of:
            - context_target: (1, context_length) float32
            - horizon_target: (horizon,) float32
            - past_only: (num_past, context_length) float32
            - past_future: (num_future, context_length + horizon) float32
        """
        start = self.valid_start_indices[idx]
        end = start + self.total_length
        ctx_end = start + self.context_length

        # Target: context is shape (1, context_length), horizon is (horizon,)
        ctx_target = torch.from_numpy(self.targets[start:ctx_end, 0:1].T.copy())
        hrz_target = torch.from_numpy(self.targets[ctx_end:end, 0].copy())

        # Past-only covariates: shape (num_po, context_length)
        po_cov = torch.from_numpy(self.past_only[start:ctx_end].T.copy())

        # Past-future covariates: shape (num_pf, context_length + horizon)
        pf_cov = torch.from_numpy(self.past_future[start:end].T.copy())

        return ctx_target, hrz_target, po_cov, pf_cov
