"""Multi-quantile pinball and Huber loss functions for time series forecasting."""

from typing import List
import torch
import torch.nn as nn


class JointQuantileHuberLoss(nn.Module):
    """Joint loss combining Pinball / Quantile loss across 9 percentiles and Huber loss on the median."""

    def __init__(
        self,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        huber_weight: float = 0.5,
        huber_beta: float = 1.0,
    ) -> None:
        """Initialize loss module."""
        super().__init__()
        self.quantile_levels = quantile_levels
        self.huber_weight = huber_weight
        self.huber_beta = huber_beta
        self.median_idx = (
            quantile_levels.index(0.5)
            if 0.5 in quantile_levels
            else len(quantile_levels) // 2
        )

    def forward(
        self, pred_quantiles: torch.Tensor, actual: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint pinball and Huber loss.

        Args:
            pred_quantiles: Tensor of shape (batch, horizon, num_quantiles).
            actual: Tensor of shape (batch, horizon).

        Returns:
            Scalar loss tensor.
        """
        pred_quantiles = pred_quantiles.contiguous()
        actual = actual.contiguous()
        batch, horizon, n_q = pred_quantiles.shape
        actual_expanded = actual.unsqueeze(-1)  # (batch, horizon, 1)

        pinball_losses = []
        for i, q in enumerate(self.quantile_levels):
            q_pred = pred_quantiles[:, :, i : i + 1]
            diff = actual_expanded - q_pred
            pinball_q = torch.maximum(q * diff, (q - 1.0) * diff)
            pinball_losses.append(pinball_q.mean())

        total_pinball = torch.stack(pinball_losses).mean()

        median_pred = pred_quantiles[:, :, self.median_idx].contiguous()
        huber_loss = torch.nn.functional.smooth_l1_loss(
            median_pred, actual, beta=self.huber_beta
        )

        return total_pinball + self.huber_weight * huber_loss
