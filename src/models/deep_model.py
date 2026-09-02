"""Deep Learning probabilistic forecasting baseline (PyTorch DeepAR-style recurrent forecaster)."""

import logging
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


class DeepARNetwork(nn.Module):
    """Recurrent neural network with Gaussian likelihood head for probabilistic forecasting."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = self.lstm(x, hidden)
        mu = self.fc_mu(out)
        sigma = self.softplus(self.fc_sigma(out)) + 1e-4
        return mu, sigma, hidden


class DeepLearningForecaster:
    """Deep Learning probabilistic forecaster based on DeepAR recurrent architecture."""

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 0.005,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_data(
        self,
        context: np.ndarray,
        past_only_covariates: Optional[np.ndarray],
        past_future_covariates: Optional[np.ndarray],
        seq_len: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
        """Construct sliding sequences for recurrent training."""
        context_len = len(context)
        mean_val = float(np.mean(context))
        std_val = float(np.std(context)) + 1e-5
        norm_context = (context - mean_val) / std_val

        features_list = [norm_context[:, np.newaxis]]

        if past_only_covariates is not None:
            po_norm = (
                past_only_covariates
                - np.mean(past_only_covariates, axis=1, keepdims=True)
            ) / (np.std(past_only_covariates, axis=1, keepdims=True) + 1e-5)
            features_list.append(po_norm.T)

        if past_future_covariates is not None:
            pf_norm = past_future_covariates[:, :context_len].T
            features_list.append(pf_norm)

        full_features = np.concatenate(features_list, axis=1).astype(np.float32)
        total_feats = full_features.shape[1]

        x_seqs, y_seqs = [], []
        for i in range(context_len - seq_len):
            x_seqs.append(full_features[i : i + seq_len])
            y_seqs.append(norm_context[i + 1 : i + seq_len + 1, np.newaxis])

        x_tensor = torch.tensor(np.array(x_seqs), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seqs), dtype=torch.float32)

        return x_tensor, y_tensor, total_feats, mean_val, std_val

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        past_only_covariates: Optional[np.ndarray] = None,
        past_future_covariates: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Train DeepAR on context window and autoregressively forecast over horizon."""
        start_time = time.perf_counter()
        x_tensor, y_tensor, input_dim, mean_val, std_val = self._prepare_data(
            context, past_only_covariates, past_future_covariates
        )

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = DeepARNetwork(
            input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                mu, sigma, _ = model(batch_x)
                # Gaussian negative log likelihood loss
                dist = torch.distributions.Normal(mu, sigma)
                loss = -dist.log_prob(batch_y).mean()
                loss.backward()
                optimizer.step()

        # Autoregressive generation over horizon with Monte Carlo sampling
        model.eval()
        num_samples = 100
        samples = np.zeros((num_samples, horizon), dtype=np.float32)

        po_norm = None
        if past_only_covariates is not None:
            po_norm = (
                past_only_covariates
                - np.mean(past_only_covariates, axis=1, keepdims=True)
            ) / (np.std(past_only_covariates, axis=1, keepdims=True) + 1e-5)
            last_po = po_norm[:, -1]
        else:
            last_po = np.empty((0,), dtype=np.float32)

        context_len = len(context)

        with torch.no_grad():
            for s in range(num_samples):
                curr_target = (context[-1] - mean_val) / std_val
                hidden = None

                # Seed hidden state with last context segment
                recent_x = x_tensor[-1:].to(self.device)
                _, _, hidden = model(recent_x)

                for step in range(horizon):
                    feat_components = [np.array([curr_target], dtype=np.float32)]
                    if len(last_po) > 0:
                        feat_components.append(last_po)
                    if past_future_covariates is not None:
                        pf_step = past_future_covariates[:, context_len + step]
                        feat_components.append(pf_step)

                    step_input = (
                        np.concatenate(feat_components)
                        .reshape(1, 1, -1)
                        .astype(np.float32)
                    )
                    step_tensor = torch.tensor(step_input, device=self.device)

                    mu, sigma, hidden = model(step_tensor, hidden)
                    mu_val = mu.item()
                    sigma_val = sigma.item()

                    # Sample from Gaussian predictive distribution
                    sampled_val = np.random.normal(mu_val, sigma_val)
                    # Denormalize
                    samples[s, step] = sampled_val * std_val + mean_val
                    curr_target = sampled_val

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info("DeepAR forecast completed in %.2f ms", elapsed_ms)

        point_pred = np.median(samples, axis=0)
        quantiles_arr = np.percentile(
            samples, [q * 100 for q in self.DEFAULT_QUANTILES], axis=0
        ).T

        return ForecastResult(
            model_name="DeepAR (Deep Learning)",
            point_forecast=point_pred.astype(np.float32),
            quantiles=quantiles_arr.astype(np.float32),
            quantile_levels=self.DEFAULT_QUANTILES,
            inference_time_ms=elapsed_ms,
        )
