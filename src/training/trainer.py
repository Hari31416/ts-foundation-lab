"""Fine-tuning trainer for Google TimesFM-3 Foundation Model."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WeatherDatasetLoader
from src.data.timeseries_dataset import WeatherSlidingDataset
from src.evaluation.visualizer import plot_training_curves
from src.training.loss import JointQuantileHuberLoss
from timesfm import TimesFM3Forecaster

logger = logging.getLogger(__name__)


class TimesFM3FineTuningTrainer:
    """Trainer class for fine-tuning TimesFM-3 with strict temporal isolation."""

    def __init__(
        self,
        checkpoint_path: Path,
        pretrained_model_id: str = "google/timesfm-3.0-pytorch",
        lr: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 5,
        device: Optional[str] = None,
        train_layers_from: int = 16,  # Fine-tune top 4 transformer layers + output head
    ) -> None:
        """Initialize fine-tuning trainer."""
        self.checkpoint_path = checkpoint_path
        self.pretrained_model_id = pretrained_model_id
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_layers_from = train_layers_from

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info("Fine-tuning configured on device: %s", self.device)
        self.loss_fn = JointQuantileHuberLoss()

    def train(self) -> pd.DataFrame:
        """Execute full training and validation fine-tuning loop and save loss plots."""
        logger.info(
            "Loading pre-trained TimesFM-3 checkpoint: %s", self.pretrained_model_id
        )
        forecaster = TimesFM3Forecaster.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_id,
            device=str(self.device),
        )
        model = forecaster.model

        # Parameter Freezing Strategy
        # Freeze pre_transformer_resblock and early transformer layers
        for param in model.pre_transformer_resblock.parameters():
            param.requires_grad = False

        for i, layer in enumerate(model.transformer_stack.layers):
            if i < self.train_layers_from:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        for param in model.output_head.parameters():
            param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(
            "TimesFM-3 Model: %d Total Params | %d Trainable Params (%.2f%%)",
            total_params,
            num_trainable,
            (num_trainable / total_params) * 100.0,
        )

        # Prepare isolated datasets
        loader = WeatherDatasetLoader()
        train_splits, val_splits, _ = loader.get_train_val_test_splits(
            train_ratio=0.70, val_ratio=0.09
        )

        train_dataset = WeatherSlidingDataset(
            targets_df=train_splits[0],
            past_only_df=train_splits[1],
            past_future_df=train_splits[2],
            context_length=512,
            horizon=96,
            stride=96,  # Stride of 96 provides ~3,000 diverse training slices
        )
        val_dataset = WeatherSlidingDataset(
            targets_df=val_splits[0],
            past_only_df=val_splits[1],
            past_future_df=val_splits[2],
            context_length=512,
            horizon=96,
            stride=192,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(
            "Datasets prepared: %d Train Windows | %d Val Windows",
            len(train_dataset),
            len(val_dataset),
        )

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        undecorated_decode = model.decode.__wrapped__
        # Disable non-differentiable iterative CPM refinement during gradient descent
        model.use_iterative_cpm_revin = False
        best_val_loss = float("inf")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        history_records = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            model.use_iterative_cpm_revin = False
            total_train_loss = 0.0
            num_batches = 0

            for batch_idx, (ctx, hrz, po, pf) in enumerate(train_loader):
                ctx = ctx.to(self.device)
                hrz = hrz.to(self.device)
                po = po.to(self.device)
                pf = pf.to(self.device)

                optimizer.zero_grad()

                logits = undecorated_decode(
                    model,
                    target=ctx,
                    horizon=96,
                    past_only_covariates=po,
                    past_future_covariates=pf,
                )

                # Target predictions are variate index 0: shape (batch, horizon, num_quantiles)
                pred_quantiles = logits[:, 0, :, :]
                loss = self.loss_fn(pred_quantiles, hrz)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            avg_train_loss = total_train_loss / max(num_batches, 1)

            # Validation pass
            model.eval()
            total_val_loss = 0.0
            val_maes = []
            val_rmses = []
            val_batches = 0

            with torch.no_grad():
                for ctx, hrz, po, pf in val_loader:
                    ctx = ctx.to(self.device)
                    hrz = hrz.to(self.device)
                    po = po.to(self.device)
                    pf = pf.to(self.device)

                    logits = undecorated_decode(
                        model,
                        target=ctx,
                        horizon=96,
                        past_only_covariates=po,
                        past_future_covariates=pf,
                    )
                    pred_quantiles = logits[:, 0, :, :]
                    val_loss = self.loss_fn(pred_quantiles, hrz)
                    total_val_loss += val_loss.item()

                    # Median point prediction is index 4 (0.5 quantile)
                    median_pred = pred_quantiles[:, :, 4]
                    mae = torch.abs(median_pred - hrz).mean().item()
                    rmse = torch.sqrt(torch.mean((median_pred - hrz) ** 2)).item()
                    val_maes.append(mae)
                    val_rmses.append(rmse)
                    val_batches += 1

            avg_val_loss = total_val_loss / max(val_batches, 1)
            avg_val_mae = float(np.mean(val_maes))
            avg_val_rmse = float(np.mean(val_rmses))

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val MAE: %.4f | Val RMSE: %.4f | LR: %.2e",
                epoch,
                self.epochs,
                avg_train_loss,
                avg_val_loss,
                avg_val_mae,
                avg_val_rmse,
                current_lr,
            )

            history_records.append(
                {
                    "Epoch": epoch,
                    "Train_Loss": avg_train_loss,
                    "Val_Loss": avg_val_loss,
                    "Val_MAE": avg_val_mae,
                    "Val_RMSE": avg_val_rmse,
                    "LR": current_lr,
                }
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(
                    "New best validation loss (%.4f). Saving checkpoint to %s",
                    best_val_loss,
                    self.checkpoint_path,
                )
                torch.save(model.state_dict(), self.checkpoint_path)

        # Save training history CSV and plot
        history_df = pd.DataFrame(history_records)
        history_csv_path = self.checkpoint_path.parent / "training_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        logger.info("Saved training history log to %s", history_csv_path)

        plot_path = self.checkpoint_path.parent / "training_loss_curves.png"
        plot_training_curves(history_df=history_df, output_path=plot_path)

        logger.info(
            "TimesFM-3 fine-tuning finished. Best Val Loss: %.4f", best_val_loss
        )
        return history_df
