"""Fine-tuning trainer for Amazon Chronos-2 Foundation Model."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from src.data.dataset import WeatherDatasetLoader

logger = logging.getLogger(__name__)


def prepare_chronos_dataset_splits(
    targets_df: pd.DataFrame,
    past_only_df: pd.DataFrame,
    past_future_df: pd.DataFrame,
    stride: int = 128,
    context_length: int = 512,
    horizon: int = 96,
) -> List[Dict]:
    """Convert partition dataframes into list of sliding-window dictionary entries for Chronos-2.

    Args:
        targets_df: Target temperature DataFrame.
        past_only_df: Past-only meteorological covariates.
        past_future_df: Cyclical calendar features.
        stride: Step stride between consecutive window slices.
        context_length: Historical context length.
        horizon: Forecast horizon length.

    Returns:
        List of dicts formatted with 'target', 'past_covariates', 'future_covariates'.
    """
    total_len = context_length + horizon
    target_vals = targets_df.iloc[:, 0].to_numpy(dtype=np.float32)
    past_cols = past_only_df.columns.tolist()
    pf_cols = past_future_df.columns.tolist()

    n_samples = len(target_vals)
    entries: List[Dict] = []

    for start_idx in range(0, n_samples - total_len + 1, stride):
        ctx_end = start_idx + context_length
        end_idx = start_idx + total_len

        ctx_target = target_vals[start_idx:ctx_end]
        past_covs = {}

        # Past-only covariates for historical context
        for col in past_cols:
            past_covs[col] = (
                past_only_df[col].iloc[start_idx:ctx_end].to_numpy(dtype=np.float32)
            )

        # Past-future covariates (past portion in past_covs, future portion in future_covs)
        future_covs = {}
        for col in pf_cols:
            past_covs[col] = (
                past_future_df[col].iloc[start_idx:ctx_end].to_numpy(dtype=np.float32)
            )
            future_covs[col] = (
                past_future_df[col].iloc[ctx_end:end_idx].to_numpy(dtype=np.float32)
            )

        entries.append(
            {
                "target": ctx_target,
                "past_covariates": past_covs,
                "future_covariates": future_covs,
            }
        )

    return entries


class Chronos2FineTuningTrainer:
    """Trainer class for fine-tuning Chronos-2 using Hugging Face PEFT LoRA."""

    def __init__(
        self,
        output_dir: Path,
        pretrained_model_id: str = "amazon/chronos-2",
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        num_steps: int = 300,
        finetune_mode: str = "lora",
        device: Optional[str] = None,
    ) -> None:
        """Initialize Chronos-2 trainer.

        Args:
            output_dir: Destination path for saving fine-tuned checkpoint.
            pretrained_model_id: Base Hugging Face model repository.
            learning_rate: Optimizer learning rate.
            batch_size: Per-device batch size.
            num_steps: Number of optimizer steps.
            finetune_mode: 'lora' or 'full'.
            device: Computing device ('cuda', 'mps', 'cpu', or None for auto).
        """
        self.output_dir = output_dir
        self.pretrained_model_id = pretrained_model_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.finetune_mode = finetune_mode

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("Chronos-2 fine-tuning configured on device: %s", self.device)

    def train(self) -> Path:
        """Execute Chronos-2 fine-tuning on isolated training partition."""
        logger.info(
            "Initializing base Chronos-2 pipeline from %s on %s...",
            self.pretrained_model_id,
            self.device,
        )
        pipeline = Chronos2Pipeline.from_pretrained(
            self.pretrained_model_id,
            device_map=self.device,
        )

        logger.info("Preparing zero-leakage training and validation partitions...")
        loader = WeatherDatasetLoader()
        train_splits, val_splits, _ = loader.get_train_val_test_splits(
            train_ratio=0.70, val_ratio=0.09
        )

        train_entries = prepare_chronos_dataset_splits(
            targets_df=train_splits[0],
            past_only_df=train_splits[1],
            past_future_df=train_splits[2],
            stride=128,
            context_length=512,
            horizon=96,
        )
        val_entries = prepare_chronos_dataset_splits(
            targets_df=val_splits[0],
            past_only_df=val_splits[1],
            past_future_df=val_splits[2],
            stride=256,
            context_length=512,
            horizon=96,
        )

        logger.info(
            "Chronos-2 Datasets: %d Train Windows | %d Val Windows",
            len(train_entries),
            len(val_entries),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = "checkpoint-final"

        logger.info(
            "Launching Chronos-2 fine-tuning (Mode: %s, Steps: %d, Batch: %d, LR: %.2e)...",
            self.finetune_mode,
            self.num_steps,
            self.batch_size,
            self.learning_rate,
        )

        finetuned_pipeline = pipeline.fit(
            inputs=train_entries,
            validation_inputs=val_entries,
            prediction_length=96,
            context_length=512,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_steps=self.num_steps,
            finetune_mode=self.finetune_mode,
            output_dir=self.output_dir,
            finetuned_ckpt_name=ckpt_name,
        )

        final_ckpt_path = self.output_dir / ckpt_name
        logger.info(
            "Chronos-2 fine-tuning completed successfully. Saved to %s", final_ckpt_path
        )

        # Parse trainer state history and plot curves
        try:
            import json
            from src.evaluation.visualizer import plot_chronos_training_curves

            state_file = (
                self.output_dir / f"checkpoint-{self.num_steps}" / "trainer_state.json"
            )
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)

                train_losses, eval_losses, lrs = {}, {}, {}
                for entry in state.get("log_history", []):
                    s = entry.get("step")
                    if "loss" in entry:
                        train_losses[s] = entry["loss"]
                        lrs[s] = entry.get("learning_rate")
                    if "eval_loss" in entry:
                        eval_losses[s] = entry["eval_loss"]

                steps = sorted(
                    list(set(list(train_losses.keys()) + list(eval_losses.keys())))
                )
                records = [
                    {
                        "Step": s,
                        "Train_Loss": train_losses.get(s),
                        "Eval_Loss": eval_losses.get(s),
                        "LR": lrs.get(s),
                    }
                    for s in steps
                ]
                history_df = pd.DataFrame(records)
                csv_path = self.output_dir.parent / "chronos_training_history.csv"
                history_df.to_csv(csv_path, index=False)
                plot_path = self.output_dir.parent / "chronos_training_loss_curves.png"
                plot_chronos_training_curves(history_df, plot_path)
                logger.info("Saved Chronos-2 training loss curves to %s", plot_path)
        except Exception as e:
            logger.warning("Could not auto-generate Chronos training curves: %s", e)

        return final_ckpt_path
