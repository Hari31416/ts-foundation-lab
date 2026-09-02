"""Standalone pipeline script for fine-tuning Google TimesFM-3 on the Jena Weather dataset."""

import argparse
import logging
from pathlib import Path

from src.training.trainer import TimesFM3FineTuningTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_timesfm")


def main() -> None:
    """Parse CLI arguments and run TimesFM-3 fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune TimesFM-3 on the Jena Climate Weather dataset with zero leakage."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--train-layers-from",
        type=int,
        default=16,
        help="Transformer layer index from which to unfreeze weights (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'mps', 'cpu', or auto if None).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "results" / "timesfm_finetuned_checkpoint.pt"

    logger.info("=== Starting TimesFM-3 Fine-Tuning Pipeline ===")
    logger.info("Output Checkpoint: %s", checkpoint_path)
    logger.info(
        "Hyperparameters: Epochs=%d, Batch Size=%d, LR=%.2e, Train Layers From=%d",
        args.epochs,
        args.batch_size,
        args.lr,
        args.train_layers_from,
    )

    trainer = TimesFM3FineTuningTrainer(
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_layers_from=args.train_layers_from,
        device=args.device,
    )

    history_df = trainer.train()
    history_csv_path = checkpoint_path.parent / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    logger.info("\n=== Training Complete ===\n%s\n", history_df.to_string(index=False))
    logger.info(
        "Saved model checkpoint to %s",
        checkpoint_path,
    )
    logger.info(
        "Saved training history metrics CSV to %s",
        history_csv_path,
    )
    logger.info(
        "Saved loss and performance plots to %s",
        checkpoint_path.parent / "training_loss_curves.png",
    )


if __name__ == "__main__":
    main()
