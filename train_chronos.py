"""Standalone CLI training script for fine-tuning Amazon Chronos-2 on the Weather dataset."""

import argparse
import logging
from pathlib import Path

from src.training.chronos_trainer import Chronos2FineTuningTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_chronos")


def main() -> None:
    """Parse CLI arguments and run Chronos-2 fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Amazon Chronos-2 on Jena Climate Weather dataset with zero leakage."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of training steps (default: 300).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Fine-tuning mode: 'lora' or 'full' (default: 'lora').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'mps', 'cpu', or auto if None).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results" / "chronos2_finetuned"

    logger.info("=== Starting Chronos-2 Fine-Tuning Pipeline ===")
    logger.info("Output Directory: %s", output_dir)
    logger.info(
        "Hyperparameters: Steps=%d, Batch Size=%d, LR=%.2e, Mode=%s, Device=%s",
        args.steps,
        args.batch_size,
        args.lr,
        args.mode,
        args.device or "auto",
    )

    trainer = Chronos2FineTuningTrainer(
        output_dir=output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_steps=args.steps,
        finetune_mode=args.mode,
        device=args.device,
    )

    ckpt_path = trainer.train()
    logger.info("Fine-tuning pipeline completed. Checkpoint saved at: %s", ckpt_path)


if __name__ == "__main__":
    main()
