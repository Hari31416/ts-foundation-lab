"""Visualization module for benchmark forecasting results and uncertainty intervals."""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import BenchmarkWindow
from src.models.timesfm_model import ForecastResult

logger = logging.getLogger(__name__)


DEFAULT_COLORS: Dict[str, str] = {
    "TimesFM-3 (Zero-Shot)": "#2563eb",
    "TimesFM-2.5 (Zero-Shot)": "#0284c7",
    "Chronos-2 (Zero-Shot)": "#ea580c",
    "TimesFM-3 (Fine-Tuned)": "#7c3aed",
    "Chronos-2 (Fine-Tuned)": "#dc2626",
    "LightGBM": "#16a34a",
    "AutoARIMA": "#d97706",
    "DeepAR (Deep Learning)": "#0891b2",
}

MODEL_LINESTYLES: Dict[str, str] = {
    "TimesFM-3 (Zero-Shot)": "-",
    "TimesFM-2.5 (Zero-Shot)": "-",
    "Chronos-2 (Zero-Shot)": "-",
    "TimesFM-3 (Fine-Tuned)": "-",
    "Chronos-2 (Fine-Tuned)": "-",
    "LightGBM": "-.",
    "AutoARIMA": ":",
    "DeepAR (Deep Learning)": "--",
}


def plot_benchmark_comparison(
    window: BenchmarkWindow,
    results: Dict[str, ForecastResult],
    output_path: Path,
    show_context_tail: int = 288,
    window_label: Optional[str] = None,
) -> None:
    """Generate visual plot of Ground Truth vs all evaluated models with uncertainty intervals.

    Args:
        window: BenchmarkWindow with ground truth context and horizon.
        results: Dictionary mapping model names to ForecastResult objects.
        output_path: Output filepath for saving the plot image.
        show_context_tail: Number of historical context points to show before cutoff.
        window_label: Optional title label describing the window (e.g., 'Window 01: 2015-03-12').
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ctx_tail = window.context_target[-show_context_tail:]
    ctx_times = window.timestamps_context[-show_context_tail:]
    hrz_times = window.timestamps_horizon
    actual_hrz = window.horizon_target

    # Top Plot: Forecasts vs Ground Truth
    ax1.plot(
        ctx_times, ctx_tail, color="#475569", label="Historical Context", linewidth=1.5
    )
    ax1.plot(
        hrz_times,
        actual_hrz,
        color="#0f172a",
        label="Ground Truth Actuals",
        linewidth=2.2,
        linestyle="--",
    )

    # Draw vertical separator at forecast cutoff
    ax1.axvline(
        x=ctx_times[-1],
        color="#94a3b8",
        linestyle=":",
        linewidth=1.5,
        label="Forecast Cutoff",
    )

    # Plot each model in results dynamically
    for model_name, res in results.items():
        color = DEFAULT_COLORS.get(model_name, "#64748b")
        linestyle = MODEL_LINESTYLES.get(model_name, "-")
        linewidth = (
            2.2
            if "Fine-Tuned" in model_name
            else (2.0 if "TimesFM" in model_name or "Chronos" in model_name else 1.8)
        )
        ax1.plot(
            hrz_times,
            res.point_forecast,
            color=color,
            label=model_name,
            linewidth=linewidth,
            linestyle=linestyle,
        )

        if res.quantiles is not None and res.quantile_levels is not None:
            if 0.1 in res.quantile_levels and 0.9 in res.quantile_levels:
                idx_10 = res.quantile_levels.index(0.1)
                idx_90 = res.quantile_levels.index(0.9)
                alpha = 0.20 if "Fine-Tuned" in model_name else 0.15
                ax1.fill_between(
                    hrz_times,
                    res.quantiles[:, idx_10],
                    res.quantiles[:, idx_90],
                    color=color,
                    alpha=alpha,
                    label=f"{model_name} 80% CI",
                )

    main_title = (
        f"{window_label} - Forecast Benchmark (Horizon = 96)"
        if window_label
        else "TimesFM-3 vs Baseline Models: Weather Forecast Benchmark (Horizon = 96)"
    )
    ax1.set_title(
        main_title,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel(f"Target: {window.target_name}", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Absolute Residuals (|Actual - Predicted|)
    for model_name, res in results.items():
        residuals = np.abs(actual_hrz - res.point_forecast)
        color = DEFAULT_COLORS.get(model_name, "#64748b")
        ax2.plot(
            hrz_times,
            residuals,
            label=f"{model_name} Error",
            color=color,
            linewidth=1.5,
        )

    ax2.set_title(
        "Forecast Absolute Error (|Actual - Predicted|)", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Timestamp", fontsize=11)
    ax2.set_ylabel("Absolute Error", fontsize=11)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forecast comparison plot to %s", output_path)


def plot_rolling_benchmark_summary(
    df_details: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate multi-panel summary figure aggregating all rolling evaluation windows.

    Args:
        df_details: DataFrame containing per-window, per-model evaluation metrics.
        df_summary: DataFrame containing aggregated cross-window metrics.
        output_path: Output image destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_mae, ax_rmse = axes[0, 0], axes[0, 1]
    ax_time, ax_crps = axes[1, 0], axes[1, 1]

    colors = DEFAULT_COLORS

    models = df_details["Model"].unique().tolist()
    windows = df_details["Window"].unique().tolist()
    x_indices = np.arange(len(windows))

    # Panel 1: MAE by Window
    bar_width = 0.2
    for i, model in enumerate(models):
        sub = df_details[df_details["Model"] == model]
        ax_mae.plot(
            x_indices,
            sub["MAE"].values,
            marker="o",
            linewidth=2.0,
            label=model,
            color=colors.get(model, "#475569"),
        )
    ax_mae.set_title(
        "Mean Absolute Error (MAE) Across Rolling Windows",
        fontsize=13,
        fontweight="bold",
    )
    ax_mae.set_xlabel("Evaluation Window Index", fontsize=11)
    ax_mae.set_ylabel("MAE (lower is better)", fontsize=11)
    ax_mae.set_xticks(x_indices)
    ax_mae.set_xticklabels([f"W{w}" for w in windows], rotation=0)
    ax_mae.legend(loc="upper right", frameon=True, fontsize=10)
    ax_mae.grid(True, alpha=0.3)

    # Panel 2: Aggregate RMSE Mean & Std Comparison (Bar Chart)
    summary_models = df_summary["Model"].tolist()
    rmse_means = df_summary["RMSE_Mean"].astype(float).tolist()
    rmse_stds = df_summary["RMSE_Std"].astype(float).tolist()
    bar_colors = [colors.get(m, "#475569") for m in summary_models]

    bars = ax_rmse.bar(
        summary_models,
        rmse_means,
        yerr=rmse_stds,
        capsize=5,
        color=bar_colors,
        alpha=0.85,
        edgecolor="#1e293b",
    )
    for bar, mean_val in zip(bars, rmse_means):
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean_val * 0.5,
            f"{mean_val:.3f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )
    ax_rmse.set_title(
        "Overall Aggregate RMSE (Mean ± Std)", fontsize=13, fontweight="bold"
    )
    ax_rmse.set_ylabel("RMSE (degC)", fontsize=11)
    ax_rmse.set_xticks(range(len(summary_models)))
    ax_rmse.set_xticklabels(summary_models, rotation=15, ha="right")
    ax_rmse.grid(True, alpha=0.3)

    # Panel 3: Inference Latency Comparison (Log Scale)
    lat_means = df_summary["Latency_Mean_ms"].astype(float).tolist()
    ax_time.bar(
        summary_models,
        lat_means,
        color=bar_colors,
        alpha=0.85,
        edgecolor="#1e293b",
    )
    ax_time.set_yscale("log")
    for idx, (m, lat) in enumerate(zip(summary_models, lat_means)):
        ax_time.text(
            idx,
            lat * 1.2,
            f"{lat:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax_time.set_title(
        "Inference Latency per Horizon=96 (Log Scale)", fontsize=13, fontweight="bold"
    )
    ax_time.set_ylabel("Latency in ms (log scale)", fontsize=11)
    ax_time.set_xticks(range(len(summary_models)))
    ax_time.set_xticklabels(summary_models, rotation=15, ha="right")
    ax_time.grid(True, alpha=0.3)

    # Panel 4: Probabilistic CRPS / Uncertainty Evaluation
    prob_sub = df_details.dropna(subset=["CRPS"])
    if not prob_sub.empty:
        for model in prob_sub["Model"].unique():
            sub = prob_sub[prob_sub["Model"] == model]
            ax_crps.plot(
                range(len(sub)),
                sub["CRPS"].values,
                marker="s",
                linewidth=2.0,
                label=f"{model} CRPS",
                color=colors.get(model, "#475569"),
            )
        ax_crps.set_title(
            "Continuous Ranked Probability Score (CRPS)", fontsize=13, fontweight="bold"
        )
        ax_crps.set_xlabel("Evaluation Window Index", fontsize=11)
        ax_crps.set_ylabel("CRPS (lower is better)", fontsize=11)
        ax_crps.set_xticks(range(len(windows)))
        ax_crps.set_xticklabels([f"W{w}" for w in windows], rotation=0)
        ax_crps.legend(loc="upper right", frameon=True, fontsize=10)
        ax_crps.grid(True, alpha=0.3)

    plt.suptitle(
        f"Multi-Variable Weather Forecasting Benchmark: Rolling-Window Evaluation ({len(windows)} Windows)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rolling benchmark summary plot to %s", output_path)


def plot_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate multi-panel figure displaying training and validation loss curves, errors, and LR schedule.

    Args:
        history_df: DataFrame containing epoch, train_loss, val_loss, val_mae, val_rmse, lr.
        output_path: Output image destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_loss, ax_metrics, ax_lr = axes[0], axes[1], axes[2]

    epochs = history_df["Epoch"].values

    # Panel 1: Loss curves
    ax_loss.plot(
        epochs,
        history_df["Train_Loss"].values,
        marker="o",
        linewidth=2.2,
        color="#2563eb",
        label="Train Joint Loss (Pinball + Huber)",
    )
    ax_loss.plot(
        epochs,
        history_df["Val_Loss"].values,
        marker="s",
        linewidth=2.2,
        color="#e11d48",
        label="Validation Joint Loss",
    )
    ax_loss.set_title(
        "Training & Validation Loss Progression", fontsize=12, fontweight="bold"
    )
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Joint Loss", fontsize=11)
    ax_loss.set_xticks(epochs)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # Panel 2: Validation MAE & RMSE
    if "Val_MAE" in history_df.columns and "Val_RMSE" in history_df.columns:
        ax_metrics.plot(
            epochs,
            history_df["Val_MAE"].values,
            marker="^",
            linewidth=2.0,
            color="#7c3aed",
            label="Val MAE (degC)",
        )
        ax_metrics.plot(
            epochs,
            history_df["Val_RMSE"].values,
            marker="D",
            linewidth=2.0,
            color="#0891b2",
            label="Val RMSE (degC)",
        )
        ax_metrics.set_title(
            "Validation Point Error (MAE & RMSE)", fontsize=12, fontweight="bold"
        )
        ax_metrics.set_xlabel("Epoch", fontsize=11)
        ax_metrics.set_ylabel("Error (degC)", fontsize=11)
        ax_metrics.set_xticks(epochs)
        ax_metrics.legend(loc="upper right", frameon=True, fontsize=10)
        ax_metrics.grid(True, alpha=0.3)

    # Panel 3: Learning Rate
    if "LR" in history_df.columns:
        ax_lr.plot(
            epochs,
            history_df["LR"].values,
            marker="o",
            linewidth=2.0,
            color="#d97706",
            label="Cosine Learning Rate",
        )
        ax_lr.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        ax_lr.set_xlabel("Epoch", fontsize=11)
        ax_lr.set_ylabel("Learning Rate", fontsize=11)
        ax_lr.set_xticks(epochs)
        ax_lr.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_lr.legend(loc="upper right", frameon=True, fontsize=10)
        ax_lr.grid(True, alpha=0.3)

    plt.suptitle(
        "TimesFM-3 Zero-Leakage Fine-Tuning Performance & Loss Curves",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves plot to %s", output_path)


def plot_chronos_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate multi-panel figure displaying Chronos-2 fine-tuning loss curves and learning rate schedule.

    Args:
        history_df: DataFrame containing step, train_loss, eval_loss, lr.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_lr = axes[0], axes[1]

    steps = history_df["Step"].values

    # Panel 1: Loss curves
    if "Train_Loss" in history_df.columns:
        ax_loss.plot(
            steps,
            history_df["Train_Loss"].values,
            marker="o",
            linewidth=2.2,
            color="#ea580c",
            label="Training Cross-Entropy Loss",
        )
    if "Eval_Loss" in history_df.columns:
        ax_loss.plot(
            steps,
            history_df["Eval_Loss"].values,
            marker="s",
            linewidth=2.2,
            color="#dc2626",
            label="Validation Cross-Entropy Loss",
        )
    ax_loss.set_title(
        "Chronos-2 Training & Validation Loss Progression",
        fontsize=12,
        fontweight="bold",
    )
    ax_loss.set_xlabel("Optimization Step", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_xticks(steps)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # Panel 2: Learning Rate Schedule
    if "LR" in history_df.columns:
        ax_lr.plot(
            steps,
            history_df["LR"].values,
            marker="o",
            linewidth=2.0,
            color="#0891b2",
            label="Linear LR Schedule",
        )
        ax_lr.set_title("Learning Rate Decay", fontsize=12, fontweight="bold")
        ax_lr.set_xlabel("Optimization Step", fontsize=11)
        ax_lr.set_ylabel("Learning Rate", fontsize=11)
        ax_lr.set_xticks(steps)
        ax_lr.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_lr.legend(loc="upper right", frameon=True, fontsize=10)
        ax_lr.grid(True, alpha=0.3)

    plt.suptitle(
        "Chronos-2 LoRA Fine-Tuning Performance & Loss Curves",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Chronos training curves plot to %s", output_path)
