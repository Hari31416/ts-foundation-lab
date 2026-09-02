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


def plot_benchmark_comparison(
    window: BenchmarkWindow,
    results: Dict[str, ForecastResult],
    output_path: Path,
    show_context_tail: int = 288,
) -> None:
    """Generate visual plot of Ground Truth vs TimesFM-3 (with uncertainty) vs LightGBM vs AutoARIMA.

    Args:
        window: BenchmarkWindow with ground truth context and horizon.
        results: Dictionary mapping model names to ForecastResult objects.
        output_path: Output filepath for saving the plot image.
        show_context_tail: Number of historical context points to show before cutoff.
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

    colors = {
        "TimesFM-3 (Zero-Shot)": "#2563eb",
        "LightGBM": "#16a34a",
        "AutoARIMA": "#d97706",
        "DeepAR (Deep Learning)": "#9333ea",
    }

    # Plot TimesFM-3 prediction and confidence interval
    if "TimesFM-3 (Zero-Shot)" in results:
        tfm_res = results["TimesFM-3 (Zero-Shot)"]
        ax1.plot(
            hrz_times,
            tfm_res.point_forecast,
            color=colors.get("TimesFM-3 (Zero-Shot)", "#2563eb"),
            label="TimesFM-3 Median",
            linewidth=2.0,
        )
        if tfm_res.quantiles is not None and tfm_res.quantile_levels is not None:
            if 0.1 in tfm_res.quantile_levels and 0.9 in tfm_res.quantile_levels:
                idx_10 = tfm_res.quantile_levels.index(0.1)
                idx_90 = tfm_res.quantile_levels.index(0.9)
                ax1.fill_between(
                    hrz_times,
                    tfm_res.quantiles[:, idx_10],
                    tfm_res.quantiles[:, idx_90],
                    color=colors.get("TimesFM-3 (Zero-Shot)", "#2563eb"),
                    alpha=0.2,
                    label="TimesFM-3 10th-90th %ile",
                )

    # Plot LightGBM
    if "LightGBM" in results:
        lgb_res = results["LightGBM"]
        ax1.plot(
            hrz_times,
            lgb_res.point_forecast,
            color=colors.get("LightGBM", "#16a34a"),
            label="LightGBM Regressor",
            linewidth=1.8,
            linestyle="-.",
        )

    # Plot AutoARIMA
    if "AutoARIMA" in results:
        arima_res = results["AutoARIMA"]
        ax1.plot(
            hrz_times,
            arima_res.point_forecast,
            color=colors.get("AutoARIMA", "#d97706"),
            label="AutoARIMA",
            linewidth=1.8,
            linestyle=":",
        )

    # Plot DeepAR if present
    if "DeepAR (Deep Learning)" in results:
        deep_res = results["DeepAR (Deep Learning)"]
        ax1.plot(
            hrz_times,
            deep_res.point_forecast,
            color=colors.get("DeepAR (Deep Learning)", "#9333ea"),
            label="DeepAR",
            linewidth=1.5,
            alpha=0.8,
        )

    ax1.set_title(
        "TimesFM-3 vs Baseline Models: Weather Forecast Benchmark (Horizon = 96)",
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
        color = colors.get(model_name, "#64748b")
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
