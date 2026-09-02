"""Main executable benchmark script comparing TimesFM-3 against classical and ML-based forecasting models across rolling evaluation windows."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.dataset import BenchmarkWindow, WeatherDatasetLoader
from src.evaluation.metrics import EvaluationMetrics, evaluate_forecast
from src.evaluation.visualizer import (
    plot_benchmark_comparison,
    plot_rolling_benchmark_summary,
)
from src.models.classical_model import ClassicalForecaster
from src.models.deep_model import DeepLearningForecaster
from src.models.timesfm_model import ForecastResult, TimesFM3ModelWrapper
from src.models.tree_model import LightGBMForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")


def run_benchmark() -> None:
    """Execute end-to-end multi-variable forecasting benchmark pipeline with rolling windows."""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    windows_dir = results_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting TimesFM-3 Rolling-Window Multi-Variable Forecasting Benchmark"
    )

    # Step 1: Load rolling windows from Weather benchmark dataset
    context_length = 512
    horizon = 96
    num_windows = 12

    logger.info(
        "Extracting %d rolling windows (Context=%d, Horizon=%d)",
        num_windows,
        context_length,
        horizon,
    )

    data_loader = WeatherDatasetLoader(cache_dir=base_dir / "data")
    rolling_windows = data_loader.get_rolling_benchmark_windows(
        num_windows=num_windows,
        context_length=context_length,
        horizon=horizon,
        start_ratio=0.70,
        end_ratio=0.98,
    )

    logger.info(
        "Successfully extracted %d rolling evaluation windows.", len(rolling_windows)
    )

    # Step 2: Initialize models once (re-used across windows)
    logger.info("Initializing models for benchmark...")
    models = {
        "TimesFM-3 (Zero-Shot)": TimesFM3ModelWrapper(),
        "AutoARIMA": ClassicalForecaster(seasonal=False, max_p=3, max_q=3),
        "LightGBM": LightGBMForecaster(n_estimators=150, learning_rate=0.05),
        "DeepAR (Deep Learning)": DeepLearningForecaster(
            hidden_dim=64, num_layers=2, epochs=25, lr=0.005
        ),
    }

    detailed_records = []
    window_summaries = []

    # Step 3: Run inference on each window
    for w_idx, (start_idx, window) in enumerate(rolling_windows, start=1):
        window_id_str = f"Window {w_idx:02d}"
        date_start = str(window.timestamps_horizon[0])[:16]
        date_end = str(window.timestamps_horizon[-1])[:16]
        window_label = f"{window_id_str} ({date_start} to {date_end})"

        logger.info(
            "=== Evaluating %s [Start Index: %d, Horizon: %s -> %s] ===",
            window_id_str,
            start_idx,
            date_start,
            date_end,
        )

        forecast_results: Dict[str, ForecastResult] = {}
        window_evaluations: List[EvaluationMetrics] = []

        for model_name, model_instance in models.items():
            logger.info("Running forecast for %s on %s", model_name, window_id_str)
            result = model_instance.forecast(
                context=window.context_target,
                horizon=horizon,
                past_only_covariates=window.past_only_context,
                past_future_covariates=window.past_future_full,
            )
            forecast_results[model_name] = result

            metrics = evaluate_forecast(
                model_name=model_name,
                actual=window.horizon_target,
                point_pred=result.point_forecast,
                quantiles=result.quantiles,
                quantile_levels=result.quantile_levels,
                inference_time_ms=result.inference_time_ms,
            )
            window_evaluations.append(metrics)

            detailed_records.append(
                {
                    "Window": w_idx,
                    "Start_Idx": start_idx,
                    "Horizon_Start": date_start,
                    "Horizon_End": date_end,
                    "Model": model_name,
                    "MAE": metrics.mae,
                    "RMSE": metrics.rmse,
                    "WAPE": metrics.wape,
                    "CRPS": metrics.crps,
                    "Coverage_80": metrics.coverage_80,
                    "Interval_Width": metrics.avg_interval_width,
                    "Latency_ms": metrics.inference_time_ms,
                }
            )

        # Save individual window metrics table
        w_df_rows = []
        for m in window_evaluations:
            w_df_rows.append(
                {
                    "Model": m.model_name,
                    "MAE": f"{m.mae:.4f}",
                    "RMSE": f"{m.rmse:.4f}",
                    "WAPE": f"{m.wape:.4f}",
                    "CRPS": f"{m.crps:.4f}" if m.crps is not None else "N/A",
                    "80% Coverage": (
                        f"{m.coverage_80 * 100:.1f}%"
                        if m.coverage_80 is not None
                        else "N/A"
                    ),
                    "Interval Width": (
                        f"{m.avg_interval_width:.4f}"
                        if m.avg_interval_width is not None
                        else "N/A"
                    ),
                    "Latency (ms)": f"{m.inference_time_ms:.2f}",
                }
            )
        window_metrics_df = pd.DataFrame(w_df_rows)
        w_csv_path = windows_dir / f"window_{w_idx:02d}_metrics.csv"
        window_metrics_df.to_csv(w_csv_path, index=False)

        # Save individual window plot
        w_plot_path = windows_dir / f"window_{w_idx:02d}_comparison.png"
        plot_benchmark_comparison(
            window=window,
            results=forecast_results,
            output_path=w_plot_path,
            show_context_tail=288,
            window_label=window_label,
        )
        logger.info("Saved %s plot and metrics.", window_id_str)

        if w_idx == 1:
            # Also save window 1 to results/benchmark_comparison.png for backward-compatibility
            plot_benchmark_comparison(
                window=window,
                results=forecast_results,
                output_path=results_dir / "benchmark_comparison.png",
                show_context_tail=288,
                window_label=f"Representative Window 01 ({date_start} to {date_end})",
            )

    # Step 4: Aggregate results across all windows
    df_details = pd.DataFrame(detailed_records)
    details_csv_path = results_dir / "rolling_window_details.csv"
    df_details.to_csv(details_csv_path, index=False)
    logger.info("Saved rolling window details to %s", details_csv_path)

    # Compute aggregate metrics per model
    model_groups = df_details.groupby("Model")
    summary_rows = []
    for model_name, grp in model_groups:
        mae_mean, mae_std = grp["MAE"].mean(), grp["MAE"].std()
        rmse_mean, rmse_std = grp["RMSE"].mean(), grp["RMSE"].std()
        wape_mean, wape_std = grp["WAPE"].mean(), grp["WAPE"].std()
        lat_mean, lat_std = grp["Latency_ms"].mean(), grp["Latency_ms"].std()

        crps_mean = (
            grp["CRPS"].dropna().mean() if not grp["CRPS"].dropna().empty else None
        )
        cov_mean = (
            grp["Coverage_80"].dropna().mean()
            if not grp["Coverage_80"].dropna().empty
            else None
        )
        width_mean = (
            grp["Interval_Width"].dropna().mean()
            if not grp["Interval_Width"].dropna().empty
            else None
        )

        summary_rows.append(
            {
                "Model": model_name,
                "MAE_Mean": f"{mae_mean:.4f}",
                "MAE_Std": f"{mae_std:.4f}",
                "RMSE_Mean": f"{rmse_mean:.4f}",
                "RMSE_Std": f"{rmse_std:.4f}",
                "WAPE_Mean": f"{wape_mean:.4f}",
                "CRPS_Mean": f"{crps_mean:.4f}" if crps_mean is not None else "N/A",
                "80%_Coverage_Mean": (
                    f"{cov_mean * 100:.1f}%" if cov_mean is not None else "N/A"
                ),
                "Interval_Width_Mean": (
                    f"{width_mean:.4f}" if width_mean is not None else "N/A"
                ),
                "Latency_Mean_ms": f"{lat_mean:.2f}",
                "Latency_Std_ms": f"{lat_std:.2f}",
            }
        )

    # Sort summary by MAE_Mean
    df_summary = (
        pd.DataFrame(summary_rows).sort_values("MAE_Mean").reset_index(drop=True)
    )
    summary_csv_path = results_dir / "benchmark_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info("Saved aggregate benchmark summary to %s", summary_csv_path)

    logger.info(
        "\n=== AGGREGATE ROLLING BENCHMARK SUMMARY ===\n%s\n",
        df_summary.to_string(index=False),
    )

    # Step 5: TimesFM-3 Uncertainty Summary across all windows
    tfm_records = df_details[df_details["Model"] == "TimesFM-3 (Zero-Shot)"]
    uncertainty_details = {
        "model": "TimesFM-3 (Zero-Shot)",
        "num_windows": num_windows,
        "horizon": horizon,
        "nominal_coverage_target": 0.80,
        "mean_empirical_80_coverage": float(tfm_records["Coverage_80"].mean()),
        "mean_interval_width": float(tfm_records["Interval_Width"].mean()),
        "mean_crps": float(tfm_records["CRPS"].mean()),
        "mean_mae": float(tfm_records["MAE"].mean()),
        "mean_rmse": float(tfm_records["RMSE"].mean()),
        "quantile_levels": TimesFM3ModelWrapper.DEFAULT_QUANTILES,
    }

    uncertainty_json_path = results_dir / "timesfm_uncertainty_assessment.json"
    with open(uncertainty_json_path, "w", encoding="utf-8") as f:
        json.dump(uncertainty_details, f, indent=2)
    logger.info("Saved TimesFM-3 uncertainty assessment to %s", uncertainty_json_path)

    # Step 6: Generate rolling summary visualization
    rolling_plot_path = results_dir / "rolling_benchmark_summary.png"
    plot_rolling_benchmark_summary(
        df_details=df_details,
        df_summary=df_summary,
        output_path=rolling_plot_path,
    )

    # Step 7: Auto-generate comprehensive README.md
    readme_path = base_dir / "README.md"
    generate_readme(
        readme_path=readme_path,
        df_summary=df_summary,
        df_details=df_details,
        uncertainty_details=uncertainty_details,
        num_windows=num_windows,
        context_length=context_length,
        horizon=horizon,
    )
    logger.info("Auto-generated README at %s", readme_path)
    logger.info("Rolling-Window Benchmark execution completed successfully.")


def generate_readme(
    readme_path: Path,
    df_summary: pd.DataFrame,
    df_details: pd.DataFrame,
    uncertainty_details: dict,
    num_windows: int,
    context_length: int,
    horizon: int,
) -> None:
    """Generate comprehensive project README with rolling-window methodology and benchmark results."""
    # Format aggregate summary table
    summary_headers = [
        "Model",
        "MAE (Mean ± Std)",
        "RMSE (Mean ± Std)",
        "WAPE",
        "CRPS",
        "80% Coverage",
        "Interval Width",
        "Latency (ms)",
    ]
    summary_rows = []
    for _, r in df_summary.iterrows():
        summary_rows.append(
            [
                r["Model"],
                f"{r['MAE_Mean']} ± {r['MAE_Std']}",
                f"{r['RMSE_Mean']} ± {r['RMSE_Std']}",
                r["WAPE_Mean"],
                r["CRPS_Mean"],
                r["80%_Coverage_Mean"],
                r["Interval_Width_Mean"],
                f"{r['Latency_Mean_ms']} ± {r['Latency_Std_ms']}",
            ]
        )

    summary_header_line = "| " + " | ".join(summary_headers) + " |"
    summary_sep_line = "| " + " | ".join(["---"] * len(summary_headers)) + " |"
    summary_data_lines = ["| " + " | ".join(row) + " |" for row in summary_rows]
    summary_table_md = "\n".join(
        [summary_header_line, summary_sep_line] + summary_data_lines
    )

    # Format window-by-window detail table
    detail_headers = [
        "Window",
        "Period (UTC)",
        "TimesFM-3 MAE",
        "AutoARIMA MAE",
        "LightGBM MAE",
        "DeepAR MAE",
    ]
    detail_rows = []
    for w_idx in sorted(df_details["Window"].unique()):
        sub_w = df_details[df_details["Window"] == w_idx]
        start_t = sub_w.iloc[0]["Horizon_Start"]
        end_t = sub_w.iloc[0]["Horizon_End"]
        period_str = f"{start_t} to {end_t}"

        def get_model_mae(m_name: str) -> str:
            m_sub = sub_w[sub_w["Model"] == m_name]
            return f"{m_sub.iloc[0]['MAE']:.4f}" if not m_sub.empty else "N/A"

        detail_rows.append(
            [
                f"Window {w_idx:02d}",
                period_str,
                get_model_mae("TimesFM-3 (Zero-Shot)"),
                get_model_mae("AutoARIMA"),
                get_model_mae("LightGBM"),
                get_model_mae("DeepAR (Deep Learning)"),
            ]
        )

    detail_header_line = "| " + " | ".join(detail_headers) + " |"
    detail_sep_line = "| " + " | ".join(["---"] * len(detail_headers)) + " |"
    detail_data_lines = ["| " + " | ".join(row) + " |" for row in detail_rows]
    detail_table_md = "\n".join(
        [detail_header_line, detail_sep_line] + detail_data_lines
    )

    cov_val = (
        f"{(uncertainty_details.get('mean_empirical_80_coverage') or 0.0) * 100.0:.2f}%"
    )
    avg_width_val = f"{(uncertainty_details.get('mean_interval_width') or 0.0):.4f}"
    crps_val = f"{(uncertainty_details.get('mean_crps') or 0.0):.4f}"

    readme_content = f"""# TimesFM-3 Multi-Variable Forecasting Benchmark

Benchmark experiment evaluating Google TimesFM-3 zero-shot foundation model against classical statistical, gradient boosted tree, and deep learning forecasting architectures on the Weather multi-variable dataset across a {num_windows}-window rolling evaluation protocol.

## Benchmark Overview

This repository benchmarks time series models under standardized context lengths and prediction horizons across multiple temporal regimes and weather seasons.

### Problem Formulation

- Dataset: Jena Climate Weather 10-Minute Resolution Benchmark (2009–2016)
- Evaluation Setup: {num_windows} rolling test windows sampled evenly across seasonal partitions
- Context Window: {context_length} historical time steps (~3.55 days)
- Prediction Horizon: {horizon} forecast time steps (16 hours)
- Target Series: T (degC) (Temperature)
- Past-Only Covariates: p (mbar), rh (%), wv (m/s), Tdew (degC), VPdef (mbar), rho (g/m**3)
- Past-Future Covariates: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, dayofyear_sin, dayofyear_cos

## Evaluated Models

- TimesFM-3 (Zero-Shot): Google foundation model (`google/timesfm-3.0-pytorch`) predicting median point forecasts and 9 quantile intervals (10th to 90th percentile) using cross-attention over multivariate past and future covariates.
- AutoARIMA: Classical statistical benchmark fitted via stepwise parameter search with dynamic calendar exogenous regressors.
- LightGBM / Tree Boosting: Gradient boosted decision trees using lag features, rolling statistics, and future calendar covariate steps.
- DeepAR (Deep Learning): Recurrent neural network with Gaussian likelihood head trained on context sequences using Monte Carlo predictive sampling.

## Aggregate Benchmark Results ({num_windows} Rolling Windows)

{summary_table_md}

## Window-by-Window Error Breakdown

{detail_table_md}

## TimesFM-3 Uncertainty Assessment (Aggregate)

- Nominal Coverage Target: 80.00% (10th to 90th percentile)
- Empirical Interval Coverage: {cov_val}
- Average Interval Width: {avg_width_val}
- Quantile CRPS: {crps_val}

## Visual Comparisons

### Cross-Window Rolling Benchmark Summary

![Rolling Benchmark Summary](results/rolling_benchmark_summary.png)

### Representative Forecast (Window 01)

![Representative Forecast Comparison](results/benchmark_comparison.png)

Individual plots and metric CSVs for all {num_windows} evaluation windows are archived in [`results/windows/`](results/windows/).

## Procedure and Workflow

1. Data Ingestion: Download and cache Jena Climate dataset, clean anomalous values, and compute cyclical timestamp features (`hour_sin`, `hour_cos`, `dayofweek_sin`, etc.).
2. Rolling Window Extraction: Extract {num_windows} evenly distributed windows, each with {context_length} context steps and {horizon} future horizon steps.
3. Partitioning: Separate columns into Target, Past-Only Covariates, and Past-Future Covariates.
4. Model Inference: Execute predictions across all 4 model classes under identical historical context and future horizon inputs for each window.
5. Evaluation: Compute MAE, RMSE, WAPE, CRPS, 10th-90th coverage calibration, and inference latency per window and in aggregate.
6. Visualization & Reporting: Generate per-window comparison plots (`results/windows/`), cross-window summary plots (`results/rolling_benchmark_summary.png`), summary CSVs, uncertainty JSON metrics, and auto-update this documentation.

## Installation and Reproduction

### Prerequisites

Install `uv` (Fast Python package and project manager).

### Running the Benchmark

```bash
uv sync
uv run python run_benchmark.py
```

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black .
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


if __name__ == "__main__":
    run_benchmark()
