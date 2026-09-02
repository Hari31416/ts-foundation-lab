"""Main executable benchmark script comparing TimesFM-3, Chronos-2, and baselines across rolling evaluation windows."""

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
from src.models.chronos_finetuned import Chronos2FineTunedWrapper
from src.models.chronos_model import Chronos2ModelWrapper
from src.models.classical_model import ClassicalForecaster
from src.models.deep_model import DeepLearningForecaster
from src.models.timesfm_finetuned import TimesFM3FineTunedWrapper
from src.models.timesfm_model import ForecastResult, TimesFM3ModelWrapper
from src.models.tree_model import LightGBMForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")


def run_benchmark() -> None:
    """Execute multi-variable forecasting benchmark pipeline across rolling evaluation windows."""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    windows_dir = results_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    tfm_checkpoint_path = results_dir / "timesfm_finetuned_checkpoint.pt"
    chronos_checkpoint_dir = results_dir / "chronos2_finetuned" / "checkpoint-final"

    logger.info(
        "Starting Multi-Variable Forecasting Rolling Benchmark with Foundation & Baseline Models"
    )

    # Step 1: Load rolling windows from Weather benchmark dataset (Test Split: 2015-2016)
    context_length = 512
    horizon = 96
    num_windows = 12

    logger.info(
        "Extracting %d rolling test windows (Context=%d, Horizon=%d)",
        num_windows,
        context_length,
        horizon,
    )

    data_loader = WeatherDatasetLoader(cache_dir=base_dir / "data")
    rolling_windows = data_loader.get_rolling_benchmark_windows(
        num_windows=num_windows,
        context_length=context_length,
        horizon=horizon,
        start_ratio=0.80,
        end_ratio=0.98,
    )

    logger.info(
        "Successfully extracted %d rolling evaluation windows.",
        len(rolling_windows),
    )

    # Step 2: Initialize all evaluated models
    logger.info("Initializing models for benchmark...")
    models = {
        "TimesFM-3 (Zero-Shot)": TimesFM3ModelWrapper(),
        "Chronos-2 (Zero-Shot)": Chronos2ModelWrapper(),
        "AutoARIMA": ClassicalForecaster(seasonal=False, max_p=3, max_q=3),
        "LightGBM": LightGBMForecaster(n_estimators=150, learning_rate=0.05),
        "DeepAR (Deep Learning)": DeepLearningForecaster(
            hidden_dim=64, num_layers=2, epochs=25, lr=0.005
        ),
    }

    if tfm_checkpoint_path.exists():
        logger.info("Found fine-tuned TimesFM-3 checkpoint at %s", tfm_checkpoint_path)
        models["TimesFM-3 (Fine-Tuned)"] = TimesFM3FineTunedWrapper(
            checkpoint_path=tfm_checkpoint_path
        )

    if chronos_checkpoint_dir.exists():
        logger.info(
            "Found fine-tuned Chronos-2 checkpoint at %s", chronos_checkpoint_dir
        )
        models["Chronos-2 (Fine-Tuned)"] = Chronos2FineTunedWrapper(
            checkpoint_path=chronos_checkpoint_dir
        )

    detailed_records = []

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
            # Also save window 1 to results/benchmark_comparison.png for preview
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
                "CRPS_Mean": (f"{crps_mean:.4f}" if crps_mean is not None else "N/A"),
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

    # Step 5: Save Foundation Models Uncertainty Summaries
    uncertainty_details = {
        "nominal_coverage_target": 0.80,
        "num_windows": num_windows,
        "horizon": horizon,
    }

    for foundation_model in [
        "TimesFM-3 (Zero-Shot)",
        "Chronos-2 (Zero-Shot)",
        "TimesFM-3 (Fine-Tuned)",
        "Chronos-2 (Fine-Tuned)",
    ]:
        sub_fm = df_details[df_details["Model"] == foundation_model]
        if not sub_fm.empty:
            uncertainty_details[foundation_model] = {
                "mean_empirical_80_coverage": float(sub_fm["Coverage_80"].mean()),
                "mean_interval_width": float(sub_fm["Interval_Width"].mean()),
                "mean_crps": float(sub_fm["CRPS"].mean()),
                "mean_mae": float(sub_fm["MAE"].mean()),
                "mean_rmse": float(sub_fm["RMSE"].mean()),
            }

    uncertainty_json_path = (
        results_dir / "foundation_models_uncertainty_assessment.json"
    )
    with open(uncertainty_json_path, "w", encoding="utf-8") as f:
        json.dump(uncertainty_details, f, indent=2)
    logger.info(
        "Saved foundation models uncertainty assessment to %s", uncertainty_json_path
    )

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
    logger.info("Benchmark execution completed successfully.")


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
    detail_headers = ["Window", "Period (UTC)"]
    evaluated_models = df_details["Model"].unique().tolist()
    detail_headers.extend(evaluated_models)

    detail_rows = []
    for w_idx in sorted(df_details["Window"].unique()):
        sub_w = df_details[df_details["Window"] == w_idx]
        start_t = sub_w.iloc[0]["Horizon_Start"]
        end_t = sub_w.iloc[0]["Horizon_End"]
        period_str = f"{start_t} to {end_t}"

        row_vals = [f"Window {w_idx:02d}", period_str]
        for m_name in evaluated_models:
            m_sub = sub_w[sub_w["Model"] == m_name]
            row_vals.append(f"{m_sub.iloc[0]['MAE']:.4f}" if not m_sub.empty else "N/A")
        detail_rows.append(row_vals)

    detail_header_line = "| " + " | ".join(detail_headers) + " |"
    detail_sep_line = "| " + " | ".join(["---"] * len(detail_headers)) + " |"
    detail_data_lines = ["| " + " | ".join(row) + " |" for row in detail_rows]
    detail_table_md = "\n".join(
        [detail_header_line, detail_sep_line] + detail_data_lines
    )

    uncertainty_lines = []
    for model_key in [
        "TimesFM-3 (Zero-Shot)",
        "Chronos-2 (Zero-Shot)",
        "TimesFM-3 (Fine-Tuned)",
    ]:
        if model_key in uncertainty_details:
            info = uncertainty_details[model_key]
            cov = f"{info['mean_empirical_80_coverage'] * 100.0:.2f}%"
            w = f"{info['mean_interval_width']:.4f}"
            c = f"{info['mean_crps']:.4f}"
            uncertainty_lines.append(
                f"- **{model_key}**: Empirical Coverage: {cov} | Interval Width: {w} | CRPS: {c}"
            )

    uncertainty_block = "\n".join(uncertainty_lines)

    readme_content = f"""# Foundation Models Multi-Variable Forecasting Benchmark: TimesFM-3 vs Chronos-2

Comprehensive forecasting benchmark evaluating Google TimesFM-3 and Amazon Chronos-2 foundation models against classical statistical, gradient boosted tree, and deep learning forecasting architectures on the Weather multi-variable dataset across a {num_windows}-window rolling evaluation protocol.

## Benchmark Overview

This repository benchmarks time series models under standardized context lengths and prediction horizons across multiple temporal regimes and weather seasons with strict zero-leakage chronological partitioning.

### Problem Formulation

- Dataset: Jena Climate Weather 10-Minute Resolution Benchmark (2009–2016)
- Evaluation Setup: {num_windows} rolling test windows sampled evenly across seasonal partitions
- Context Window: {context_length} historical time steps (~3.55 days)
- Prediction Horizon: {horizon} forecast time steps (16 hours)
- Target Series: T (degC) (Temperature)
- Past-Only Covariates: p (mbar), rh (%), wv (m/s), Tdew (degC), VPdef (mbar), rho (g/m**3)
- Past-Future Covariates: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, dayofyear_sin, dayofyear_cos

## Evaluated Models

- TimesFM-3 (Zero-Shot): Google foundation model (`google/timesfm-3.0-pytorch`) predicting point forecasts and 9 quantile intervals (10th to 90th percentile) using cross-attention over multivariate past and future covariates.
- Chronos-2 (Zero-Shot): Amazon foundation model (`amazon/chronos-2`) predicting point forecasts and 9 quantile intervals (10th to 90th percentile) using covariate-informed attention over past and future regressors.
- AutoARIMA: Classical statistical benchmark fitted via stepwise parameter search with dynamic calendar exogenous regressors.
- LightGBM / Tree Boosting: Gradient boosted decision trees using lag features, rolling statistics, and future calendar covariate steps.
- DeepAR (Deep Learning): Recurrent neural network with Gaussian likelihood head trained on context sequences using Monte Carlo predictive sampling.

## Aggregate Benchmark Results ({num_windows} Rolling Windows)

{summary_table_md}

## Window-by-Window Error Breakdown

{detail_table_md}

## Foundation Models Uncertainty Assessment (Aggregate)

- Nominal Coverage Target: 80.00% (10th to 90th percentile)
{uncertainty_block}

## Visual Comparisons

### Cross-Window Rolling Benchmark Summary

![Rolling Benchmark Summary](results/rolling_benchmark_summary.png)

### Representative Forecast (Window 01)

![Representative Forecast Comparison](results/benchmark_comparison.png)

Individual plots and metric CSVs for all {num_windows} evaluation windows are archived in [`results/windows/`](results/windows/).

## Procedure and Workflow

1. Data Ingestion & Partitioning: Download and cache Jena Climate dataset, clean anomalous values, and strictly partition into train, validation, and test sets.
2. Rolling Window Extraction: Extract {num_windows} evenly distributed windows across the test split, each with {context_length} context steps and {horizon} future horizon steps.
3. Model Inference (`run_benchmark.py`): Execute predictions across all evaluated model classes under identical historical context and future horizon inputs for each window.
4. Evaluation: Compute MAE, RMSE, WAPE, CRPS, 10th-90th coverage calibration, and inference latency per window and in aggregate.
5. Visualization & Reporting: Generate per-window comparison plots (`results/windows/`), cross-window summary plots (`results/rolling_benchmark_summary.png`), summary CSVs, uncertainty JSON metrics, and auto-update this documentation.

## Installation and Reproduction

### Prerequisites

Install `uv` (Fast Python package and project manager).

### Running the Benchmark

```bash
uv sync
uv run python run_benchmark.py
```

### Running the Optional Fine-Tuning Pipeline

```bash
uv run python train_timesfm.py --epochs 3 --lr 1e-4
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
