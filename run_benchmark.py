"""Main executable benchmark script comparing TimesFM-3 against classical and ML-based forecasting models."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.dataset import BenchmarkWindow, WeatherDatasetLoader
from src.evaluation.metrics import EvaluationMetrics, evaluate_forecast
from src.evaluation.visualizer import plot_benchmark_comparison
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
    """Execute end-to-end multi-variable forecasting benchmark pipeline."""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting TimesFM-3 Multi-Variable Forecasting Benchmark")

    # Step 1: Load and partition Weather benchmark dataset
    context_length = 512
    horizon = 96
    logger.info(
        "Extracting Weather dataset slice (Context=%d, Horizon=%d)",
        context_length,
        horizon,
    )

    data_loader = WeatherDatasetLoader(cache_dir=base_dir / "data")
    window = data_loader.get_benchmark_window(
        context_length=context_length, horizon=horizon
    )

    logger.info(
        "Target: '%s' | Past-only covariates: %s | Past-future covariates: %s",
        window.target_name,
        window.past_only_names,
        window.past_future_names,
    )
    logger.info(
        "Context target shape: %s | Horizon target shape: %s",
        window.context_target.shape,
        window.horizon_target.shape,
    )

    # Step 2: Initialize models
    models = {
        "TimesFM-3 (Zero-Shot)": TimesFM3ModelWrapper(),
        "AutoARIMA": ClassicalForecaster(seasonal=False, max_p=3, max_q=3),
        "LightGBM": LightGBMForecaster(n_estimators=150, learning_rate=0.05),
        "DeepAR (Deep Learning)": DeepLearningForecaster(
            hidden_dim=64, num_layers=2, epochs=25, lr=0.005
        ),
    }

    # Step 3: Run forecast inference for each model
    forecast_results: Dict[str, ForecastResult] = {}
    evaluation_list: List[EvaluationMetrics] = []

    for model_name, model_instance in models.items():
        logger.info("Running forecast for model: %s", model_name)
        result = model_instance.forecast(
            context=window.context_target,
            horizon=horizon,
            past_only_covariates=window.past_only_context,
            past_future_covariates=window.past_future_full,
        )
        forecast_results[model_name] = result

        # Compute metrics
        metrics = evaluate_forecast(
            model_name=model_name,
            actual=window.horizon_target,
            point_pred=result.point_forecast,
            quantiles=result.quantiles,
            quantile_levels=result.quantile_levels,
            inference_time_ms=result.inference_time_ms,
        )
        evaluation_list.append(metrics)

    # Step 4: Build and save summary tables
    summary_rows = []
    for m in evaluation_list:
        summary_rows.append(
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
    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = results_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved summary metrics to %s", summary_csv_path)

    # Log summary table in clean text format
    logger.info(
        "\n=== BENCHMARK SUMMARY METRICS ===\n%s\n", summary_df.to_string(index=False)
    )

    # Step 5: Uncertainty Assessment for TimesFM-3
    tfm_metrics = next(
        m for m in evaluation_list if m.model_name == "TimesFM-3 (Zero-Shot)"
    )
    tfm_result = forecast_results["TimesFM-3 (Zero-Shot)"]

    uncertainty_details = {
        "model": "TimesFM-3 (Zero-Shot)",
        "horizon": horizon,
        "nominal_coverage_target": 0.80,
        "empirical_80_coverage": tfm_metrics.coverage_80,
        "average_interval_width": tfm_metrics.avg_interval_width,
        "crps": tfm_metrics.crps,
        "quantile_levels": tfm_result.quantile_levels,
    }

    uncertainty_json_path = results_dir / "timesfm_uncertainty_assessment.json"
    with open(uncertainty_json_path, "w", encoding="utf-8") as f:
        json.dump(uncertainty_details, f, indent=2)
    logger.info("Saved TimesFM-3 uncertainty assessment to %s", uncertainty_json_path)

    logger.info(
        "\n=== TIMESFM-3 UNCERTAINTY ASSESSMENT ===\n"
        "Nominal 10th-90th Coverage Target: 80.0%%\n"
        "Empirical 10th-90th Coverage: %.2f%%\n"
        "Average 10th-90th Interval Width: %.4f\n"
        "Quantile CRPS: %.4f\n",
        (tfm_metrics.coverage_80 or 0.0) * 100.0,
        tfm_metrics.avg_interval_width or 0.0,
        tfm_metrics.crps or 0.0,
    )

    # Step 6: Generate visual comparison plot
    plot_path = results_dir / "benchmark_comparison.png"
    plot_benchmark_comparison(
        window=window,
        results=forecast_results,
        output_path=plot_path,
        show_context_tail=288,
    )

    # Step 7: Auto-generate README.md with procedure and latest results
    readme_path = base_dir / "README.md"
    generate_readme(
        readme_path=readme_path,
        summary_df=summary_df,
        uncertainty_details=uncertainty_details,
        window=window,
        context_length=context_length,
        horizon=horizon,
    )
    logger.info("Auto-generated README at %s", readme_path)
    logger.info("Benchmark execution completed successfully.")


def generate_readme(
    readme_path: Path,
    summary_df: pd.DataFrame,
    uncertainty_details: dict,
    window: BenchmarkWindow,
    context_length: int,
    horizon: int,
) -> None:
    """Generate comprehensive project README with methodology and latest benchmark results."""
    # Format markdown table
    headers = summary_df.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in summary_df.iterrows():
        data_lines.append("| " + " | ".join(str(val) for val in row) + " |")
    markdown_table = "\n".join([header_line, separator_line] + data_lines)

    coverage_val = (
        f"{(uncertainty_details.get('empirical_80_coverage') or 0.0) * 100.0:.2f}%"
    )
    avg_width_val = f"{(uncertainty_details.get('average_interval_width') or 0.0):.4f}"
    crps_val = f"{(uncertainty_details.get('crps') or 0.0):.4f}"

    readme_content = f"""# TimesFM-3 Multi-Variable Forecasting Benchmark

Benchmark experiment evaluating Google TimesFM-3 zero-shot foundation model against classical statistical, gradient boosted tree, and deep learning forecasting architectures on the Weather multi-variable dataset.

## Benchmark Overview

This repository benchmarks time series models under standardized context lengths and prediction horizons using multivariate weather features.

### Problem Formulation

- Dataset: Jena Climate Weather 10-Minute Resolution Benchmark
- Context Window: {context_length} historical time steps (~3.55 days)
- Prediction Horizon: {horizon} forecast time steps (16 hours)
- Target Series: {window.target_name} (Temperature)
- Past-Only Covariates: {', '.join(window.past_only_names)}
- Past-Future Covariates: {', '.join(window.past_future_names)}

## Evaluated Models

- TimesFM-3 (Zero-Shot): Google foundation model (`google/timesfm-3.0-pytorch`) predicting median point forecasts and 9 quantile intervals (10th to 90th percentile) using cross-attention over multivariate past and future covariates.
- AutoARIMA: Classical statistical benchmark fitted via stepwise parameter search with dynamic calendar exogenous regressors.
- LightGBM / Tree Boosting: Gradient boosted decision trees using lag features, rolling statistics, and future calendar covariate steps.
- DeepAR (Deep Learning): Recurrent neural network with Gaussian likelihood head trained on context sequences using Monte Carlo predictive sampling.

## Benchmark Results

{markdown_table}

## TimesFM-3 Uncertainty Assessment

- Nominal Coverage Target: 80.00% (10th to 90th percentile)
- Empirical Interval Coverage: {coverage_val}
- Average Interval Width: {avg_width_val}
- Quantile CRPS: {crps_val}

## Visual Comparison

![Forecast Benchmark Comparison](results/benchmark_comparison.png)

## Procedure and Workflow

1. Data Ingestion: Download and cache Jena Climate dataset, clean anomalous values, and compute cyclical timestamp features (`hour_sin`, `hour_cos`, `dayofweek_sin`, etc.).
2. Window Extraction: Extract {context_length} context steps and {horizon} future horizon steps.
3. Partitioning: Separate columns into Target, Past-Only Covariates, and Past-Future Covariates.
4. Model Inference: Execute predictions across all 4 model classes under identical historical context and future horizon inputs.
5. Evaluation: Compute MAE, RMSE, WAPE, CRPS, 10th-90th coverage calibration, and inference latency.
6. Visualization & Reporting: Generate high-resolution plots, summary CSVs, uncertainty JSON metrics, and auto-update this documentation.

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
