# TimesFM-3 Multi-Variable Forecasting Benchmark

Benchmark experiment evaluating Google TimesFM-3 zero-shot foundation model against classical statistical, gradient boosted tree, and deep learning forecasting architectures on the Weather multi-variable dataset.

## Benchmark Overview

This repository benchmarks time series models under standardized context lengths and prediction horizons using multivariate weather features.

### Problem Formulation

- Dataset: Jena Climate Weather 10-Minute Resolution Benchmark
- Context Window: 512 historical time steps (~3.55 days)
- Prediction Horizon: 96 forecast time steps (16 hours)
- Target Series: T (degC) (Temperature)
- Past-Only Covariates: p (mbar), rh (%), wv (m/s), Tdew (degC), VPdef (mbar), rho (g/m**3)
- Past-Future Covariates: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, dayofyear_sin, dayofyear_cos

## Evaluated Models

- TimesFM-3 (Zero-Shot): Google foundation model (`google/timesfm-3.0-pytorch`) predicting median point forecasts and 9 quantile intervals (10th to 90th percentile) using cross-attention over multivariate past and future covariates.
- AutoARIMA: Classical statistical benchmark fitted via stepwise parameter search with dynamic calendar exogenous regressors.
- LightGBM / Tree Boosting: Gradient boosted decision trees using lag features, rolling statistics, and future calendar covariate steps.
- DeepAR (Deep Learning): Recurrent neural network with Gaussian likelihood head trained on context sequences using Monte Carlo predictive sampling.

## Benchmark Results

| Model                  | MAE    | RMSE   | WAPE   | CRPS   | 80% Coverage | Interval Width | Latency (ms) |
| ---------------------- | ------ | ------ | ------ | ------ | ------------ | -------------- | ------------ |
| TimesFM-3 (Zero-Shot)  | 1.1278 | 1.2513 | 0.1389 | 0.8102 | 67.7%        | 2.5581         | 186.06       |
| AutoARIMA              | 0.6297 | 0.7785 | 0.0775 | 0.4750 | 70.8%        | 1.6644         | 1256.92      |
| LightGBM               | 1.2459 | 1.5344 | 0.1534 | 1.2257 | 4.2%         | 0.1586         | 2129.59      |
| DeepAR (Deep Learning) | 1.2829 | 1.4595 | 0.1580 | 1.2675 | 1.0%         | 0.1015         | 4681.77      |

## TimesFM-3 Uncertainty Assessment

- Nominal Coverage Target: 80.00% (10th to 90th percentile)
- Empirical Interval Coverage: 67.71%
- Average Interval Width: 2.5581
- Quantile CRPS: 0.8102

## Visual Comparison

![Forecast Benchmark Comparison](results/benchmark_comparison.png)

## Procedure and Workflow

1. Data Ingestion: Download and cache Jena Climate dataset, clean anomalous values, and compute cyclical timestamp features (`hour_sin`, `hour_cos`, `dayofweek_sin`, etc.).
2. Window Extraction: Extract 512 context steps and 96 future horizon steps.
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
