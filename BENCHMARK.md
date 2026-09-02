# Foundation Models Multi-Variable Forecasting Benchmark: TimesFM-3 vs Chronos-2

Comprehensive forecasting benchmark evaluating Google TimesFM-3 and Amazon Chronos-2 foundation models against classical statistical, gradient boosted tree, and deep learning forecasting architectures on the Weather multi-variable dataset across a 12-window rolling evaluation protocol.

## Benchmark Overview

This repository benchmarks time series models under standardized context lengths and prediction horizons across multiple temporal regimes and weather seasons with strict zero-leakage chronological partitioning.

### Problem Formulation

- Dataset: Jena Climate Weather 10-Minute Resolution Benchmark (2009–2016)
- Evaluation Setup: 12 rolling test windows sampled evenly across seasonal partitions
- Context Window: 512 historical time steps (~3.55 days)
- Prediction Horizon: 96 forecast time steps (16 hours)
- Target Series: T (degC) (Temperature)
- Past-Only Covariates: p (mbar), rh (%), wv (m/s), Tdew (degC), VPdef (mbar), rho (g/m**3)
- Past-Future Covariates: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, dayofyear_sin, dayofyear_cos

## Evaluated Models

- TimesFM-3 (Zero-Shot): Google foundation model (`google/timesfm-3.0-pytorch`) predicting point forecasts and 9 quantile intervals (10th to 90th percentile) using cross-attention over multivariate past and future covariates.
- TimesFM-2.5 (Zero-Shot): Google foundation model (`google/timesfm-2.5-200m-pytorch`, Apache 2.0) predicting point forecasts and 9 quantile intervals over univariate historical context.
- Chronos-2 (Zero-Shot): Amazon foundation model (`amazon/chronos-2`, Apache 2.0) predicting point forecasts and 9 quantile intervals (10th to 90th percentile) using covariate-informed attention over past and future regressors.
- TimesFM-3 (Fine-Tuned): Model fine-tuned on train split with multi-quantile pinball loss.
- Chronos-2 (Fine-Tuned): Model fine-tuned using LoRA parameter-efficient adapters.
- AutoARIMA: Classical statistical benchmark fitted via stepwise parameter search with dynamic calendar exogenous regressors.
- LightGBM / Tree Boosting: Gradient boosted decision trees using lag features, rolling statistics, and future calendar covariate steps.
- DeepAR (Deep Learning): Recurrent neural network with Gaussian likelihood head trained on context sequences using Monte Carlo predictive sampling.

## Aggregate Benchmark Results (12 Rolling Windows)

| Model                   | MAE (Mean ± Std) | RMSE (Mean ± Std) | WAPE   | CRPS   | 80% Coverage | Interval Width | Latency (ms)     |
| ----------------------- | ---------------- | ----------------- | ------ | ------ | ------------ | -------------- | ---------------- |
| TimesFM-3 (Fine-Tuned)  | 1.3192 ± 1.0496  | 1.5173 ± 1.0999   | 0.2339 | 1.0459 | 78.5%        | 4.1372         | 212.24 ± 4.98    |
| Chronos-2 (Fine-Tuned)  | 1.5282 ± 1.1053  | 1.7299 ± 1.1947   | 0.3493 | 1.2003 | 81.0%        | 4.7100         | 140.74 ± 9.00    |
| Chronos-2 (Zero-Shot)   | 1.6451 ± 1.1778  | 1.8552 ± 1.2623   | 0.3451 | 1.2826 | 79.2%        | 4.9302         | 79.90 ± 11.12    |
| TimesFM-3 (Zero-Shot)   | 1.6575 ± 0.9837  | 1.8745 ± 1.0753   | 0.3154 | 1.2603 | 76.5%        | 5.1417         | 224.43 ± 32.45   |
| TimesFM-2.5 (Zero-Shot) | 2.2388 ± 1.8151  | 2.5689 ± 2.0438   | 0.6457 | 1.7397 | 63.8%        | 5.5547         | 155.36 ± 39.43   |
| DeepAR (Deep Learning)  | 2.7530 ± 1.8265  | 3.1707 ± 2.1274   | 0.5490 | 2.6754 | 9.7%         | 0.5238         | 4035.04 ± 62.76  |
| AutoARIMA               | 2.8941 ± 3.1537  | 3.2738 ± 3.1569   | 0.6191 | 2.4257 | 57.9%        | 4.2375         | 1799.18 ± 482.10 |
| LightGBM                | 3.2134 ± 2.3375  | 3.6784 ± 2.7252   | 0.6036 | 3.0695 | 10.0%        | 0.8735         | 2267.94 ± 449.32 |

## Window-by-Window Error Breakdown

| Window    | Period (UTC)                         | TimesFM-3 (Zero-Shot) | TimesFM-2.5 (Zero-Shot) | Chronos-2 (Zero-Shot) | AutoARIMA | LightGBM | DeepAR (Deep Learning) | TimesFM-3 (Fine-Tuned) | Chronos-2 (Fine-Tuned) |
| --------- | ------------------------------------ | --------------------- | ----------------------- | --------------------- | --------- | -------- | ---------------------- | ---------------------- | ---------------------- |
| Window 01 | 2015-05-28 08:50 to 2015-05-29 00:40 | 3.9952                | 4.3621                  | 4.4873                | 2.3794    | 4.1295   | 4.2739                 | 4.4549                 | 4.1757                 |
| Window 02 | 2015-07-14 18:30 to 2015-07-15 10:20 | 0.9374                | 0.7264                  | 0.9835                | 2.3312    | 1.2403   | 0.6810                 | 1.3739                 | 0.9493                 |
| Window 03 | 2015-08-31 04:10 to 2015-08-31 20:00 | 3.1982                | 6.2059                  | 2.5220                | 3.6664    | 9.5666   | 7.4607                 | 0.6688                 | 1.2039                 |
| Window 04 | 2015-10-17 14:00 to 2015-10-18 05:50 | 1.6340                | 0.5497                  | 0.1943                | 0.4709    | 0.9248   | 0.6399                 | 0.6630                 | 0.1633                 |
| Window 05 | 2015-12-03 23:40 to 2015-12-04 15:30 | 1.8314                | 0.8261                  | 1.3351                | 12.5609   | 1.2002   | 1.2784                 | 1.7241                 | 2.7383                 |
| Window 06 | 2016-01-20 09:30 to 2016-01-21 01:20 | 1.5549                | 3.5142                  | 1.8866                | 2.0979    | 2.5379   | 2.0943                 | 0.9576                 | 2.1416                 |
| Window 07 | 2016-03-07 19:10 to 2016-03-08 11:00 | 0.7091                | 2.7410                  | 0.7383                | 2.1411    | 2.1946   | 2.4432                 | 0.6250                 | 0.5884                 |
| Window 08 | 2016-04-24 05:00 to 2016-04-24 20:50 | 1.3127                | 2.8867                  | 2.7801                | 2.1718    | 3.5274   | 2.9035                 | 1.0721                 | 1.2796                 |
| Window 09 | 2016-06-10 14:40 to 2016-06-11 06:30 | 1.5119                | 0.5958                  | 1.8158                | 2.7751    | 5.0244   | 2.8304                 | 0.8164                 | 1.6946                 |
| Window 10 | 2016-07-28 00:30 to 2016-07-28 16:20 | 1.2249                | 1.6198                  | 1.2805                | 1.6533    | 3.0715   | 3.2479                 | 1.5610                 | 1.0943                 |
| Window 11 | 2016-09-13 10:10 to 2016-09-14 02:00 | 0.7369                | 0.3863                  | 0.5382                | 1.5256    | 2.5977   | 3.0567                 | 1.0474                 | 0.5047                 |
| Window 12 | 2016-11-02 22:10 to 2016-11-03 14:00 | 1.2436                | 2.4519                  | 1.1798                | 0.9555    | 2.5460   | 2.1263                 | 0.8662                 | 1.8050                 |

## Foundation Models Uncertainty Assessment (Aggregate)

- Nominal Coverage Target: 80.00% (10th to 90th percentile)
- **TimesFM-3 (Zero-Shot)**: Empirical Coverage: 76.48% | Interval Width: 5.1417 | CRPS: 1.2603
- **TimesFM-2.5 (Zero-Shot)**: Empirical Coverage: 63.80% | Interval Width: 5.5547 | CRPS: 1.7397
- **Chronos-2 (Zero-Shot)**: Empirical Coverage: 79.17% | Interval Width: 4.9302 | CRPS: 1.2826
- **TimesFM-3 (Fine-Tuned)**: Empirical Coverage: 78.47% | Interval Width: 4.1372 | CRPS: 1.0459
- **Chronos-2 (Fine-Tuned)**: Empirical Coverage: 80.99% | Interval Width: 4.7100 | CRPS: 1.2003

## Visual Comparisons

### Cross-Window Rolling Benchmark Summary

![Rolling Benchmark Summary](results/rolling_benchmark_summary.png)

### Representative Forecast (Window 01)

![Representative Forecast Comparison](results/benchmark_comparison.png)

Individual plots and metric CSVs for all 12 evaluation windows are archived in [`results/windows/`](results/windows/).

## Procedure and Workflow

1. Data Ingestion & Partitioning: Download and cache Jena Climate dataset, clean anomalous values, and strictly partition into train, validation, and test sets.
2. Rolling Window Extraction: Extract 12 evenly distributed windows across the test split, each with 512 context steps and 96 future horizon steps.
3. Model Inference (`run_benchmark.py`): Execute predictions across all evaluated model classes under identical historical context and future horizon inputs for each window.
4. Evaluation: Compute MAE, RMSE, WAPE, CRPS, 10th-90th coverage calibration, and inference latency per window and in aggregate.
5. Visualization & Reporting: Generate per-window comparison plots (`results/windows/`), cross-window summary plots (`results/rolling_benchmark_summary.png`), summary CSVs, uncertainty JSON metrics, and auto-update this documentation.
