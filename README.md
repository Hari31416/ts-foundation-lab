# ts-foundation-lab: Universal Time Series Foundation Forecasting Studio & Benchmark Hub

Comprehensive time series foundation model laboratory featuring zero-shot multi-variable forecasting with Google TimesFM-3 and Amazon Chronos-2, an interactive Gradio web application for arbitrary user CSV datasets, and standardized multi-window benchmarking against statistical, tree-based, and deep learning baselines.

Live Hugging Face Space: [https://huggingface.co/spaces/hari31416/ts-foundation-lab](https://huggingface.co/spaces/hari31416/ts-foundation-lab)

## Features


- Interactive Universal Forecaster: Drag-and-drop web application (`app.py`) capable of parsing arbitrary time series CSVs with automatic timestamp detection, target selection, and past/future covariate handling.
- Preloaded Benchmark Datasets: Instant experimentation with 8 diverse real-world time series spanning electricity demand, solar cycles, retail sales, weather, and airline passenger volumes.
- Foundation Models: Side-by-side inference with Google TimesFM-3 (`google/timesfm-3.0-pytorch`) and Amazon Chronos-2 (`amazon/chronos-2`) predicting point estimates and 80% prediction intervals (10th to 90th percentile).
- Multi-Panel Visualizations: Interactive Plotly subplots displaying historical lookback tails, forecast horizons, ground truth actuals, and per-model uncertainty bands.
- Rigorous Benchmarking: Standardized 12-window rolling evaluation protocol against AutoARIMA, LightGBM, DeepAR, and fine-tuned variants (full details in [`BENCHMARK.md`](BENCHMARK.md)).
- Parameter-Efficient Fine-Tuning: Multi-quantile pinball loss fine-tuning for TimesFM-3 and LoRA adapter fine-tuning for Chronos-2.

## Quickstart

### Prerequisites

Install `uv` (Fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/ts-foundation-lab.git
cd ts-foundation-lab
uv sync
```

## Running the Web Application

Launch the interactive Gradio forecasting dashboard:

```bash
uv run python app.py --port 7860
```

Open your browser at `http://localhost:7860`. You can:

- Upload any custom time series CSV file or pick one of the 8 preloaded benchmark presets.
- Select target columns, timestamp indexing, and past/future covariates.
- Toggle between Backtesting Mode (evaluating against hidden ground truth) and Future Mode (extrapolating into the future).
- Inspect multi-panel Plotly charts and download generated prediction CSVs.

## Benchmark Results Summary

Across 12 rolling seasonal test windows on the Jena Climate benchmark:

| Model                  | MAE    | RMSE   | WAPE   | CRPS   | 80% Coverage | Latency (ms) |
| :--------------------- | :----- | :----- | :----- | :----- | :----------- | :----------- |
| TimesFM-3 (Fine-Tuned) | 1.3192 | 1.5173 | 0.2339 | 1.0459 | 78.5%        | 211.7 ms     |
| Chronos-2 (Fine-Tuned) | 1.5282 | 1.7299 | 0.3493 | 1.2003 | 81.0%        | 139.1 ms     |
| Chronos-2 (Zero-Shot)  | 1.6451 | 1.8552 | 0.3451 | 1.2826 | 79.2%        | 81.6 ms      |
| TimesFM-3 (Zero-Shot)  | 1.6575 | 1.8745 | 0.3154 | 1.2603 | 76.5%        | 215.2 ms     |
| DeepAR (Deep Learning) | 2.7669 | 3.1989 | 0.5296 | 2.6827 | 8.7%         | 3951.1 ms    |
| AutoARIMA              | 2.8941 | 3.2738 | 0.6191 | 2.4257 | 57.9%        | 1808.3 ms    |
| LightGBM               | 3.2134 | 3.6784 | 0.6036 | 3.0695 | 10.0%        | 1972.7 ms    |

For detailed breakdown, per-window tables, and loss curves, see [`BENCHMARK.md`](BENCHMARK.md).

## Running the Benchmark Pipeline

To execute the 7-model rolling evaluation protocol locally:

```bash
uv run python run_benchmark.py
```

## Running Fine-Tuning

### TimesFM-3 Fine-Tuning

```bash
uv run python train_timesfm.py --epochs 3 --lr 1e-4
```

### Chronos-2 LoRA Fine-Tuning

```bash
uv run python train_chronos.py --steps 300 --lr 1e-4 --mode lora
```

## Repository Structure

- `app.py`: Gradio web application for universal time series forecasting.
- `run_benchmark.py`: End-to-end rolling-window benchmark execution engine.
- `train_timesfm.py`: TimesFM-3 multi-quantile loss fine-tuning script.
- `train_chronos.py`: Chronos-2 LoRA fine-tuning script.
- `src/ui/engine.py`: Universal forecasting engine with schema inference and Plotly visualizations.
- `src/models/`: Model wrappers for TimesFM-3, Chronos-2, AutoARIMA, LightGBM, and DeepAR.
- `sample_data/`: 8 preloaded benchmark datasets in CSV format.
- `tests/`: Pytest suite for pipelines, models, and UI engine.
- `BENCHMARK.md`: Comprehensive benchmark evaluation documentation.

## Running Tests

```bash
uv run pytest tests/
```