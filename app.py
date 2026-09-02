"""Universal Time Series Forecasting Gradio Web Application powered by TimesFM-3 and Chronos-2."""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import time

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.data.generate_samples import generate_sample_datasets
from src.ui.engine import (
    UniversalForecastingEngine,
    get_column_options,
    load_dataset_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gradio_app")

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
JENA_PATH = BASE_DIR / "data" / "jena_climate_2009_2016.csv"

# Ensure sample demo datasets exist
if (
    not (SAMPLES_DIR / "daily_retail_sales.csv").exists()
    or not (SAMPLES_DIR / "hourly_energy_grid.csv").exists()
):
    generate_sample_datasets()

PRESET_CONFIGS = {
    "Hourly Energy Grid Demand (Sample)": {
        "path": SAMPLES_DIR / "hourly_energy_grid.csv",
        "target": "energy_demand_mw",
        "timestamp": "timestamp",
        "past_only": ["temperature_c", "solar_output_mw"],
        "past_future": ["is_weekend"],
        "horizon": 96,
        "context": 256,
    },
    "Daily Retail Store Sales (Sample)": {
        "path": SAMPLES_DIR / "daily_retail_sales.csv",
        "target": "store_sales_usd",
        "timestamp": "date",
        "past_only": ["foot_traffic_count"],
        "past_future": ["promotion_active", "discount_percentage"],
        "horizon": 48,
        "context": 128,
    },
}

if JENA_PATH.exists():
    PRESET_CONFIGS["Jena Climate Weather (Benchmark)"] = {
        "path": JENA_PATH,
        "target": "T (degC)",
        "timestamp": "Date Time",
        "past_only": [
            "p (mbar)",
            "rh (%)",
            "wv (m/s)",
            "Tdew (degC)",
            "VPdef (mbar)",
            "rho (g/m**3)",
        ],
        "past_future": [],
        "horizon": 96,
        "context": 256,
    }

engine = UniversalForecastingEngine(results_dir=BASE_DIR / "results")


def on_preset_selected(
    preset_name: str,
) -> Tuple[
    Optional[pd.DataFrame],
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    str,
]:
    """Load preset dataset and automatically configure target, timestamp, and covariates."""
    if not preset_name or preset_name not in PRESET_CONFIGS:
        return (
            None,
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(),
            gr.Dropdown(),
            "",
        )

    cfg = PRESET_CONFIGS[preset_name]
    file_path = cfg["path"]
    try:
        df = load_dataset_file(file_path)
        # Limit rows for responsive loading if massive
        if len(df) > 5000:
            df = df.iloc[-5000:].reset_index(drop=True)

        dt_cols, num_cols, def_time, def_target = get_column_options(df)

        # Select configured defaults if present in dataframe
        sel_time = cfg["timestamp"] if cfg["timestamp"] in dt_cols else def_time
        sel_target = cfg["target"] if cfg["target"] in num_cols else def_target
        sel_past = [c for c in cfg["past_only"] if c in num_cols and c != sel_target]
        sel_future = [
            c for c in cfg["past_future"] if c in num_cols and c != sel_target
        ]
        sel_horizon = cfg.get("horizon", 96)
        sel_context = cfg.get("context", 256)

        status_msg = f"Loaded **{preset_name}** ({len(df):,} rows, {len(df.columns)} columns). Configured schema automatically."

        return (
            df,
            gr.Dropdown(choices=dt_cols, value=sel_time, interactive=True),
            gr.Dropdown(choices=num_cols, value=sel_target, interactive=True),
            gr.Dropdown(
                choices=num_cols,
                value=sel_past,
                multiselect=True,
                interactive=True,
            ),
            gr.Dropdown(
                choices=num_cols,
                value=sel_future,
                multiselect=True,
                interactive=True,
            ),
            gr.Dropdown(value=sel_horizon, interactive=True),
            gr.Dropdown(value=sel_context, interactive=True),
            status_msg,
        )
    except Exception as e:
        return (
            None,
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(),
            gr.Dropdown(),
            f"Error loading preset: {e}",
        )


def on_file_uploaded(
    file_obj,
) -> Tuple[
    Optional[pd.DataFrame],
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    str,
]:
    """Handle custom CSV upload and populate dropdown choices."""
    if file_obj is None:
        return (
            None,
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(multiselect=True),
            "",
        )

    file_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    try:
        df = load_dataset_file(file_path)
        dt_cols, num_cols, def_time, def_target = get_column_options(df)
        status_msg = f"Uploaded **{file_path.name}** ({len(df):,} rows, {len(df.columns)} columns)."

        # Past covariates default: all numeric columns except the target
        default_covs = [c for c in num_cols if c != def_target]

        return (
            df,
            gr.Dropdown(choices=dt_cols, value=def_time, interactive=True),
            gr.Dropdown(choices=num_cols, value=def_target, interactive=True),
            gr.Dropdown(
                choices=num_cols,
                value=default_covs,
                multiselect=True,
                interactive=True,
            ),
            gr.Dropdown(
                choices=num_cols,
                value=[],
                multiselect=True,
                interactive=True,
            ),
            status_msg,
        )
    except Exception as e:
        return (
            None,
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(multiselect=True),
            gr.Dropdown(multiselect=True),
            f"Error parsing uploaded file: {e}",
        )


def run_forecast_pipeline(
    df: Optional[pd.DataFrame],
    timestamp_col: Optional[str],
    target_col: Optional[str],
    past_only_cols: Optional[List[str]],
    past_future_cols: Optional[List[str]],
    context_length: int,
    horizon: int,
    selected_models: List[str],
    backtest_mode: bool,
) -> Tuple[go.Figure, pd.DataFrame, Optional[str], str]:
    """Run model inference and generate visualizations and export files."""
    if df is None or df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data loaded. Please upload a CSV or select a preset."
        )
        return (
            empty_fig,
            pd.DataFrame(),
            None,
            "Please upload a CSV or select a sample dataset first.",
        )

    if not target_col:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Please select a Target Column.")
        return empty_fig, pd.DataFrame(), None, "Target column is required."

    if not selected_models:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Please select at least one forecasting model.")
        return empty_fig, pd.DataFrame(), None, "At least one model must be selected."

    try:
        start_t = time.perf_counter()
        fig, metrics_df, pred_df = engine.run_forecasting(
            df=df,
            target_col=target_col,
            timestamp_col=timestamp_col,
            past_only_cols=past_only_cols,
            past_future_cols=past_future_cols,
            context_length=context_length,
            horizon=horizon,
            selected_models=selected_models,
            backtest_mode=backtest_mode,
        )
        total_time_ms = (time.perf_counter() - start_t) * 1000.0

        # Save predictions to temporary CSV for download
        temp_dir = Path(tempfile.gettempdir())
        export_csv_path = temp_dir / "forecast_predictions.csv"
        pred_df.to_csv(export_csv_path, index=False)

        mode_str = (
            "Backtesting Mode (Evaluated against ground truth)"
            if backtest_mode
            else "Future Forecast Mode"
        )
        status_text = f"Forecasting completed in {total_time_ms:.1f} ms ({mode_str})."

        return fig, metrics_df, str(export_csv_path), status_text
    except Exception as e:
        logger.exception("Forecasting execution error: %s", e)
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, pd.DataFrame(), None, f"Error: {str(e)}"


def build_app() -> gr.Blocks:
    """Build modern, sleek Gradio UI interface for zero-shot foundation forecasting."""
    theme = gr.themes.Default(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.25rem;
        border: 1px solid #334155;
    }
    .header-title { font-size: 1.75rem; font-weight: 800; margin: 0 0 0.35rem 0; color: #f8fafc; }
    .header-subtitle { font-size: 0.95rem; color: #94a3b8; margin: 0; }
    .predict-btn {
        background: #2563eb !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 0.5rem !important;
    }
    """

    with gr.Blocks(
        title="Foundation Time Series Forecaster", theme=theme, css=custom_css
    ) as demo:
        dataset_state = gr.State()

        # Header Banner
        gr.HTML("""
            <div class="header-banner">
                <div class="header-title">Foundation Time Series Forecaster</div>
                <div class="header-subtitle">Zero-shot multi-variable time series forecasting powered by Google TimesFM-3 and Amazon Chronos-2.</div>
            </div>
            """)

        with gr.Row(equal_height=False):
            # Left Column: Controls
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### 1. Data Source")
                    with gr.Tabs():
                        with gr.TabItem("Upload CSV"):
                            file_input = gr.File(
                                label="Upload any time series CSV",
                                file_types=[".csv"],
                                type="filepath",
                            )
                        with gr.TabItem("Sample Datasets"):
                            preset_dropdown = gr.Dropdown(
                                choices=list(PRESET_CONFIGS.keys()),
                                label="Select Demo Dataset",
                                value=None,
                            )

                    data_status = gr.Markdown("No dataset loaded yet.")

                with gr.Group():
                    gr.Markdown("### 2. Schema Configuration")
                    target_select = gr.Dropdown(
                        label="Target Column (Required)",
                        choices=[],
                        value=None,
                        info="Numerical series to forecast",
                    )
                    time_select = gr.Dropdown(
                        label="Timestamp Column (Optional)",
                        choices=[],
                        value=None,
                        info="Datetime column for temporal indexing",
                    )
                    past_covs_select = gr.Dropdown(
                        label="Past-Only Covariates (Optional)",
                        choices=[],
                        multiselect=True,
                        info="Historical features known only in the past",
                    )
                    future_covs_select = gr.Dropdown(
                        label="Past & Future Covariates (Optional)",
                        choices=[],
                        multiselect=True,
                        info="Known future regressors (calendar, promotions)",
                    )

                with gr.Group():
                    gr.Markdown("### 3. Forecasting Parameters & Models")
                    with gr.Row():
                        horizon_select = gr.Dropdown(
                            choices=[12, 24, 48, 96, 128, 192, 256],
                            value=96,
                            label="Horizon (Steps)",
                            info="Discrete prediction window",
                        )
                        context_select = gr.Dropdown(
                            choices=[64, 128, 256, 384, 512, 1024],
                            value=256,
                            label="Context (Steps)",
                            info="Discrete lookback context",
                        )

                    backtest_checkbox = gr.Checkbox(
                        label="Backtest Mode (Holdout Ground Truth)",
                        value=True,
                        info="Holds out last Horizon steps to compute error metrics",
                    )

                    models_checkbox = gr.CheckboxGroup(
                        choices=[
                            "TimesFM-3 (Zero-Shot)",
                            "Chronos-2 (Zero-Shot)",
                        ],
                        value=["TimesFM-3 (Zero-Shot)", "Chronos-2 (Zero-Shot)"],
                        label="Zero-Shot Foundation Models",
                    )

                predict_btn = gr.Button(
                    "Generate Forecast",
                    variant="primary",
                    size="lg",
                    elem_classes=["predict-btn"],
                )

            # Right Column: Visualizations & Metrics
            with gr.Column(scale=6):
                with gr.Group():
                    gr.Markdown("### Multi-Panel Forecast Visualizer")
                    plot_output = gr.Plot(label="Forecast Visualizer", container=True)
                    execution_status = gr.Markdown("")

                with gr.Group():
                    gr.Markdown("### Evaluation Metrics (Backtest Mode)")
                    metrics_table = gr.DataFrame(
                        headers=[
                            "Model",
                            "MAE",
                            "RMSE",
                            "WAPE",
                            "CRPS",
                            "80% Coverage",
                            "Interval Width",
                            "Latency (ms)",
                        ],
                        label="Zero-Shot Benchmark Metrics",
                        interactive=False,
                    )

                    download_btn = gr.DownloadButton(
                        label="Download Predictions (CSV)",
                        visible=True,
                        size="md",
                    )

        # Wire Event Handlers
        preset_dropdown.change(
            fn=on_preset_selected,
            inputs=[preset_dropdown],
            outputs=[
                dataset_state,
                time_select,
                target_select,
                past_covs_select,
                future_covs_select,
                horizon_select,
                context_select,
                data_status,
            ],
        )

        file_input.change(
            fn=on_file_uploaded,
            inputs=[file_input],
            outputs=[
                dataset_state,
                time_select,
                target_select,
                past_covs_select,
                future_covs_select,
                data_status,
            ],
        )

        predict_btn.click(
            fn=run_forecast_pipeline,
            inputs=[
                dataset_state,
                time_select,
                target_select,
                past_covs_select,
                future_covs_select,
                context_select,
                horizon_select,
                models_checkbox,
                backtest_checkbox,
            ],
            outputs=[
                plot_output,
                metrics_table,
                download_btn,
                execution_status,
            ],
        )

    return demo


def main() -> None:
    """CLI entrypoint for running Gradio server."""
    parser = argparse.ArgumentParser(
        description="Run TimesFM-3 & Chronos-2 Universal Forecaster App."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio app on (default: 7860).",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio share link."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Server hostname."
    )
    args = parser.parse_args()

    demo = build_app()
    logger.info("Launching Gradio App on http://%s:%d...", args.server_name, args.port)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
