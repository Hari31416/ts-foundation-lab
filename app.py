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
SAMPLES_DIR = BASE_DIR / "sample_data"

# Ensure sample demo datasets exist
if (
    not (SAMPLES_DIR / "daily_retail_sales.csv").exists()
    or not (SAMPLES_DIR / "hourly_energy_grid.csv").exists()
):
    generate_sample_datasets()

PRESET_CONFIGS = {
    "Electricity Transformer Temperature (ETTh1 Benchmark)": {
        "path": SAMPLES_DIR / "ett_electricity_transformer.csv",
        "target": "OT",
        "timestamp": "date",
        "past_only": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        "past_future": [],
        "horizon": 96,
        "context": 256,
    },
    "Hourly Energy Grid Demand (Sample)": {
        "path": SAMPLES_DIR / "hourly_energy_grid.csv",
        "target": "energy_demand_mw",
        "timestamp": "timestamp",
        "past_only": ["temperature_c", "solar_output_mw"],
        "past_future": ["is_weekend"],
        "horizon": 96,
        "context": 256,
    },
    "Melbourne Daily Minimum Temperatures": {
        "path": SAMPLES_DIR / "melbourne_daily_temperatures.csv",
        "target": "Daily minimum temperatures",
        "timestamp": "Date",
        "past_only": [],
        "past_future": [],
        "horizon": 96,
        "context": 256,
    },
    "Daily Total Female Births": {
        "path": SAMPLES_DIR / "daily_female_births.csv",
        "target": "Births",
        "timestamp": "Date",
        "past_only": [],
        "past_future": [],
        "horizon": 48,
        "context": 128,
    },
    "Monthly Solar Sunspots (1749–1983)": {
        "path": SAMPLES_DIR / "monthly_sunspots.csv",
        "target": "Sunspots",
        "timestamp": "Month",
        "past_only": [],
        "past_future": [],
        "horizon": 48,
        "context": 256,
    },
    "Monthly Airline Passengers (Box-Jenkins)": {
        "path": SAMPLES_DIR / "monthly_airline_passengers.csv",
        "target": "Passengers",
        "timestamp": "Month",
        "past_only": [],
        "past_future": [],
        "horizon": 24,
        "context": 64,
    },
    "Monthly Quebec Car Sales": {
        "path": SAMPLES_DIR / "monthly_car_sales.csv",
        "target": "Sales",
        "timestamp": "Month",
        "past_only": [],
        "past_future": [],
        "horizon": 24,
        "context": 64,
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


try:
    import spaces  # type: ignore

    GPU_DECORATOR = spaces.GPU
except (ImportError, AttributeError):

    def GPU_DECORATOR(func=None, **kwargs):
        return func if func is not None else lambda f: f


@GPU_DECORATOR
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

    with gr.Blocks(title="Foundation Time Series Forecaster") as demo:
        dataset_state = gr.State()

        # Top Header Banner with Embedded Styles
        gr.HTML("""
            <style>
                .header-card {
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    color: #ffffff;
                    padding: 24px 28px;
                    border-radius: 12px;
                    border: 1px solid #334155;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
                }
                .header-main-title {
                    font-size: 26px;
                    font-weight: 800;
                    color: #f8fafc;
                    letter-spacing: -0.5px;
                    margin-bottom: 6px;
                    line-height: 1.2;
                }
                .header-sub-title {
                    font-size: 14.5px;
                    color: #94a3b8;
                    margin-bottom: 12px;
                    font-weight: 400;
                }
                .badge-pill {
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 9999px;
                    font-size: 12px;
                    font-weight: 600;
                    margin-right: 6px;
                }
                .badge-google {
                    background: rgba(59, 130, 246, 0.18);
                    color: #93c5fd;
                    border: 1px solid rgba(59, 130, 246, 0.35);
                }
                .badge-amazon {
                    background: rgba(249, 115, 22, 0.18);
                    color: #fdba74;
                    border: 1px solid rgba(249, 115, 22, 0.35);
                }
            </style>
            <div class="header-card">
                <div class="header-main-title">Foundation Time Series Forecaster</div>
                <div class="header-sub-title">Zero-shot multi-variable time series forecasting powered by state-of-the-art foundation models.</div>
                <div>
                    <span class="badge-pill badge-google">Google TimesFM-3 (Zero-Shot)</span>
                    <span class="badge-pill badge-amazon">Amazon Chronos-2 (Zero-Shot)</span>
                </div>
            </div>
            """)

        # Layman Guide & Terminology Accordion
        with gr.Accordion(
            "Concepts & Terminology Guide (Click to Expand)",
            open=False,
        ):
            gr.Markdown("""
                ### 1. Target vs. Covariates (What are they?)
                - **Target Column (Required)**: The exact quantity you want to predict into the future (e.g., *Tomorrow's Energy Demand*, *Next Month's Sales*, or *Tomorrow's Temperature*).
                - **Past-Only Covariates (Optional)**: Extra helper signals that you only know up to today, not tomorrow (e.g., *Yesterday's Weather*, *Website Traffic Count*). The models look at their past patterns to better understand the target.
                - **Past & Future Covariates (Optional)**: Special helper features that you know both in the past **AND** already know for future dates (e.g., *Is Weekend / Holiday*, *Scheduled Marketing Promo*, *Price Discounts*). Because these are known ahead of time, the model can anticipate demand surges.

                ### 2. Context vs. Horizon (How far back & forward?)
                - **Historical Context (Lookback)**: How many past time steps the model reads before making a forecast (e.g., reading the last 256 hours).
                - **Forecast Horizon**: How many steps into the future you want the model to predict (e.g., predicting 96 hours ahead).

                ### 3. What is Backtesting Mode?
                - **Backtesting (Holdout Testing)**: Think of this like giving the model a test with known answers. The app hides the last *Horizon* steps from the model, asks it to forecast them, and then compares its guesses against the real numbers.
                - This allows the app to calculate real-world accuracy scorecards like **MAE (Average Error)** and **RMSE** to prove how well each model performed.
                - **Future Mode (Unchecked)**: When backtesting is turned off, the model uses all latest data up to the very last timestamp to extrapolate into the unknown future.

                ### 4. How are Large Datasets Handled?
                - If you upload a huge CSV with 100,000+ rows, you don't need to wait minutes. The engine automatically slices the most recent *Context* window directly before the forecast cutoff. This keeps predictions instantaneous (~100–200 ms) regardless of file size.

                ### 5. What do the Shaded Bands Mean? (80% Confidence Interval)
                - Foundation models don't just give a single guess—they produce a probability distribution. The shaded colored band shows the **80% range of likely outcomes** (from the 10th percentile to the 90th percentile). A narrow band means high model certainty, while a wide band warns of higher volatility.
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


demo = build_app()


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
        "--server-name", type=str, default="0.0.0.0", help="Server hostname."
    )
    args = parser.parse_args()
    port = int(os.environ.get("PORT", args.port))

    logger.info("Launching Gradio App on http://%s:%d...", args.server_name, port)
    demo.launch(
        server_name=args.server_name,
        server_port=port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
