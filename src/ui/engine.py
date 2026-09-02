"""Universal Forecasting Engine for Gradio UI powered by TimesFM-3 and Chronos-2."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.evaluation.metrics import evaluate_forecast
from src.models.chronos_finetuned import Chronos2FineTunedWrapper
from src.models.chronos_model import Chronos2ModelWrapper
from src.models.timesfm2_5_model import TimesFM2p5ModelWrapper
from src.models.timesfm_finetuned import TimesFM3FineTunedWrapper
from src.models.timesfm_model import ForecastResult, TimesFM3ModelWrapper

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "Ground Truth": "#0f172a",
    "Context (History)": "#475569",
    "TimesFM-3 (Zero-Shot)": "#2563eb",
    "TimesFM-2.5 (Zero-Shot)": "#0284c7",
    "Chronos-2 (Zero-Shot)": "#ea580c",
    "TimesFM-3 (Fine-Tuned)": "#7c3aed",
    "Chronos-2 (Fine-Tuned)": "#dc2626",
}


def load_dataset_file(file_path: Path) -> pd.DataFrame:
    """Load uploaded CSV file into DataFrame with encoding fallbacks and auto-cleaning."""
    try:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")

        # Auto-clean any columns that are numbers with trailing/leading symbols (?, $, commas)
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).iloc[:50]
                # If sample doesn't look like dates (no hyphens/slashes with dates)
                try:
                    pd.to_datetime(sample)
                    continue
                except Exception:
                    pass

                # Try cleaning common numeric noise
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.replace("?", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().sum() >= 0.5 * len(df):
                    df[col] = converted

        return df
    except Exception as e:
        logger.error("Error reading CSV file %s: %s", file_path, e)
        raise ValueError(f"Failed to read CSV: {e}")


def get_column_options(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Inspect DataFrame and extract datetime and numeric column names.

    Returns:
        (datetime_cols, numeric_cols, default_time_col, default_target_col)
    """
    datetime_cols = []
    numeric_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Check if column can be parsed as datetime
            is_dt = False
            try:
                pd.to_datetime(df[col].dropna().iloc[:30])
                datetime_cols.append(col)
                is_dt = True
            except Exception:
                pass

            if not is_dt:
                # Try numeric coercion
                try:
                    cleaned_s = (
                        df[col]
                        .astype(str)
                        .str.replace("?", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.strip()
                    )
                    num_converted = pd.to_numeric(cleaned_s, errors="coerce")
                    if num_converted.notna().sum() >= 0.5 * len(df):
                        numeric_cols.append(col)
                except Exception:
                    pass

    # If no numeric column detected, fall back to non-datetime columns
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if c not in datetime_cols]

    default_time_col = datetime_cols[0] if datetime_cols else None
    default_target_col = numeric_cols[0] if numeric_cols else None

    return datetime_cols, numeric_cols, default_time_col, default_target_col


class UniversalForecastingEngine:
    """Manages model instances and handles arbitrary multi-variable time series inference."""

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        """Initialize models cache."""
        self.results_dir = (
            results_dir or Path(__file__).resolve().parent.parent.parent / "results"
        )
        self._models: Dict[str, Any] = {}

    def get_model(self, model_name: str) -> Any:
        """Lazy-load and cache model instance."""
        if model_name in self._models:
            return self._models[model_name]

        logger.info("Instantiating %s for UI inference...", model_name)
        if model_name == "TimesFM-3 (Zero-Shot)":
            self._models[model_name] = TimesFM3ModelWrapper()
        elif model_name == "TimesFM-2.5 (Zero-Shot)":
            self._models[model_name] = TimesFM2p5ModelWrapper()
        elif model_name == "Chronos-2 (Zero-Shot)":
            self._models[model_name] = Chronos2ModelWrapper()

        elif model_name == "TimesFM-3 (Fine-Tuned)":
            ckpt = self.results_dir / "timesfm_finetuned_checkpoint.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Fine-tuned TimesFM-3 checkpoint not found at {ckpt}. Train via `uv run python train_timesfm.py`."
                )
            self._models[model_name] = TimesFM3FineTunedWrapper(checkpoint_path=ckpt)
        elif model_name == "Chronos-2 (Fine-Tuned)":
            ckpt_dir = self.results_dir / "chronos2_finetuned" / "checkpoint-final"
            if not ckpt_dir.exists():
                raise FileNotFoundError(
                    f"Fine-tuned Chronos-2 checkpoint not found at {ckpt_dir}. Train via `uv run python train_chronos.py`."
                )
            self._models[model_name] = Chronos2FineTunedWrapper(
                checkpoint_path=ckpt_dir
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return self._models[model_name]

    def run_forecasting(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: Optional[str],
        past_only_cols: Optional[List[str]],
        past_future_cols: Optional[List[str]],
        context_length: int,
        horizon: int,
        selected_models: List[str],
        backtest_mode: bool = False,
    ) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
        """Execute forecasting across selected models for arbitrary user data.

        Returns:
            (plotly_figure, metrics_dataframe, predictions_dataframe)
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")
        if not selected_models:
            raise ValueError("Please select at least one forecasting model.")

        past_only_cols = [
            c for c in (past_only_cols or []) if c in df.columns and c != target_col
        ]
        past_future_cols = [
            c for c in (past_future_cols or []) if c in df.columns and c != target_col
        ]

        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            cleaned_target = (
                df[target_col]
                .astype(str)
                .str.replace("?", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[target_col] = pd.to_numeric(cleaned_target, errors="coerce")

        df_clean = df.dropna(subset=[target_col]).copy()

        # Coerce any covariate columns to numeric
        for c in past_only_cols + past_future_cols:
            if c in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[c]):
                cleaned_c = (
                    df_clean[c]
                    .astype(str)
                    .str.replace("?", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df_clean[c] = pd.to_numeric(cleaned_c, errors="coerce").fillna(0.0)

        n_total = len(df_clean)

        if backtest_mode:
            required_len = context_length + horizon
            if n_total < required_len:
                raise ValueError(
                    f"Dataset has {n_total} rows, but backtest mode requires at least {required_len} rows (Context={context_length} + Horizon={horizon})."
                )
            context_slice = df_clean.iloc[-(context_length + horizon) : -horizon]
            horizon_slice = df_clean.iloc[-horizon:]
            actual_horizon = horizon_slice[target_col].to_numpy(dtype=np.float32)
        else:
            if n_total < context_length:
                raise ValueError(
                    f"Dataset has {n_total} rows, but context length requires at least {context_length} rows."
                )
            context_slice = df_clean.iloc[-context_length:]
            horizon_slice = None
            actual_horizon = None

        context_target = context_slice[target_col].to_numpy(dtype=np.float32)

        # Parse timestamps or generate synthetic indices
        if timestamp_col and timestamp_col in df_clean.columns:
            try:
                context_times = pd.to_datetime(context_slice[timestamp_col]).tolist()
                if backtest_mode and horizon_slice is not None:
                    horizon_times = pd.to_datetime(
                        horizon_slice[timestamp_col]
                    ).tolist()
                else:
                    # Extrapolate future timestamps
                    time_diff = context_times[-1] - context_times[-2]
                    horizon_times = [
                        context_times[-1] + time_diff * (i + 1) for i in range(horizon)
                    ]
            except Exception:
                context_times = list(range(len(context_target)))
                horizon_times = list(
                    range(len(context_target), len(context_target) + horizon)
                )
        else:
            context_times = list(range(len(context_target)))
            horizon_times = list(
                range(len(context_target), len(context_target) + horizon)
            )

        # Format past-only covariates (shape: num_covs, context_length)
        past_only_arr = None
        if past_only_cols:
            past_only_arr = context_slice[past_only_cols].to_numpy(dtype=np.float32).T

        # Format past-future covariates
        past_future_arr = None
        if past_future_cols:
            if backtest_mode and horizon_slice is not None:
                full_slice = pd.concat([context_slice, horizon_slice])
                past_future_arr = (
                    full_slice[past_future_cols].to_numpy(dtype=np.float32).T
                )
            else:
                # If future values not provided, use calendar extrapolation or repeat last values
                ctx_pf = context_slice[past_future_cols].to_numpy(dtype=np.float32).T
                future_extrap = np.tile(ctx_pf[:, -1:], (1, horizon))
                past_future_arr = np.concatenate([ctx_pf, future_extrap], axis=1)

        forecast_results: Dict[str, ForecastResult] = {}
        metrics_rows = []

        # Execute model predictions
        for model_name in selected_models:
            model = self.get_model(model_name)
            result: ForecastResult = model.forecast(
                context=context_target,
                horizon=horizon,
                past_only_covariates=past_only_arr,
                past_future_covariates=past_future_arr,
            )
            forecast_results[model_name] = result

            if backtest_mode and actual_horizon is not None:
                eval_m = evaluate_forecast(
                    model_name=model_name,
                    actual=actual_horizon,
                    point_pred=result.point_forecast,
                    quantiles=result.quantiles,
                    quantile_levels=result.quantile_levels,
                    inference_time_ms=result.inference_time_ms,
                )
                metrics_rows.append(
                    {
                        "Model": model_name,
                        "MAE": f"{eval_m.mae:.4f}",
                        "RMSE": f"{eval_m.rmse:.4f}",
                        "WAPE": f"{eval_m.wape:.4f}",
                        "CRPS": (
                            f"{eval_m.crps:.4f}" if eval_m.crps is not None else "N/A"
                        ),
                        "80% Coverage": (
                            f"{eval_m.coverage_80 * 100:.1f}%"
                            if eval_m.coverage_80 is not None
                            else "N/A"
                        ),
                        "Interval Width": (
                            f"{eval_m.avg_interval_width:.4f}"
                            if eval_m.avg_interval_width is not None
                            else "N/A"
                        ),
                        "Latency (ms)": f"{eval_m.inference_time_ms:.2f}",
                    }
                )

        metrics_df = (
            pd.DataFrame(metrics_rows)
            if metrics_rows
            else pd.DataFrame(columns=["Model", "Latency (ms)"])
        )

        # Build Predictions DataFrame for download
        pred_dict: Dict[str, Any] = {"Timestamp": horizon_times}
        if backtest_mode and actual_horizon is not None:
            pred_dict["Actual"] = np.round(actual_horizon, 4)

        for model_name, res in forecast_results.items():
            col_safe = (
                model_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )
            pred_dict[f"{col_safe}_Point"] = np.round(res.point_forecast, 4)
            if res.quantiles is not None and res.quantile_levels is not None:
                for q_idx, q_val in enumerate(res.quantile_levels):
                    pred_dict[f"{col_safe}_q{int(q_val * 100):02d}"] = np.round(
                        res.quantiles[:, q_idx], 4
                    )

        predictions_df = pd.DataFrame(pred_dict)

        # Build Interactive Plotly Figure
        fig = self._build_plotly_figure(
            context_times=context_times,
            context_target=context_target,
            horizon_times=horizon_times,
            actual_horizon=actual_horizon,
            forecast_results=forecast_results,
            target_name=target_col,
            backtest_mode=backtest_mode,
        )

        # Build In-Table Heatmap HTML Scorecard
        metrics_html = self.generate_metrics_html_table(
            metrics_rows=metrics_rows,
            backtest_mode=backtest_mode,
        )

        return fig, metrics_df, metrics_html, predictions_df

    @staticmethod
    def generate_metrics_html_table(
        metrics_rows: List[Dict[str, Any]],
        backtest_mode: bool = True,
    ) -> str:
        """Generate a sleek, responsive HTML table with in-cell heatmap background coloring."""
        if not metrics_rows:
            return '<div style="color: #94a3b8; padding: 12px; font-size: 13px;">No evaluation metrics available (run in Backtest Mode to evaluate).</div>'

        # Extract numeric columns for min/max ranking
        metric_cols = [
            "MAE",
            "RMSE",
            "WAPE",
            "CRPS",
            "80% Coverage",
            "Interval Width",
            "Latency (ms)",
        ]

        # Parse numeric matrices for heat evaluation
        col_values: Dict[str, List[Optional[float]]] = {c: [] for c in metric_cols}
        for r in metrics_rows:
            for c in metric_cols:
                raw = str(r.get(c, "N/A")).replace("%", "").strip()
                try:
                    col_values[c].append(float(raw))
                except Exception:
                    col_values[c].append(None)

        def get_cell_style(col_name: str, val: Optional[float]) -> str:
            """Compute heatmap pill styling based on relative column performance."""
            if val is None:
                return "color: #64748b; font-size: 13px;"

            valid_vals = [v for v in col_values[col_name] if v is not None]
            if len(valid_vals) <= 1:
                return "color: #e2e8f0; font-weight: 500; font-size: 13px;"

            min_v = min(valid_vals)
            max_v = max(valid_vals)

            if col_name == "80% Coverage":
                # Closer to 80.0 is better
                diffs = [abs(v - 80.0) for v in valid_vals]
                curr_diff = abs(val - 80.0)
                if curr_diff == min(diffs):
                    # Best coverage
                    return "background: rgba(16, 185, 129, 0.22); color: #34d399; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.45); padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"
                elif curr_diff == max(diffs):
                    return "background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"
                else:
                    return "background: rgba(148, 163, 184, 0.12); color: #cbd5e1; padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"

            # Lower is better for error and latency
            if max_v == min_v:
                return "color: #e2e8f0; font-weight: 500; font-size: 13px;"

            # Best in column (minimum error or fastest latency)
            if val == min_v:
                return "background: rgba(16, 185, 129, 0.22); color: #34d399; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.45); padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"
            elif val == max_v:
                # Worst in column
                return "background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"
            else:
                # Middle
                return "background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); padding: 3px 8px; border-radius: 6px; display: inline-block; font-size: 12.5px;"

        headers_html = "".join(
            f'<th style="padding: 10px 14px; text-align: {"left" if i==0 else "center"}; color: #94a3b8; font-weight: 700; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #334155;">{h}</th>'
            for i, h in enumerate(["Model"] + metric_cols)
        )

        rows_html = ""
        for row_idx, r in enumerate(metrics_rows):
            bg_row = "#0f172a" if row_idx % 2 == 0 else "#1e293b"
            model_name = r.get("Model", "")
            cells_html = f'<td style="padding: 10px 14px; font-weight: 600; color: #f8fafc; font-size: 13px; text-align: left; border-bottom: 1px solid #1e293b;"><span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {COLOR_MAP.get(model_name, "#3b82f6")}; margin-right: 8px;"></span>{model_name}</td>'

            for c in metric_cols:
                raw_str = str(r.get(c, "N/A"))
                raw_val = None
                try:
                    raw_val = float(raw_str.replace("%", "").strip())
                except Exception:
                    pass

                style = get_cell_style(c, raw_val)
                cells_html += f'<td style="padding: 8px 12px; text-align: center; border-bottom: 1px solid #1e293b;"><span style="{style}">{raw_str}</span></td>'

            rows_html += f'<tr style="background: {bg_row}; transition: background 0.15s;">{cells_html}</tr>'

        return f"""
        <div style="overflow-x: auto; border: 1px solid #334155; border-radius: 8px; margin-top: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            <table style="width: 100%; border-collapse: collapse; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">
                <thead>
                    <tr style="background: #0f172a;">{headers_html}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div style="display: flex; gap: 16px; padding: 8px 14px; background: #0f172a; border-top: 1px solid #1e293b; font-size: 11px; color: #94a3b8;">
                <span><span style="display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 2px; margin-right: 4px;"></span> Best in Column</span>
                <span><span style="display: inline-block; width: 8px; height: 8px; background: #f59e0b; border-radius: 2px; margin-right: 4px;"></span> Intermediate</span>
                <span><span style="display: inline-block; width: 8px; height: 8px; background: #ef4444; border-radius: 2px; margin-right: 4px;"></span> Lowest Rank</span>
            </div>
        </div>
        """

    def _build_plotly_figure(
        self,
        context_times: list,
        context_target: np.ndarray,
        horizon_times: list,
        actual_horizon: Optional[np.ndarray],
        forecast_results: Dict[str, ForecastResult],
        target_name: str,
        backtest_mode: bool,
    ) -> go.Figure:
        """Create clean, professional 2-panel figure matching the benchmark comparison plot layout."""
        from plotly.subplots import make_subplots

        # Tail length for historical context (show last 250 points for optimal context visibility)
        tail_len = min(250, len(context_target))
        ctx_x = context_times[-tail_len:]
        ctx_y = context_target[-tail_len:]

        bottom_title = (
            "Pointwise Absolute Error (|Forecast - Actual|)"
            if (backtest_mode and actual_horizon is not None)
            else "Model Prediction Uncertainty (80% Interval Width: q90 - q10)"
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.72, 0.28],
            subplot_titles=[
                f"Multi-Model Zero-Shot Forecast & 80% Uncertainty: {target_name}",
                bottom_title,
            ],
        )

        # 1. Historical Context (Row 1)
        fig.add_trace(
            go.Scatter(
                x=ctx_x,
                y=ctx_y,
                mode="lines",
                name="Historical Context",
                line=dict(color=COLOR_MAP["Context (History)"], width=1.8),
                hovertemplate="<b>Context</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Ground Truth Actuals (Row 1)
        if backtest_mode and actual_horizon is not None:
            fig.add_trace(
                go.Scatter(
                    x=horizon_times,
                    y=actual_horizon,
                    mode="lines",
                    name="Ground Truth Actuals",
                    line=dict(color=COLOR_MAP["Ground Truth"], width=2.4, dash="dash"),
                    hovertemplate="<b>Actual</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 3. Model 80% CI Bands and Point Predictions (Row 1)
        for model_name, res in forecast_results.items():
            base_color = COLOR_MAP.get(model_name, "#2563eb")

            # 80% Confidence Interval Shaded Area
            if res.quantiles is not None and res.quantile_levels is not None:
                if 0.1 in res.quantile_levels and 0.9 in res.quantile_levels:
                    idx_10 = res.quantile_levels.index(0.1)
                    idx_90 = res.quantile_levels.index(0.9)
                    q10 = res.quantiles[:, idx_10]
                    q90 = res.quantiles[:, idx_90]

                    fig.add_trace(
                        go.Scatter(
                            x=horizon_times,
                            y=q90,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )
                    rgba_color = self._hex_to_rgba(base_color, alpha=0.18)
                    fig.add_trace(
                        go.Scatter(
                            x=horizon_times,
                            y=q10,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=rgba_color,
                            name=f"{model_name} (80% CI)",
                            showlegend=True,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )

            # Model Point Forecast
            fig.add_trace(
                go.Scatter(
                    x=horizon_times,
                    y=res.point_forecast,
                    mode="lines",
                    name=model_name,
                    line=dict(color=base_color, width=2.2),
                    hovertemplate=f"<b>{model_name}</b><br>Time: %{{x}}<br>Pred: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Vertical Cutoff Line
        if len(ctx_x) > 0:
            fig.add_vline(
                x=ctx_x[-1],
                line_width=1.5,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="Cutoff",
                annotation_position="top left",
                row=1,
                col=1,
            )

        # 4. ROW 2: Pointwise Residuals / Absolute Error or Uncertainty Width
        for model_name, res in forecast_results.items():
            base_color = COLOR_MAP.get(model_name, "#2563eb")

            if backtest_mode and actual_horizon is not None:
                abs_err = np.abs(res.point_forecast - actual_horizon)
                fig.add_trace(
                    go.Scatter(
                        x=horizon_times,
                        y=abs_err,
                        mode="lines",
                        name=f"{model_name} Error",
                        line=dict(color=base_color, width=1.8),
                        showlegend=False,
                        hovertemplate=f"<b>{model_name} Error</b><br>Time: %{{x}}<br>|Err|: %{{y:.2f}}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
            elif res.quantiles is not None and res.quantile_levels is not None:
                if 0.1 in res.quantile_levels and 0.9 in res.quantile_levels:
                    idx_10 = res.quantile_levels.index(0.1)
                    idx_90 = res.quantile_levels.index(0.9)
                    ci_width = res.quantiles[:, idx_90] - res.quantiles[:, idx_10]
                    fig.add_trace(
                        go.Scatter(
                            x=horizon_times,
                            y=ci_width,
                            mode="lines",
                            name=f"{model_name} Width",
                            line=dict(color=base_color, width=1.8),
                            showlegend=False,
                            hovertemplate=f"<b>{model_name} Width</b><br>Time: %{{x}}<br>Spread: %{{y:.2f}}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

        # Figure Dimensions and Theme Styling
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=55, r=30, t=65, b=45),
            height=680,
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
        )

        return fig

    @staticmethod
    def _hex_to_rgba(hex_code: str, alpha: float = 0.2) -> str:
        """Convert HEX color code to CSS rgba string."""
        hex_code = hex_code.lstrip("#")
        if len(hex_code) == 6:
            r, g, b = tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        return f"rgba(37,99,235,{alpha})"
