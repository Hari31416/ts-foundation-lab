"""Dataset loading, preprocessing, and partitioning module for the Weather benchmark dataset."""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkWindow:
    """Container for time series benchmark window."""

    context_target: np.ndarray  # Shape: (context_length,)
    horizon_target: np.ndarray  # Shape: (horizon,)
    past_only_context: np.ndarray  # Shape: (num_past_cov, context_length)
    past_future_full: np.ndarray  # Shape: (num_future_cov, context_length + horizon)
    timestamps_context: pd.DatetimeIndex
    timestamps_horizon: pd.DatetimeIndex
    target_name: str
    past_only_names: List[str]
    past_future_names: List[str]


class WeatherDatasetLoader:
    """Loader and preprocessor for the standard Jena Climate Weather benchmark dataset."""

    DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    ZIP_FILENAME = "jena_climate_2009_2016.csv.zip"
    CSV_FILENAME = "jena_climate_2009_2016.csv"

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize loader with optional cache directory.

        Args:
            cache_dir: Directory to cache downloaded raw dataset.
        """
        self.cache_dir = (
            cache_dir or Path(__file__).resolve().parent.parent.parent / "data"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.cache_dir / self.CSV_FILENAME

    def download_and_load(self) -> pd.DataFrame:
        """Download raw dataset if not present and return cleaned dataframe.

        Returns:
            pd.DataFrame: Cleaned time series dataset indexed by Datetime.
        """
        if not self.csv_path.exists():
            logger.info("Downloading Weather dataset from %s", self.DATA_URL)
            try:
                response = requests.get(self.DATA_URL, timeout=60)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    zip_file.extract(self.CSV_FILENAME, path=self.cache_dir)
                logger.info("Extracted dataset to %s", self.csv_path)
            except Exception as exc:
                logger.error("Failed to download Weather dataset: %s", exc)
                raise RuntimeError(
                    f"Could not download Weather dataset: {exc}"
                ) from exc
        else:
            logger.info("Found cached Weather dataset at %s", self.csv_path)

        df = pd.read_csv(self.csv_path)
        df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
        df = df.sort_values("Date Time").reset_index(drop=True)
        df = df.set_index("Date Time")

        # Handle anomalous wind values (-9999) if present
        if "wv (m/s)" in df.columns:
            df["wv (m/s)"] = df["wv (m/s)"].clip(lower=0.0)
        if "max. wv (m/s)" in df.columns:
            df["max. wv (m/s)"] = df["max. wv (m/s)"].clip(lower=0.0)

        # Forward fill any potential missing values
        df = df.ffill().bfill()
        return df

    def extract_covariates(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Separate dataset into Target, Past-Only Covariates, and Past-Future Covariates.

        Args:
            df: Input cleaned weather dataframe.

        Returns:
            Tuple of (targets_df, past_only_df, past_future_df)
        """
        target_col = "T (degC)"
        past_only_cols = [
            "p (mbar)",
            "rh (%)",
            "wv (m/s)",
            "Tdew (degC)",
            "VPdef (mbar)",
            "rho (g/m**3)",
        ]

        # Verify columns exist
        available_past_cols = [col for col in past_only_cols if col in df.columns]

        # Compute cyclical calendar signals for past-future covariates
        timestamps = df.index
        hour = timestamps.hour + timestamps.minute / 60.0
        day_of_week = timestamps.dayofweek
        day_of_year = timestamps.dayofyear

        past_future_dict = {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dayofweek_sin": np.sin(2 * np.pi * day_of_week / 7.0),
            "dayofweek_cos": np.cos(2 * np.pi * day_of_week / 7.0),
            "dayofyear_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "dayofyear_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        }
        past_future_df = pd.DataFrame(past_future_dict, index=timestamps)

        targets_df = df[[target_col]]
        past_only_df = df[available_past_cols]

        return targets_df, past_only_df, past_future_df

    def get_benchmark_window(
        self,
        context_length: int = 512,
        horizon: int = 96,
        start_idx: Optional[int] = None,
    ) -> BenchmarkWindow:
        """Extract a single standardized benchmark window with context and horizon.

        Args:
            context_length: Historical context steps (>= 512).
            horizon: Prediction horizon steps (96).
            start_idx: Optional start index. If None, uses a stable test slice.

        Returns:
            BenchmarkWindow dataclass with all partitioned components.
        """
        df = self.download_and_load()
        targets_df, past_only_df, past_future_df = self.extract_covariates(df)

        total_length = context_length + horizon
        if start_idx is None:
            # Pick a representative test window in the final 10% of data
            test_start = int(len(df) * 0.85)
            start_idx = test_start

        if start_idx + total_length > len(df):
            raise ValueError(
                f"Window range [{start_idx}:{start_idx + total_length}] exceeds dataset size {len(df)}"
            )

        window_targets = targets_df.iloc[start_idx : start_idx + total_length]
        window_past_only = past_only_df.iloc[start_idx : start_idx + total_length]
        window_past_future = past_future_df.iloc[start_idx : start_idx + total_length]

        context_target = window_targets.iloc[:context_length, 0].to_numpy(
            dtype=np.float32
        )
        horizon_target = window_targets.iloc[context_length:, 0].to_numpy(
            dtype=np.float32
        )

        # Past-only covariates: shape (num_covariates, context_length)
        past_only_context = (
            window_past_only.iloc[:context_length].to_numpy(dtype=np.float32).T
        )

        # Past-future covariates: shape (num_covariates, context_length + horizon)
        past_future_full = window_past_future.to_numpy(dtype=np.float32).T

        timestamps_context = window_targets.index[:context_length]
        timestamps_horizon = window_targets.index[context_length:]

        return BenchmarkWindow(
            context_target=context_target,
            horizon_target=horizon_target,
            past_only_context=past_only_context,
            past_future_full=past_future_full,
            timestamps_context=timestamps_context,
            timestamps_horizon=timestamps_horizon,
            target_name=targets_df.columns[0],
            past_only_names=list(past_only_df.columns),
            past_future_names=list(past_future_df.columns),
        )

    def get_rolling_benchmark_windows(
        self,
        num_windows: int = 12,
        context_length: int = 512,
        horizon: int = 96,
        start_ratio: float = 0.80,
        end_ratio: float = 0.98,
    ) -> List[Tuple[int, BenchmarkWindow]]:
        """Extract multiple evenly-spaced rolling benchmark windows across the evaluation partition.

        Args:
            num_windows: Number of rolling windows to extract (e.g., 12).
            context_length: Historical context steps (512).
            horizon: Forecast horizon steps (96).
            start_ratio: Proportion of dataset to begin sampling from.
            end_ratio: Proportion of dataset to end sampling at.

        Returns:
            List of tuples (start_idx, BenchmarkWindow).
        """
        df = self.download_and_load()
        targets_df, past_only_df, past_future_df = self.extract_covariates(df)
        total_length = context_length + horizon
        n_rows = len(df)

        min_start = int(n_rows * start_ratio)
        max_start = int(n_rows * end_ratio) - total_length

        if max_start <= min_start:
            raise ValueError(
                f"Invalid range for rolling windows: [{min_start}, {max_start}]"
            )

        step_indices = np.linspace(min_start, max_start, num_windows, dtype=int)
        windows: List[Tuple[int, BenchmarkWindow]] = []

        for start_idx in step_indices:
            window_targets = targets_df.iloc[start_idx : start_idx + total_length]
            window_past_only = past_only_df.iloc[start_idx : start_idx + total_length]
            window_past_future = past_future_df.iloc[
                start_idx : start_idx + total_length
            ]

            context_target = window_targets.iloc[:context_length, 0].to_numpy(
                dtype=np.float32
            )
            horizon_target = window_targets.iloc[context_length:, 0].to_numpy(
                dtype=np.float32
            )

            past_only_context = (
                window_past_only.iloc[:context_length].to_numpy(dtype=np.float32).T
            )
            past_future_full = window_past_future.to_numpy(dtype=np.float32).T

            timestamps_context = window_targets.index[:context_length]
            timestamps_horizon = window_targets.index[context_length:]

            window = BenchmarkWindow(
                context_target=context_target,
                horizon_target=horizon_target,
                past_only_context=past_only_context,
                past_future_full=past_future_full,
                timestamps_context=timestamps_context,
                timestamps_horizon=timestamps_horizon,
                target_name=targets_df.columns[0],
                past_only_names=list(past_only_df.columns),
                past_future_names=list(past_future_df.columns),
            )
            windows.append((int(start_idx), window))

        return windows

    def get_train_val_test_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.09,
        context_length: int = 512,
        horizon: int = 96,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        """Extract chronologically isolated train, val, and test partitions with zero data leakage.

        Returns:
            Tuple of (train_splits, val_splits, test_splits), where each is (targets, past_only, past_future).
        """
        df = self.download_and_load()
        targets_df, past_only_df, past_future_df = self.extract_covariates(df)
        n_rows = len(df)
        buffer_len = context_length + horizon

        train_end = int(n_rows * train_ratio)
        val_start = train_end + buffer_len
        val_end = val_start + int(n_rows * val_ratio)
        test_start = val_end + buffer_len

        logger.info(
            "Temporal Splits: Train [0:%d], Val [%d:%d], Test [%d:%d] (Buffer=%d steps)",
            train_end,
            val_start,
            val_end,
            test_start,
            n_rows,
            buffer_len,
        )

        train_split = (
            targets_df.iloc[:train_end],
            past_only_df.iloc[:train_end],
            past_future_df.iloc[:train_end],
        )
        val_split = (
            targets_df.iloc[val_start:val_end],
            past_only_df.iloc[val_start:val_end],
            past_future_df.iloc[val_start:val_end],
        )
        test_split = (
            targets_df.iloc[test_start:],
            past_only_df.iloc[test_start:],
            past_future_df.iloc[test_start:],
        )

        return train_split, val_split, test_split
