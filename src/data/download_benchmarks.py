"""Script to download diverse real-world time series benchmark datasets."""

import logging
from pathlib import Path
import urllib.request
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    # 1. Electricity Transformer Temperature (ETTh1 Benchmark - 17,420 rows, multivariate)
    "ett_electricity_transformer.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    # 2. Melbourne Daily Minimum Temperatures (3,650 rows)
    "melbourne_daily_temperatures.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    # 3. Monthly Sunspots Solar Cycles (2,820 rows, 1749–1983)
    "monthly_sunspots.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
    # 4. Monthly Airline Passenger Numbers (144 rows, classic trend + seasonality)
    "monthly_airline_passengers.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    # 5. Monthly Quebec Car Sales (108 rows)
    "monthly_car_sales.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv",
}


def download_datasets() -> None:
    """Download and sanitize real-world datasets."""
    for filename, url in DATASETS.items():
        out_path = SAMPLES_DIR / filename
        logger.info("Downloading %s from %s...", filename, url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                content = response.read().decode("utf-8", errors="replace")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Sanitize / clean with pandas
            df = pd.read_csv(out_path)

            # Fix dirty strings if any
            for col in df.columns:
                if df[col].dtype == object and col.lower() not in [
                    "date",
                    "datetime",
                    "timestamp",
                    "month",
                ]:
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

            df.to_csv(out_path, index=False)
            logger.info(
                "Successfully saved %s (%d rows, %d columns)",
                filename,
                len(df),
                len(df.columns),
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", filename, e)


if __name__ == "__main__":
    download_datasets()
