"""Script to generate demo time series datasets for the Gradio app."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_datasets() -> None:
    """Generate sample datasets for retail sales and energy demand."""
    samples_dir = Path(__file__).resolve().parent.parent.parent / "sample_data"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hourly Energy Demand Dataset (1 year, hourly)
    np.random.seed(42)
    dates_energy = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    n_energy = len(dates_energy)

    hours = dates_energy.hour.to_numpy()
    days = dates_energy.dayofweek.to_numpy()
    dayofyear = dates_energy.dayofyear.to_numpy()

    # Temperature seasonal component
    temp = (
        15.0
        + 12.0 * np.sin(2 * np.pi * (dayofyear - 100) / 365.25)
        + 5.0 * np.sin(2 * np.pi * (hours - 8) / 24)
        + np.random.normal(0, 2.0, n_energy)
    )

    # Energy demand driven by temperature (AC in summer, heating in winter), diurnal cycles, weekday/weekend
    base_demand = 500.0 + 200.0 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekday_effect = np.where(days < 5, 80.0, -40.0)
    weather_effect = 10.0 * np.abs(temp - 18.0)
    noise = np.random.normal(0, 25.0, n_energy)

    energy_demand = base_demand + weekday_effect + weather_effect + noise
    solar_generation = np.maximum(
        0,
        300.0 * np.sin(np.pi * (hours - 6) / 12) * (1 - 0.3 * np.random.rand(n_energy)),
    )
    solar_generation = np.where((hours >= 6) & (hours <= 18), solar_generation, 0.0)

    df_energy = pd.DataFrame(
        {
            "timestamp": dates_energy.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_demand_mw": np.round(energy_demand, 2),
            "temperature_c": np.round(temp, 2),
            "solar_output_mw": np.round(solar_generation, 2),
            "is_weekend": (days >= 5).astype(int),
        }
    )
    df_energy.to_csv(samples_dir / "hourly_energy_grid.csv", index=False)

    # 2. Daily Retail Store Sales (2 years, daily)
    dates_sales = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n_sales = len(dates_sales)
    t = np.arange(n_sales)

    trend = 1000.0 + 1.5 * t
    weekly_season = 300.0 * np.sin(2 * np.pi * dates_sales.dayofweek / 7)
    annual_season = 400.0 * np.sin(2 * np.pi * dates_sales.dayofyear / 365.25)
    promo = np.random.binomial(1, 0.15, n_sales)
    promo_boost = promo * 600.0
    discount_pct = promo * np.random.uniform(10, 30, n_sales)
    sales_noise = np.random.normal(0, 80.0, n_sales)

    daily_sales = trend + weekly_season + annual_season + promo_boost + sales_noise
    foot_traffic = np.clip(
        daily_sales * 0.45 + np.random.normal(0, 40.0, n_sales), 50, None
    )

    df_sales = pd.DataFrame(
        {
            "date": dates_sales.strftime("%Y-%m-%d"),
            "store_sales_usd": np.round(daily_sales, 2),
            "foot_traffic_count": np.round(foot_traffic, 0).astype(int),
            "promotion_active": promo,
            "discount_percentage": np.round(discount_pct, 1),
        }
    )
    df_sales.to_csv(samples_dir / "daily_retail_sales.csv", index=False)


if __name__ == "__main__":
    generate_sample_datasets()
