import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "processed" / "energy_daily_kwh.csv"
output_file = BASE_DIR / "data" / "processed" / "energy_features_kwh_weather.csv"

print("Loading processed kWh dataset...")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
df["day_of_year"] = df.index.dayofyear

np.random.seed(42)

df["temperature"] = (
    15
    + 10 * np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    + np.random.normal(0, 2, len(df))
)

df["humidity"] = (
    65
    - 10 * np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    + np.random.normal(0, 5, len(df))
)

df["temperature"] = df["temperature"].clip(lower=-5, upper=35)
df["humidity"] = df["humidity"].clip(lower=20, upper=100)

df["lag_1"] = df["energy_kwh"].shift(1)
df["lag_7"] = df["energy_kwh"].shift(7)
df["lag_14"] = df["energy_kwh"].shift(14)

df["rolling_mean_7"] = df["energy_kwh"].rolling(7).mean()
df["rolling_std_7"] = df["energy_kwh"].rolling(7).std()
df["rolling_mean_14"] = df["energy_kwh"].rolling(14).mean()

df = df.dropna()

print("Feature dataset shape:", df.shape)

df.to_csv(output_file)

print("Feature dataset saved to:", output_file)
print("\nFirst 5 rows:")
print(df.head())