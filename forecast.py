import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "processed" / "energy_features_kwh_weather.csv"
output_csv = BASE_DIR / "outputs" / "metrics" / "forecast_with_ci.csv"
output_plot = BASE_DIR / "outputs" / "figures" / "forecast_with_ci.png"

output_csv.parent.mkdir(parents=True, exist_ok=True)
output_plot.parent.mkdir(parents=True, exist_ok=True)

print("Loading feature dataset...")

df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

X = df.drop("energy_kwh", axis=1)
y = df["energy_kwh"]

train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
errors = y_test - y_pred_test
std_dev = float(np.std(errors))

print("Forecasting next 7 days...\n")

model.fit(X, y)

future_predictions = []
current = df.iloc[-1:].copy()
last_date = df.index[-1]

for step in range(7):
    pred = float(model.predict(current.drop(columns=["energy_kwh"]))[0])
    future_predictions.append(pred)

    next_date = last_date + timedelta(days=step + 1)
    new_row = pd.DataFrame(index=[next_date])

    new_row["day_of_week"] = next_date.dayofweek
    new_row["month"] = next_date.month
    new_row["is_weekend"] = int(next_date.dayofweek >= 5)
    new_row["day_of_year"] = next_date.timetuple().tm_yday

    new_row["temperature"] = current["temperature"].iloc[0]
    new_row["humidity"] = current["humidity"].iloc[0]

    new_row["lag_1"] = pred
    new_row["lag_7"] = current["lag_1"].iloc[0]
    new_row["lag_14"] = current["lag_7"].iloc[0]

    rolling_values = [
        pred,
        current["lag_1"].iloc[0],
        current["lag_7"].iloc[0],
        current["lag_14"].iloc[0],
        current["rolling_mean_7"].iloc[0],
        current["rolling_mean_14"].iloc[0],
        current["energy_kwh"].iloc[0]
    ]

    new_row["rolling_mean_7"] = np.mean(rolling_values)
    new_row["rolling_std_7"] = np.std(rolling_values)
    new_row["rolling_mean_14"] = np.mean([
        pred,
        current["lag_1"].iloc[0],
        current["lag_7"].iloc[0],
        current["lag_14"].iloc[0],
        current["rolling_mean_7"].iloc[0],
        current["rolling_mean_14"].iloc[0]
    ])

    new_row["energy_kwh"] = pred
    current = new_row

future_dates = [last_date + timedelta(days=i + 1) for i in range(7)]
future_predictions = np.array(future_predictions)

upper_bound = future_predictions + 2 * std_dev
lower_bound = future_predictions - 2 * std_dev

forecast_df = pd.DataFrame({
    "date": future_dates,
    "prediction_kwh": future_predictions,
    "lower_ci_kwh": lower_bound,
    "upper_ci_kwh": upper_bound
})
forecast_df.to_csv(output_csv, index=False)

print("Next 7 days forecast (kWh):")
for i, row in forecast_df.iterrows():
    print(
        f"Day {i+1}: "
        f"{row['prediction_kwh']:.2f} kWh "
        f"(95% CI: {row['lower_ci_kwh']:.2f} to {row['upper_ci_kwh']:.2f})"
    )

recent_actual = df["energy_kwh"].iloc[-30:]

plt.figure(figsize=(10, 5))
plt.plot(recent_actual.index, recent_actual.values, label="Recent Actual")
plt.plot(future_dates, future_predictions, marker="o", label="Forecast")
plt.fill_between(
    future_dates,
    lower_bound,
    upper_bound,
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.title("7-Day Energy Forecast with Confidence Interval")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot)
plt.show()

print("\nForecast saved to:", output_csv)
print("Plot saved to:", output_plot)