import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "processed" / "energy_features_kwh_weather.csv"

metrics_dir = BASE_DIR / "outputs" / "metrics"
models_dir = BASE_DIR / "outputs" / "models"
metrics_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading feature dataset with weather features...")

df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

X = df.drop("energy_kwh", axis=1)
y = df["energy_kwh"]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# Naive baseline
y_naive = df["lag_1"]

naive_mae = mean_absolute_error(y, y_naive)
naive_rmse = np.sqrt(mean_squared_error(y, y_naive))
naive_r2 = r2_score(y, y_naive)
naive_mape = mean_absolute_percentage_error(y, y_naive)

print("\nNaive Persistence Baseline")
print("MAE:", round(naive_mae, 3))
print("RMSE:", round(naive_rmse, 3))
print("R²:", round(naive_r2, 3))
print("MAPE:", round(naive_mape, 3))
print("-" * 30)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror"
    )
}

tscv = TimeSeriesSplit(n_splits=5)
results = []

print("\nRunning TimeSeries cross-validation...\n")

results.append([
    "Naive Persistence",
    naive_mae,
    naive_rmse,
    naive_r2,
    naive_mape,
    0.0,
    0.0,
    0.0,
    0.0
])

best_model_name = None
best_model_mean_mae = float("inf")

for name, model in models.items():
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    mape_scores = []

    fold = 1

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)

        print(f"{name} | Fold {fold}")
        print("MAE:", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("R²:", round(r2, 3))
        print("MAPE:", round(mape, 3))
        print("-" * 30)

        fold += 1

    mean_mae = np.mean(mae_scores)

    if mean_mae < best_model_mean_mae:
        best_model_mean_mae = mean_mae
        best_model_name = name

    results.append([
        name,
        np.mean(mae_scores),
        np.mean(rmse_scores),
        np.mean(r2_scores),
        np.mean(mape_scores),
        np.std(mae_scores),
        np.std(rmse_scores),
        np.std(r2_scores),
        np.std(mape_scores)
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Model", "Mean_MAE", "Mean_RMSE", "Mean_R2", "Mean_MAPE",
        "Std_MAE", "Std_RMSE", "Std_R2", "Std_MAPE"
    ]
)

results_path = metrics_dir / "timeseries_cv_results_weather.csv"
results_df.to_csv(results_path, index=False)

print("\nCross-validation summary:")
print(results_df)
print("\nResults saved to:", results_path)

print("\nRunning Random Forest hyperparameter tuning...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error"
)
grid.fit(X, y)

best_rf_params = grid.best_params_
print("Best Random Forest parameters:", best_rf_params)

# Final production model
final_model = RandomForestRegressor(
    n_estimators=best_rf_params["n_estimators"],
    max_depth=best_rf_params["max_depth"],
    random_state=42
)
final_model.fit(X, y)

model_path = models_dir / "rf_model.pkl"
joblib.dump(final_model, model_path)

print("Saved trained Random Forest model to:", model_path)
print("Best cross-validation model by MAE:", best_model_name)