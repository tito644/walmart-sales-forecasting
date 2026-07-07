"""
train_model.py
==============
Run this script ONCE to train and save the XGBoost model on your local machine.
This ensures the saved model is compatible with your local XGBoost version.

Usage:
    python train_model.py

Output:
    models/xgb_sales_model.pkl
    models/model_features.pkl
    models/model_metrics.pkl   (RMSE, MAE, MAPE on the 12-week test set)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

# ── 1. Load processed data ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "processed", "walmart_clean.csv")

print("Loading data...")
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# ── 2. Feature engineering (lag features per store) ───────────────────────────
print("Building lag features...")
df["lag_1"] = df.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(1))
df["lag_52"] = df.groupby("Store")["Weekly_Sales"].transform(lambda s: s.shift(52))
df["rolling_mean_4"] = df.groupby("Store")["Weekly_Sales"].transform(
    lambda s: s.shift(1).rolling(4).mean()
)

model_df = df.dropna(subset=["lag_1", "lag_52", "rolling_mean_4"]).reset_index(drop=True)
print(f"  Rows after dropping NaN lag rows: {len(model_df)}")

# ── 3. Time-based train/test split (last 12 weeks = test) ─────────────────────
cutoff_date = model_df["Date"].max() - pd.Timedelta(weeks=12)
train = model_df[model_df["Date"] <= cutoff_date].copy()
test = model_df[model_df["Date"] > cutoff_date].copy()

print(f"  Train: {len(train)} rows | Test: {len(test)} rows")
print(f"  Cutoff: {cutoff_date.date()}")

# ── 4. Train XGBoost with TimeSeriesSplit GridSearch ──────────────────────────
FEATURES = [
    "Store", "Month", "Week", "Holiday_Flag",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "lag_1", "lag_52", "rolling_mean_4",
]
TARGET = "Weekly_Sales"

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print("\nTraining XGBoost (GridSearch + TimeSeriesSplit)...")
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")
grid = GridSearchCV(
    xgb_base, param_grid, cv=tscv,
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"\nBest hyperparameters: {grid.best_params_}")

# ── 5. Evaluate on test set ───────────────────────────────────────────────────
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print(f"\nTest set evaluation:")
print(f"  RMSE : ${rmse:>10,.0f}")
print(f"  MAE  : ${mae:>10,.0f}")
print(f"  MAPE : {mape:.2f}%")

# ── 6. Save model, features, and metrics ─────────────────────────────────────
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(models_dir, "xgb_sales_model.pkl"))
joblib.dump(FEATURES, os.path.join(models_dir, "model_features.pkl"))
joblib.dump(
    {"rmse": rmse, "mae": mae, "mape": mape, "best_params": grid.best_params_},
    os.path.join(models_dir, "model_metrics.pkl"),
)

print("\n✅ Model saved successfully:")
print(f"   {os.path.join(models_dir, 'xgb_sales_model.pkl')}")
print(f"   {os.path.join(models_dir, 'model_features.pkl')}")
print(f"   {os.path.join(models_dir, 'model_metrics.pkl')}")
print("\nYou can now run the Streamlit app: streamlit run app/app.py")
