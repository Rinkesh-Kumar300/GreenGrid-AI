"""
GreenGrid AI -- Energy Consumption Forecasting: Model Training
==============================================================
Trains and compares two regression models:
  1. Random Forest Regressor
  2. Gradient Boosting Regressor

Evaluation metrics: MAE, RMSE, R²
Best model saved to: models/forecast_model.pkl

Usage:
  python ml/train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "energy_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "forecast_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "forecast_feature_columns.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "forecast_model_metadata.pkl")

RANDOM_STATE = 42


def engineer_features(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"]              = ts.dt.hour
    df["day_of_week"]       = ts.dt.dayofweek
    df["month"]             = ts.dt.month
    df["day_of_year"]       = ts.dt.day_of_year
    df["is_weekend"]        = (ts.dt.dayofweek >= 5).astype(int)
    df["building_type_enc"] = df["building_type"].map({"office": 0, "residential": 1, "retail": 2}).fillna(0)
    df["temp_x_occupants"]  = df["temperature_c"] * df["occupants"]
    df["ac_x_temp"]         = df["ac_usage"]       * df["temperature_c"]
    return df


def get_feature_columns():
    return [
        "hour", "day_of_week", "month", "day_of_year", "is_weekend",
        "temperature_c", "occupants", "ac_usage", "appliance_usage", "peak_hour",
        "building_type_enc", "temp_x_occupants", "ac_x_temp",
    ]


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("\nLoading dataset ...")
    if not os.path.isfile(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows")
    df = df[df["is_anomaly"] == 0].copy()
    print(f"  Using {len(df):,} normal rows (anomalies excluded from training)")

    df = engineer_features(df)
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df["consumption_kwh"].values

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE),
    }

    print("\n" + "=" * 60)
    metrics_list = []
    trained_models = {}
    for name, model in models.items():
        print(f"\n  [{name}] Training ...")
        model.fit(X_train, y_train)
        train_m = evaluate(f"{name} (train)", model.predict(X_train), y_train)
        test_m  = evaluate(f"{name} (test)",  model.predict(X_test),  y_test)
        print(f"    Train -> MAE:{train_m['MAE']:.4f}  RMSE:{train_m['RMSE']:.4f}  R2:{train_m['R2']:.4f}")
        print(f"    Test  -> MAE:{test_m['MAE']:.4f}  RMSE:{test_m['RMSE']:.4f}  R2:{test_m['R2']:.4f}")
        metrics_list.append({**test_m, "model_key": name})
        trained_models[name] = model

    print("\n" + "=" * 60)
    print(f"  {'Model':<22}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}")
    for m in metrics_list:
        print(f"  {m['model']:<22}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {m['R2']:>8.4f}")

    best      = min(metrics_list, key=lambda m: m["MAE"])
    best_name = best["model_key"]
    best_model= trained_models[best_name]
    print(f"\n  Best model: {best_name}  (test MAE: {best['MAE']:.4f})")

    if hasattr(best_model, "feature_importances_"):
        fi = sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\n  Feature importances:")
        for feat, imp in fi:
            print(f"    {feat:<22}  {imp:.4f}  {'#'*int(imp*50)}")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump({"model_name": best_name, "mae": best["MAE"], "rmse": best["RMSE"],
                 "r2": best["R2"], "feature_cols": feature_cols,
                 "train_rows": len(X_train), "test_rows": len(X_test)}, METADATA_PATH)
    print(f"\nSaved -> {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
