"""
GreenGrid AI -- Energy Consumption Forecasting: Prediction
==========================================================
Loads the saved best model and generates predictions.

Public API
----------
load_model()          -> (model, feature_cols)
predict_single(row)   -> float  (predicted kWh for one hour)
predict_next_24h(row) -> list[dict]  (24-hour ahead forecast)

Usage (module):
    from ml.predict import predict_single, predict_next_24h

Usage (CLI):
    python -m ml.predict --forecast24
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "forecast_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "forecast_feature_columns.pkl")

_model        = None
_feature_cols = None

_BUILDING_LABEL = {"office": 0, "residential": 1, "retail": 2}
_PEAK_HOURS     = set(range(8, 11)) | set(range(17, 21))


def load_model():
    global _model, _feature_cols
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run  python ml/train_model.py  first.")
        _model        = joblib.load(MODEL_PATH)
        _feature_cols = joblib.load(FEATURES_PATH)
    return _model, _feature_cols


def _build_feature_row(row):
    _, feature_cols = load_model()
    ts = pd.to_datetime(row["timestamp"])
    building_enc    = _BUILDING_LABEL.get(str(row.get("building_type", "office")).lower(), 0)
    temperature_c   = float(row.get("temperature_c",    20.0))
    occupants       = float(row.get("occupants",         0))
    ac_usage        = float(row.get("ac_usage",          0.0))
    appliance_usage = float(row.get("appliance_usage",   0.0))
    peak_hour       = int(row.get("peak_hour", 1 if ts.hour in _PEAK_HOURS else 0))
    engineered = {
        "hour"             : ts.hour,
        "day_of_week"      : ts.dayofweek,
        "month"            : ts.month,
        "day_of_year"      : ts.day_of_year,
        "is_weekend"       : int(ts.dayofweek >= 5),
        "temperature_c"    : temperature_c,
        "occupants"        : occupants,
        "ac_usage"         : ac_usage,
        "appliance_usage"  : appliance_usage,
        "peak_hour"        : peak_hour,
        "building_type_enc": building_enc,
        "temp_x_occupants" : temperature_c * occupants,
        "ac_x_temp"        : ac_usage      * temperature_c,
    }
    return np.array([engineered[col] for col in feature_cols]).reshape(1, -1)


def predict_single(row):
    """Return predicted consumption_kwh (float) for one input row dict."""
    model, _ = load_model()
    return float(round(model.predict(_build_feature_row(row))[0], 3))


def predict_next_24h(last_row):
    """Return list of 24 dicts: [{'timestamp': ..., 'predicted_kwh': ...}]."""
    model, _      = load_model()
    base_ts        = pd.to_datetime(last_row["timestamp"]) + pd.Timedelta(hours=1)
    building_type  = str(last_row.get("building_type", "office")).lower()
    base_temp      = float(last_row.get("temperature_c", 18.0))
    results = []
    for offset in range(24):
        ts   = base_ts + pd.Timedelta(hours=offset)
        hour = ts.hour
        temp = base_temp + 5 * np.cos(2 * np.pi * (hour - 14) / 24)
        temp = round(float(temp), 1)
        if building_type == "office":
            occupants = 50 if (ts.dayofweek < 5 and 8 <= hour < 18) else \
                        10 if (ts.dayofweek < 5 and (7 <= hour < 8 or 18 <= hour < 20)) else 0
        elif building_type == "residential":
            occupants = 20 if (7 <= hour < 9 or 17 <= hour < 23) else 8
        else:
            occupants = 40 if 10 <= hour < 21 else 5
        ac_usage        = round(max(0.5 * max(temp - 22.0, 0) + 0.02 * occupants, 0.0), 2)
        appliance_usage = round(0.05 * occupants + (1.0 if hour in _PEAK_HOURS else 0.3), 2)
        peak_hour       = 1 if hour in _PEAK_HOURS else 0
        row = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "temperature_c": temp,
               "building_type": building_type, "occupants": occupants,
               "ac_usage": ac_usage, "appliance_usage": appliance_usage, "peak_hour": peak_hour}
        pred = float(round(model.predict(_build_feature_row(row))[0], 3))
        results.append({"timestamp": row["timestamp"], "predicted_kwh": pred})
    return results


if __name__ == "__main__":
    if "--forecast24" in sys.argv:
        import csv
        with open(os.path.join(BASE_DIR, "data", "energy_data.csv"), newline="") as f:
            last = list(csv.DictReader(f))[-1]
        print(f"24-hour forecast from: {last['timestamp']}")
        for entry in predict_next_24h(last):
            print(f"  {entry['timestamp']}  {entry['predicted_kwh']:>8.3f} kWh")
    else:
        row = {"timestamp": "2023-08-15 14:00:00", "temperature_c": 32.5,
               "building_type": "office", "occupants": 55,
               "ac_usage": 4.2, "appliance_usage": 3.1, "peak_hour": 0}
        print(f"Predicted: {predict_single(row)} kWh")
