"""
GreenGrid AI -- Anomaly Detection Module
=========================================
Detects abnormal energy consumption by comparing the ACTUAL reading
against the PREDICTED value from the trained forecasting model.

How it works
------------
1. Run the actual sensor readings through the saved model to get the expected value.
2. Calculate the gap:  difference = actual - predicted
3. Convert the gap to a percentage of the predicted value.
4. Classify: Normal / Elevated / Abnormal using configurable thresholds.
5. Check each input feature against simple thresholds to list likely causes.

Public API
----------
detect(actual_kwh: float, row: dict) -> dict

Usage:
    from ml.anomaly import detect
    result = detect(actual_kwh=22.5, row={...})
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predict import predict_single

# ── Configurable classification thresholds ────────────────────────────────────
ELEVATED_THRESHOLD_PCT = 10.0   # % above predicted -> "Elevated"
ABNORMAL_THRESHOLD_PCT = 25.0   # % above predicted -> "Abnormal"

# ── Configurable factor-detection thresholds ──────────────────────────────────
FACTOR_THRESHOLDS = {
    "temperature_c"   : ("High temperature",     30.0),
    "ac_usage"        : ("High AC usage",          5.0),
    "occupants"       : ("High occupancy",         60.0),
    "appliance_usage" : ("High appliance usage",   4.0),
    "peak_hour"       : ("Peak-hour consumption",  0.5),
}


def calculate_difference(actual: float, predicted: float) -> float:
    """Return actual - predicted, rounded to 3 dp."""
    return round(actual - predicted, 3)


def calculate_deviation_percent(actual: float, predicted: float) -> float:
    """Return ((actual - predicted) / predicted) * 100. Returns 0 if predicted == 0."""
    if predicted == 0:
        return 0.0
    return round(((actual - predicted) / predicted) * 100, 2)


def classify_status(deviation_pct: float) -> str:
    """Map deviation percentage to Normal / Elevated / Abnormal."""
    if deviation_pct < ELEVATED_THRESHOLD_PCT:
        return "Normal"
    elif deviation_pct < ABNORMAL_THRESHOLD_PCT:
        return "Elevated"
    else:
        return "Abnormal"


def identify_factors(row: dict) -> list:
    """Return list of human-readable factor labels for features above threshold."""
    factors = []
    for feature_key, (label, threshold) in FACTOR_THRESHOLDS.items():
        try:
            value = float(row.get(feature_key, 0))
        except (TypeError, ValueError):
            value = 0.0
        if value > threshold:
            factors.append(label)
    return factors


def detect(actual_kwh: float, row: dict) -> dict:
    """
    Full anomaly analysis for one hourly reading.

    Parameters
    ----------
    actual_kwh : float  -- measured energy consumption (kWh)
    row        : dict   -- feature dict (timestamp, temperature_c, building_type,
                           occupants, ac_usage, appliance_usage, peak_hour)

    Returns
    -------
    dict with keys: actual_consumption, predicted_consumption, difference_kwh,
                    deviation_percent, status, possible_factors
    """
    predicted_kwh = predict_single(row)
    diff    = calculate_difference(actual_kwh, predicted_kwh)
    dev_pct = calculate_deviation_percent(actual_kwh, predicted_kwh)
    return {
        "actual_consumption"   : round(actual_kwh, 3),
        "predicted_consumption": predicted_kwh,
        "difference_kwh"       : diff,
        "deviation_percent"    : dev_pct,
        "status"               : classify_status(dev_pct),
        "possible_factors"     : identify_factors(row),
    }


if __name__ == "__main__":
    import json
    scenarios = [
        {"label": "Normal (cool night, empty office)", "actual_kwh": 3.1,
         "row": {"timestamp": "2023-01-10 02:00:00", "temperature_c": 5.0,
                 "building_type": "office", "occupants": 0,
                 "ac_usage": 0.1, "appliance_usage": 0.5, "peak_hour": 0}},
        {"label": "Elevated (busy afternoon, warm day)", "actual_kwh": 14.0,
         "row": {"timestamp": "2023-06-20 15:00:00", "temperature_c": 28.0,
                 "building_type": "office", "occupants": 65,
                 "ac_usage": 4.5, "appliance_usage": 3.8, "peak_hour": 0}},
        {"label": "Abnormal (heat wave + peak hour)", "actual_kwh": 38.0,
         "row": {"timestamp": "2023-08-15 18:00:00", "temperature_c": 38.0,
                 "building_type": "office", "occupants": 80,
                 "ac_usage": 12.0, "appliance_usage": 6.5, "peak_hour": 1}},
    ]
    print("=" * 60)
    print("  GreenGrid AI -- Anomaly Detection Demo")
    print("=" * 60)
    for s in scenarios:
        result = detect(s["actual_kwh"], s["row"])
        print(f"\n  Scenario: {s['label']}")
        print("  " + json.dumps(result, indent=4).replace("\n", "\n  "))
