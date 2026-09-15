"""
GreenGrid AI — Synthetic Energy Dataset Generator
==================================================
Generates a realistic hourly energy consumption CSV for a simulated building.

Columns produced:
  timestamp        — ISO 8601 datetime, hourly intervals
  temperature_c    — outdoor temperature in Celsius (seasonal + diurnal pattern)
  building_type    — one of: office, residential, retail
  occupants        — number of people in the building at that hour
  ac_usage         — AC power draw in kW (correlated with temperature + occupancy)
  appliance_usage  — other appliance load in kW (correlated with occupancy + hour)
  peak_hour        — 1 if the hour falls in a utility peak period, else 0
  consumption_kwh  — total energy consumed that hour (target variable)
  is_anomaly       — 1 if this row was artificially injected as an anomaly, else 0

Usage:
  python data/generate_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

START_DATE   = "2023-01-01"
PERIODS      = 8760
ANOMALY_RATE = 0.03
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "energy_data.csv")

BUILDING_TYPES = ["office", "residential", "retail"]
PEAK_HOURS = set(range(8, 11)) | set(range(17, 21))


def make_temperature(timestamps):
    day_of_year = timestamps.day_of_year.to_numpy()
    hour        = timestamps.hour.to_numpy()
    seasonal = 15 * np.cos(2 * np.pi * (day_of_year - 196) / 365)
    diurnal  =  5 * np.cos(2 * np.pi * (hour - 14)        / 24)
    base     = 18.0
    noise    = np.random.normal(0, 1.5, size=len(timestamps))
    return np.round(base + seasonal + diurnal + noise, 1)


def make_occupancy(timestamps, building):
    occupancy = np.zeros(len(timestamps), dtype=int)
    for i, ts in enumerate(timestamps):
        h   = ts.hour
        dow = ts.dayofweek
        is_weekend = dow >= 5
        if building == "office":
            if not is_weekend and 8 <= h < 18:
                occupancy[i] = random.randint(30, 80)
            elif not is_weekend and (7 <= h < 8 or 18 <= h < 20):
                occupancy[i] = random.randint(5, 20)
        elif building == "residential":
            if 7 <= h < 9 or 17 <= h < 23:
                occupancy[i] = random.randint(10, 40)
            elif 9 <= h < 17 and is_weekend:
                occupancy[i] = random.randint(15, 35)
            elif 23 <= h or h < 6:
                occupancy[i] = random.randint(5, 15)
        elif building == "retail":
            if 10 <= h < 21:
                base = 40 if is_weekend else 20
                occupancy[i] = random.randint(base, base + 60)
            elif 8 <= h < 10 or 21 <= h < 22:
                occupancy[i] = random.randint(2, 10)
    return occupancy


def generate(periods=PERIODS):
    timestamps = pd.date_range(start=START_DATE, periods=periods, freq="h")
    building   = "office"
    temperature = make_temperature(timestamps)
    occupants   = make_occupancy(timestamps, building)
    peak_hour   = np.array([1 if ts.hour in PEAK_HOURS else 0 for ts in timestamps])

    temp_excess = np.maximum(temperature - 22.0, 0)
    ac_usage = np.round(np.maximum(
        0.5 * temp_excess + 0.02 * occupants + np.random.uniform(0, 0.5, size=periods), 0), 2)

    appliance_usage = np.round(np.maximum(
        0.05 * occupants + 1.0 * peak_hour + np.random.uniform(0.2, 1.5, size=periods), 0.2), 2)

    base_load   = np.random.uniform(1.5, 3.0, size=periods)
    consumption = np.round(np.maximum(
        base_load + ac_usage + appliance_usage + 0.5 * peak_hour
        + np.random.normal(0, 0.3, size=periods), 0.5), 3)

    n_anomalies = int(periods * ANOMALY_RATE)
    anomaly_idx = np.random.choice(periods, size=n_anomalies, replace=False)
    is_anomaly  = np.zeros(periods, dtype=int)
    for idx in anomaly_idx:
        multiplier           = np.random.uniform(2.8, 5.0)
        consumption[idx]     = round(consumption[idx] * multiplier, 3)
        ac_usage[idx]        = round(ac_usage[idx]    * np.random.uniform(1.5, 3.0), 2)
        appliance_usage[idx] = round(appliance_usage[idx] * np.random.uniform(1.5, 2.5), 2)
        is_anomaly[idx]      = 1

    return pd.DataFrame({
        "timestamp"       : timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_c"   : temperature,
        "building_type"   : building,
        "occupants"       : occupants,
        "ac_usage"        : ac_usage,
        "appliance_usage" : appliance_usage,
        "peak_hour"       : peak_hour,
        "consumption_kwh" : consumption,
        "is_anomaly"      : is_anomaly,
    })


if __name__ == "__main__":
    print("Generating synthetic energy dataset ...")
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Anomalies : {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Date range: {df['timestamp'].iloc[0]}  to  {df['timestamp'].iloc[-1]}")
    print(df["consumption_kwh"].describe().to_string())
    print("\nSample rows (first 5):")
    print(df.head(5).to_string(index=False))
