"""
GreenGrid AI -- Dataset Validation Script
==========================================
Validates energy_data.csv against all Sub-Task 1 acceptance criteria.
Run with:  python data/validate_dataset.py
"""

import os
import sys
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "energy_data.csv")

EXPECTED_COLUMNS = [
    "timestamp", "temperature_c", "building_type", "occupants",
    "ac_usage", "appliance_usage", "peak_hour", "consumption_kwh", "is_anomaly",
]
NUMERIC_COLUMNS = [
    "temperature_c", "occupants", "ac_usage", "appliance_usage",
    "peak_hour", "consumption_kwh", "is_anomaly",
]
VALID_BUILDING_TYPES = {"office", "residential", "retail"}

results = []

def check(name, passed, detail=""):
    results.append((name, passed, detail))

file_exists = os.path.isfile(CSV_PATH)
check("File exists", file_exists, CSV_PATH)
if not file_exists:
    for name, passed, detail in results:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}  {detail}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

check("Row count >= 1000", len(df) >= 1000, f"{len(df):,} rows found")

missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
check("All expected columns present", len(missing_cols) == 0,
      f"Missing: {missing_cols}" if missing_cols else f"All {len(EXPECTED_COLUMNS)} columns present")

null_counts = df[EXPECTED_COLUMNS].isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]
check("No missing values", len(cols_with_nulls) == 0,
      "No nulls found" if len(cols_with_nulls) == 0 else f"Nulls in: {cols_with_nulls.to_dict()}")

bad_numeric = [col for col in NUMERIC_COLUMNS if col in df.columns
               and pd.to_numeric(df[col], errors="coerce").isna().sum() > 0]
check("Numeric columns contain valid numbers", len(bad_numeric) == 0,
      "All numeric" if not bad_numeric else f"Non-numeric in: {bad_numeric}")

if "timestamp" in df.columns:
    parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
    check("Timestamp values are valid datetimes", parsed_ts.isna().sum() == 0,
          f"Range: {parsed_ts.min()}  to  {parsed_ts.max()}")
    check("Timestamps are unique", df["timestamp"].duplicated().sum() == 0,
          "All unique" if df["timestamp"].duplicated().sum() == 0
          else f"{df['timestamp'].duplicated().sum()} duplicates")

if "peak_hour" in df.columns:
    invalid = df[~df["peak_hour"].isin([0, 1])]
    check("peak_hour is binary", len(invalid) == 0,
          f"0s: {(df['peak_hour']==0).sum():,}  |  1s: {(df['peak_hour']==1).sum():,}")

if "consumption_kwh" in df.columns:
    non_pos = (df["consumption_kwh"] <= 0).sum()
    check("consumption_kwh is positive", non_pos == 0,
          f"Min: {df['consumption_kwh'].min():.3f}  Max: {df['consumption_kwh'].max():.3f}")

if "is_anomaly" in df.columns:
    n = int(df["is_anomaly"].sum())
    check("is_anomaly is binary", len(df[~df["is_anomaly"].isin([0, 1])]) == 0, "All 0 or 1")
    check("Anomaly records exist", n > 0, f"{n:,} anomaly rows ({df['is_anomaly'].mean()*100:.1f}%)")

if "is_anomaly" in df.columns and "consumption_kwh" in df.columns:
    nm = df[df["is_anomaly"] == 0]["consumption_kwh"].mean()
    am = df[df["is_anomaly"] == 1]["consumption_kwh"].mean()
    ratio = am / nm if nm > 0 else 0
    check("Anomaly mean is notably higher than normal", ratio >= 2.0,
          f"Normal: {nm:.2f} kWh  Anomaly: {am:.2f} kWh  Ratio: {ratio:.1f}x")

print()
print("=" * 68)
print("  GreenGrid AI -- Dataset Validation Results")
print("=" * 68)
for name, passed, detail in results:
    print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    if detail:
        print(f"         {detail}")
    print()
passed_count = sum(1 for _, p, _ in results if p)
print("=" * 68)
print(f"  Result: {passed_count}/{len(results)} checks passed")
print("=" * 68)
if passed_count < len(results):
    sys.exit(1)
