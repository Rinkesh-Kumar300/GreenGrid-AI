# data/

This folder holds the synthetic energy consumption dataset used by GreenGrid AI.

## Files

| File | Description |
|------|-------------|
| `generate_dataset.py` | Python script that creates `energy_data.csv` |
| `validate_dataset.py` | Validates the CSV against acceptance criteria |
| `energy_data.csv` | Generated dataset — **8,760 rows** (one per hour for a full year) |

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime string | Hour the reading was taken |
| `temperature_c` | float | Outdoor temperature in Celsius |
| `building_type` | string | `office`, `residential`, or `retail` |
| `occupants` | int | Number of people in the building |
| `ac_usage` | float | Air-conditioning load in kW |
| `appliance_usage` | float | Other appliance load in kW |
| `peak_hour` | int (0/1) | `1` during utility peak hours (08-10, 17-20) |
| `consumption_kwh` | float | **Target** — total energy consumed that hour |
| `is_anomaly` | int (0/1) | `1` for artificially injected spikes (~3%) |

## Regenerate

```bash
python data/generate_dataset.py
python data/validate_dataset.py
```
