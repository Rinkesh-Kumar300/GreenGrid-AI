"""
GreenGrid AI -- Unit Tests: Anomaly Detection
=============================================
Tests for helper functions in ml/anomaly.py.
Run with:  pytest tests/test_anomaly.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.anomaly import (
    calculate_difference,
    calculate_deviation_percent,
    classify_status,
    identify_factors,
    detect,
    ELEVATED_THRESHOLD_PCT,
    ABNORMAL_THRESHOLD_PCT,
)


class TestCalculateDifference:
    def test_over_consumption(self):
        assert calculate_difference(15.0, 10.0) == pytest.approx(5.0)

    def test_under_consumption(self):
        assert calculate_difference(8.0, 10.0) == pytest.approx(-2.0)

    def test_exact_match(self):
        assert calculate_difference(10.0, 10.0) == pytest.approx(0.0)

    def test_rounding(self):
        assert calculate_difference(10.1234, 10.0) == pytest.approx(0.123, abs=0.001)

    def test_zero_actual(self):
        assert calculate_difference(0.0, 5.0) == pytest.approx(-5.0)


class TestCalculateDeviationPercent:
    def test_twenty_percent_over(self):
        assert calculate_deviation_percent(12.0, 10.0) == pytest.approx(20.0)

    def test_twenty_percent_under(self):
        assert calculate_deviation_percent(8.0, 10.0) == pytest.approx(-20.0)

    def test_zero_deviation(self):
        assert calculate_deviation_percent(10.0, 10.0) == pytest.approx(0.0)

    def test_zero_predicted(self):
        assert calculate_deviation_percent(5.0, 0.0) == 0.0

    def test_hundred_percent_over(self):
        assert calculate_deviation_percent(20.0, 10.0) == pytest.approx(100.0)

    def test_rounding(self):
        assert calculate_deviation_percent(10.1, 10.0) == pytest.approx(1.0, abs=0.01)


class TestClassifyStatus:
    def test_normal_zero(self):
        assert classify_status(0.0) == "Normal"

    def test_normal_just_below_elevated(self):
        assert classify_status(ELEVATED_THRESHOLD_PCT - 0.01) == "Normal"

    def test_elevated_at_threshold(self):
        assert classify_status(ELEVATED_THRESHOLD_PCT) == "Elevated"

    def test_elevated_midpoint(self):
        assert classify_status((ELEVATED_THRESHOLD_PCT + ABNORMAL_THRESHOLD_PCT) / 2) == "Elevated"

    def test_elevated_just_below_abnormal(self):
        assert classify_status(ABNORMAL_THRESHOLD_PCT - 0.01) == "Elevated"

    def test_abnormal_at_threshold(self):
        assert classify_status(ABNORMAL_THRESHOLD_PCT) == "Abnormal"

    def test_abnormal_large_value(self):
        assert classify_status(300.0) == "Abnormal"

    def test_negative_deviation_is_normal(self):
        assert classify_status(-50.0) == "Normal"


class TestIdentifyFactors:
    def _row(self, **overrides):
        base = {"temperature_c": 15.0, "ac_usage": 0.5, "occupants": 10,
                "appliance_usage": 1.0, "peak_hour": 0}
        base.update(overrides)
        return base

    def test_no_factors_on_low_row(self):
        assert identify_factors(self._row()) == []

    def test_high_temperature(self):
        assert "High temperature" in identify_factors(self._row(temperature_c=35.0))

    def test_high_ac(self):
        assert "High AC usage" in identify_factors(self._row(ac_usage=7.0))

    def test_high_occupancy(self):
        assert "High occupancy" in identify_factors(self._row(occupants=70))

    def test_high_appliance_usage(self):
        assert "High appliance usage" in identify_factors(self._row(appliance_usage=5.0))

    def test_peak_hour(self):
        assert "Peak-hour consumption" in identify_factors(self._row(peak_hour=1))

    def test_multiple_factors(self):
        factors = identify_factors(self._row(temperature_c=35.0, ac_usage=9.0, peak_hour=1))
        assert "High temperature"      in factors
        assert "High AC usage"         in factors
        assert "Peak-hour consumption" in factors

    def test_string_values_are_handled(self):
        factors = identify_factors(self._row(temperature_c="35.0", ac_usage="8.0"))
        assert "High temperature" in factors
        assert "High AC usage"    in factors


class TestDetect:
    _ROW = {"timestamp": "2023-08-15 14:00:00", "temperature_c": 36.0,
            "building_type": "office", "occupants": 75,
            "ac_usage": 9.0, "appliance_usage": 5.5, "peak_hour": 1}

    def test_result_has_all_required_keys(self):
        result = detect(38.0, self._ROW)
        assert {"actual_consumption", "predicted_consumption", "difference_kwh",
                "deviation_percent", "status", "possible_factors"}.issubset(result.keys())

    def test_actual_consumption_matches_input(self):
        assert detect(38.0, self._ROW)["actual_consumption"] == pytest.approx(38.0)

    def test_predicted_consumption_is_positive(self):
        assert detect(38.0, self._ROW)["predicted_consumption"] > 0

    def test_difference_equals_actual_minus_predicted(self):
        r = detect(38.0, self._ROW)
        assert r["difference_kwh"] == pytest.approx(r["actual_consumption"] - r["predicted_consumption"], abs=0.01)

    def test_status_is_valid_string(self):
        assert detect(38.0, self._ROW)["status"] in {"Normal", "Elevated", "Abnormal"}

    def test_possible_factors_is_list(self):
        assert isinstance(detect(38.0, self._ROW)["possible_factors"], list)

    def test_high_actual_not_normal(self):
        assert detect(38.0, self._ROW)["status"] in {"Elevated", "Abnormal"}

    def test_normal_row_is_normal(self):
        normal_row = {"timestamp": "2023-01-10 02:00:00", "temperature_c": 5.0,
                      "building_type": "office", "occupants": 0,
                      "ac_usage": 0.1, "appliance_usage": 0.5, "peak_hour": 0}
        from ml.predict import predict_single
        predicted = predict_single(normal_row)
        assert detect(predicted * 1.05, normal_row)["status"] == "Normal"
