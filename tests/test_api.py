"""
GreenGrid AI -- API Tests
===========================
Tests for the FastAPI backend using FastAPI's built-in TestClient.

Test coverage
-------------
  TestHealth        -- GET /health returns 200 and expected fields
  TestAnalyzeValid  -- POST /analyze with valid input returns correct shape
  TestAnalyzeInvalid-- POST /analyze with bad input returns 422 validation errors
  TestAnalyzeNoOllama -- POST /analyze still returns a result when Ollama is down

Run with:
    pytest tests/test_api.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Shared valid payload ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "timestamp"          : "2023-08-15 14:00:00",
    "temperature"        : 32.0,
    "building_type"      : "office",
    "occupants"          : 40,
    "ac_usage"           : 70.0,
    "appliance_usage"    : 30.0,
    "peak_hour"          : 1,
    "actual_consumption" : 145.0,
}

# Expected keys in a successful /analyze response
EXPECTED_ANALYZE_KEYS = {
    "actual_consumption",
    "predicted_consumption",
    "difference_kwh",
    "deviation_percent",
    "status",
    "possible_factors",
    "rag_guidance",
    "recommendation",
    "estimated_savings_pct",
    "estimated_savings",
}


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_field(self):
        resp = client.get("/health")
        assert "status" in resp.json()

    def test_status_is_string(self):
        data = client.get("/health").json()
        assert isinstance(data["status"], str)

    def test_response_has_all_required_fields(self):
        data = client.get("/health").json()
        for field in ("status", "model_loaded", "rag_loaded",
                      "ollama_available", "ollama_model"):
            assert field in data, f"Missing field: {field}"

    def test_model_loaded_is_bool(self):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)

    def test_rag_loaded_is_bool(self):
        data = client.get("/health").json()
        assert isinstance(data["rag_loaded"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  — valid input
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeValid:
    def test_returns_200(self):
        resp = client.post("/analyze", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_all_expected_keys(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert EXPECTED_ANALYZE_KEYS.issubset(data.keys())

    def test_actual_consumption_matches_input(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["actual_consumption"] == pytest.approx(145.0)

    def test_predicted_consumption_is_positive(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["predicted_consumption"] > 0

    def test_status_is_valid_string(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["status"] in {"Normal", "Elevated", "Abnormal"}

    def test_possible_factors_is_list(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert isinstance(data["possible_factors"], list)

    def test_rag_guidance_is_list(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert isinstance(data["rag_guidance"], list)

    def test_rag_guidance_items_have_correct_shape(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        for item in data["rag_guidance"]:
            for key in ("source", "section", "text", "score"):
                assert key in item, f"RAG item missing key: {key}"

    def test_estimated_savings_pct_is_non_negative(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["estimated_savings_pct"] >= 0

    def test_estimated_savings_is_string(self):
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert isinstance(data["estimated_savings"], str)

    def test_deviation_percent_calculation(self):
        """deviation_percent should equal (actual-predicted)/predicted*100."""
        data = client.post("/analyze", json=VALID_PAYLOAD).json()
        expected = ((data["actual_consumption"] - data["predicted_consumption"])
                    / data["predicted_consumption"] * 100)
        assert data["deviation_percent"] == pytest.approx(expected, abs=0.1)

    def test_timestamp_defaults_when_omitted(self):
        """Sending a payload without timestamp should still succeed."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "timestamp"}
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200

    def test_building_type_case_insensitive(self):
        """'Office', 'OFFICE', 'office' should all be accepted."""
        for bt in ("Office", "OFFICE", "office"):
            payload = {**VALID_PAYLOAD, "building_type": bt}
            resp = client.post("/analyze", json=payload)
            assert resp.status_code == 200, f"Failed for building_type='{bt}'"


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  — invalid input (validation errors)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeInvalid:
    def _post(self, overrides):
        payload = {**VALID_PAYLOAD, **overrides}
        return client.post("/analyze", json=payload)

    def test_missing_actual_consumption_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "actual_consumption"}
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 422

    def test_missing_temperature_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "temperature"}
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 422

    def test_negative_actual_consumption_returns_422(self):
        assert self._post({"actual_consumption": -5.0}).status_code == 422

    def test_zero_actual_consumption_returns_422(self):
        assert self._post({"actual_consumption": 0}).status_code == 422

    def test_temperature_below_min_returns_422(self):
        assert self._post({"temperature": -100}).status_code == 422

    def test_temperature_above_max_returns_422(self):
        assert self._post({"temperature": 100}).status_code == 422

    def test_ac_usage_above_100_returns_422(self):
        assert self._post({"ac_usage": 150}).status_code == 422

    def test_ac_usage_below_0_returns_422(self):
        assert self._post({"ac_usage": -1}).status_code == 422

    def test_invalid_building_type_returns_422(self):
        assert self._post({"building_type": "factory"}).status_code == 422

    def test_peak_hour_invalid_value_returns_422(self):
        assert self._post({"peak_hour": 5}).status_code == 422

    def test_negative_occupants_returns_422(self):
        assert self._post({"occupants": -1}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  — Ollama unavailable (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeNoOllama:
    """
    Simulate Ollama being unreachable by patching the agent's check function.
    The endpoint should still return 200 with all numeric fields populated —
    only the 'recommendation' field will be None and 'error' will be set.
    """

    def test_returns_200_when_ollama_down(self):
        with patch("agent.energy_agent._check_ollama_available",
                   return_value=(False, "Ollama not running (mocked)")):
            resp = client.post("/analyze", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_recommendation_is_none_when_ollama_down(self):
        with patch("agent.energy_agent._check_ollama_available",
                   return_value=(False, "Ollama not running (mocked)")):
            data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["recommendation"] is None

    def test_error_field_is_set_when_ollama_down(self):
        with patch("agent.energy_agent._check_ollama_available",
                   return_value=(False, "Ollama not running (mocked)")):
            data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["error"] is not None
        assert len(data["error"]) > 0

    def test_numeric_fields_still_populated_when_ollama_down(self):
        """Anomaly detection and RAG should work regardless of Ollama."""
        with patch("agent.energy_agent._check_ollama_available",
                   return_value=(False, "Ollama not running (mocked)")):
            data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert data["actual_consumption"] > 0
        assert data["predicted_consumption"] > 0
        assert data["status"] in {"Normal", "Elevated", "Abnormal"}

    def test_rag_guidance_still_returned_when_ollama_down(self):
        with patch("agent.energy_agent._check_ollama_available",
                   return_value=(False, "Ollama not running (mocked)")):
            data = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert isinstance(data["rag_guidance"], list)
