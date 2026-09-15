"""
GreenGrid AI -- API Route Handlers
=====================================
Implements GET /health and POST /analyze.

/analyze is now fully connected: ML → anomaly detection → RAG → Ollama agent.
Ollama errors are handled gracefully — the endpoint always returns 200.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter
from agent.energy_agent import run_agent
from backend.models import AnalyzeRequest, AnalyzeResponse, HealthResponse, RAGGuidanceItem

router = APIRouter()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health():
    """Returns 200 with status 'healthy' when the API is running."""
    return HealthResponse(status="healthy", version="1.0.0")


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse energy consumption",
    description=(
        "Accepts current building sensor readings and actual energy consumption. "
        "Returns an anomaly status, possible contributing factors, "
        "a recommendation, and an estimated savings range."
    ),
    tags=["Analysis"],
)
def analyze(payload: AnalyzeRequest):
    """
    Full pipeline via agent/energy_agent.run_agent():
      1. Builds the feature row (converts % inputs → kW)
      2. ml/anomaly.detect()    → predicted kWh, deviation, status, factors
      3. rag/retriever.retrieve() → top-3 knowledge-base sections
      4. ollama.chat(llama3)    → plain-English recommendation
    Ollama errors are caught inside run_agent() and surfaced in the
    'error' field — the endpoint always returns HTTP 200.
    """

    # ── Build the feature row ─────────────────────────────────────────────────
    # ac_usage / appliance_usage arrive as 0-100 percentages; the ML model
    # was trained on kW values (max ~20 kW AC, ~10 kW appliances).
    row = {
        "timestamp"       : payload.timestamp or "2023-01-01 00:00:00",
        "temperature_c"   : payload.temperature,
        "building_type"   : payload.building_type,
        "occupants"       : payload.occupants,
        "ac_usage"        : round(payload.ac_usage        / 100.0 * 20.0, 2),
        "appliance_usage" : round(payload.appliance_usage / 100.0 * 10.0, 2),
        "peak_hour"       : payload.peak_hour,
    }

    # ── Run the full agent pipeline ───────────────────────────────────────────
    # run_agent() never raises — it always returns a complete dict even when
    # Ollama is unavailable (recommendation=None, error field set).
    result = run_agent(actual_kwh=payload.actual_consumption, row=row)

    # ── Shape RAG guidance into typed response objects ────────────────────────
    rag_items = [
        RAGGuidanceItem(
            source  = r["source"],
            section = r["section"],
            text    = r["text"],
            score   = r["score"],
        )
        for r in result.get("retrieved_guidance", [])
    ]

    # ── Build human-readable savings string from the numeric estimate ─────────
    pct = result.get("estimated_savings_pct", 0.0)
    if pct > 0:
        pct_lo = max(0, round(pct - 3))
        pct_hi = round(pct + 3)
        savings_str = f"{pct_lo}-{pct_hi}%"
    else:
        savings_str = "0%"

    return AnalyzeResponse(
        actual_consumption   = result["actual_consumption"],
        predicted_consumption= result["predicted_consumption"],
        deviation_percent    = result["deviation_percent"],
        status               = result["status"],
        possible_factors     = result["possible_factors"],
        rag_guidance         = rag_items,
        recommendation       = result.get("recommendation"),   # None when Ollama is down
        estimated_savings    = savings_str,
    )
