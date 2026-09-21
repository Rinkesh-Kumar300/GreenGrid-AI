"""
GreenGrid AI -- API Route Handlers
=====================================
Implements GET /health and POST /analyze.

 /analyze performs ML-based energy analysis,
 anomaly detection, and RAG-based energy guidance.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from fastapi import APIRouter

from ml.anomaly import detect
from rag.retriever import retrieve

from backend.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    RAGGuidanceItem,
)

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

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=True,
        rag_loaded=True,
    )


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse energy consumption",
    description=(
        "Accepts current building sensor readings and actual energy "
        "consumption. Returns an anomaly status, possible contributing "
        "factors, relevant energy-saving guidance, and an estimated "
        "savings range."
    ),
    tags=["Analysis"],
)
def analyze(payload: AnalyzeRequest):

    # ── Build the feature row ─────────────────────────────────────────────────
    # AC and appliance usage arrive as percentages.
    # The ML model uses kW values.

    row = {
        "timestamp": payload.timestamp or "2023-01-01 00:00:00",
        "temperature_c": payload.temperature,
        "building_type": payload.building_type,
        "occupants": payload.occupants,
        "ac_usage": round(
            payload.ac_usage / 100.0 * 20.0,
            2
        ),
        "appliance_usage": round(
            payload.appliance_usage / 100.0 * 10.0,
            2
        ),
        "peak_hour": payload.peak_hour,
    }


    # ── ML + anomaly detection ────────────────────────────────────────────────

    result = detect(
        actual_kwh=payload.actual_consumption,
        row=row
    )


    # ── RAG energy-saving guidance ────────────────────────────────────────────
    # RAG expects a text query, not the row dictionary.
    # The retriever uses TF-IDF + cosine similarity.

    query = (
        f"Energy consumption is "
        f"{result['deviation_percent']}% "
        f"{'higher' if result['difference_kwh'] > 0 else 'lower'} "
        f"than expected. "
        f"Temperature is {payload.temperature} C. "
        f"AC usage is {payload.ac_usage}%. "
        f"Appliance usage is {payload.appliance_usage}%. "
        f"Occupants are {payload.occupants}. "
        f"Peak hour is {payload.peak_hour}."
    )

    retrieved_guidance = retrieve(
        query,
        n_results=3
    )

    result["retrieved_guidance"] = retrieved_guidance


    # ── Estimate potential savings ───────────────────────────────────────────

    if result["status"] == "Abnormal":
        result["estimated_savings_pct"] = 15.0

    elif result["status"] == "Elevated":
        result["estimated_savings_pct"] = 10.0

    else:
        result["estimated_savings_pct"] = 5.0


    # ── Shape RAG guidance into typed response objects ────────────────────────

    rag_items = [
        RAGGuidanceItem(
            source=r["source"],
            section=r["section"],
            text=r["text"],
            score=r["score"],
        )
        for r in result.get("retrieved_guidance", [])
    ]


    # ── Build human-readable savings string ───────────────────────────────────

    pct = result.get(
        "estimated_savings_pct",
        0.0
    )

    if pct > 0:
        pct_lo = max(0, round(pct - 3))
        pct_hi = round(pct + 3)
        savings_str = f"{pct_lo}-{pct_hi}%"

    else:
        savings_str = "0%"


    # ── Return final response ─────────────────────────────────────────────────

    return AnalyzeResponse(
        actual_consumption=result["actual_consumption"],
        predicted_consumption=result["predicted_consumption"],
        difference_kwh=result["difference_kwh"],
        deviation_percent=result["deviation_percent"],
        status=result["status"],
        possible_factors=result["possible_factors"],
        rag_guidance=rag_items,
        estimated_savings_pct=result.get(
            "estimated_savings_pct",
        0.0
        ),
        estimated_savings=savings_str,
    )