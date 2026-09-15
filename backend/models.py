"""
GreenGrid AI -- Backend Pydantic Models
=========================================
Request and response shapes for the FastAPI endpoints.
Pydantic validates all incoming data automatically.
"""

from pydantic import BaseModel, Field, field_validator
from typing   import Optional, List, Any


# ── /analyze Request ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Input payload for POST /analyze."""

    timestamp: Optional[str] = Field(
        default=None,
        description="Optional ISO datetime, e.g. '2023-08-15 14:00:00'.",
        examples=["2023-08-15 14:00:00"],
    )
    temperature: float = Field(
        ..., ge=-30, le=60,
        description="Outdoor temperature in Celsius.",
        examples=[32.0],
    )
    building_type: str = Field(
        default="office",
        description="One of: office, residential, retail.",
        examples=["office"],
    )
    occupants: int = Field(
        ..., ge=0, le=5000,
        description="Number of people in the building.",
        examples=[40],
    )
    ac_usage: float = Field(
        ..., ge=0, le=100,
        description="AC usage as a percentage of maximum capacity (0-100).",
        examples=[70.0],
    )
    appliance_usage: float = Field(
        ..., ge=0, le=100,
        description="Appliance load as a percentage (0-100).",
        examples=[30.0],
    )
    peak_hour: int = Field(
        ..., ge=0, le=1,
        description="1 if this is a utility peak hour, 0 otherwise.",
        examples=[1],
    )
    actual_consumption: float = Field(
        ..., gt=0,
        description="Measured energy consumption in kWh (must be > 0).",
        examples=[145.0],
    )

    @field_validator("building_type")
    @classmethod
    def validate_building_type(cls, v: str) -> str:
        allowed = {"office", "residential", "retail"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"building_type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower


# ── /analyze Response ─────────────────────────────────────────────────────────

class RAGGuidanceItem(BaseModel):
    """One retrieved knowledge-base section."""
    source : str   = Field(description="Source document filename.")
    section: str   = Field(description="Section heading.")
    text   : str   = Field(description="Full section text.")
    score  : float = Field(description="Cosine distance (lower = more relevant).")


class AnalyzeResponse(BaseModel):
    """JSON response returned by POST /analyze."""
    actual_consumption   : float                  = Field(description="Measured kWh from the request.")
    predicted_consumption: float                  = Field(description="Model-predicted kWh.")
    deviation_percent    : float                  = Field(description="Percentage deviation from predicted.")
    status               : str                    = Field(description="Normal | Elevated | Abnormal")
    possible_factors     : List[str]              = Field(description="Contributing factors identified.")
    rag_guidance         : List[RAGGuidanceItem]  = Field(default=[], description="Retrieved knowledge-base sections.")
    recommendation       : Optional[str]          = Field(default=None, description="AI recommendation (available once Ollama is connected).")
    estimated_savings    : str                    = Field(description="Estimated savings range, e.g. '8-12%'.")


# ── /health Response ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status : str = Field(examples=["healthy"])
    version: str = Field(examples=["1.0.0"])
