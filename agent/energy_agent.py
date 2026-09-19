"""
GreenGrid AI -- AI Recommendation Agent
=========================================
Ties together the anomaly-detection module, the RAG retriever, and the
local Ollama LLM to produce a personalised energy-saving recommendation.

Complete data flow
------------------
  1. Caller supplies: the actual kWh reading + feature dict (same shape
     used throughout the project).
  2. The agent runs anomaly detection  ->  gets status, deviation %, factors.
  3. The agent queries the RAG retriever using the detected factors as the
     search query  ->  gets top-3 relevant knowledge-base sections.
  4. The agent builds a structured prompt that contains all of the above.
  5. The agent sends the prompt to Ollama (llama3 by default).
  6. Ollama returns a plain-English recommendation.
  7. The agent computes a numeric estimated savings figure.
  8. Everything is packed into a structured result dict and returned.

Public API
----------
run_agent(actual_kwh: float, row: dict) -> dict
    The single entry point.  Returns a dict with keys:
        actual_consumption     float   -- kWh you passed in
        predicted_consumption  float   -- what the model expected
        difference_kwh         float   -- actual - predicted
        deviation_percent      float   -- % over/under expected
        status                 str     -- Normal / Elevated / Abnormal
        possible_factors       list    -- human-readable causes
        retrieved_guidance     list    -- top RAG results (dicts)
        recommendation         str     -- LLM-generated advice
        estimated_savings_pct  float   -- estimated recoverable %
        error                  str     -- set if something went wrong, else None

Usage example
-------------
    from agent.energy_agent import run_agent

    result = run_agent(
        actual_kwh=38.0,
        row={
            "timestamp"       : "2023-08-15 18:00:00",
            "temperature_c"   : 38.0,
            "building_type"   : "office",
            "occupants"       : 80,
            "ac_usage"        : 12.0,
            "appliance_usage" : 6.5,
            "peak_hour"       : 1,
        }
    )
    print(result["recommendation"])
"""

import os
import sys

# Allow running this file directly from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.anomaly    import detect
from rag.retriever import retrieve, format_results
from agent.config  import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    RAG_N_RESULTS,
    SYSTEM_PROMPT,
    SAVINGS_RECOVERY_FACTOR,
)


# ── Ollama client setup ───────────────────────────────────────────────────────
# We import the ollama library lazily (inside the function that needs it) so
# that the rest of the agent still works for testing even if ollama is not
# installed. This makes unit-testing the non-LLM parts much easier.


def _check_ollama_available() -> tuple[bool, str]:
    """
    Check whether the Ollama server is running and the required model is
    available, without making a real inference call.

    Returns (True, "") if everything is fine, or (False, error_message).
    """
    try:
        import ollama
    except ImportError:
        return False, (
            "The 'ollama' Python package is not installed.\n"
            "Fix:  pip install ollama"
        )

    try:
        # ollama.list() returns the locally available models.
        # This call fails quickly if the Ollama server is not running.
        available_models = [m.model for m in ollama.list().models]
    except Exception as exc:
        return False, (
            f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
            f"Make sure Ollama is running:  ollama serve\n"
            f"Detail: {exc}"
        )

    # Check that the specific model is pulled and ready
    # Model names may include a tag like "llama3:latest"
    model_ready = any(
        OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL)
        for m in available_models
    )
    if not model_ready:
        return False, (
            f"Model '{OLLAMA_MODEL}' is not installed in Ollama.\n"
            f"Fix:  ollama pull {OLLAMA_MODEL}\n"
            f"Available models: {available_models}"
        )

    return True, ""


# ── RAG query builder ─────────────────────────────────────────────────────────

def _build_rag_query(factors: list, status: str) -> str:
    """
    Convert the list of detected factors into a natural-language search query
    for the RAG retriever.  The retriever performs semantic search, so a short
    descriptive sentence works better than a plain list of keywords.
    """
    if not factors:
        return "general building energy saving best practices"

    # Join the factors into a readable sentence fragment
    factors_text = ", ".join(f.lower() for f in factors)
    return (
        f"Energy consumption is {status.lower()} and the following factors are detected: "
        f"{factors_text}. How to reduce energy use?"
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_user_prompt(anomaly_result: dict, rag_results: list, row: dict) -> str:
    """
    Assemble all available context into a single structured user prompt.

    The prompt deliberately separates:
      - The measured situation (numbers)
      - The retrieved knowledge (text the LLM MUST use)
      - The explicit instruction (what to produce)

    This structure helps the LLM stay grounded in the retrieved knowledge
    rather than generating plausible-sounding but unsupported advice.
    """
    a = anomaly_result  # shorthand

    # ── Section 1: Measured situation ─────────────────────────────────────────
    situation = f"""CURRENT ENERGY SITUATION
-------------------------
Building type    : {row.get('building_type', 'office')}
Timestamp        : {row.get('timestamp', 'unknown')}
Temperature      : {row.get('temperature_c', '?')} C
Occupants        : {row.get('occupants', '?')}
AC usage (kW)    : {row.get('ac_usage', '?')}
Appliance use(kW): {row.get('appliance_usage', '?')}
Peak hour        : {'Yes' if row.get('peak_hour', 0) else 'No'}

ENERGY MEASUREMENTS
-------------------
Actual consumption   : {a['actual_consumption']} kWh
Predicted consumption: {a['predicted_consumption']} kWh
Difference           : {a['difference_kwh']:+.3f} kWh
Deviation            : {a['deviation_percent']:+.2f}%
Status               : {a['status']}"""

    # ── Section 2: Possible causes ────────────────────────────────────────────
    if a["possible_factors"]:
        factors_text = "\n".join(f"  - {f}" for f in a["possible_factors"])
        causes = f"\nPOSSIBLE CONTRIBUTING FACTORS\n------------------------------\n{factors_text}"
    else:
        causes = "\nPOSSIBLE CONTRIBUTING FACTORS\n------------------------------\n  None detected."

    # ── Section 3: Retrieved knowledge base excerpts ──────────────────────────
    if rag_results:
        excerpts = "\n".join(
            f"[Source: {r['source']} | Section: {r['section']}]\n{r['text']}\n"
            for r in rag_results
        )
        knowledge = (
            "\nRELEVANT KNOWLEDGE BASE EXCERPTS\n"
            "---------------------------------\n"
            "Use ONLY the following excerpts to support your recommendations:\n\n"
            + excerpts
        )
    else:
        knowledge = (
            "\nRELEVANT KNOWLEDGE BASE EXCERPTS\n"
            "---------------------------------\n"
            "No relevant excerpts found. Use general energy efficiency principles."
        )

    # ── Section 4: Instruction ────────────────────────────────────────────────
    instruction = """
TASK
----
Based on the situation and the knowledge base excerpts above, provide:

1. A one-sentence summary of the energy situation.
2. The possible contributing factors (use the detected factors listed above).
3. A bullet-point list of recommended actions grounded in the knowledge excerpts.
4. An estimated potential energy reduction percentage range (e.g. "8-12%").

Format your response clearly with these four numbered sections.
Do not add information that is not supported by the knowledge excerpts."""

    return situation + causes + knowledge + instruction


# ── Savings estimator ─────────────────────────────────────────────────────────

def _estimate_savings(anomaly_result: dict) -> float:
    """
    Estimate the recoverable savings as a percentage of actual consumption.

    Logic:
      - If status is Normal: no savings available (0%).
      - If Elevated or Abnormal: the excess = actual - predicted.
        We assume SAVINGS_RECOVERY_FACTOR (70%) of the excess is recoverable.
        Savings % = (excess * recovery_factor / actual) * 100

    This is a simple heuristic — the LLM also provides a qualitative range.
    """
    if anomaly_result["status"] == "Normal":
        return 0.0

    actual    = anomaly_result["actual_consumption"]
    excess    = anomaly_result["difference_kwh"]

    if actual <= 0 or excess <= 0:
        return 0.0

    savings_kwh = excess * SAVINGS_RECOVERY_FACTOR
    savings_pct = (savings_kwh / actual) * 100
    return round(savings_pct, 1)


# ── Main agent entry point ────────────────────────────────────────────────────

def run_agent(actual_kwh: float, row: dict) -> dict:
    """
    Run the full recommendation pipeline for one energy reading.

    Parameters
    ----------
    actual_kwh : float  -- the measured energy consumption for this hour (kWh)
    row        : dict   -- feature dict (timestamp, temperature_c, building_type,
                           occupants, ac_usage, appliance_usage, peak_hour)

    Returns
    -------
    dict  -- see module docstring for the full key list
    """

    # ── Step 1: Input validation ───────────────────────────────────────────────
    if not isinstance(actual_kwh, (int, float)) or actual_kwh < 0:
        return _error_result("actual_kwh must be a non-negative number.")

    required_keys = ["timestamp", "temperature_c", "building_type",
                     "occupants", "ac_usage", "appliance_usage", "peak_hour"]
    missing = [k for k in required_keys if k not in row]
    if missing:
        return _error_result(f"Missing required input fields: {missing}")

    # ── Step 2: Anomaly detection ──────────────────────────────────────────────
    # Compare actual vs. predicted and classify the status
    try:
        anomaly_result = detect(actual_kwh, row)
    except Exception as exc:
        return _error_result(f"Anomaly detection failed: {exc}")

    # ── Step 3: RAG retrieval ──────────────────────────────────────────────────
    # Build a search query from the detected factors and retrieve relevant tips
    rag_query   = _build_rag_query(anomaly_result["possible_factors"], anomaly_result["status"])
    try:
        rag_results = retrieve(rag_query, n_results=RAG_N_RESULTS)
    except FileNotFoundError as exc:
        # The vector store has not been built yet — still return anomaly result
        # but with empty guidance and a warning
        rag_results = []
        rag_warning = str(exc)
    except Exception as exc:
        rag_results = []
        rag_warning = f"RAG retrieval failed: {exc}"
    else:
        rag_warning = None

    # ── Step 4: Check Ollama availability ─────────────────────────────────────
    

    # ── Step 5: Build the prompt ───────────────────────────────────────────────
    user_prompt = _build_user_prompt(anomaly_result, rag_results, row)

    # ── Step 6: Call OpenAI ────────────────────────────────────────────────────
    recommendation = None
    ollama_error   = None
    try:
        from openai import OpenAI

        client   = OpenAI()   # reads OPENAI_API_KEY from env automatically
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        recommendation = response.choices[0].message.content.strip()

    except Exception as exc:
        ollama_error = f"OpenAI inference failed: {exc}"

    # ── Step 7: Assemble and return the final result ───────────────────────────
    return {
        "actual_consumption": anomaly_result["actual_consumption"],
        "predicted_consumption": anomaly_result["predicted_consumption"],
        "difference_kwh": anomaly_result["difference_kwh"],
        "deviation_percent": anomaly_result["deviation_percent"],
        "status": anomaly_result["status"],
        "possible_factors": anomaly_result["possible_factors"],
        "retrieved_guidance": rag_results,
        "recommendation": recommendation,
        "estimated_savings_pct": _estimate_savings(anomaly_result),
        "error": ollama_error,
    }