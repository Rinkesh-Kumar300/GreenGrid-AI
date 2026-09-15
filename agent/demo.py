"""
GreenGrid AI -- Agent End-to-End Demo
======================================
Demonstrates the complete pipeline from raw sensor readings all the
way through to an AI-generated energy recommendation.

Complete flow
-------------
  Raw input
    -> ml/anomaly.py      (detect anomaly, identify factors)
    -> rag/retriever.py   (fetch relevant knowledge-base sections)
    -> agent/energy_agent.py -> Ollama llama3  (generate recommendation)
    -> Structured result printed to console

Prerequisites
-------------
1. Dataset generated:
      python data/generate_dataset.py
2. ML model trained:
      python ml/train_model.py
3. RAG knowledge base built:
      python rag/ingest.py
4. Ollama installed and running:
      ollama serve              (in a separate terminal, or as a background service)
      ollama pull llama3        (downloads the model once, ~4 GB)

Run this demo:
      python agent/demo.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.energy_agent import run_agent
from agent.config       import OLLAMA_MODEL, OLLAMA_HOST


# ── Demo scenarios ────────────────────────────────────────────────────────────
# Three realistic scenarios to show the full range of agent behaviour.

SCENARIOS = [
    {
        "name"       : "Scenario 1 -- Normal consumption (cool winter night)",
        "actual_kwh" : 3.2,
        "row"        : {
            "timestamp"       : "2023-01-15 02:00:00",
            "temperature_c"   : 4.0,
            "building_type"   : "office",
            "occupants"       : 0,
            "ac_usage"        : 0.1,
            "appliance_usage" : 0.4,
            "peak_hour"       : 0,
        },
    },
    {
        "name"       : "Scenario 2 -- Elevated consumption (busy afternoon, warm day)",
        "actual_kwh" : 15.5,
        "row"        : {
            "timestamp"       : "2023-06-22 15:00:00",
            "temperature_c"   : 29.0,
            "building_type"   : "office",
            "occupants"       : 65,
            "ac_usage"        : 4.8,
            "appliance_usage" : 3.9,
            "peak_hour"       : 0,
        },
    },
    {
        "name"       : "Scenario 3 -- Abnormal consumption (heat wave + peak hour)",
        "actual_kwh" : 38.0,
        "row"        : {
            "timestamp"       : "2023-08-15 18:00:00",
            "temperature_c"   : 38.0,
            "building_type"   : "office",
            "occupants"       : 80,
            "ac_usage"        : 12.0,
            "appliance_usage" : 6.5,
            "peak_hour"       : 1,
        },
    },
]


# ── Display helpers ───────────────────────────────────────────────────────────

def _divider(char="=", width=68):
    return char * width


def _print_result(scenario_name: str, result: dict):
    """Print one scenario result in a clear, readable format."""
    print()
    print(_divider())
    print(f"  {scenario_name}")
    print(_divider())

    # ── Measurements ─────────────────────────────────────────────────────────
    print()
    print("  MEASUREMENTS")
    print(f"    Actual consumption   : {result['actual_consumption']} kWh")
    print(f"    Predicted consumption: {result['predicted_consumption']} kWh")
    print(f"    Difference           : {result['difference_kwh']:+.3f} kWh")
    print(f"    Deviation            : {result['deviation_percent']:+.2f}%")
    print(f"    Status               : {result['status']}")

    # ── Possible factors ──────────────────────────────────────────────────────
    print()
    print("  POSSIBLE CONTRIBUTING FACTORS")
    if result["possible_factors"]:
        for f in result["possible_factors"]:
            print(f"    - {f}")
    else:
        print("    None detected")

    # ── Retrieved knowledge ───────────────────────────────────────────────────
    print()
    print("  RETRIEVED KNOWLEDGE-BASE SECTIONS")
    if result["retrieved_guidance"]:
        for r in result["retrieved_guidance"]:
            print(f"    [{r['source']}]  {r['section']}  (score: {r['score']})")
    else:
        print("    None retrieved")

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print("  AI RECOMMENDATION")
    if result["recommendation"]:
        # Indent every line of the multi-line recommendation
        for line in result["recommendation"].splitlines():
            print(f"    {line}")
    else:
        print("    (No recommendation — Ollama not available)")

    # ── Estimated savings ─────────────────────────────────────────────────────
    print()
    print(f"  ESTIMATED POTENTIAL SAVINGS: {result['estimated_savings_pct']}%")

    # ── Error / warning ───────────────────────────────────────────────────────
    if result.get("error"):
        print()
        print(f"  NOTE: {result['error']}")

    print(_divider())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print(_divider("="))
    print("  GreenGrid AI -- Agent End-to-End Demo")
    print(f"  Model : {OLLAMA_MODEL}  |  Host: {OLLAMA_HOST}")
    print(_divider("="))

    # Run only the most interesting scenario (Scenario 3) by default to save time.
    # Uncomment the loop below to run all three scenarios.
    demo_scenarios = [SCENARIOS[2]]   # Abnormal scenario
    # demo_scenarios = SCENARIOS      # <- uncomment to run all three

    for scenario in demo_scenarios:
        print(f"\n  Running: {scenario['name']} ...")
        result = run_agent(scenario["actual_kwh"], scenario["row"])
        _print_result(scenario["name"], result)

    print()
    print("Demo complete.")
    print()
    print("To run all three scenarios, edit agent/demo.py and")
    print("change  demo_scenarios = [SCENARIOS[2]]  to  demo_scenarios = SCENARIOS")


if __name__ == "__main__":
    main()
