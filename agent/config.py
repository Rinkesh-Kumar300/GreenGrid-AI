"""
GreenGrid AI -- Agent Configuration
=====================================
Central place for all settings the AI agent needs.
Change values here to affect the entire agent without
touching the main logic in energy_agent.py.
"""

# ── Ollama settings ───────────────────────────────────────────────────────────

# The local LLM model to use. Must be pulled with:  ollama pull llama3
OLLAMA_MODEL = "llama3"

# Ollama runs a local HTTP server on this host and port by default.
# Only change these if you have configured Ollama differently.
OLLAMA_HOST = "http://localhost:11434"

# ── RAG settings ──────────────────────────────────────────────────────────────

# How many knowledge-base chunks to retrieve for each query.
# 3 gives the LLM enough context without making the prompt too long.
RAG_N_RESULTS = 3

# ── Prompt settings ───────────────────────────────────────────────────────────

# The system prompt tells the LLM what role it plays and what rules to follow.
# Keep this short and clear so the model stays on topic.
SYSTEM_PROMPT = """You are GreenGrid AI, an expert energy efficiency advisor for commercial buildings.

Your job is to analyse the energy situation described and provide clear, actionable recommendations.

Rules you MUST follow:
- Base all recommendations on the provided knowledge-base excerpts only.
- Do NOT invent guidelines that are not supported by the provided knowledge.
- Be concise. Use bullet points for actions.
- Always include an estimated savings percentage range.
- Write in plain English that a building manager can act on immediately.
- If consumption is normal, acknowledge it and suggest preventive best practices."""

# ── Savings estimation ────────────────────────────────────────────────────────

# Simple heuristic: if an anomaly is detected, the potential saving is
# estimated as a fraction of the excess consumption.
# 0.7 means "70% of the excess could realistically be recovered".
SAVINGS_RECOVERY_FACTOR = 0.70
