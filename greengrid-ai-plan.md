# GreenGrid AI — Implementation Plan

> **Goal:** Build an Energy Consumption Forecasting & Optimization Agent aligned with SDG 7.
> **Scope:** MVP — synthetic data, local ML, local LLM via Ollama, simple HTML dashboard.
> **Approach:** Build one layer at a time, bottom-up: data → ML → RAG → agent → API → UI.

---

## Top-Level Overview

GreenGrid AI is a web application that:
1. Loads historical (synthetic) energy consumption data.
2. Forecasts future energy usage using a machine learning model.
3. Detects abnormal consumption spikes using anomaly detection.
4. Identifies likely causes (high AC, high temperature, high occupancy, etc.).
5. Retrieves relevant energy-saving guidelines from a local knowledge base (RAG).
6. Passes all of the above to a local LLM (Ollama llama3) to generate actionable recommendations.
7. Estimates potential energy savings.
8. Displays everything on a simple HTML/Chart.js dashboard via a FastAPI backend.

The entire project runs **100% locally** — no API keys, no cloud dependencies.

---

## Project Folder Structure

```
GreenGrid-AI/
│
├── data/
│   ├── generate_dataset.py
│   └── energy_data.csv
│
├── rag/
│   ├── documents/
│   │   ├── energy_guidelines.txt
│   │   ├── peak_hours.txt
│   │   ├── ac_efficiency.txt
│   │   └── office_energy_saving.txt
│   ├── ingest.py
│   ├── retriever.py
│   └── chroma_db/
│
├── ml/
│   ├── train_model.py
│   ├── predict.py
│   └── anomaly.py
│
├── agent/
│   └── recommendation_agent.py
│
├── backend/
│   ├── main.py
│   └── schemas.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── models/
│   ├── forecast_model.pkl
│   ├── forecast_feature_columns.pkl
│   └── forecast_model_metadata.pkl
│
├── tests/
│   ├── test_anomaly.py
│   └── test_rag.py
│
├── requirements.txt
└── README.md
```

---

## Sub-Tasks

### Sub-Task 1 — Generate the Synthetic Dataset
**Status:** `[x] done`

### Sub-Task 2 — Train the ML Models
**Status:** `[x] done`

### Sub-Task 3 — Anomaly Detection
**Status:** `[x] done`

### Sub-Task 4 — Build the RAG Vector Store
**Status:** `[x] done`

### Sub-Task 5 — Build the AI Recommendation Agent
**Status:** `[ ] pending`

### Sub-Task 6 — Build the FastAPI Backend
**Status:** `[ ] pending`

### Sub-Task 7 — Build the Frontend Dashboard
**Status:** `[ ] pending`

### Sub-Task 8 — Write Tests
**Status:** `[ ] pending`

### Sub-Task 9 — Write README and requirements.txt
**Status:** `[ ] pending`

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Dataset | Synthetic CSV | No external dependencies |
| Forecasting | Gradient Boosting Regressor | Best MAE on test set |
| Anomaly Detection | Forecast-comparison + thresholds | Simple, interpretable |
| Knowledge Base | Plain .txt files | Easy to edit |
| Vector Store | ChromaDB | Local, no server needed |
| LLM | Ollama llama3 | Runs locally, no API key |
| API | FastAPI | Fast, automatic docs |
| Frontend | HTML + Chart.js | No build tools |
