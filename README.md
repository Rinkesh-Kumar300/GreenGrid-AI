# GreenGrid AI

> **Energy Consumption Forecasting & Optimization Agent — aligned with UN SDG 7: Affordable and Clean Energy**

GreenGrid AI analyses historical building energy data, forecasts future consumption,
detects abnormal usage, identifies likely causes, retrieves relevant energy-saving
guidance from a local knowledge base (RAG), and uses a local LLM to generate
plain-English recommendations — all running 100% locally, no API keys required.

---

## What it does

1. **Forecasting** — predicts hourly energy consumption using a Gradient Boosting model trained on synthetic data
2. **Anomaly detection** — flags consumption that is Normal, Elevated, or Abnormal versus the prediction
3. **Factor identification** — detects contributing causes (high AC, temperature, occupancy, peak hours, etc.)
4. **RAG retrieval** — searches a local ChromaDB knowledge base for relevant energy-saving guidance
5. **AI recommendation** — sends the full context to Ollama (llama3) and generates actionable advice
6. **Dashboard** — simple HTML/CSS/JS frontend served directly from the `frontend/` folder

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 or higher | [python.org](https://www.python.org/downloads/) |
| Ollama | latest | [ollama.com](https://ollama.com) — runs llama3 locally |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/GreenGrid-AI.git
cd GreenGrid-AI

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## Ollama Setup

Ollama is the local LLM server. Install it once, pull the model once — it stays on disk.

```bash
# Download and install Ollama from https://ollama.com

# Pull the llama3 model (~4 GB, one-time download)
ollama pull llama3

# Start the Ollama server (must be running before the backend)
ollama serve
```

> **Note:** `ollama serve` needs to be running in a separate terminal whenever you use the dashboard.
> The backend handles Ollama being unavailable gracefully — all other fields still work without it.

---

## First-Time Setup

Run these steps once in order. After that, only steps 4–5 are needed on every restart.

```bash
# Step 1 — Generate the synthetic dataset (creates data/energy_data.csv)
python data/generate_dataset.py

# Step 2 — Train the ML forecasting model (creates models/forecast_model.pkl)
python ml/train_model.py

# Step 3 — Build the RAG vector store (creates rag/chroma_db/)
python rag/ingest.py
```

---

## Running the Application

### Start the backend API

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/health` | Health check — confirms the API is running |
| `http://localhost:8000/analyze` | Main POST endpoint (JSON) |
| `http://localhost:8000/docs` | Interactive Swagger UI — try the API in the browser |
| `http://localhost:8000/redoc` | ReDoc API documentation |

### Open the dashboard

Open `frontend/index.html` directly in your browser — no server or build step needed.

On Windows you can double-click the file, or run:

```bash
start frontend/index.html
```

On macOS / Linux:

```bash
open frontend/index.html
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Anomaly detection unit tests only
pytest tests/test_anomaly.py -v
```

---

## Example API Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 32,
    "building_type": "office",
    "occupants": 40,
    "ac_usage": 70,
    "appliance_usage": 30,
    "peak_hour": 1,
    "actual_consumption": 18.5
  }'
```

---

## Project Structure

```
GreenGrid-AI/
│
├── data/
│   ├── generate_dataset.py   # generates energy_data.csv
│   ├── validate_dataset.py   # validates the CSV
│   └── energy_data.csv       # synthetic hourly dataset (8,760 rows)
│
├── ml/
│   ├── train_model.py        # trains forecasting model, saves to models/
│   ├── predict.py            # predict_single() and predict_next_24h()
│   └── anomaly.py            # detect() — compares actual vs predicted
│
├── models/
│   ├── forecast_model.pkl             # saved Gradient Boosting model
│   ├── forecast_feature_columns.pkl   # feature column order
│   └── forecast_model_metadata.pkl    # metrics and config
│
├── rag/
│   ├── documents/            # plain-text energy-saving knowledge base
│   │   ├── energy_guidelines.txt
│   │   ├── peak_hours.txt
│   │   ├── ac_efficiency.txt
│   │   └── office_energy_saving.txt
│   ├── ingest.py             # chunks documents and loads into ChromaDB
│   ├── retriever.py          # retrieve(query) → top-N relevant chunks
│   └── chroma_db/            # ChromaDB vector store (created by ingest.py)
│
├── agent/
│   ├── config.py             # Ollama model name, prompt settings
│   ├── energy_agent.py       # run_agent() — full pipeline entry point
│   └── demo.py               # standalone demo script
│
├── backend/
│   ├── main.py               # FastAPI app, CORS
│   ├── models.py             # Pydantic request/response models
│   └── routes.py             # GET /health, POST /analyze
│
├── frontend/
│   ├── index.html            # dashboard page
│   ├── style.css             # styles
│   └── app.js                # fetch() → /analyze → render results
│
├── tests/
│   ├── test_anomaly.py       # 35 unit tests for ml/anomaly.py
│   └── test_api.py           # FastAPI endpoint tests
│
├── greengrid-ai-plan.md      # implementation plan
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```

---

## Implementation Status

| Sub-Task | Description | Status |
|----------|-------------|--------|
| 1 | Synthetic dataset + project structure | ✓ Done |
| 2 | ML forecasting model | ✓ Done |
| 3 | Anomaly detection | ✓ Done |
| 4 | RAG knowledge base (ChromaDB) | ✓ Done |
| 5 | AI recommendation agent (Ollama) | ✓ Done |
| 6 | FastAPI backend (`/health`, `/analyze`) | ✓ Done |
| 7 | HTML dashboard | ✓ Done |

---

## SDG 7 Alignment

This project directly supports **UN Sustainable Development Goal 7: Affordable and Clean Energy**
by helping building operators identify and reduce wasteful energy consumption through
data-driven forecasting and AI-powered recommendations.
