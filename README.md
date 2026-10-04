# GreenGrid AI

> **Energy Consumption Forecasting & Optimization Agent — aligned with UN SDG 7: Affordable and Clean Energy**

**Live Demo:** [greengrid-ai.vercel.app](https://greengrid-ai.vercel.app/)

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

GreenGrid-AI/
│
├── agent/
│   └── energy_agent.py
│
├── backend/
│   ├── main.py
│   ├── models.py
│   └── routes.py
│
├── data/
│   └── energy_data.csv
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── ml/
│   ├── predict.py
│   └── anomaly.py
│
├── models/
│   └── energy_model.pkl
│
├── rag/
│   ├── retriever.py
│   ├── ac_efficiency.txt
│   ├── peak_hours.txt
│   ├── temperature_management.txt
│   └── occupancy_management.txt
│
├── tests/
│   ├── test_api.py
│   └── test_anomaly.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── runtime.txt
├── Procfile
├── nixpacks.toml
├── netlify.toml
├── mise.toml
├── greengrid-ai-plan.md
└── start_greengrid.bat```

---

## Implementation Status

| Sub-Task | Description | Status |
|----------|-------------|--------|
| 1 | Synthetic dataset + project structure | ✓ Done |
| 2 | ML forecasting model | ✓ Done |
| 3 | Anomaly detection | ✓ Done |
| 4 | RAG knowledge base (ChromaDB) | ✓ Done |
| 5 | FastAPI backend (`/health`, `/analyze`) | ✓ Done |
| 6 | HTML dashboard | ✓ Done |

---

## SDG 7 Alignment

This project directly supports **UN Sustainable Development Goal 7: Affordable and Clean Energy**
by helping building operators identify and reduce wasteful energy consumption through
data-driven forecasting and AI-powered recommendations.
