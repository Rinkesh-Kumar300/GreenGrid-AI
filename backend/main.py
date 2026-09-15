"""
GreenGrid AI -- FastAPI Application
=====================================
Creates the app, adds CORS, and registers routes.

Run locally:
    uvicorn backend.main:app --reload

  API:        http://localhost:8000
  Swagger UI: http://localhost:8000/docs
  ReDoc:      http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title      = "GreenGrid AI API",
    description= (
        "Energy Consumption Forecasting & Optimization Agent — SDG 7.\n\n"
        "**POST /analyze** accepts building sensor readings and returns "
        "an anomaly analysis, contributing factors, a recommendation, "
        "and an estimated energy savings range."
    ),
    version    = "1.0.0",
    docs_url   = "/docs",
    redoc_url  = "/redoc",
)

# Allow all origins so the HTML frontend can call the API from any port or
# from a file:// URL without browser CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

from backend.routes import router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
