from backend.main import app


@app.get("/", tags=["System"])
def deployment_root():
	return {
		"name": "GreenGrid AI API",
		"status": "online",
		"health": "/health",
		"api_health": "/api/health",
		"docs": "/docs",
	}
