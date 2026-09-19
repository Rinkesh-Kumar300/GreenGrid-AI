from fastapi import FastAPI

app = FastAPI()

try:
    from backend.main import app as real_app
    app = real_app
except Exception as e:
    error_message = repr(e)

    @app.get("/health")
    def health():
        return {
            "status": "import_failed",
            "error": error_message
        }