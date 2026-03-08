import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.api.endpoints import router as experiments_router
from app.api.middleware import ExceptionHandlerMiddleware
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api")

app = FastAPI(title=settings.APP_NAME)
logger.info(f"Starting {settings.APP_NAME}...")

# Add middleware
app.add_middleware(ExceptionHandlerMiddleware)

# Register routers
app.include_router(experiments_router)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
def health_check() -> dict[str, str]:
    """Check the health and connectivity of the API."""
    return {
        "status": "healthy",
        "timestamp": os.getenv("TIME", "unknown"),
        "api_connectivity": "ok"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
