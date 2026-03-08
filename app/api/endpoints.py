import uuid
import logging
from typing import Any
from fastapi import APIRouter, Depends
from app.core.models import AIModel, ExperimentType
from app.api.schemas import ChatRequest
from app.services.experiment_service import ExperimentService

logger = logging.getLogger("api")
router = APIRouter()

def get_experiment_service() -> ExperimentService:
    return ExperimentService()

@router.get("/metadata")
def get_metadata() -> dict[str, Any]:
    """Return available models and experiment types."""
    return {
        "models": [model.value for model in AIModel],
        "experiment_types": [exp.value for exp in ExperimentType]
    }

@router.post("/chat")
def chat(
    request: ChatRequest,
    service: ExperimentService = Depends(get_experiment_service)
) -> dict[str, Any]:
    """
    Perform a chat completion with real-time analysis and a unique request ID.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Received chat request [{request_id}] for model: {request.model.value}")
    
    result = service.analyze_text(
        prompt=request.prompt,
        model=request.model,
        request_id=request_id,
        experiment_type=request.experiment_type
    )
    
    return {
        "request_id": request_id,
        "response": result["response"],
        "log_analysis": result["log_analysis"]
    }

@router.get("/results")
def get_results(
    service: ExperimentService = Depends(get_experiment_service)
) -> list[dict[str, Any]]:
    """Retrieve all logged experiment results."""
    return service.get_results()
