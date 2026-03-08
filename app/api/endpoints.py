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
        experiment_type=request.experiment_type,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        return_logprobs=request.return_logprobs
    )
    
    response_data = {
        "request_id": request_id,
        "response": result["response"],
        "log_analysis": result["log_analysis"]
    }

    if request.return_logprobs:
        if "logprobs" in result:
            response_data["logprobs"] = result["logprobs"]
            response_data["logprobs_supported"] = True
        else:
            # Provider does not support logprobs (e.g. Claude, Gemini)
            response_data["logprobs"] = None
            response_data["logprobs_supported"] = False
            response_data["logprobs_note"] = (
                f"Logprobs are only available for OpenAI models. "
                f"'{request.model.value}' does not support token-level probabilities."
            )
        
    return response_data

@router.get("/results")
def get_results(
    service: ExperimentService = Depends(get_experiment_service)
) -> list[dict[str, Any]]:
    """Retrieve all logged experiment results."""
    return service.get_results()
