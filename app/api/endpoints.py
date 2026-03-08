import uuid
import logging
from typing import Any
from fastapi import APIRouter, Depends
from app.core.models import AIModel, ExperimentType
from app.api.schemas import ChatRequest, ToolCallRequest
from app.services.experiment_service import ExperimentService
from app.services.tool_service import ToolCallingService, TOOLS

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


@router.get("/tools/schemas")
def get_tool_schemas() -> dict[str, Any]:
    """Return the JSON schemas for all available tools."""
    return {
        "tools": [t["function"] for t in TOOLS],
        "count": len(TOOLS)
    }


@router.post("/tool-call")
def run_tool_call(request: ToolCallRequest) -> dict[str, Any]:
    """
    Execute a full tool-calling loop:
    1. Send prompt + tool schemas to the model.
    2. Parse the tool call request from the model.
    3. Validate and execute the tool.
    4. Return the tool result + final model answer.
    """
    service = ToolCallingService()
    enabled = request.enabled_tools

    # Demo mode: inject a malformed args to trigger the failure-case handler
    if request.force_error:
        from app.services.tool_service import ToolArgumentError
        try:
            from app.services.tool_service import _validate_and_execute_tool
            _validate_and_execute_tool("get_weather", '{"unit": 42}')  # bad args
        except ToolArgumentError as exc:
            return {
                "final_response": "The tool call failed during argument validation.",
                "tool_called": "get_weather",
                "tool_args": {"unit": 42},
                "tool_result": None,
                "tool_error": str(exc),
                "steps": [
                    {"step": "user_prompt", "content": request.prompt},
                    {"step": "tool_call_requested", "tool": "get_weather", "args_raw": '{"unit": 42}'},
                    {"step": "tool_error", "tool": "get_weather", "error": str(exc)},
                    {"step": "final_answer", "content": "The tool call failed during argument validation."}
                ]
            }

    result = service.run_tool_loop(
        prompt=request.prompt,
        model=request.model,
        enabled_tools=enabled
    )
    return result
