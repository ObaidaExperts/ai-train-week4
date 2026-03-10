from pydantic import BaseModel
from app.core.models import AIModel, ExperimentType

class ChatRequest(BaseModel):
    prompt: str
    model: AIModel = AIModel.GPT_4O
    experiment_type: ExperimentType = ExperimentType.BASELINE
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    return_logprobs: bool = False


class ToolCallRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    enabled_tools: list[str] | None = None  # None means all tools
    force_error: bool = False  # Inject a malformed arg to demo error handling


class AgenticFlowRequest(BaseModel):
    """Request for Single vs Agentic Flow (trip planning)."""

    user_request: str
    model: str = "gpt-4o"


class MultiSDKRequest(BaseModel):
    """Request for Multi-SDK model execution (same task across providers)."""

    user_request: str
    provider: str  # openai | anthropic | gemini | vllm
    model: str | None = None  # Optional; uses provider default if omitted


class MultiSDKRunAllRequest(BaseModel):
    """Request to run the same task across multiple providers."""

    user_request: str
    providers: list[str] | None = None  # None = all providers

