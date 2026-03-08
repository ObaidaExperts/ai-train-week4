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

