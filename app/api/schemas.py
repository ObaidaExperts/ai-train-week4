from pydantic import BaseModel
from app.core.models import AIModel, ExperimentType

class ChatRequest(BaseModel):
    prompt: str
    model: AIModel = AIModel.GPT_4O
    experiment_type: ExperimentType = ExperimentType.BASELINE
