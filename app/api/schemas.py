from pydantic import BaseModel
from app.core.models import AIModel

class ChatRequest(BaseModel):
    prompt: str
    model: AIModel = AIModel.GPT_4O
