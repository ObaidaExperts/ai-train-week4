import uuid
import logging
from datetime import datetime
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.models import AIModel
from app.core.tokenizer import TokenCounter
from app.services.repository import ResultsRepository

logger = logging.getLogger("service")

class ExperimentService:
    """Service to run tokenization and context experiments."""

    def __init__(
        self, 
        repository: ResultsRepository | None = None,
        client: OpenAI | None = None
    ) -> None:
        """
        Initialize the service with its dependencies.
        """
        self.repository = repository or ResultsRepository(settings.RESULTS_FILE)
        self.client = client or OpenAI(api_key=settings.OPENAI_API_KEY)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: AIModel | str = AIModel.GPT_4O
    ) -> float:
        """Calculate cost based on model pricing (per 1M tokens)."""
        model_type = model if isinstance(model, AIModel) else AIModel(model)
        rates = model_type.pricing
        
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def analyze_text(
        self,
        prompt: str,
        model: AIModel | str = AIModel.GPT_4O,
        request_id: str | None = None
    ) -> dict[str, Any]:
        """Perform a real chat completion and return the response and analysis."""
        model_type = model if isinstance(model, AIModel) else AIModel(model)
        model_str = model_type.value
        
        # 1. Input Analysis
        input_tokens = TokenCounter.count_openai_tokens(prompt, model_str)
        context_limit = model_type.context_limit
        percent_used = (input_tokens / context_limit) * 100 if context_limit > 0 else 0
        
        # 2. Execution
        response_text = ""
        output_tokens = 0
        status = "Success"
        
        logger.info(f"Executing AI completion for model: {model_str}")
        try:
            if model_type.is_gemini:
                # Gemini Simulation
                response_text = f"This is a simulated Gemini response for: '{prompt[:50]}...'"
                output_tokens = min(500, input_tokens // 5)
                status = "Success (Gemini Simulation)"
            else:
                response = self.client.chat.completions.create(
                    model=model_str,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
                output_tokens = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            logger.error(f"AI execution failed for {model_str}: {e}")
            raise  # Let middleware handle it or wrap in custom exception

        # 3. Cost Calculation
        cost = self.calculate_cost(input_tokens, output_tokens, model_str)
        
        analysis = {
            "model": model_str,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "context_limit": context_limit,
            "usage_percentage": round(percent_used, 2),
            "is_near_limit": percent_used > 80,
            "is_over_limit": input_tokens > context_limit and context_limit > 0,
            "status": status
        }
        
        # log via repository
        self.repository.log_result({
            "Timestamp": datetime.now().isoformat(),
            "Request_ID": request_id or "unknown",
            "Model": model_str,
            "Prompt": prompt,
            "Response": response_text,
            "Input_Tokens": input_tokens,
            "Output_Tokens": output_tokens,
            "Cost_USD": cost,
            "Status": status
        })
        
        return {
            "response": response_text,
            "log_analysis": analysis
        }

    def get_results(self) -> list[dict[str, Any]]:
        """Retrieve all logged experiment results via the repository."""
        return self.repository.get_all_results()
