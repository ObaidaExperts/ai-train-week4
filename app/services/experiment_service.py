import uuid
import logging
from datetime import datetime
from typing import Any

from openai import OpenAI
from anthropic import Anthropic

from app.core.config import settings
from app.core.models import AIModel, ExperimentType
from app.core.tokenizer import TokenCounter
from app.services.repository import ResultsRepository

logger = logging.getLogger("service")

class ExperimentService:
    """Service to run tokenization and context experiments."""

    def __init__(
        self, 
        repository: ResultsRepository | None = None,
        openai_client: OpenAI | None = None,
        anthropic_client: Anthropic | None = None
    ) -> None:
        """
        Initialize the service with its dependencies.
        """
        self.repository = repository or ResultsRepository(settings.RESULTS_FILE)
        self.openai_client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic_client = anthropic_client or Anthropic(api_key=settings.ANTHROPIC_API_KEY)

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
        request_id: str | None = None,
        experiment_type: ExperimentType | str = ExperimentType.BASELINE,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None
    ) -> dict[str, Any]:
        """Perform a real chat completion and return the response and analysis."""
        model_type = model if isinstance(model, AIModel) else AIModel(model)
        model_str = model_type.value
        exp_type = experiment_type if isinstance(experiment_type, ExperimentType) else ExperimentType(experiment_type)
        
        # 1. Input Analysis
        # Note: Using tiktoken as a general approximation for Claude if needed, 
        # but better to use specific tokenizer if available.
        input_tokens = TokenCounter.count_openai_tokens(prompt, model_str)
        context_limit = model_type.context_limit
        percent_used = (input_tokens / context_limit) * 100 if context_limit > 0 else 0
        
        # 2. Execution
        response_text = ""
        output_tokens = 0
        status = "Success"
        
        logger.info(f"Executing AI completion for model: {model_str} [{exp_type.value}]")
        try:
            if model_type.is_gemini:
                # Gemini Simulation
                response_text = f"This is a simulated Gemini response for: '{prompt[:50]}...'"
                output_tokens = min(500, input_tokens // 5)
                status = "Success (Gemini Simulation)"
            elif model_type.is_anthropic:
                # Claude Execution
                params = {
                    "model": model_str,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if top_p is not None and top_p != 1.0:
                    # If Top-P is actively being modified (not default 1.0), use it instead of temperature
                    params["top_p"] = top_p
                elif temperature is not None:
                    # Otherwise, use temperature
                    params["temperature"] = temperature
                    
                if top_k is not None:
                    params["top_k"] = top_k

                # Note: System prompts are handled differently in the new Messages API, 
                # but for simplicity in this exercise we'll keep the prompt in the user message.
                
                message = self.anthropic_client.messages.create(**params)
                response_text = message.content[0].text
                output_tokens = message.usage.output_tokens
                input_tokens = message.usage.input_tokens # Use actual tokens from API
            else:
                # OpenAI Execution
                params = {
                    "model": model_str,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if temperature is not None: params["temperature"] = temperature
                if top_p is not None: params["top_p"] = top_p
                # OpenAI doesn't natively support top_k in ChatCompletion
                
                response = self.openai_client.chat.completions.create(**params)
                response_text = response.choices[0].message.content
                output_tokens = response.usage.completion_tokens if response.usage else 0
                input_tokens = response.usage.prompt_tokens if response.usage else input_tokens
        except Exception as e:
            status = f"Error: {str(e)}"
            logger.error(f"AI execution failed for {model_str}: {e}")
            
            # Log the failure before raising
            self.repository.log_result({
                "Timestamp": datetime.now().isoformat(),
                "Request_ID": request_id or "unknown",
                "Experiment_Type": exp_type.value,
                "Model": model_str,
                "Prompt": prompt,
                "Response": f"ERROR: {str(e)}",
                "Input_Tokens": input_tokens,
                "Output_Tokens": 0,
                "Cost_USD": self.calculate_cost(input_tokens, 0, model_str),
                "Status": status,
                "Temperature": temperature,
                "Top_P": top_p,
                "Top_K": top_k
            })
            raise

        # 3. Cost Calculation
        cost = self.calculate_cost(input_tokens, output_tokens, model_str)
        
        analysis = {
            "model": model_str,
            "experiment_type": exp_type.value,
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
            "Experiment_Type": exp_type.value,
            "Model": model_str,
            "Prompt": prompt,
            "Response": response_text,
            "Input_Tokens": input_tokens,
            "Output_Tokens": output_tokens,
            "Cost_USD": cost,
            "Status": status,
            "Temperature": temperature,
            "Top_P": top_p,
            "Top_K": top_k
        })
        
        return {
            "response": response_text,
            "log_analysis": analysis
        }

    def get_results(self) -> list[dict[str, Any]]:
        """Retrieve all logged experiment results via the repository."""
        return self.repository.get_all_results()
