from typing import Any

import tiktoken


from app.core.models import AIModel


class TokenCounter:
    """Utility class to count tokens for different models."""

    @staticmethod
    def count_openai_tokens(text: str, model: str = AIModel.GPT_4O.value) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base if model not found
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))

    @staticmethod
    def estimate_gemini_tokens(text: str) -> int:
        """
        Estimate tokens for Gemini models.
        Note: Real Gemini token counting requires an API call
        or specific library logic.
        This is a rough estimation (approx 4 chars per token).
        """
        return len(text) // 4

    @classmethod
    def get_token_report(cls, text: str) -> dict[str, Any]:
        """Generate a report of token counts for major models."""
        report = {
            "characters": len(text),
            "gemini_estimate": cls.estimate_gemini_tokens(text)
        }
        
        for model in AIModel:
            report[model.value] = cls.count_openai_tokens(text, model.value)
            
        return report
