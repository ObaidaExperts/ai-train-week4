from enum import Enum


class ExperimentType(str, Enum):
    """Common experiment labels for categorization."""
    STRESS_TEST = "Stress Test"
    BASELINE = "Baseline"
    PRICING_COMPARISON = "Pricing Comparison"
    LIMIT_TEST = "Limit Test"


class AIModel(str, Enum):
    """Supported AI models for token counting and experiments."""
    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    
    # Gemini Models
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_0_PRO = "gemini-1.0-pro"

    @property
    def context_limit(self) -> int:
        """Return the context window limit for the model."""
        limits = {
            # OpenAI Models
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "o1-preview": 128_000,
            "o1-mini": 128_000,
            "gpt-4-turbo": 128_000,
            "gpt-4": 8_192,
            "gpt-3.5-turbo": 16_385,
            "text-embedding-3-small": 8_191,
            "text-embedding-3-large": 8_191,
            # Gemini Models
            "gemini-1.5-pro": 2_000_000,
            "gemini-1.5-flash": 1_000_000,
            "gemini-1.0-pro": 32_768,
        }
        return limits.get(self.value, 0)

    @property
    def pricing(self) -> dict[str, float]:
        """Return the pricing per 1 million tokens (input and output) for the model."""
        prices = {
            # OpenAI Models
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "o1-preview": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "text-embedding-3-small": {"input": 0.02, "output": 0.0},
            "text-embedding-3-large": {"input": 0.13, "output": 0.0},
            # Gemini Models
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
        }
        return prices.get(self.value, {"input": 0.0, "output": 0.0})

    @property
    def is_gemini(self) -> bool:
        """Check if model is a Gemini model."""
        return self.value.startswith("gemini")

    @property
    def is_openai(self) -> bool:
        """Check if model is an OpenAI model."""
        return not self.is_gemini
