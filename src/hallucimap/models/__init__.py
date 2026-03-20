"""LLM adapters for hallucimap.

All adapters share the :class:`BaseLLMAdapter` interface and can be used
interchangeably with any probe or scorer.
"""

from hallucimap.models.anthropic_adapter import AnthropicAdapter
from hallucimap.models.base import BaseLLMAdapter, CompletionRequest, CompletionResponse
from hallucimap.models.hf_adapter import HFAdapter
from hallucimap.models.openai_adapter import OpenAIAdapter

__all__ = [
    "BaseLLMAdapter",
    "CompletionRequest",
    "CompletionResponse",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HFAdapter",
]
