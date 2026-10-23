"""Proxy package for downstream LLM provider forwarding."""

from app.proxy.factory import (
    ENV_DOWNSTREAM_PROVIDER,
    ENV_DOWNSTREAM_TIMEOUT_SECONDS,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    get_downstream_provider,
    reset_downstream_provider,
    set_downstream_provider,
    validate_downstream_provider_config,
)
from app.proxy.mock_service import (
    MockDownstreamProvider,
    forward_to_mock_downstream,
    get_mock_call_count,
    reset_mock_call_count,
)
from app.proxy.ollama_service import OllamaDownstreamProvider
from app.proxy.provider import DownstreamProvider, DownstreamProviderError
from app.schemas import DownstreamResponse, MockDownstreamResponse

__all__ = [
    "DownstreamProvider",
    "DownstreamProviderError",
    "DownstreamResponse",
    "ENV_DOWNSTREAM_PROVIDER",
    "ENV_DOWNSTREAM_TIMEOUT_SECONDS",
    "ENV_OLLAMA_BASE_URL",
    "ENV_OLLAMA_MODEL",
    "MockDownstreamProvider",
    "MockDownstreamResponse",
    "OllamaDownstreamProvider",
    "forward_to_mock_downstream",
    "get_downstream_provider",
    "get_mock_call_count",
    "reset_downstream_provider",
    "reset_mock_call_count",
    "set_downstream_provider",
    "validate_downstream_provider_config",
]
