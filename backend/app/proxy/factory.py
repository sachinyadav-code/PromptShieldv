"""Configuration and factory for downstream LLM providers."""

import os
import threading
from typing import Optional

from app.proxy.mock_service import MockDownstreamProvider
from app.proxy.ollama_service import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    OllamaDownstreamProvider,
)
from app.proxy.provider import DownstreamProvider

ENV_DOWNSTREAM_PROVIDER = "PROMPTSHIELD_DOWNSTREAM_PROVIDER"
ENV_OLLAMA_BASE_URL = "PROMPTSHIELD_OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "PROMPTSHIELD_OLLAMA_MODEL"
ENV_DOWNSTREAM_TIMEOUT_SECONDS = "PROMPTSHIELD_DOWNSTREAM_TIMEOUT_SECONDS"

_provider_lock = threading.RLock()
_programmatic_provider: Optional[DownstreamProvider] = None


def validate_downstream_provider_config() -> str:
    """Validate downstream provider configuration without making external network calls.

    Returns:
        Validated provider identifier ('mock' or 'ollama').

    Raises:
        ValueError: If provider choice, URL, model, or timeout is invalid.
    """
    provider_raw = os.environ.get(ENV_DOWNSTREAM_PROVIDER, "mock").strip().lower()
    if provider_raw not in ("mock", "ollama"):
        raise ValueError(
            f"Unsupported downstream provider '{provider_raw}'. "
            f"Supported providers: 'mock', 'ollama'."
        )

    timeout_raw = os.environ.get(ENV_DOWNSTREAM_TIMEOUT_SECONDS)
    if timeout_raw is not None and timeout_raw.strip():
        try:
            val = float(timeout_raw.strip())
            if val < MIN_TIMEOUT_SECONDS or val > MAX_TIMEOUT_SECONDS:
                raise ValueError(
                    f"Timeout must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds"
                )
        except ValueError as exc:
            raise ValueError(
                f"Invalid {ENV_DOWNSTREAM_TIMEOUT_SECONDS}: '{timeout_raw}'. "
                f"Must be a float between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}."
            ) from exc

    if provider_raw == "ollama":
        base_url = os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL).strip()
        if not base_url or not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ValueError(
                f"Invalid {ENV_OLLAMA_BASE_URL}: '{base_url}'. Must be a valid HTTP or HTTPS URL."
            )

        model = os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL).strip()
        if not model:
            raise ValueError(f"Invalid {ENV_OLLAMA_MODEL}: model name cannot be empty.")

    return provider_raw


def get_downstream_provider() -> DownstreamProvider:
    """Return the active downstream provider (programmatic override or environment-configured)."""
    with _provider_lock:
        if _programmatic_provider is not None:
            return _programmatic_provider

        provider_name = validate_downstream_provider_config()
        if provider_name == "ollama":
            base_url = os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL).strip()
            model = os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_MODEL).strip()
            timeout_raw = os.environ.get(ENV_DOWNSTREAM_TIMEOUT_SECONDS)
            timeout = (
                float(timeout_raw.strip())
                if timeout_raw is not None and timeout_raw.strip()
                else DEFAULT_TIMEOUT_SECONDS
            )
            return OllamaDownstreamProvider(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout,
            )
        return MockDownstreamProvider()


def set_downstream_provider(provider: Optional[DownstreamProvider]) -> None:
    """Set a programmatic downstream provider override (useful for testing)."""
    global _programmatic_provider
    with _provider_lock:
        _programmatic_provider = provider


def reset_downstream_provider() -> None:
    """Reset programmatic downstream provider override."""
    global _programmatic_provider
    with _provider_lock:
        _programmatic_provider = None
