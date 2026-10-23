"""Ollama downstream provider connecting to local Ollama HTTP API."""

import logging
from typing import Any, Dict
import httpx

from app.proxy.provider import DownstreamProvider, DownstreamProviderError
from app.schemas import DownstreamResponse

logger = logging.getLogger("promptshield.proxy.ollama")

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_TIMEOUT_SECONDS = 30.0
MIN_TIMEOUT_SECONDS = 1.0
MAX_TIMEOUT_SECONDS = 300.0


class OllamaDownstreamProvider(DownstreamProvider):
    """Downstream LLM provider connecting to a local Ollama HTTP instance."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)

    def generate(self, prompt: str) -> DownstreamResponse:
        """Process prompt through local Ollama HTTP API: POST /api/generate.

        Executes strictly once with bounded timeouts. Does not perform automatic retries.

        Args:
            prompt: Text prompt string allowed by the gateway.

        Returns:
            DownstreamResponse with simulated=False, provider="ollama", message=completion.

        Raises:
            DownstreamProviderError: On connection failure, timeout, non-200 status, or malformed response.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        connect_timeout = min(5.0, self.timeout_seconds)
        timeout = httpx.Timeout(self.timeout_seconds, connect=connect_timeout)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            logger.warning("Downstream Ollama connection failed")
            raise DownstreamProviderError(
                "Downstream Ollama service connection failed",
                error_code="DOWNSTREAM_CONNECTION_FAILED",
            ) from exc
        except httpx.TimeoutException as exc:
            logger.warning("Downstream Ollama request timed out")
            raise DownstreamProviderError(
                "Downstream Ollama request timed out",
                error_code="DOWNSTREAM_TIMEOUT",
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Downstream Ollama network request error: %s", type(exc).__name__)
            raise DownstreamProviderError(
                "Downstream Ollama request failed",
                error_code="DOWNSTREAM_CALL_FAILED",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected downstream Ollama client failure: %s", type(exc).__name__)
            raise DownstreamProviderError(
                "Downstream provider call failed unexpectedly",
                error_code="DOWNSTREAM_CALL_FAILED",
            ) from exc

        if response.status_code != 200:
            logger.warning("Downstream Ollama returned non-200 status code: %d", response.status_code)
            raise DownstreamProviderError(
                f"Downstream Ollama returned HTTP {response.status_code}",
                error_code="DOWNSTREAM_HTTP_ERROR",
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Downstream Ollama returned non-JSON response")
            raise DownstreamProviderError(
                "Downstream Ollama returned invalid JSON",
                error_code="DOWNSTREAM_INVALID_RESPONSE",
            ) from exc

        if not isinstance(data, dict):
            logger.warning("Downstream Ollama response is not a JSON object")
            raise DownstreamProviderError(
                "Downstream Ollama response is not a valid JSON object",
                error_code="DOWNSTREAM_INVALID_RESPONSE",
            )

        raw_response = data.get("response")
        if not isinstance(raw_response, str):
            logger.warning("Downstream Ollama response missing or non-string 'response' field")
            raise DownstreamProviderError(
                "Downstream Ollama response missing string 'response' field",
                error_code="DOWNSTREAM_INVALID_RESPONSE",
            )

        return DownstreamResponse(
            simulated=False,
            provider="ollama",
            message=raw_response,
            received_prompt_length=len(prompt),
        )
