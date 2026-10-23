"""Provider abstraction and models for downstream LLM execution."""

from abc import ABC, abstractmethod
from app.schemas import DownstreamResponse


class DownstreamProviderError(Exception):
    """Base exception for downstream provider communication and parsing errors."""

    def __init__(self, message: str, error_code: str = "DOWNSTREAM_CALL_FAILED"):
        super().__init__(message)
        self.error_code = error_code


class DownstreamProvider(ABC):
    """Abstract interface for downstream LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> DownstreamResponse:
        """Process prompt through downstream provider and return a validated DownstreamResponse.

        Args:
            prompt: Allowed prompt string to send to downstream service.

        Returns:
            DownstreamResponse with simulated flag, provider identifier, message, and length.

        Raises:
            DownstreamProviderError: On connection failure, timeout, HTTP error, or malformed response.
        """
        ...
