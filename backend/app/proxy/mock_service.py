"""Mock downstream service for PromptShield.

Simulates a safe downstream LLM response. Keeps an internal invocation counter
to verify that blocked requests never reach the downstream service.
Does not connect to external networks, APIs, databases, or shell tools.
"""

import threading
from app.proxy.provider import DownstreamProvider
from app.schemas import DownstreamResponse, MockDownstreamResponse

_lock = threading.Lock()
_call_count = 0


def get_mock_call_count() -> int:
    """Return the total number of calls made to the mock downstream service."""
    with _lock:
        return _call_count


def reset_mock_call_count() -> None:
    """Reset the mock call counter to zero (used by test suites)."""
    global _call_count
    with _lock:
        _call_count = 0


def forward_to_mock_downstream(prompt: str) -> DownstreamResponse:
    """Process forwarded prompt through simulated downstream service.

    Increments the call counter and returns a clearly labeled synthetic response.
    """
    global _call_count
    with _lock:
        _call_count += 1

    prompt_length = len(prompt)
    message = (
        f"[SIMULATION] Request reached mock downstream service. "
        f"Simulated response generated for payload of {prompt_length} characters. "
        f"This is an educational simulation, not a live AI model completion."
    )

    return DownstreamResponse(
        simulated=True,
        provider="mock-downstream-service",
        message=message,
        received_prompt_length=prompt_length,
    )


class MockDownstreamProvider(DownstreamProvider):
    """Simulated downstream LLM service implementing DownstreamProvider."""

    def generate(self, prompt: str) -> DownstreamResponse:
        """Forward prompt through simulated mock service."""
        return forward_to_mock_downstream(prompt)
