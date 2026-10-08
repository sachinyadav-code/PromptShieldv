"""Proxy package for mock downstream forwarding."""

from app.proxy.mock_service import (
    forward_to_mock_downstream,
    get_mock_call_count,
    reset_mock_call_count,
)

__all__ = [
    "forward_to_mock_downstream",
    "get_mock_call_count",
    "reset_mock_call_count",
]
