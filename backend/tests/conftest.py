"""Pytest fixtures for PromptShield security gateway test suite."""

import pytest
from fastapi.testclient import TestClient

from app.audit.logger import clear_audit_events
from app.main import app
from app.policies.engine import reset_ml_enforcement_mode
from app.proxy.mock_service import reset_mock_call_count


@pytest.fixture(autouse=True)
def reset_gateway_state():
    """Reset mock downstream invocation counter, audit logs, and policy mode before each test."""
    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()
    yield
    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client
