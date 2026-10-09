"""Pytest fixtures for PromptShield security gateway test suite."""

import os
import pytest
from fastapi.testclient import TestClient

from app.audit.logger import clear_audit_events
from app.auth import ENV_API_KEYS, ENV_AUTH_ENABLED
from app.main import app, reset_rag_corpus
from app.policies.engine import reset_ml_enforcement_mode
from app.proxy.mock_service import reset_mock_call_count

TEST_API_KEY = "ps-test-suite-key-2026"


@pytest.fixture(autouse=True)
def reset_gateway_state():
    """Reset mock downstream invocation counter, audit logs, policy mode, RAG corpus, and auth env before each test."""
    os.environ[ENV_AUTH_ENABLED] = "true"
    os.environ[ENV_API_KEYS] = TEST_API_KEY

    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()
    reset_rag_corpus()
    yield
    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()
    reset_rag_corpus()



@pytest.fixture
def client():
    """FastAPI TestClient fixture with default test authorization header."""
    with TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"}) as test_client:
        yield test_client
