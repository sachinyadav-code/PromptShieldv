"""Pytest fixtures for PromptShield security gateway test suite."""

import os
import pytest
from fastapi.testclient import TestClient

from app.audit import clear_audit_events, reset_audit_store
from app.audit.storage import ENV_AUDIT_DB_PATH
from app.auth import ENV_API_KEYS, ENV_AUTH_ENABLED
from app.limits import reset_limits_state
from app.main import app, reset_rag_corpus
from app.policies.engine import reset_ml_enforcement_mode
from app.proxy import reset_downstream_provider, reset_mock_call_count

TEST_API_KEY = "ps-test-suite-key-2026"


@pytest.fixture(autouse=True)
def reset_gateway_state(tmp_path):
    """Reset mock downstream counter, audit logs, policy mode, RAG corpus, auth env, and limits before each test.

    Directs audit storage to a safe, isolated temporary SQLite database for each test.
    """
    test_db = tmp_path / "test_audit.db"
    os.environ[ENV_AUDIT_DB_PATH] = str(test_db)
    os.environ[ENV_AUTH_ENABLED] = "true"
    os.environ[ENV_API_KEYS] = TEST_API_KEY

    reset_audit_store()
    reset_downstream_provider()
    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()
    reset_rag_corpus()
    reset_limits_state()
    yield
    reset_mock_call_count()
    clear_audit_events()
    reset_ml_enforcement_mode()
    reset_rag_corpus()
    reset_limits_state()
    reset_downstream_provider()
    reset_audit_store()




@pytest.fixture
def client():
    """FastAPI TestClient fixture with default test authorization header."""
    with TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"}) as test_client:
        yield test_client
