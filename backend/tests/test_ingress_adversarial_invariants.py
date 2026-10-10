"""Adversarial regression invariants tests for PromptShield Phase 7 Stage 7.5.

Focuses on:
1. ML Enforcement and Downstream Forwarding Isolation:
   - Genuine deterministic-ALLOW + ML-BLOCK scenario for /api/v1/proxy.
   - Genuine deterministic-ALLOW + ML-BLOCK scenario for /api/v1/rag/proxy.
   - Asserts final decision is BLOCK, decision_source is 'ml_enforcement',
     and mock downstream service is NEVER called.
2. Input-Length Boundary Invariants:
   - Exact 10,000-character benign prompt against /api/v1/inspect and /api/v1/proxy.
   - Exact 10,001-character prompt against /api/v1/inspect and /api/v1/proxy.
     Asserts HTTP 422 validation failure before ingress processing, preventing downstream forward.
   - Direct IngressTransformationError boundary: verify generate_ingress_views fails closed
     with PROMPT_LENGTH_EXCEEDED on > 10,000 input, and records that exact error code in audit telemetry.
"""

from unittest.mock import patch
import pytest
from starlette.testclient import TestClient

from app.audit.logger import clear_audit_events, get_audit_events
from app.audit.storage import reset_audit_store
from app.auth import ENV_API_KEYS
from app.detection.ingress import (
    MAX_INGRESS_INPUT_LENGTH,
    IngressTransformationError,
    extract_ingress_telemetry,
    generate_ingress_views,
)
from app.detection.ml_classifier import MLInferenceResult, get_classifier
from app.main import app, reset_rag_corpus
from app.policies.engine import reset_ml_enforcement_mode, set_ml_enforcement_mode
from app.proxy.mock_service import get_mock_call_count, reset_mock_call_count

TEST_API_KEY = "test-stage75-adversarial-key"
HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    """Ensure clean isolated audit database, environment, and counters for each test."""
    db_file = tmp_path / "test_stage75_invariants.db"
    monkeypatch.setenv("PROMPTSHIELD_AUDIT_DB_PATH", str(db_file))
    monkeypatch.setenv(ENV_API_KEYS, TEST_API_KEY)
    reset_audit_store()
    clear_audit_events()
    reset_mock_call_count()
    reset_ml_enforcement_mode()
    reset_rag_corpus()
    yield
    reset_ml_enforcement_mode()
    reset_audit_store()
    reset_rag_corpus()
    reset_mock_call_count()


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# 1. ML Enforcement Overrule and Downstream Isolation
# ---------------------------------------------------------------------------


def test_proxy_ml_enforcement_overrule_blocks_and_never_forwards(client, monkeypatch):
    """In block_on_threshold mode, a prompt with deterministic ALLOW but ML BLOCK

    must return BLOCK with decision_source='ml_enforcement' and NEVER call downstream.
    """
    set_ml_enforcement_mode("block_on_threshold")
    calls_before = get_mock_call_count()

    classifier = get_classifier()

    # Synthetic prompt with zero deterministic rule matches
    benign_text_prompt = "Tell me a bedtime story about a curious robotic turtle."

    # Mock ML classifier to return high risk (injection) exceeding threshold
    def mock_predict(text: str) -> MLInferenceResult:
        return MLInferenceResult(
            available=True,
            score=0.92,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
            model_version="v1",
            error_code=None,
        )

    monkeypatch.setattr(classifier, "predict", mock_predict)

    with patch("app.main.forward_to_mock_downstream") as mock_forward:
        res = client.post("/api/v1/proxy", json={"prompt": benign_text_prompt}, headers=HEADERS)
        assert res.status_code == 200
        data = res.json()

        # Decision invariants (ProxyResponse schema)
        assert data["decision"] == "BLOCK"
        assert data["decision_source"] == "ml_enforcement"
        assert data["matched_rules"] == []  # No deterministic rules matched
        assert data["downstream_response"] is None
        assert "ML prompt injection detection threshold met" in data["explanation"]
        assert data["ml_assessment"]["available"] is True
        assert data["ml_assessment"]["would_block"] is True
        assert data["ml_assessment"]["score"] == 0.92

        # Isolation invariant: mock service never called
        mock_forward.assert_not_called()
        assert get_mock_call_count() == calls_before


def test_rag_proxy_ml_enforcement_overrule_blocks_and_never_forwards(client, monkeypatch):
    """In block_on_threshold mode, a RAG query with deterministic ALLOW but ML BLOCK

    must return BLOCK with decision_source='ml_enforcement' and NEVER call downstream.
    """
    set_ml_enforcement_mode("block_on_threshold")
    calls_before = get_mock_call_count()

    classifier = get_classifier()

    # Query with zero deterministic rule matches against query and corpus
    benign_query = "What is the recommended maintenance schedule for water heaters?"

    def mock_predict(text: str) -> MLInferenceResult:
        return MLInferenceResult(
            available=True,
            score=0.88,
            threshold=0.50,
            predicted_label="injection",
            would_block=True,
            model_version="v1",
            error_code=None,
        )

    monkeypatch.setattr(classifier, "predict", mock_predict)

    with patch("app.main.forward_to_mock_downstream") as mock_forward:
        res = client.post("/api/v1/rag/proxy", json={"query": benign_query, "top_k": 2}, headers=HEADERS)
        assert res.status_code == 200
        data = res.json()

        # Decision invariants (RAGProxyResponse schema)
        assert data["decision"] == "BLOCK"
        assert data["decision_source"] == "ml_enforcement"
        assert data["risk_score"] == 0.88
        assert data["forwarded"] is False
        assert data["downstream_response"] is None
        assert data["matched_rules"] == []
        assert data["ml_assessment"]["available"] is True
        assert data["ml_assessment"]["would_block"] is True
        assert data["ml_assessment"]["score"] == 0.88

        # Downstream isolation invariant
        mock_forward.assert_not_called()
        assert get_mock_call_count() == calls_before


# ---------------------------------------------------------------------------
# 2. Input Length Boundary Invariants
# ---------------------------------------------------------------------------


def test_exact_max_length_boundary_input_processed_cleanly(client):
    """An input of exactly MAX_INGRESS_INPUT_LENGTH (10,000) characters is accepted

    and processed cleanly by /api/v1/inspect and /api/v1/proxy.
    """
    # 10,000 characters of benign repeated words without matched rules
    base = "valid benign text word "
    repeated = (base * ((MAX_INGRESS_INPUT_LENGTH // len(base)) + 1))[:MAX_INGRESS_INPUT_LENGTH]
    assert len(repeated) == 10_000

    # 1. /api/v1/inspect accepts and evaluates
    res_inspect = client.post("/api/v1/inspect", json={"prompt": repeated}, headers=HEADERS)
    assert res_inspect.status_code == 200
    data_inspect = res_inspect.json()
    assert data_inspect["decision"] == "ALLOW"
    assert data_inspect["matched_rules"] == []

    # 2. /api/v1/proxy accepts, evaluates, and forwards downstream
    calls_before = get_mock_call_count()
    res_proxy = client.post("/api/v1/proxy", json={"prompt": repeated}, headers=HEADERS)
    assert res_proxy.status_code == 200
    data_proxy = res_proxy.json()
    assert data_proxy["decision"] == "ALLOW"
    assert data_proxy["downstream_response"] is not None
    assert get_mock_call_count() == calls_before + 1


def test_oversized_boundary_input_fails_closed_and_telemetry_verifies(client):
    """An input of exactly 10,001 characters is rejected at the API schema boundary

    with HTTP 422, never invoking downstream service; and internal ingress generator
    strictly raises IngressTransformationError('PROMPT_LENGTH_EXCEEDED').
    """
    oversized = "a" * (MAX_INGRESS_INPUT_LENGTH + 1)
    assert len(oversized) == 10_001

    # 1. API Schema Boundary: rejected with 422, zero downstream calls
    calls_before = get_mock_call_count()
    with patch("app.main.forward_to_mock_downstream") as mock_forward:
        res_proxy = client.post("/api/v1/proxy", json={"prompt": oversized}, headers=HEADERS)
        assert res_proxy.status_code == 422
        mock_forward.assert_not_called()
        assert get_mock_call_count() == calls_before

        res_inspect = client.post("/api/v1/inspect", json={"prompt": oversized}, headers=HEADERS)
        assert res_inspect.status_code == 422

    # 2. Ingress Defense Layer Boundary: directly verify IngressTransformationError
    with pytest.raises(IngressTransformationError) as exc_info:
        generate_ingress_views(oversized)
    assert exc_info.value.error_code == "PROMPT_LENGTH_EXCEEDED"

    # 3. Telemetry Extraction Invariant: error_code is mapped accurately without leaking text
    telemetry = extract_ingress_telemetry(error_code=exc_info.value.error_code, surface="prompt")
    assert telemetry["ingress_transformation_status"] == "TRANSFORMATION_FAILED"
    assert telemetry["ingress_transformation_error_code"] == "PROMPT_LENGTH_EXCEEDED"
    assert telemetry["ingress_views_count"] == 1
    assert telemetry["ingress_views_evaluated"] == ["raw"]
    assert "a" * 100 not in str(telemetry)
