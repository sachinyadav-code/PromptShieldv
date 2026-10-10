"""Tests for PromptShield Phase 7 Stage 7.4: Security Audit Telemetry for Multi-View Inspection.

Validates that:
1. Multi-view ingress metadata is safely extracted and logged across /inspect, /proxy,
   /rag/inspect, and /rag/proxy without leaking raw, normalized, or decoded prompt text.
2. Ingress views evaluated are strictly allowlisted (raw, normalized, decoded,
   combined_normalized_decoded, decoded_normalized) and bounded to [1, 5].
3. Transformation failures record TRANSFORMATION_FAILED and strictly allowlisted error codes
   while blocking fail-closed.
4. Persistent SQLite storage preserves, indexes, and queries ingress telemetry columns.
5. Ingress telemetry preserves zero-leakage privacy guarantees across all audit logs.
6. RAG document content and classifier exception messages never leak into ingress telemetry.
7. Legacy SQLite database schema without ingress telemetry columns migrates safely and idempotently.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from app.audit.logger import (
    clear_audit_events,
    get_audit_events,
    log_rag_security_event,
    log_security_event,
)
from app.audit.storage import SQLiteAuditStore, reset_audit_store
from app.auth import ENV_API_KEYS
from app.detection.ingress import (
    ALLOWED_INGRESS_ERROR_CODES,
    ALLOWED_INGRESS_VIEWS,
    extract_ingress_telemetry,
    generate_ingress_views,
)
from app.main import app, reset_rag_corpus, set_rag_corpus
from app.rag import Document
from app.schemas import MLAssessment

TEST_API_KEY = "admin-test-key-stage74"
HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture(autouse=True)
def clean_audit_state(tmp_path, monkeypatch):
    """Ensure clean isolated SQLite database and in-memory buffer for each test."""
    db_file = tmp_path / "test_stage74_audit.db"
    monkeypatch.setenv("PROMPTSHIELD_AUDIT_DB_PATH", str(db_file))
    monkeypatch.setenv(ENV_API_KEYS, TEST_API_KEY)
    reset_audit_store()
    clear_audit_events()
    reset_rag_corpus()
    yield
    reset_audit_store()
    reset_rag_corpus()


def test_extract_ingress_telemetry_allowlisted_views_and_bounded_count():
    """Verify extract_ingress_telemetry bounds view count and strictly allowlists views."""
    views = generate_ingress_views("Benign prompt with no obfuscation")
    telemetry = extract_ingress_telemetry(views=views, surface="prompt")

    assert telemetry["ingress_pipeline_version"] == "v1"
    assert telemetry["ingress_surface"] == "prompt"
    assert telemetry["ingress_transformation_status"] == "SUCCESS"
    assert telemetry["ingress_transformation_error_code"] is None
    assert 1 <= telemetry["ingress_views_count"] <= 5
    for view_name in telemetry["ingress_views_evaluated"]:
        assert view_name in ALLOWED_INGRESS_VIEWS


def test_extract_ingress_telemetry_transformation_failure():
    """Verify extract_ingress_telemetry handles transformation failure safely."""
    telemetry = extract_ingress_telemetry(
        error_code="MALFORMED_ENCODING",
        surface="rag_query",
    )

    assert telemetry["ingress_pipeline_version"] == "v1"
    assert telemetry["ingress_surface"] == "rag_query"
    assert telemetry["ingress_transformation_status"] == "TRANSFORMATION_FAILED"
    assert telemetry["ingress_transformation_error_code"] == "MALFORMED_ENCODING"
    assert telemetry["ingress_views_evaluated"] == ["raw"]
    assert telemetry["ingress_views_count"] == 1


def test_strict_ingress_error_code_allowlisting_and_fallback():
    """Verify arbitrary error codes and exception messages are sanitized to safe allowlisted codes."""
    # 1. Valid existing error codes are preserved
    for valid_code in ALLOWED_INGRESS_ERROR_CODES:
        tel = extract_ingress_telemetry(error_code=valid_code, surface="prompt")
        assert tel["ingress_transformation_error_code"] == valid_code

    # 2. Unknown or arbitrary exception text falls back to safe MALFORMED_ENCODING
    arbitrary_code = "SECRET_EXCEPTION_TEXT_CANARY_foo_bar_123"
    tel = extract_ingress_telemetry(error_code=arbitrary_code, surface="prompt")
    assert tel["ingress_transformation_error_code"] == "MALFORMED_ENCODING"

    # 3. Inside logger sanitization boundary, un-allowlisted or malformed codes fall back
    event = log_security_event(
        request_id="req-test-err-boundary",
        endpoint="/api/v1/inspect",
        decision="BLOCK",
        matched_rule_ids=["RULE_MALFORMED_ENCODING"],
        risk_score=0.95,
        forwarded=False,
        prompt="test",
        ingress_telemetry={
            "ingress_pipeline_version": "v1",
            "ingress_surface": "prompt",
            "ingress_views_evaluated": ["raw"],
            "ingress_views_count": 1,
            "ingress_transformation_status": "TRANSFORMATION_FAILED",
            "ingress_transformation_error_code": "SECRET_ATTACK_TRACE_CANARY",
        },
    )
    assert event["ingress_transformation_error_code"] == "MALFORMED_ENCODING"
    assert "SECRET_ATTACK_TRACE_CANARY" not in json.dumps(event)


def test_zero_leakage_in_audit_event_and_storage(tmp_path, monkeypatch):
    """Verify sensitive raw prompt and query text never leak into logged telemetry fields."""
    with TestClient(app) as client:
        sensitive_prompt = "SUPER_SECRET_TOKEN_DO_NOT_LEAK_INTO_AUDIT_12345"

        resp = client.post(
            "/api/v1/inspect",
            json={"prompt": sensitive_prompt},
            headers=HEADERS,
        )
        assert resp.status_code == 200

        events = get_audit_events()
        assert len(events) == 1
        event = events[0]

        # Verify telemetry metadata is recorded
        assert event.get("ingress_pipeline_version") == "v1"
        assert event.get("ingress_surface") == "prompt"
        assert event.get("ingress_transformation_status") == "SUCCESS"
        assert "raw" in event.get("ingress_views_evaluated", [])
        assert 1 <= event.get("ingress_views_count", 0) <= 5

        # Strictly assert secret token is absent from every field in event
        serialized = json.dumps(event)
        assert "SUPER_SECRET_TOKEN" not in serialized
        assert sensitive_prompt not in serialized


def test_inspect_endpoint_persists_ingress_telemetry_on_malformed_encoding():
    """Verify malformed input triggers fail-closed block and TRANSFORMATION_FAILED telemetry."""
    with TestClient(app) as client:
        # Invalid UTF-8 trail byte (%80) triggers IngressTransformationError
        malformed_prompt = "malformed %80 attack"

        resp = client.post(
            "/api/v1/inspect",
            json={"prompt": malformed_prompt},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "BLOCK"
        assert "RULE_MALFORMED_ENCODING" in data["matched_rules"]

        events = get_audit_events()
        assert len(events) == 1
        event = events[0]
        assert event["decision"] == "BLOCK"
        assert event.get("ingress_transformation_status") == "TRANSFORMATION_FAILED"
        assert event.get("ingress_transformation_error_code") == "MALFORMED_ENCODING"
        assert event.get("ingress_views_evaluated") == ["raw"]
        assert event.get("ingress_views_count") == 1


def test_proxy_endpoint_persists_ingress_telemetry_on_malformed_encoding():
    """Verify /proxy endpoint logs TRANSFORMATION_FAILED telemetry and never forwards downstream."""
    with TestClient(app) as client:
        malformed_prompt = "malformed %80 attack"

        resp = client.post(
            "/api/v1/proxy",
            json={"prompt": malformed_prompt},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "BLOCK"
        assert "RULE_MALFORMED_ENCODING" in data["matched_rules"]

        events = get_audit_events()
        assert len(events) == 1
        event = events[0]
        assert event["forwarded"] is False
        assert event.get("ingress_transformation_status") == "TRANSFORMATION_FAILED"
        assert event.get("ingress_transformation_error_code") == "MALFORMED_ENCODING"


def test_rag_inspect_and_proxy_persist_ingress_telemetry():
    """Verify /rag/inspect and /rag/proxy record surface='rag_query' ingress telemetry."""
    with TestClient(app) as client:
        query = "What is the policy for expense reports?"

        # RAG Inspect
        resp_inspect = client.post(
            "/api/v1/rag/inspect",
            json={"query": query, "top_k": 3},
            headers=HEADERS,
        )
        assert resp_inspect.status_code == 200

        # RAG Proxy
        resp_proxy = client.post(
            "/api/v1/rag/proxy",
            json={"query": query, "top_k": 3},
            headers=HEADERS,
        )
        assert resp_proxy.status_code == 200

        events = get_audit_events()
        assert len(events) == 2

        for event in events:
            assert event["event_type"] == "rag_inspection"
            assert event.get("ingress_surface") == "rag_query"
            assert event.get("ingress_pipeline_version") == "v1"
            assert event.get("ingress_transformation_status") == "SUCCESS"
            assert 1 <= event.get("ingress_views_count", 0) <= 5
            assert "raw" in event.get("ingress_views_evaluated", [])


def test_audit_query_endpoint_returns_ingress_telemetry_fields():
    """Verify GET /api/v1/audit/events returns the new optional ingress telemetry fields."""
    with TestClient(app) as client:
        # Generate an audit event via /inspect
        resp_inspect = client.post(
            "/api/v1/inspect",
            json={"prompt": "Harmless benign prompt"},
            headers=HEADERS,
        )
        assert resp_inspect.status_code == 200

        # Query audit events
        resp_query = client.get(
            "/api/v1/audit/events?limit=10",
            headers=HEADERS,
        )
        assert resp_query.status_code == 200
        body = resp_query.json()
        assert body["total_count"] >= 1
        first_event = body["events"][0]

        assert first_event["ingress_pipeline_version"] == "v1"
        assert first_event["ingress_surface"] == "prompt"
        assert first_event["ingress_transformation_status"] == "SUCCESS"
        assert 1 <= first_event["ingress_views_count"] <= 5
        assert isinstance(first_event["ingress_views_evaluated"], list)


def test_rag_document_content_non_leakage_in_ingress_telemetry():
    """Verify canary strings in retrieved RAG documents never leak into ingress telemetry or audit fields."""
    canary_doc_content = "CANARY_CONFIDENTIAL_RAG_CONTENT_98765_DO_NOT_LEAK"
    custom_doc = Document(
        doc_id="doc_canary_001",
        title="Canary Financial Document",
        content=f"Important policy guidelines. Secret payload: {canary_doc_content}",
        source="internal_kb",
    )
    set_rag_corpus([custom_doc])

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/rag/inspect",
            json={"query": "financial policy guidelines", "top_k": 1},
            headers=HEADERS,
        )
        assert resp.status_code == 200

        events = get_audit_events()
        assert len(events) == 1
        event = events[0]

        # Ingress telemetry must reflect the user's query, never document content
        assert event["ingress_surface"] == "rag_query"
        assert event["ingress_transformation_status"] == "SUCCESS"

        # Explicitly assert canary does not leak into any audit event or ingress field
        serialized_event = json.dumps(event)
        assert canary_doc_content not in serialized_event
        assert "CANARY_CONFIDENTIAL" not in serialized_event


def test_classifier_exception_non_leakage_in_ingress_telemetry():
    """Verify classifier exception containing secret canary never enters telemetry or crashes inspection."""
    canary_exception_secret = "CANARY_CLASSIFIER_INTERNAL_EXCEPTION_TRACE_SECRET_456"

    mock_classifier = MagicMock()
    mock_classifier.predict.side_effect = RuntimeError(f"Model hardware failure: {canary_exception_secret}")
    mock_classifier.threshold = 0.50

    with patch("app.detection.ingress.get_classifier", return_value=mock_classifier), \
         patch("app.main.get_classifier", return_value=mock_classifier):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/inspect",
                json={"prompt": "Hello benign assistant"},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()

            # ML assessment degrades gracefully to unavailable
            assert data["ml_assessment"]["available"] is False
            assert data["ml_assessment"]["error_code"] == "INFERENCE_FAILED"

            events = get_audit_events()
            assert len(events) == 1
            event = events[0]

            assert event["ml_available"] is False
            assert event["ml_risk_score"] is None

            # Strictly assert canary secret is omitted from serialized event
            serialized_event = json.dumps(event)
            assert canary_exception_secret not in serialized_event
            assert "CANARY_CLASSIFIER" not in serialized_event


def test_legacy_sqlite_schema_migration_and_compatibility(tmp_path, monkeypatch):
    """Verify existing pre-Stage-7.4 database schema migrates safely and preserves existing rows."""
    legacy_db_file = tmp_path / "legacy_audit_test.db"

    # 1. Create table using pre-Stage-7.4 schema (lacking ingress telemetry columns)
    legacy_schema = """
    CREATE TABLE audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        client_request_id TEXT,
        timestamp TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        decision TEXT,
        decision_source TEXT,
        event_type TEXT NOT NULL,
        matched_rule_ids TEXT,
        heuristic_risk_score REAL,
        forwarded INTEGER,
        prompt_length INTEGER,
        prompt_sha256 TEXT,
        query_length INTEGER,
        query_sha256 TEXT,
        retrieved_doc_ids TEXT,
        num_retrieved INTEGER,
        is_complete INTEGER,
        error_code TEXT,
        outcome TEXT,
        finding_count INTEGER,
        output_status TEXT,
        output_findings_count INTEGER,
        output_matched_rule_ids TEXT,
        output_error_code TEXT,
        created_at TEXT NOT NULL
    );
    """
    conn = sqlite3.connect(str(legacy_db_file))
    conn.executescript(legacy_schema)

    # Insert an existing legacy audit event row
    conn.execute(
        """
        INSERT INTO audit_events (
            event_id, request_id, client_request_id, timestamp, endpoint, decision,
            event_type, heuristic_risk_score, forwarded, prompt_length, prompt_sha256, created_at
        ) VALUES (
            'legacy-event-001', 'legacy-req-001', 'client-001', '2026-01-01T00:00:00Z',
            '/api/v1/inspect', 'ALLOW', 'security_inspection', 0.0, 0, 10, 'dummyhash', '2026-01-01T00:00:00Z'
        );
        """
    )
    conn.commit()
    conn.close()

    # 2. Point store to legacy database and run initialize_schema()
    monkeypatch.setenv("PROMPTSHIELD_AUDIT_DB_PATH", str(legacy_db_file))
    reset_audit_store()
    store = SQLiteAuditStore(legacy_db_file)
    store.initialize_schema()

    # 3. Repeated initialization must be idempotent and safe
    store.initialize_schema()

    # Verify table has all 8 new columns
    conn = sqlite3.connect(str(legacy_db_file))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(audit_events);").fetchall()}
    conn.close()

    expected_new_cols = {
        "ingress_pipeline_version",
        "ingress_surface",
        "ingress_views_evaluated",
        "ingress_views_count",
        "ingress_transformation_status",
        "ingress_transformation_error_code",
        "ml_available",
        "ml_risk_score",
    }
    assert expected_new_cols.issubset(columns)

    # 4. Verify existing row is readable and deserializes safely to default/None
    events, count = store.query_events(limit=10)
    assert count == 1
    assert len(events) == 1
    legacy_row = events[0]

    assert legacy_row["event_id"] == "legacy-event-001"
    assert legacy_row["decision"] == "ALLOW"
    assert legacy_row.get("ingress_pipeline_version") is None
    assert legacy_row.get("ingress_surface") is None
    assert legacy_row.get("ingress_views_evaluated") is None
    assert legacy_row.get("ingress_views_count") is None
    assert legacy_row.get("ingress_transformation_status") is None
    assert legacy_row.get("ingress_transformation_error_code") is None
    assert legacy_row.get("ml_available") is None
    assert legacy_row.get("ml_risk_score") is None
