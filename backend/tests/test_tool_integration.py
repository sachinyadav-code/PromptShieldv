"""Integration tests for PromptShield Phase 3.3 Tool-Call Guard endpoints.

Verifies:
1. Authentication & authorization enforcement on /api/v1/tools/validate and /api/v1/tools/execute.
2. Valid execution of harmless demonstration tools (calculator, sample_data_lookup).
3. Deny-by-default rejection of unknown/unregistered tools.
4. Strict argument schema validation (missing, extra, invalid type, out-of-bounds).
5. Proof that the validation endpoint never executes any tool handler.
6. Proof that the execution endpoint never invokes handlers after validation rejection.
7. DLP inspection blocking synthetic sensitive tool outputs.
8. Output inspection errors failing closed.
9. Privacy-preserving audit events (zero raw arguments, zero raw results, deterministic hash & keys).
10. Audit persistence failure fail-closed behavior (clean 500 error).
11. Rate limiting enforcement on tool endpoints.
12. Backward compatibility: existing /api/v1/proxy and /api/v1/rag/proxy behavior remains unchanged.
"""

import hashlib
import json
from typing import Optional
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, create_model
import pytest

from app.audit import clear_audit_events, get_audit_events, get_audit_store, reset_audit_store
from app.audit.storage import AuditStorageError
from app.auth import HEADER_CONTENT_TYPE_OPTIONS
from app.limits import (
    ENV_RATE_LIMIT_ENABLED,
    ENV_RATE_LIMIT_REQUESTS,
    ENV_RATE_LIMIT_WINDOW_SECONDS,
    reset_limits_state,
)
from app.main import app
from app.output.schemas import OutputFinding, OutputScanResult
from app.output.service import OutputEnforcementResult
from app.tools import (
    ToolDefinition,
    get_tool_registry,
    reset_tool_registry,
)
from tests.conftest import TEST_API_KEY


@pytest.fixture(autouse=True)
def clean_tools_and_audit():
    """Ensure every test runs with clean tool registry and audit events."""
    reset_tool_registry()
    clear_audit_events()
    yield
    reset_tool_registry()
    clear_audit_events()


# ---------------------------------------------------------------------------
# 1. Authentication & Authorization Enforcement
# ---------------------------------------------------------------------------


def test_tools_validate_requires_authentication():
    """Verify POST /api/v1/tools/validate rejects unauthenticated and invalid credentials with 401."""
    with TestClient(app) as unauth_client:
        # Missing auth header
        res_no_auth = unauth_client.post(
            "/api/v1/tools/validate",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
        )
        assert res_no_auth.status_code == 401
        assert "API key required" in res_no_auth.json()["detail"]

        # Invalid auth token
        res_bad_auth = unauth_client.post(
            "/api/v1/tools/validate",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
            headers={"Authorization": "Bearer invalid-wrong-key"},
        )
        assert res_bad_auth.status_code == 401
        assert "Invalid API key" in res_bad_auth.json()["detail"]

        # Valid auth header succeeds
        res_valid = unauth_client.post(
            "/api/v1/tools/validate",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res_valid.status_code == 200
        assert res_valid.json()["decision"] == "ALLOW"


def test_tools_execute_requires_authentication():
    """Verify POST /api/v1/tools/execute rejects unauthenticated and invalid credentials with 401."""
    with TestClient(app) as unauth_client:
        # Missing auth header
        res_no_auth = unauth_client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 5}},
        )
        assert res_no_auth.status_code == 401
        assert "API key required" in res_no_auth.json()["detail"]

        # Invalid API key
        res_bad_auth = unauth_client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 5}},
            headers={"X-API-Key": "completely-bogus-key"},
        )
        assert res_bad_auth.status_code == 401
        assert "Invalid API key" in res_bad_auth.json()["detail"]

        # Valid auth header succeeds
        res_valid = unauth_client.post(
            "/api/v1/tools/execute",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 5, "b": 5}},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res_valid.status_code == 200
        assert res_valid.json()["success"] is True


# ---------------------------------------------------------------------------
# 2. Harmless Demo Tool Execution (Calculator & Sample Lookup)
# ---------------------------------------------------------------------------


def test_tools_execute_valid_calculator_all_operations(client):
    """Verify calculator executes all operations safely via POST /api/v1/tools/execute."""
    cases = [
        ("add", 10.0, 5.0, 15.0),
        ("subtract", 20.0, 7.0, 13.0),
        ("multiply", 6.0, 7.0, 42.0),
        ("divide", 100.0, 4.0, 25.0),
    ]
    for op, a, b, expected in cases:
        res = client.post(
            "/api/v1/tools/execute",
            json={
                "tool_name": "calculator",
                "arguments": {"operation": op, "a": a, "b": b},
                "call_id": f"call_{op}_1",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["tool_name"] == "calculator"
        assert body["client_request_id"] == f"call_{op}_1"
        assert body["output"]["operation"] == op
        assert body["output"]["result"] == expected
        assert body["error"] is None
        assert body["error_code"] is None


def test_tools_execute_calculator_division_by_zero(client):
    """Verify division by zero is handled safely without crashing or exposing internal stack traces."""
    res = client.post(
        "/api/v1/tools/execute",
        json={
            "tool_name": "calculator",
            "arguments": {"operation": "divide", "a": 10.0, "b": 0.0},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["output"] is None
    assert "Division by zero is undefined" in body["error"]
    assert body["error_code"] == "TOOL_EXECUTION_ERROR"


def test_tools_execute_sample_data_lookup(client):
    """Verify sample data lookup demo tool works correctly for existing and missing records."""
    # Existing record
    res_found = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "sample_data_lookup", "arguments": {"record_id": "item_001"}},
    )
    assert res_found.status_code == 200
    body_found = res_found.json()
    assert body_found["success"] is True
    assert body_found["output"]["found"] is True
    assert body_found["output"]["record"]["name"] == "Solar Inverter A1"

    # Non-existent record
    res_missing = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "sample_data_lookup", "arguments": {"record_id": "nonexistent_99"}},
    )
    assert res_missing.status_code == 200
    body_missing = res_missing.json()
    assert body_missing["success"] is True
    assert body_missing["output"]["found"] is False


# ---------------------------------------------------------------------------
# 3. Deny-by-Default: Unknown Tool Rejection
# ---------------------------------------------------------------------------


def test_tools_validate_unknown_tool_rejection(client):
    """Verify validate endpoint blocks unregistered tools with TOOL_NOT_ALLOWED."""
    res = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "bash_shell", "arguments": {"cmd": "id"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "BLOCK"
    assert body["error_code"] == "TOOL_NOT_ALLOWED"
    assert "not in the allowed registry" in body["explanation"]
    assert body["validated_arguments"] is None


def test_tools_execute_unknown_tool_rejection(client):
    """Verify execute endpoint blocks unregistered tools with TOOL_NOT_ALLOWED."""
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "system_exec", "arguments": {"command": "rm -rf /"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["output"] is None
    assert body["error_code"] == "TOOL_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# 4. Strict Argument Validation (Missing, Extra, Type, Bounds)
# ---------------------------------------------------------------------------


def test_tools_argument_validation_missing_arguments(client):
    """Verify missing required parameters are rejected with INVALID_ARGUMENTS."""
    res = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 10.0}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "BLOCK"
    assert body["error_code"] == "INVALID_ARGUMENTS"
    assert "Missing required argument 'b'" in body["explanation"]


def test_tools_argument_validation_extra_arguments_forbidden(client):
    """Verify extra unexpected arguments (extra='forbid') are rejected."""
    res = client.post(
        "/api/v1/tools/validate",
        json={
            "tool_name": "calculator",
            "arguments": {"operation": "add", "a": 1.0, "b": 2.0, "bypass_flag": True},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "BLOCK"
    assert body["error_code"] == "INVALID_ARGUMENTS"
    assert "Unexpected extra argument 'bypass_flag'" in body["explanation"]


def test_tools_argument_validation_type_and_bounds_rejection(client):
    """Verify invalid types and out-of-bound numbers are rejected."""
    # Type mismatch
    res_type = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": "not-a-number", "b": 2.0}},
    )
    assert res_type.status_code == 200
    assert res_type.json()["decision"] == "BLOCK"

    # Out of bounds (> 1e9)
    res_bounds = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1e12, "b": 1.0}},
    )
    assert res_bounds.status_code == 200
    assert res_bounds.json()["decision"] == "BLOCK"
    assert res_bounds.json()["error_code"] == "INVALID_ARGUMENTS"


# ---------------------------------------------------------------------------
# 5. Handler Isolation: Validation Never Executes Handler
# ---------------------------------------------------------------------------


def test_validation_endpoint_never_executes_handler(client):
    """Verify POST /api/v1/tools/validate never invokes the tool execution handler."""
    mock_handler = MagicMock(return_value={"result": 100})
    registry = get_tool_registry()

    class CustomArgs(BaseModel):
        val: int
        model_config = ConfigDict(extra="forbid")

    registry.register_tool(
        ToolDefinition(
            name="custom_test_tool",
            description="Test tool with mock handler",
            category="READ_ONLY",
            args_schema=CustomArgs,
        ),
        handler=mock_handler,
    )

    # Valid validation request
    res = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "custom_test_tool", "arguments": {"val": 42}},
    )
    assert res.status_code == 200
    assert res.json()["decision"] == "ALLOW"

    # CRITICAL: Handler was NEVER called
    mock_handler.assert_not_called()


def test_execution_endpoint_never_invokes_handler_on_rejected_call(client):
    """Verify POST /api/v1/tools/execute never calls handler if validation fails."""
    mock_handler = MagicMock(return_value={"result": 100})
    registry = get_tool_registry()

    class CustomArgs(BaseModel):
        val: int
        model_config = ConfigDict(extra="forbid")

    registry.register_tool(
        ToolDefinition(
            name="custom_test_tool",
            description="Test tool with mock handler",
            category="READ_ONLY",
            args_schema=CustomArgs,
        ),
        handler=mock_handler,
    )

    # Invalid arguments (extra field)
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "custom_test_tool", "arguments": {"val": 42, "extra": "evil"}},
    )
    assert res.status_code == 200
    assert res.json()["success"] is False

    # CRITICAL: Handler was NEVER called
    mock_handler.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Output DLP Security on Tool Results
# ---------------------------------------------------------------------------


def test_tools_execute_dlp_blocks_critical_sensitive_output(client):
    """Verify output DLP scans serialized tool results and blocks critical data (private keys, cards)."""
    # Register a tool that returns a synthetic private key
    class SecretLeakArgs(BaseModel):
        dummy: str = "ok"
        model_config = ConfigDict(extra="forbid")

    def leaky_handler(args: SecretLeakArgs):
        return {
            "key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0m1q" + "A" * 64 + "\n-----END RSA PRIVATE KEY-----",
        }

    registry = get_tool_registry()
    registry.register_tool(
        ToolDefinition(
            name="leaky_tool",
            description="Test tool returning sensitive credentials",
            category="READ_ONLY",
            args_schema=SecretLeakArgs,
        ),
        handler=leaky_handler,
    )

    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "leaky_tool", "arguments": {}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["output"] is None
    assert "Critical sensitive data detected" in body["error"]
    assert body["error_code"] == "OUTPUT_BLOCKED_BY_POLICY"


def test_tools_execute_dlp_inspection_error_fails_closed(client, monkeypatch):
    """Verify that if output DLP scanning encounters an error, execution fails closed."""
    # Monkeypatch enforce_output_security to simulate a scanner failure
    def mock_enforce_scan_error(*args, **kwargs):
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation="Output blocked by security policy: Downstream response could not be verified.",
            error_code="OUTPUT_SCAN_ERROR",
            risk_score=1.0,
            is_complete=False,
        )

    monkeypatch.setattr("app.main.enforce_output_security", mock_enforce_scan_error)

    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 1}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["output"] is None
    assert body["error_code"] == "OUTPUT_SCAN_ERROR"


# ---------------------------------------------------------------------------
# 7. Privacy-Preserving Audit Logging
# ---------------------------------------------------------------------------


def test_audit_event_recorded_without_raw_argument_or_result_leakage(client):
    """Verify tool audit events record tool_name, argument_keys, count, SHA-256 hash, and zero raw values."""
    sensitive_value = "super_secret_sensitive_arg_value_987654321"

    class SensitiveInputArgs(BaseModel):
        secret_token: str
        model_config = ConfigDict(extra="forbid")

    registry = get_tool_registry()
    registry.register_tool(
        ToolDefinition(
            name="sensitive_echo",
            description="Echo demo tool",
            category="READ_ONLY",
            args_schema=SensitiveInputArgs,
        ),
        handler=lambda args: {"status": "ok"},
    )

    res = client.post(
        "/api/v1/tools/execute",
        json={
            "tool_name": "sensitive_echo",
            "arguments": {"secret_token": sensitive_value},
            "call_id": "call_audit_test_42",
        },
    )
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Check audit events recorded
    events = get_audit_events()
    tool_events = [e for e in events if e.get("event_type") == "tool_execution"]
    assert len(tool_events) >= 1
    event = tool_events[-1]

    assert event["tool_name"] == "sensitive_echo"
    assert event["client_request_id"] == "call_audit_test_42"
    assert event["decision"] == "ALLOW"
    assert event["decision_source"] == "tool_guard"
    assert event["argument_keys"] == ["secret_token"]
    assert event["argument_count"] == 1
    assert event["argument_sha256"] is not None

    # CRITICAL PRIVACY CHECKS: Raw argument values and raw outputs NEVER appear in audit event
    event_str = json.dumps(event)
    assert sensitive_value not in event_str
    # Verify no prompt_sha256 or query_sha256 field confusion
    assert "prompt_sha256" not in event or event.get("prompt_sha256") is None
    assert "query_sha256" not in event or event.get("query_sha256") is None


# ---------------------------------------------------------------------------
# 8. Audit Persistence Failure Fail-Closed Behavior
# ---------------------------------------------------------------------------


def test_tools_audit_persistence_failure_returns_clean_500(client, monkeypatch):
    """Verify gateway fails closed with HTTP 500 without leaking SQL/disk paths if audit storage fails."""
    store = get_audit_store()
    monkeypatch.setattr(
        store,
        "save_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AuditStorageError("Simulated write failure on audit storage")),
    )

    with TestClient(app, raise_server_exceptions=False) as tc:
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        res = tc.post(
            "/api/v1/tools/execute",
            json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
            headers=headers,
        )
        assert res.status_code == 500
        assert res.json()["detail"] == "Internal gateway error occurred."
        # No leak of exception or SQLite details
        assert "Simulated write failure" not in res.text
        assert "sqlite" not in res.text.lower()
        assert res.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS


# ---------------------------------------------------------------------------
# 9. Rate Limiting Enforcement on Tool Endpoints
# ---------------------------------------------------------------------------


def test_tools_rate_limiting_enforcement(client, monkeypatch):
    """Verify tool endpoints enforce rate limiting (HTTP 429) when request threshold is exceeded."""
    monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
    monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "2")
    monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, "60")
    reset_limits_state()

    # First request: ALLOW (200)
    res1 = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 1}},
    )
    assert res1.status_code == 200

    # Second request: ALLOW (200)
    res2 = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 2, "b": 2}},
    )
    assert res2.status_code == 200

    # Third request: RATE LIMITED (429)
    res3 = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 3, "b": 3}},
    )
    assert res3.status_code == 429
    assert res3.json()["detail"] == "Rate limit exceeded. Please retry later."
    assert "Retry-After" in res3.headers


# ---------------------------------------------------------------------------
# 10. Existing Proxy and RAG Proxy Invariance
# ---------------------------------------------------------------------------


def test_existing_proxy_and_rag_proxy_unchanged(client):
    """Verify /api/v1/proxy and /api/v1/rag/proxy operate normally and are unchanged by tool additions."""
    # 1. /api/v1/proxy
    proxy_res = client.post(
        "/api/v1/proxy",
        json={"prompt": "Explain quantum computing in one short paragraph."},
    )
    assert proxy_res.status_code == 200
    proxy_body = proxy_res.json()
    assert proxy_body["decision"] == "ALLOW"
    assert proxy_body["downstream_response"] is not None
    assert proxy_body["decision_source"] == "allow_default"

    # 2. /api/v1/rag/proxy
    rag_res = client.post(
        "/api/v1/rag/proxy",
        json={"query": "How do solar panels work?"},
    )
    assert rag_res.status_code == 200
    rag_body = rag_res.json()
    assert rag_body["decision"] in ("ALLOW", "REVIEW")


# ---------------------------------------------------------------------------
# 11. Minimal Tool Input Hardening (MAX_TOOL_ARGUMENT_KEYS = 50)
# ---------------------------------------------------------------------------


def test_tools_validate_oversized_arguments_rejected_422(client):
    """Verify POST /api/v1/tools/validate rejects requests with > 50 argument keys with HTTP 422."""
    oversized = {f"k_{i}": i for i in range(51)}
    res = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "calculator", "arguments": oversized},
    )
    assert res.status_code == 422
    assert "exceed maximum allowed limit of 50 keys" in res.text


def test_tools_execute_oversized_arguments_rejected_422(client):
    """Verify POST /api/v1/tools/execute rejects requests with > 50 argument keys with HTTP 422."""
    oversized = {f"k_{i}": i for i in range(51)}
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "calculator", "arguments": oversized},
    )
    assert res.status_code == 422
    assert "exceed maximum allowed limit of 50 keys" in res.text


def test_tools_execute_oversized_request_never_invokes_handler(client):
    """Verify an oversized request (> 50 keys) never invokes the underlying tool execution handler."""
    mock_handler = MagicMock(return_value={"status": "ok"})
    registry = get_tool_registry()

    fields = {f"k_{i}": (Optional[int], None) for i in range(60)}
    BigSchema = create_model("BigSchema", __config__=None, __base__=BaseModel, **fields)
    BigSchema.model_config = ConfigDict(extra="forbid")

    registry.register_tool(
        ToolDefinition(
            name="big_tool",
            description="Tool accepting many fields",
            category="READ_ONLY",
            args_schema=BigSchema,
        ),
        handler=mock_handler,
    )

    oversized = {f"k_{i}": i for i in range(51)}
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "big_tool", "arguments": oversized},
    )
    assert res.status_code == 422
    mock_handler.assert_not_called()


def test_tools_execute_50_keys_allowed_on_supporting_tool(client):
    """Verify a tool whose schema supports 50 arguments succeeds with exactly 50 keys."""
    mock_handler = MagicMock(return_value={"count": 50, "status": "ok"})
    registry = get_tool_registry()

    fields = {f"field_{i}": (Optional[int], None) for i in range(50)}
    FiftyFieldArgs = create_model("FiftyFieldArgs", __config__=None, __base__=BaseModel, **fields)
    FiftyFieldArgs.model_config = ConfigDict(extra="forbid")

    registry.register_tool(
        ToolDefinition(
            name="fifty_key_api_tool",
            description="Tool accepting exactly 50 fields",
            category="READ_ONLY",
            args_schema=FiftyFieldArgs,
        ),
        handler=mock_handler,
    )

    args_50 = {f"field_{i}": i for i in range(50)}
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "fifty_key_api_tool", "arguments": args_50},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["output"] == {"count": 50, "status": "ok"}
    mock_handler.assert_called_once()


def test_normal_calculator_and_sample_lookup_requests_unaffected(client):
    """Verify normal valid calculator and sample data lookup requests remain unaffected."""
    # Calculator
    calc_res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "calculator", "arguments": {"operation": "add", "a": 12.5, "b": 7.5}},
    )
    assert calc_res.status_code == 200
    assert calc_res.json()["success"] is True
    assert calc_res.json()["output"]["result"] == 20.0

    # Sample data lookup
    lookup_res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "sample_data_lookup", "arguments": {"record_id": "item_002"}},
    )
    assert lookup_res.status_code == 200
    assert lookup_res.json()["success"] is True
    assert lookup_res.json()["output"]["found"] is True
    assert lookup_res.json()["output"]["record"]["category"] == "Sensors"



# ---------------------------------------------------------------------------
# 12. Phase 3.4 Durable Audit Persistence Verification
# ---------------------------------------------------------------------------


def test_tools_execute_persists_tool_security_metadata_to_durable_store(client):
    """Verify POST /api/v1/tools/execute persists tool_name, argument_count, argument_keys,
    and argument_sha256 to durable storage, queryable via GET /api/v1/audit/events.
    """
    clear_audit_events()
    res = client.post(
        "/api/v1/tools/execute",
        json={"tool_name": "calculator", "arguments": {"operation": "multiply", "a": 3.0, "b": 4.0}},
    )
    assert res.status_code == 200
    assert res.json()["success"] is True

    audit_res = client.get("/api/v1/audit/events?event_type=tool_execution")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["total_count"] >= 1
    event = audit_data["events"][0]
    assert event["event_type"] == "tool_execution"
    assert event["tool_name"] == "calculator"
    assert event["argument_count"] == 3
    assert set(event["argument_keys"]) == {"operation", "a", "b"}
    assert event["argument_sha256"] is not None
    assert len(event["argument_sha256"]) == 64


def test_tools_validate_persists_tool_security_metadata_to_durable_store(client):
    """Verify POST /api/v1/tools/validate persists tool security metadata for block decision."""
    clear_audit_events()
    res = client.post(
        "/api/v1/tools/validate",
        json={"tool_name": "unauthorized_tool", "arguments": {"param1": "test"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "BLOCK"
    assert body["error_code"] == "TOOL_NOT_ALLOWED"

    audit_res = client.get("/api/v1/audit/events?event_type=tool_validation")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["total_count"] >= 1
    event = audit_data["events"][0]
    assert event["event_type"] == "tool_validation"
    assert event["decision"] == "BLOCK"
    assert event["tool_name"] == "unauthorized_tool"
    assert event["argument_count"] == 1
    assert event["argument_keys"] == ["param1"]
    assert event["argument_sha256"] is not None
    assert len(event["argument_sha256"]) == 64
