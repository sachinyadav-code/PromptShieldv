"""Structured security event logging for PromptShield Phase 1-3.

Important Security & Storage Notice:
- In-Memory Ring Buffer: Audit events are maintained strictly in an in-memory
  FIFO buffer (maximum 1,000 events) for local inspection, test validation,
  and forensic correlation during the current process lifecycle.
- Non-Persistent: Audit events are NOT written to a persistent database or disk
  store and do NOT persist across application restarts.
- Not Tamper-Proof: This in-memory store does NOT provide cryptographic
  immutability, append-only guarantees, or formal regulatory compliance certifications.
- Privacy Guarantees: Full raw user prompts, user search queries, retrieved document
  contents, document metadata, credentials, and model-generated responses are NEVER
  stored in audit event payloads. Only content lengths, SHA-256 cryptographic fingerprints,
  and sanitized identifiers are recorded.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from app.audit.storage import (
    AuditStorageError,
    SQLiteAuditStore,
    get_audit_store,
)
from app.detection.ingress import ALLOWED_INGRESS_ERROR_CODES

logger = logging.getLogger("promptshield.audit")

# Maximum capacity for in-memory FIFO audit ring buffer
MAX_AUDIT_EVENTS = 1000
_audit_events: List[Dict[str, Any]] = []
_audit_lock = threading.Lock()

# Deterministic allowlist: letters, digits, underscore, hyphen, dot, colon, plus
AUDIT_ID_ALLOWED_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9_\-.:+]")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize_audit_id(val: Any, max_length: int = 128, fallback: str = "") -> str:
    """Sanitize identifier string to prevent log injection and field forging (CWE-117).

    Rules:
    - Strips non-printable ASCII and control characters (\x00-\x1f, \x7f-\x9f, including \n, \r, \t).
    - Normalizes characters outside the safe allowlist [a-zA-Z0-9_-.:+] to underscore '_'.
    - Bounded strictly to max_length characters.
    - If empty or entirely stripped of characters, returns safe fallback without logging raw input.
    """
    if val is None:
        return fallback
    val_str = str(val)
    # Strip control characters and newlines first
    no_ctrl = CONTROL_CHARS_PATTERN.sub("", val_str)
    # Normalize disallowed characters to '_'
    safe_chars = AUDIT_ID_ALLOWED_CHARS_PATTERN.sub("_", no_ctrl)
    truncated = safe_chars[:max_length]
    return truncated if truncated else fallback


def _compute_prompt_hash(prompt: str) -> str:
    """Compute SHA-256 fingerprint of prompt without storing raw content."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def log_security_event(
    request_id: str,
    endpoint: str,
    decision: str,
    matched_rule_ids: List[str],
    risk_score: float,
    forwarded: bool,
    prompt: str,
    client_request_id: Optional[str] = None,
    decision_source: Optional[str] = None,
    output_status: Optional[str] = None,
    output_findings_count: Optional[int] = None,
    output_matched_rule_ids: Optional[List[str]] = None,
    output_error_code: Optional[str] = None,
    ingress_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a structured security audit event.

    Avoids storing raw prompt text to prevent sensitive data accumulation.
    Stores prompt length and SHA-256 fingerprint for forensic correlation.
    """
    sanitized_req_id = sanitize_audit_id(request_id, max_length=128, fallback="unknown_req")
    sanitized_client_req_id = (
        sanitize_audit_id(client_request_id, max_length=128, fallback="unknown_client")
        if client_request_id is not None
        else None
    )
    sanitized_endpoint = CONTROL_CHARS_PATTERN.sub("", str(endpoint))[:64] or "/unknown"
    sanitized_decision = sanitize_audit_id(decision, max_length=32, fallback="UNKNOWN")
    sanitized_decision_source = (
        sanitize_audit_id(decision_source, max_length=64, fallback="unknown_source")
        if decision_source is not None
        else None
    )
    sanitized_matched_rule_ids = [
        sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
        for r in matched_rule_ids
    ][:100]

    event = {
        "event_id": sanitized_req_id,
        "request_id": sanitized_req_id,
        "client_request_id": sanitized_client_req_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": sanitized_endpoint,
        "decision": sanitized_decision,
        "decision_source": sanitized_decision_source,
        "matched_rule_ids": sanitized_matched_rule_ids,
        "heuristic_risk_score": round(max(0.0, min(float(risk_score), 1.0)), 2),
        "forwarded": bool(forwarded),
        "prompt_length": max(0, len(prompt)),
        "prompt_sha256": _compute_prompt_hash(prompt),
    }

    if output_status is not None:
        event["output_status"] = sanitize_audit_id(output_status, max_length=32, fallback="UNKNOWN")
        event["output_findings_count"] = max(0, int(output_findings_count or 0))
        event["output_matched_rule_ids"] = [
            sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
            for r in (output_matched_rule_ids or [])
        ][:100]
        event["output_error_code"] = (
            sanitize_audit_id(output_error_code, max_length=64, fallback="unknown_error")
            if output_error_code is not None
            else None
        )

    if ingress_telemetry:
        event["ingress_pipeline_version"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_pipeline_version"), max_length=16, fallback="v1"
        )
        event["ingress_surface"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_surface"), max_length=32, fallback="prompt"
        )
        views = ingress_telemetry.get("ingress_views_evaluated")
        if isinstance(views, (list, tuple)):
            allowed = {"raw", "normalized", "decoded", "combined_normalized_decoded", "decoded_normalized"}
            event["ingress_views_evaluated"] = [str(v) for v in views if str(v) in allowed][:5]
        else:
            event["ingress_views_evaluated"] = ["raw"]
        raw_count = ingress_telemetry.get("ingress_views_count")
        event["ingress_views_count"] = max(1, min(int(raw_count if raw_count is not None else 1), 5))
        event["ingress_transformation_status"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_transformation_status"), max_length=32, fallback="SUCCESS"
        )
        err_code = ingress_telemetry.get("ingress_transformation_error_code")
        if err_code is not None and isinstance(err_code, str) and err_code in ALLOWED_INGRESS_ERROR_CODES:
            event["ingress_transformation_error_code"] = err_code
        elif err_code is not None:
            event["ingress_transformation_error_code"] = "MALFORMED_ENCODING"
        else:
            event["ingress_transformation_error_code"] = None
        if ingress_telemetry.get("ml_available") is not None:
            event["ml_available"] = bool(ingress_telemetry["ml_available"])
        else:
            event["ml_available"] = None
        if ingress_telemetry.get("ml_risk_score") is not None:
            event["ml_risk_score"] = round(max(0.0, min(float(ingress_telemetry["ml_risk_score"]), 1.0)), 4)
        else:
            event["ml_risk_score"] = None

    event["event_type"] = "security_inspection"

    with _audit_lock:
        if len(_audit_events) >= MAX_AUDIT_EVENTS:
            _audit_events.pop(0)
        _audit_events.append(copy.deepcopy(event))

    # Persist to durable SQLite storage
    store = get_audit_store()
    store.save_event(event)

    logger.info(json.dumps(event))
    return event


def log_rag_security_event(
    request_id: str,
    endpoint: str,
    decision: str,
    matched_rule_ids: List[str],
    risk_score: float,
    forwarded: bool,
    query: str,
    retrieved_doc_ids: List[str],
    num_retrieved: int,
    is_complete: bool,
    client_request_id: Optional[str] = None,
    decision_source: Optional[str] = None,
    error_code: Optional[str] = None,
    output_status: Optional[str] = None,
    output_findings_count: Optional[int] = None,
    output_matched_rule_ids: Optional[List[str]] = None,
    output_error_code: Optional[str] = None,
    ingress_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a structured security audit event for RAG requests.

    Avoids storing raw query text, retrieved document contents, metadata, or secrets.
    Stores query length, SHA-256 fingerprint, and sanitized doc IDs for forensics.
    """
    sanitized_req_id = sanitize_audit_id(request_id, max_length=128, fallback="unknown_req")
    sanitized_client_request_id = (
        sanitize_audit_id(client_request_id, max_length=128, fallback="unknown_client")
        if client_request_id is not None
        else None
    )
    sanitized_doc_ids = [
        sanitize_audit_id(did, max_length=64, fallback="unknown_doc") for did in retrieved_doc_ids
    ][:50]
    sanitized_endpoint = CONTROL_CHARS_PATTERN.sub("", str(endpoint))[:64] or "/unknown"
    sanitized_decision = sanitize_audit_id(decision, max_length=32, fallback="UNKNOWN")
    sanitized_decision_source = (
        sanitize_audit_id(decision_source, max_length=64, fallback="unknown_source")
        if decision_source is not None
        else None
    )
    sanitized_error_code = (
        sanitize_audit_id(error_code, max_length=64, fallback="unknown_error")
        if error_code is not None
        else None
    )
    sanitized_matched_rule_ids = [
        sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
        for r in matched_rule_ids
    ][:100]

    event = {
        "event_id": sanitized_req_id,
        "request_id": sanitized_req_id,
        "client_request_id": sanitized_client_request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": sanitized_endpoint,
        "decision": sanitized_decision,
        "decision_source": sanitized_decision_source,
        "matched_rule_ids": sanitized_matched_rule_ids,
        "heuristic_risk_score": round(max(0.0, min(float(risk_score), 1.0)), 2),
        "forwarded": bool(forwarded),
        "query_length": max(0, len(query)),
        "query_sha256": _compute_prompt_hash(query),
        "retrieved_doc_ids": sanitized_doc_ids,
        "num_retrieved": max(0, min(int(num_retrieved), 1000)),
        "is_complete": bool(is_complete),
        "error_code": sanitized_error_code,
    }

    if output_status is not None:
        event["output_status"] = sanitize_audit_id(output_status, max_length=32, fallback="UNKNOWN")
        event["output_findings_count"] = max(0, int(output_findings_count or 0))
        event["output_matched_rule_ids"] = [
            sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
            for r in (output_matched_rule_ids or [])
        ][:100]
        event["output_error_code"] = (
            sanitize_audit_id(output_error_code, max_length=64, fallback="unknown_error")
            if output_error_code is not None
            else None
        )

    if ingress_telemetry:
        event["ingress_pipeline_version"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_pipeline_version"), max_length=16, fallback="v1"
        )
        event["ingress_surface"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_surface"), max_length=32, fallback="rag_query"
        )
        views = ingress_telemetry.get("ingress_views_evaluated")
        if isinstance(views, (list, tuple)):
            allowed = {"raw", "normalized", "decoded", "combined_normalized_decoded", "decoded_normalized"}
            event["ingress_views_evaluated"] = [str(v) for v in views if str(v) in allowed][:5]
        else:
            event["ingress_views_evaluated"] = ["raw"]
        raw_count = ingress_telemetry.get("ingress_views_count")
        event["ingress_views_count"] = max(1, min(int(raw_count if raw_count is not None else 1), 5))
        event["ingress_transformation_status"] = sanitize_audit_id(
            ingress_telemetry.get("ingress_transformation_status"), max_length=32, fallback="SUCCESS"
        )
        err_code = ingress_telemetry.get("ingress_transformation_error_code")
        if err_code is not None and isinstance(err_code, str) and err_code in ALLOWED_INGRESS_ERROR_CODES:
            event["ingress_transformation_error_code"] = err_code
        elif err_code is not None:
            event["ingress_transformation_error_code"] = "MALFORMED_ENCODING"
        else:
            event["ingress_transformation_error_code"] = None
        if ingress_telemetry.get("ml_available") is not None:
            event["ml_available"] = bool(ingress_telemetry["ml_available"])
        else:
            event["ml_available"] = None
        if ingress_telemetry.get("ml_risk_score") is not None:
            event["ml_risk_score"] = round(max(0.0, min(float(ingress_telemetry["ml_risk_score"]), 1.0)), 4)
        else:
            event["ml_risk_score"] = None

    event["event_type"] = "rag_inspection"

    with _audit_lock:
        if len(_audit_events) >= MAX_AUDIT_EVENTS:
            _audit_events.pop(0)
        _audit_events.append(copy.deepcopy(event))

    # Persist to durable SQLite storage
    store = get_audit_store()
    store.save_event(event)

    logger.info(json.dumps(event))
    return event


def log_output_security_event(
    request_id: str,
    endpoint: str,
    outcome: str,
    matched_rule_ids: Optional[List[str]] = None,
    finding_count: int = 0,
    client_request_id: Optional[str] = None,
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a dedicated structured security audit event for output DLP inspection.

    IMPORTANT PRIVACY INVARIANTS:
    Never logs raw downstream responses, matched secret values, redacted excerpts,
    or credentials. Logs only sanitized identifiers and aggregate metrics.
    """
    sanitized_req_id = sanitize_audit_id(request_id, max_length=128, fallback="unknown_req")
    sanitized_client_req_id = (
        sanitize_audit_id(client_request_id, max_length=128, fallback="unknown_client")
        if client_request_id is not None
        else None
    )
    sanitized_endpoint = CONTROL_CHARS_PATTERN.sub("", str(endpoint))[:64] or "/unknown"
    sanitized_outcome = sanitize_audit_id(outcome, max_length=32, fallback="UNKNOWN")
    sanitized_error_code = (
        sanitize_audit_id(error_code, max_length=64, fallback="unknown_error")
        if error_code is not None
        else None
    )
    sanitized_rules = [
        sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
        for r in (matched_rule_ids or [])
    ][:100]

    event = {
        "event_id": f"{sanitized_req_id}_output",
        "event_type": "output_security",
        "request_id": sanitized_req_id,
        "client_request_id": sanitized_client_req_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": sanitized_endpoint,
        "outcome": sanitized_outcome,
        "matched_rule_ids": sanitized_rules,
        "finding_count": max(0, int(finding_count)),
        "error_code": sanitized_error_code,
    }

    with _audit_lock:
        if len(_audit_events) >= MAX_AUDIT_EVENTS:
            _audit_events.pop(0)
        _audit_events.append(copy.deepcopy(event))

    # Persist to durable SQLite storage
    store = get_audit_store()
    store.save_event(event)

    logger.info(json.dumps(event))
    return event


def log_tool_security_event(
    request_id: str,
    endpoint: str,
    event_type: str,
    tool_name: str,
    decision: str,
    client_request_id: Optional[str] = None,
    error_code: Optional[str] = None,
    argument_keys: Optional[List[str]] = None,
    argument_count: int = 0,
    argument_sha256: Optional[str] = None,
    output_status: Optional[str] = None,
    output_findings_count: Optional[int] = None,
    output_matched_rule_ids: Optional[List[str]] = None,
    output_error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a dedicated structured security audit event for tool operations.

    IMPORTANT PRIVACY INVARIANTS:
    Never logs raw tool arguments, raw tool results, credentials, or secrets.
    Logs sanitized tool names, argument keys, argument counts, argument SHA-256 fingerprint,
    and output security audit metrics.
    """
    sanitized_req_id = sanitize_audit_id(request_id, max_length=128, fallback="unknown_req")
    sanitized_client_req_id = (
        sanitize_audit_id(client_request_id, max_length=128, fallback="unknown_client")
        if client_request_id is not None
        else None
    )
    sanitized_endpoint = CONTROL_CHARS_PATTERN.sub("", str(endpoint))[:64] or "/unknown"
    sanitized_tool_name = sanitize_audit_id(tool_name, max_length=64, fallback="unknown_tool")
    sanitized_decision = sanitize_audit_id(decision, max_length=32, fallback="UNKNOWN")
    sanitized_error_code = (
        sanitize_audit_id(error_code, max_length=64, fallback="unknown_error")
        if error_code is not None
        else None
    )
    sanitized_arg_keys = [
        sanitize_audit_id(k, max_length=64, fallback="arg")
        for k in (argument_keys or [])
    ][:50]
    sanitized_arg_sha256 = (
        sanitize_audit_id(argument_sha256, max_length=64, fallback="")
        if argument_sha256
        else None
    )

    event = {
        "event_id": f"{sanitized_req_id}_{event_type}",
        "event_type": sanitize_audit_id(event_type, max_length=32, fallback="tool_operation"),
        "request_id": sanitized_req_id,
        "client_request_id": sanitized_client_req_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": sanitized_endpoint,
        "decision": sanitized_decision,
        "decision_source": "tool_guard",
        "tool_name": sanitized_tool_name,
        "argument_keys": sanitized_arg_keys,
        "argument_count": max(0, int(argument_count)),
        "argument_sha256": sanitized_arg_sha256,
        "error_code": sanitized_error_code,
    }

    if output_status is not None:
        event["output_status"] = sanitize_audit_id(output_status, max_length=32, fallback="UNKNOWN")
        event["output_findings_count"] = max(0, int(output_findings_count or 0))
        event["output_matched_rule_ids"] = [
            sanitize_audit_id(r, max_length=64, fallback="unknown_rule")
            for r in (output_matched_rule_ids or [])
        ][:100]
        event["output_error_code"] = (
            sanitize_audit_id(output_error_code, max_length=64, fallback="unknown_error")
            if output_error_code is not None
            else None
        )

    with _audit_lock:
        if len(_audit_events) >= MAX_AUDIT_EVENTS:
            _audit_events.pop(0)
        _audit_events.append(copy.deepcopy(event))

    # Persist to durable storage (fails closed if storage raises AuditStorageError)
    store = get_audit_store()
    store.save_event(event)

    logger.info(json.dumps(event))
    return event


def get_audit_events() -> List[Dict[str, Any]]:
    """Return an isolated deep copy of all recorded audit events (used by tests and debugging)."""
    with _audit_lock:
        return copy.deepcopy(_audit_events)


def clear_audit_events() -> None:
    """Reset the audit event log (both in-memory buffer and persistent store)."""
    global _audit_events
    with _audit_lock:
        _audit_events.clear()
    store = get_audit_store()
    store.clear()
