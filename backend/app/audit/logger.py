"""Structured security event logging for PromptShield Phase 1.

Important Notice:
Audit events are maintained in-memory for Phase 1 inspection and testing.
They are not persistent across server restarts and are not cryptographically
tamper-proof. Full raw prompts and credentials are never stored.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("promptshield.audit")

# In-memory storage for test validation and local audit inspection
_audit_events: List[Dict[str, Any]] = []


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
) -> Dict[str, Any]:
    """Record a structured security audit event.

    Avoids storing raw prompt text to prevent sensitive data accumulation.
    Stores prompt length and SHA-256 fingerprint for forensic correlation.
    """
    event = {
        "event_id": request_id,
        "request_id": request_id,
        "client_request_id": client_request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "decision": decision,
        "decision_source": decision_source,
        "matched_rule_ids": matched_rule_ids,
        "heuristic_risk_score": risk_score,
        "forwarded": forwarded,
        "prompt_length": len(prompt),
        "prompt_sha256": _compute_prompt_hash(prompt),
    }

    _audit_events.append(event)
    logger.info(json.dumps(event))
    return event


def get_audit_events() -> List[Dict[str, Any]]:
    """Return all recorded audit events (used by tests and debugging)."""
    return list(_audit_events)


def clear_audit_events() -> None:
    """Reset the in-memory audit event log."""
    global _audit_events
    _audit_events.clear()
