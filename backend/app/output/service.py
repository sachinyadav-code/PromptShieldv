"""Output security enforcement service for PromptShield Phase 4.

Coordinates output scanning, critical-vs-redactable policy evaluation,
redaction of sensitive spans, and fail-closed error handling for downstream
responses before they are returned to clients.
"""

from dataclasses import dataclass
import logging
from typing import List, Literal, Optional

from app.output.redactor import RedactionError, redact_output
from app.output.rules import (
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_PAYMENT_CARD,
    DLP_RULE_PRIVATE_KEY_HEADER,
    OutputFindingLimitExceededError,
    OutputLengthExceededError,
    OutputScanError,
)
from app.output.scanner import scan_output
from app.output.schemas import OutputScanResult, RedactionResult
from app.schemas import MockDownstreamResponse

logger = logging.getLogger("promptshield.output.service")

# Rule IDs explicitly designated as critical (un-redactable egress hazards)
CRITICAL_DLP_RULE_IDS = {
    DLP_RULE_PRIVATE_KEY_HEADER,
    DLP_RULE_DATABASE_URI_CREDENTIAL,
    DLP_RULE_PAYMENT_CARD,
}

# Finding categories explicitly designated as critical egress hazards
# regardless of individual rule ID or severity classification variation
CRITICAL_DLP_CATEGORIES = {
    "PRIVATE_KEY",
    "DATABASE_CREDENTIAL",
    "PAYMENT_CARD",
}

# Standardized sanitized denial explanations (never leak matched tokens or values)
EXPLANATION_CRITICAL_BLOCKED = (
    "Output blocked by security policy: Critical sensitive data detected in downstream response."
)
EXPLANATION_SCAN_FAILED = (
    "Output blocked by security policy: Downstream response could not be verified by security inspection."
)


def is_critical_finding(finding: object) -> bool:
    """Evaluate whether an individual DLP finding constitutes a critical un-redactable hazard.

    Defense-in-depth criteria:
    1. Finding severity is explicitly 'CRITICAL'.
    2. Finding rule_id is in CRITICAL_DLP_RULE_IDS.
    3. Finding category is in CRITICAL_DLP_CATEGORIES.

    Evaluates both dataclass/object attributes and dict keys safely using getattr/get
    to protect against finding schema variations and avoid accidental egress bypasses.

    Args:
        finding: An OutputFinding instance or dict representing a DLP finding.

    Returns:
        True if the finding matches any critical criteria; False otherwise.
    """
    if isinstance(finding, dict):
        severity = finding.get("severity")
        rule_id = finding.get("rule_id")
        category = finding.get("category")
    else:
        severity = getattr(finding, "severity", None)
        rule_id = getattr(finding, "rule_id", None)
        category = getattr(finding, "category", None)

    return bool(
        severity == "CRITICAL"
        or (rule_id is not None and rule_id in CRITICAL_DLP_RULE_IDS)
        or (category is not None and category in CRITICAL_DLP_CATEGORIES)
    )


@dataclass(frozen=True)
class OutputEnforcementResult:
    """Outcome of enforcing output security against a downstream response.

    Attributes:
        effective_decision: High-level policy decision ('ALLOW' or 'BLOCK').
        downstream_response: Sanitized or original response, or None if blocked.
        audit_outcome: Technical audit classification ('clean', 'redacted', 'blocked', 'scan_error').
        matched_rule_ids: Distinct list of matched DLP rule IDs.
        finding_count: Total count of detected findings.
        explanation: Sanitized explanation when blocked, or None if allowed.
        error_code: Safe categorical error identifier if scan/redaction failed.
        risk_score: Security risk score (0.0 for clean/redacted, 1.0 for blocked/failed).
        is_complete: True if inspection completed without resource limit overflows or exceptions.
        scan_result: Technical scan result if scan completed, else None.
        redaction_result: Technical redaction result if redaction occurred, else None.
    """

    effective_decision: Literal["ALLOW", "BLOCK"]
    downstream_response: Optional[MockDownstreamResponse]
    audit_outcome: Literal["clean", "redacted", "blocked", "scan_error"]
    matched_rule_ids: List[str]
    finding_count: int
    explanation: Optional[str] = None
    error_code: Optional[str] = None
    risk_score: float = 0.0
    is_complete: bool = True
    scan_result: Optional[OutputScanResult] = None
    redaction_result: Optional[RedactionResult] = None


def enforce_output_security(
    downstream_response: MockDownstreamResponse,
) -> OutputEnforcementResult:
    """Inspect and sanitize downstream response content before client return.

    Trust Boundary Architecture:
    - `downstream_response.message`: Untrusted downstream egress text crossing
      the security boundary; actively scanned and sanitized by DLP output rules.
    - `downstream_response.provider`: Trusted static gateway metadata preserved
      from mock service infrastructure without inspection.

    Execution logic:
    1. Extract client-visible content (`downstream_response.message`).
    2. Scan content with `scan_output()` enforcing default safety boundaries.
    3. If scan is clean: return original downstream response unchanged.
    4. If critical findings exist: block response, return None for downstream content.
    5. If only redactable findings exist: redact message with `redact_output()`,
       reconstruct `MockDownstreamResponse`, preserve ALLOW decision.
    6. If scan or redaction fails: fail-closed with BLOCK decision, return None.

    Args:
        downstream_response: The simulated downstream response from mock service.

    Returns:
        OutputEnforcementResult containing the effective policy decision,
        sanitized downstream response (or None), audit metadata, and metrics.
    """
    if downstream_response is None or not hasattr(downstream_response, "message"):
        logger.warning("Downstream response missing or malformed; failing closed.")
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="MALFORMED_DOWNSTREAM_RESPONSE",
            risk_score=1.0,
            is_complete=False,
        )

    # Extract client-visible egress content across the trust boundary.
    # Note: `provider` is trusted static internal gateway metadata and is not scanned.
    text_to_scan = downstream_response.message
    if not isinstance(text_to_scan, str):
        logger.warning("Downstream response message is not a string; failing closed.")
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="INVALID_MESSAGE_TYPE",
            risk_score=1.0,
            is_complete=False,
        )

    # 1. Execute output DLP scan under fail-closed error boundaries
    try:
        scan_res = scan_output(text_to_scan)
    except OutputLengthExceededError as exc:
        logger.warning("Downstream output length limit exceeded: %s chars", exc.length)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="OUTPUT_LENGTH_EXCEEDED",
            risk_score=1.0,
            is_complete=False,
        )
    except OutputFindingLimitExceededError as exc:
        logger.warning("Downstream output finding count limit exceeded: %s findings", exc.count)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="FINDING_LIMIT_EXCEEDED",
            risk_score=1.0,
            is_complete=False,
        )
    except OutputScanError as exc:
        logger.warning("Downstream output scan error: %s", type(exc).__name__)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="OUTPUT_SCAN_ERROR",
            risk_score=1.0,
            is_complete=False,
        )
    except Exception as exc:
        logger.error("Unexpected downstream output scanning failure: %s", type(exc).__name__)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=[],
            finding_count=0,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="OUTPUT_INSPECTION_FAILED",
            risk_score=1.0,
            is_complete=False,
        )

    # 2. Case A: Clean, complete scan (zero findings)
    if scan_res.is_clean:
        return OutputEnforcementResult(
            effective_decision="ALLOW",
            downstream_response=downstream_response,
            audit_outcome="clean",
            matched_rule_ids=[],
            finding_count=0,
            explanation=None,
            error_code=None,
            risk_score=0.0,
            is_complete=True,
            scan_result=scan_res,
        )

    matched_rule_ids = scan_res.matched_rule_ids

    # 3. Case B: Critical findings detected -> BLOCK
    # Multi-layered check: severity 'CRITICAL', rule ID in CRITICAL_DLP_RULE_IDS,
    # or finding category in CRITICAL_DLP_CATEGORIES.
    has_critical_findings = any(
        is_critical_finding(f)
        for f in scan_res.findings
    )
    if has_critical_findings:
        logger.info("Downstream response contains critical findings: %s", matched_rule_ids)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="blocked",
            matched_rule_ids=matched_rule_ids,
            finding_count=scan_res.finding_count,
            explanation=EXPLANATION_CRITICAL_BLOCKED,
            error_code=None,
            risk_score=1.0,
            is_complete=True,
            scan_result=scan_res,
        )

    # 4. Case C: Redactable findings only -> REDACT and ALLOW
    try:
        redaction_res = redact_output(text_to_scan, scan_res.findings)
    except RedactionError as exc:
        logger.warning("Downstream output redaction failed: %s", type(exc).__name__)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=matched_rule_ids,
            finding_count=scan_res.finding_count,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="REDACTION_FAILED",
            risk_score=1.0,
            is_complete=False,
            scan_result=scan_res,
        )
    except Exception as exc:
        logger.error("Unexpected redaction failure: %s", type(exc).__name__)
        return OutputEnforcementResult(
            effective_decision="BLOCK",
            downstream_response=None,
            audit_outcome="scan_error",
            matched_rule_ids=matched_rule_ids,
            finding_count=scan_res.finding_count,
            explanation=EXPLANATION_SCAN_FAILED,
            error_code="REDACTION_UNEXPECTED_FAILURE",
            risk_score=1.0,
            is_complete=False,
            scan_result=scan_res,
        )

    # Reconstruct MockDownstreamResponse with sanitized text while preserving types.
    # Note: `provider` is preserved from trusted static gateway metadata.
    sanitized_response = MockDownstreamResponse(
        simulated=downstream_response.simulated,
        provider=downstream_response.provider,
        message=redaction_res.redacted_text,
        received_prompt_length=downstream_response.received_prompt_length,
    )

    return OutputEnforcementResult(
        effective_decision="ALLOW",
        downstream_response=sanitized_response,
        audit_outcome="redacted",
        matched_rule_ids=matched_rule_ids,
        finding_count=scan_res.finding_count,
        explanation=None,
        error_code=None,
        risk_score=0.0,
        is_complete=True,
        scan_result=scan_res,
        redaction_result=redaction_res,
    )
