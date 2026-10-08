"""FastAPI application entrypoint for PromptShield Security Gateway (Phase 1)."""

import logging
from typing import List
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.audit.logger import log_security_event
from app.detection.rules import get_configured_rules, scan_prompt
from app.policies.engine import evaluate_policy
from app.proxy.mock_service import forward_to_mock_downstream
from app.schemas import (
    HealthResponse,
    InspectResponse,
    PromptRequest,
    ProxyResponse,
    RuleInfo,
)

logger = logging.getLogger("promptshield.gateway")

app = FastAPI(
    title="PromptShield Security Gateway",
    description="Educational Enterprise LLM Security Gateway — Phase 1 Core Proxy & Rules",
    version="0.1.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled internal exceptions without exposing sensitive server internals."""
    logger.error("Unhandled gateway exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal gateway error occurred."},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health check indicator confirming the gateway is running."""
    return HealthResponse(
        status="ok",
        service="promptshield-gateway",
        phase=1,
    )


@app.get("/api/v1/rules", response_model=List[RuleInfo])
def list_rules() -> List[RuleInfo]:
    """Return list of active deterministic detection rules and configured actions."""
    rules = get_configured_rules()
    return [
        RuleInfo(
            rule_id=r.rule_id,
            name=r.name,
            description=r.description,
            severity=r.severity,
            action=r.action,
        )
        for r in rules
    ]


@app.post("/api/v1/inspect", response_model=InspectResponse)
def inspect_prompt(payload: PromptRequest) -> InspectResponse:
    """Inspect user prompt against security rules and return an authoritative decision.

    Returns HTTP 200 for valid prompts, whether the policy decision is ALLOW or BLOCK.
    Validation errors (empty prompt, whitespace-only, overlength) return HTTP 422.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)
    decision = evaluate_policy(matched_rules)

    log_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/inspect",
        decision=decision.decision,
        matched_rule_ids=decision.matched_rule_ids,
        risk_score=decision.risk_score,
        forwarded=False,
        prompt=payload.prompt,
        client_request_id=payload.request_id,
    )

    return InspectResponse(
        decision=decision.decision,
        risk_score=decision.risk_score,
        matched_rules=decision.matched_rule_ids,
        explanation=decision.explanation,
        request_id=server_request_id,
        client_request_id=payload.request_id,
    )


@app.post("/api/v1/proxy", response_model=ProxyResponse)
def proxy_prompt(payload: PromptRequest) -> ProxyResponse:
    """Inspect prompt and forward to mock downstream service ONLY if allowed.

    If decision is BLOCK, the mock service is never called.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)
    decision = evaluate_policy(matched_rules)

    downstream_result = None
    forwarded = False

    if decision.decision == "ALLOW":
        downstream_result = forward_to_mock_downstream(payload.prompt)
        forwarded = True

    log_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/proxy",
        decision=decision.decision,
        matched_rule_ids=decision.matched_rule_ids,
        risk_score=decision.risk_score,
        forwarded=forwarded,
        prompt=payload.prompt,
        client_request_id=payload.request_id,
    )

    return ProxyResponse(
        decision=decision.decision,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        matched_rules=decision.matched_rule_ids,
        explanation=decision.explanation,
        downstream_response=downstream_result,
    )
