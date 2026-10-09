from contextlib import asynccontextmanager
import logging
from typing import List
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.audit.logger import log_security_event
from app.detection.ml_classifier import get_classifier
from app.detection.rules import get_configured_rules, scan_prompt
from app.policies.engine import evaluate_policy, validate_startup_configuration
from app.proxy.mock_service import forward_to_mock_downstream
from app.schemas import (
    HealthResponse,
    InspectResponse,
    MLAssessment,
    PromptRequest,
    ProxyResponse,
    RuleInfo,
)

logger = logging.getLogger("promptshield.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan context manager.

    Validates effective ML enforcement policy configuration on startup before accepting requests.
    """
    logger.info("Initializing PromptShield Gateway...")
    mode = validate_startup_configuration()
    logger.info("PromptShield Gateway initialized with ML enforcement mode: %s", mode)
    yield
    logger.info("Shutting down PromptShield Gateway...")


app = FastAPI(
    title="PromptShield Security Gateway",
    description="Educational Enterprise LLM Security Gateway — Phase 1 Core Proxy & Rules",
    version="0.1.0",
    lifespan=lifespan,
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


def _get_ml_assessment(prompt: str) -> MLAssessment:
    """Safely evaluate prompt with the ML classifier and return structured MLAssessment.

    Resilient error handling:
    - Never raises an unhandled exception or crashes the gateway request.
    - Never leaks raw prompt text, filesystem paths, or stack traces.
    - Preserves the configured threshold and model version.
    - Gracefully degrades to an unavailable assessment if ML is disabled or failing.
    """
    try:
        classifier = get_classifier()
        inference_result = classifier.predict(prompt)
        return MLAssessment.from_inference_result(inference_result)
    except Exception as exc:
        logger.warning("ML assessment failed unexpectedly: %s", type(exc).__name__)
        fallback_threshold = 0.50
        try:
            fallback_threshold = get_classifier().threshold
        except Exception:
            pass
        return MLAssessment.unavailable(
            threshold=fallback_threshold,
            error_code="INFERENCE_FAILED",
        )


@app.post("/api/v1/inspect", response_model=InspectResponse)
def inspect_prompt(payload: PromptRequest) -> InspectResponse:
    """Inspect user prompt against security rules and return an authoritative decision.

    Returns HTTP 200 for valid prompts, whether the policy decision is ALLOW or BLOCK.
    Validation errors (empty prompt, whitespace-only, overlength) return HTTP 422.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)
    ml_assessment = _get_ml_assessment(payload.prompt)
    decision = evaluate_policy(matched_rules, ml_assessment=ml_assessment)

    log_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/inspect",
        decision=decision.decision,
        matched_rule_ids=decision.matched_rule_ids,
        risk_score=decision.risk_score,
        forwarded=False,
        prompt=payload.prompt,
        client_request_id=payload.request_id,
        decision_source=decision.decision_source,
    )

    return InspectResponse(
        decision=decision.decision,
        risk_score=decision.risk_score,
        matched_rules=decision.matched_rule_ids,
        explanation=decision.explanation,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        ml_assessment=ml_assessment,
        decision_source=decision.decision_source,
    )


@app.post("/api/v1/proxy", response_model=ProxyResponse)
def proxy_prompt(payload: PromptRequest) -> ProxyResponse:
    """Inspect prompt and forward to mock downstream service ONLY if allowed.

    If decision is BLOCK, the mock service is never called.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)
    ml_assessment = _get_ml_assessment(payload.prompt)
    decision = evaluate_policy(matched_rules, ml_assessment=ml_assessment)

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
        decision_source=decision.decision_source,
    )

    return ProxyResponse(
        decision=decision.decision,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        matched_rules=decision.matched_rule_ids,
        explanation=decision.explanation,
        downstream_response=downstream_result,
        ml_assessment=ml_assessment,
        decision_source=decision.decision_source,
    )
