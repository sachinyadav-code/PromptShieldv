from contextlib import asynccontextmanager
import logging
import threading
from typing import List, Optional, Sequence, Tuple
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.audit import (
    AuditStorageError,
    get_audit_store,
    log_rag_security_event,
    log_security_event,
)
from app.auth import (
    HEADER_CONTENT_TYPE_OPTIONS,
    HEADER_FRAME_OPTIONS,
    HEADER_REFERRER_POLICY,
    authenticate_request,
    get_allowed_origins,
)
from app.limits import (
    enforce_concurrency_limit,
    enforce_rate_limit,
    validate_limits_configuration,
)
from app.detection.ingress import (
    IngressTransformationError,
    RULE_MALFORMED_ENCODING,
    evaluate_ingress_prompt_ml,
    extract_ingress_telemetry,
    generate_ingress_views,
    inspect_ingress_prompt,
)
from app.detection.ml_classifier import get_classifier
from app.detection.rules import get_configured_rules, scan_prompt
from app.output import enforce_output_security
from app.policies.engine import (
    evaluate_policy,
    evaluate_rag_policy,
    validate_startup_configuration,
)
from app.proxy.mock_service import forward_to_mock_downstream
from app.rag import (
    CorpusLoadError,
    Document,
    inspect_rag_request,
    load_corpus,
    retrieve_documents,
)
from app.schemas import (
    AuditEventResponse,
    AuditQueryResponse,
    DocumentInspectionSummary,
    HealthResponse,
    InspectResponse,
    MLAssessment,
    PromptRequest,
    ProxyResponse,
    RAGFindingSummary,
    RAGInspectResponse,
    RAGProxyResponse,
    RAGRequest,
    RetrievedDocumentSummary,
    RuleInfo,
)

logger = logging.getLogger("promptshield.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan context manager.

    Validates effective ML enforcement policy, limits configuration, and initializes
    durable audit storage on startup before accepting requests.
    """
    logger.info("Initializing PromptShield Gateway...")
    mode = validate_startup_configuration()
    validate_limits_configuration()
    try:
        audit_store = get_audit_store()
        audit_store.initialize_schema()
    except Exception as e:
        logger.error("Failed to initialize audit database on startup: %s", type(e).__name__)
        raise RuntimeError("Audit storage initialization failed on startup.") from e

    origins = get_allowed_origins()
    logger.info("PromptShield Gateway initialized with ML enforcement mode: %s; CORS allowed origins: %s", mode, origins)
    yield
    logger.info("Shutting down PromptShield Gateway...")


app = FastAPI(
    title="PromptShield Security Gateway",
    description="Educational Enterprise LLM Security Gateway — Phase 1 Core Proxy & Rules",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. Configurable CORS policy middleware
class DynamicCORSMiddleware:
    """Dynamically applies CORSMiddleware based on PROMPTSHIELD_ALLOWED_ORIGINS.

    Allows runtime and per-test environment variable configuration of allowed origins.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        origins = get_allowed_origins()
        if origins and scope["type"] == "http":
            cors = CORSMiddleware(
                self.app,
                allow_origins=origins,
                allow_credentials=False,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Authorization", "X-API-Key", "Content-Type"],
            )
            await cors(scope, receive, send)
        else:
            await self.app(scope, receive, send)


app.add_middleware(DynamicCORSMiddleware)


# 2. HTTP Security response headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    """Consistently apply enterprise HTTP security response headers."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = HEADER_CONTENT_TYPE_OPTIONS
    response.headers["X-Frame-Options"] = HEADER_FRAME_OPTIONS
    response.headers["Referrer-Policy"] = HEADER_REFERRER_POLICY
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled internal exceptions without exposing sensitive server internals."""
    logger.error("Unhandled gateway exception on %s: %s", request.url.path, type(exc).__name__, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal gateway error occurred."},
        headers={
            "X-Content-Type-Options": HEADER_CONTENT_TYPE_OPTIONS,
            "X-Frame-Options": HEADER_FRAME_OPTIONS,
            "Referrer-Policy": HEADER_REFERRER_POLICY,
        },
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health check indicator confirming the gateway is running."""
    return HealthResponse(
        status="ok",
        service="promptshield-gateway",
        phase=1,
    )


@app.get(
    "/api/v1/rules",
    response_model=List[RuleInfo],
    dependencies=[Depends(authenticate_request), Depends(enforce_rate_limit)],
)
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


@app.get(
    "/api/v1/audit/events",
    response_model=AuditQueryResponse,
    dependencies=[Depends(authenticate_request), Depends(enforce_rate_limit)],
)
def query_audit_events(
    limit: int = 50,
    offset: int = 0,
    endpoint: Optional[str] = None,
    decision: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> AuditQueryResponse:
    """Query persisted security audit events with bounded pagination and filtering.

    Requires API-key authentication and respects gateway rate-limiting.
    """
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter 'limit' must be between 1 and 100.",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter 'offset' must be greater than or equal to 0.",
        )

    # Validate decision if provided
    if decision is not None and decision.strip().upper() not in ("ALLOW", "BLOCK", "REVIEW"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid decision filter '{decision}'. Allowed values: ALLOW, BLOCK, REVIEW.",
        )

    try:
        store = get_audit_store()
        events_dicts, total_count = store.query_events(
            limit=limit,
            offset=offset,
            endpoint=endpoint,
            decision=decision,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
        )
        events_models = [AuditEventResponse(**e) for e in events_dicts]
        return AuditQueryResponse(
            events=events_models,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
    except AuditStorageError as ase:
        logger.error("Audit store query failed: %s", type(ase).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit events from persistent storage.",
        ) from ase



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


@app.post(
    "/api/v1/inspect",
    response_model=InspectResponse,
    dependencies=[
        Depends(authenticate_request),
        Depends(enforce_rate_limit),
        Depends(enforce_concurrency_limit),
    ],
)
def inspect_prompt(payload: PromptRequest) -> InspectResponse:
    """Inspect user prompt against security rules and return an authoritative decision.

    Returns HTTP 200 for valid prompts, whether the policy decision is ALLOW or BLOCK.
    Validation errors (empty prompt, whitespace-only, overlength) return HTTP 422.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)

    try:
        ingress_result = inspect_ingress_prompt(payload.prompt)
        matched_dict = {r.rule_id: r for r in matched_rules}
        for r in ingress_result.matched_rules:
            if r.rule_id not in matched_dict:
                matched_dict[r.rule_id] = r
        all_matched = list(matched_dict.values())
        ml_assessment = ingress_result.ml_assessment
        decision = evaluate_policy(all_matched, ml_assessment=ml_assessment)
        ingress_telemetry = extract_ingress_telemetry(
            views=ingress_result.views,
            surface="prompt",
            ml_assessment=ml_assessment,
        )
    except IngressTransformationError as err:
        logger.warning("Ingress transformation error on /api/v1/inspect: %s (%s)", err.error_code, err.detail)
        matched_dict = {r.rule_id: r for r in matched_rules}
        matched_dict[RULE_MALFORMED_ENCODING.rule_id] = RULE_MALFORMED_ENCODING
        all_matched = list(matched_dict.values())
        ml_assessment = _get_ml_assessment(payload.prompt)
        decision = evaluate_policy(all_matched, ml_assessment=ml_assessment)
        ingress_telemetry = extract_ingress_telemetry(
            error_code=err.error_code,
            surface="prompt",
            ml_assessment=ml_assessment,
        )

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
        ingress_telemetry=ingress_telemetry,
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


@app.post(
    "/api/v1/proxy",
    response_model=ProxyResponse,
    dependencies=[
        Depends(authenticate_request),
        Depends(enforce_rate_limit),
        Depends(enforce_concurrency_limit),
    ],
)
def proxy_prompt(payload: PromptRequest) -> ProxyResponse:

    """Inspect prompt and forward to mock downstream service ONLY if allowed.

    If decision is BLOCK, the mock service is never called.
    Inspects downstream output for sensitive data leakage before returning to client.
    """
    server_request_id = str(uuid.uuid4())
    matched_rules = scan_prompt(payload.prompt)

    try:
        ingress_result = inspect_ingress_prompt(payload.prompt)
        matched_dict = {r.rule_id: r for r in matched_rules}
        for r in ingress_result.matched_rules:
            if r.rule_id not in matched_dict:
                matched_dict[r.rule_id] = r
        all_matched = list(matched_dict.values())
        ml_assessment = ingress_result.ml_assessment
        decision = evaluate_policy(all_matched, ml_assessment=ml_assessment)
        ingress_telemetry = extract_ingress_telemetry(
            views=ingress_result.views,
            surface="prompt",
            ml_assessment=ml_assessment,
        )
    except IngressTransformationError as err:
        logger.warning("Ingress transformation error on /api/v1/proxy: %s (%s)", err.error_code, err.detail)
        matched_dict = {r.rule_id: r for r in matched_rules}
        matched_dict[RULE_MALFORMED_ENCODING.rule_id] = RULE_MALFORMED_ENCODING
        all_matched = list(matched_dict.values())
        ml_assessment = _get_ml_assessment(payload.prompt)
        decision = evaluate_policy(all_matched, ml_assessment=ml_assessment)
        ingress_telemetry = extract_ingress_telemetry(
            error_code=err.error_code,
            surface="prompt",
            ml_assessment=ml_assessment,
        )

    downstream_result = None
    forwarded = False

    if decision.decision == "ALLOW":
        try:
            downstream_result = forward_to_mock_downstream(payload.prompt)
            forwarded = True
        except Exception as exc:
            logger.error("Downstream service call failed on /api/v1/proxy: %s", type(exc).__name__)
            log_security_event(
                request_id=server_request_id,
                endpoint="/api/v1/proxy",
                decision="BLOCK",
                matched_rule_ids=decision.matched_rule_ids,
                risk_score=1.0,
                forwarded=False,
                prompt=payload.prompt,
                client_request_id=payload.request_id,
                decision_source="deterministic_rule",
                output_status="scan_error",
                output_findings_count=0,
                output_matched_rule_ids=[],
                output_error_code="DOWNSTREAM_CALL_FAILED",
                ingress_telemetry=ingress_telemetry,
            )
            return ProxyResponse(
                decision="BLOCK",
                request_id=server_request_id,
                client_request_id=payload.request_id,
                matched_rules=decision.matched_rule_ids,
                explanation="Output blocked by security policy: Downstream response could not be verified by security inspection.",
                downstream_response=None,
                ml_assessment=ml_assessment,
                decision_source="deterministic_rule",
            )

    # 1. If request was blocked at input, audit and return immediately
    if not forwarded:
        log_security_event(
            request_id=server_request_id,
            endpoint="/api/v1/proxy",
            decision=decision.decision,
            matched_rule_ids=decision.matched_rule_ids,
            risk_score=decision.risk_score,
            forwarded=False,
            prompt=payload.prompt,
            client_request_id=payload.request_id,
            decision_source=decision.decision_source,
            ingress_telemetry=ingress_telemetry,
        )
        return ProxyResponse(
            decision=decision.decision,
            request_id=server_request_id,
            client_request_id=payload.request_id,
            matched_rules=decision.matched_rule_ids,
            explanation=decision.explanation,
            downstream_response=None,
            ml_assessment=ml_assessment,
            decision_source=decision.decision_source,
        )

    # 2. Inspect downstream output for DLP violations
    output_enforcement = enforce_output_security(downstream_result)

    if output_enforcement.effective_decision == "BLOCK":
        effective_decision = "BLOCK"
        effective_downstream = None
        effective_risk = output_enforcement.risk_score
        effective_explanation = output_enforcement.explanation or "Output blocked by security policy."
        effective_decision_source = "deterministic_rule"
    else:
        effective_decision = "ALLOW"
        effective_downstream = output_enforcement.downstream_response
        effective_risk = decision.risk_score
        effective_explanation = decision.explanation
        effective_decision_source = decision.decision_source

    all_matched_rules = list(dict.fromkeys(decision.matched_rule_ids + output_enforcement.matched_rule_ids))

    log_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/proxy",
        decision=effective_decision,
        matched_rule_ids=all_matched_rules,
        risk_score=effective_risk,
        forwarded=True,
        prompt=payload.prompt,
        client_request_id=payload.request_id,
        decision_source=effective_decision_source,
        output_status=output_enforcement.audit_outcome,
        output_findings_count=output_enforcement.finding_count,
        output_matched_rule_ids=output_enforcement.matched_rule_ids,
        output_error_code=output_enforcement.error_code,
        ingress_telemetry=ingress_telemetry,
    )

    return ProxyResponse(
        decision=effective_decision,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        matched_rules=all_matched_rules,
        explanation=effective_explanation,
        downstream_response=effective_downstream,
        ml_assessment=ml_assessment,
        decision_source=effective_decision_source,
    )


# ---------------------------------------------------------------------------
# RAG Corpus Management and Downstream Assembly Helpers
# ---------------------------------------------------------------------------

_cached_corpus: Optional[Tuple[Document, ...]] = None
_corpus_lock = threading.Lock()


def get_rag_corpus() -> Tuple[Document, ...]:
    """Return the cached RAG corpus or load from the default repository path."""
    global _cached_corpus
    with _corpus_lock:
        if _cached_corpus is None:
            _cached_corpus = tuple(load_corpus())
        return _cached_corpus


def set_rag_corpus(corpus: Sequence[Document]) -> None:
    """Set the RAG corpus in-memory (used by test suites)."""
    global _cached_corpus
    with _corpus_lock:
        _cached_corpus = tuple(corpus)


def reset_rag_corpus() -> None:
    """Reset the cached RAG corpus so it reloads on next call."""
    global _cached_corpus
    with _corpus_lock:
        _cached_corpus = None


def assemble_rag_downstream_prompt(query: str, documents: Sequence[Document]) -> str:
    """Safely format prompt context for mock downstream forwarding.

    Retrieved document content is untrusted context and labeled as such.
    """
    if not documents:
        return f"User Query:\n{query}"

    doc_sections = [
        f"Document [{doc.doc_id}] ({doc.title}):\n{doc.content}"
        for doc in documents
    ]
    context_str = "\n\n---\n\n".join(doc_sections)
    return f"Context:\n{context_str}\n\nUser Query:\n{query}"


# ---------------------------------------------------------------------------
# RAG Gateway Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/rag/inspect",
    response_model=RAGInspectResponse,
    dependencies=[
        Depends(authenticate_request),
        Depends(enforce_rate_limit),
        Depends(enforce_concurrency_limit),
    ],
)
def inspect_rag(payload: RAGRequest) -> RAGInspectResponse:
    """Inspect user query and retrieved context without forwarding to mock downstream."""
    server_request_id = str(uuid.uuid4())

    # 1. Load corpus (fail closed if loading fails)
    try:
        views = generate_ingress_views(payload.query)
        ingress_telemetry = extract_ingress_telemetry(views=views, surface="rag_query")
    except IngressTransformationError as ite:
        ingress_telemetry = extract_ingress_telemetry(error_code=ite.error_code, surface="rag_query")

    try:
        corpus = get_rag_corpus()
    except (CorpusLoadError, FileNotFoundError, OSError) as exc:
        logger.warning("Corpus loading failed: %s", type(exc).__name__)
        log_rag_security_event(
            request_id=server_request_id,
            endpoint="/api/v1/rag/inspect",
            decision="REVIEW",
            matched_rule_ids=[],
            risk_score=1.0,
            forwarded=False,
            query=payload.query,
            retrieved_doc_ids=[],
            num_retrieved=0,
            is_complete=False,
            client_request_id=payload.request_id,
            decision_source="rag_inspection",
            error_code="CORPUS_LOAD_ERROR",
            ingress_telemetry=ingress_telemetry,
        )
        return RAGInspectResponse(
            decision="REVIEW",
            decision_source="rag_inspection",
            risk_score=1.0,
            request_id=server_request_id,
            client_request_id=payload.request_id,
            is_complete=False,
            direct_query_assessment="REVIEW",
            combined_context_assessment="REVIEW",
            matched_rules=[],
            explanation="RAG security inspection could not complete: Document corpus unavailable.",
            retrieved_documents=[],
            document_assessments=[],
            findings=[],
            ml_assessment=None,
        )

    # 2. Retrieve candidate documents and evaluate security policies
    try:
        retrieval_results = retrieve_documents(
            query=payload.query,
            documents=corpus,
            top_k=payload.top_k,
        )
        retrieved_docs = [r.document for r in retrieval_results]

        # 3. Direct query ML assessment across ingress views
        ml_assessment = evaluate_ingress_prompt_ml(payload.query, classifier=get_classifier())

        # Update telemetry with ML metrics
        if "ingress_transformation_status" in ingress_telemetry and ingress_telemetry["ingress_transformation_status"] == "SUCCESS":
            ingress_telemetry = extract_ingress_telemetry(
                views=views,
                surface="rag_query",
                ml_assessment=ml_assessment,
            )
        else:
            ingress_telemetry["ml_available"] = bool(ml_assessment.available)
            if ml_assessment.available and ml_assessment.score is not None:
                ingress_telemetry["ml_risk_score"] = round(max(0.0, min(float(ml_assessment.score), 1.0)), 4)

        # 4. Multi-level RAG security inspection
        report = inspect_rag_request(
            query=payload.query,
            documents=retrieved_docs,
        )

        # 5. Hybrid policy evaluation
        policy_decision = evaluate_rag_policy(
            report=report,
            ml_assessment=ml_assessment,
        )
    except Exception as exc:
        logger.error("RAG inspection pipeline failure on /api/v1/rag/inspect: %s", type(exc).__name__)
        log_rag_security_event(
            request_id=server_request_id,
            endpoint="/api/v1/rag/inspect",
            decision="REVIEW",
            matched_rule_ids=[],
            risk_score=1.0,
            forwarded=False,
            query=payload.query,
            retrieved_doc_ids=[],
            num_retrieved=0,
            is_complete=False,
            client_request_id=payload.request_id,
            decision_source="rag_inspection",
            error_code="INSPECTION_FAILED",
            ingress_telemetry=ingress_telemetry,
        )
        return RAGInspectResponse(
            decision="REVIEW",
            decision_source="rag_inspection",
            risk_score=1.0,
            request_id=server_request_id,
            client_request_id=payload.request_id,
            is_complete=False,
            direct_query_assessment="REVIEW",
            combined_context_assessment="REVIEW",
            matched_rules=[],
            explanation="RAG security inspection could not complete due to an internal processing error.",
            retrieved_documents=[],
            document_assessments=[],
            findings=[],
            ml_assessment=None,
        )

    # 6. Structured audit logging (zero raw query or content logged)
    retrieved_doc_ids = [d.doc_id for d in retrieved_docs]
    log_rag_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/rag/inspect",
        decision=policy_decision.decision,
        matched_rule_ids=policy_decision.matched_rule_ids,
        risk_score=policy_decision.risk_score,
        forwarded=False,
        query=payload.query,
        retrieved_doc_ids=retrieved_doc_ids,
        num_retrieved=len(retrieved_docs),
        is_complete=report.is_complete,
        client_request_id=payload.request_id,
        decision_source=policy_decision.decision_source,
        ingress_telemetry=ingress_telemetry,
    )

    # 7. Build structured response
    retrieved_summaries = [
        RetrievedDocumentSummary(
            doc_id=r.doc_id,
            title=r.title,
            source=r.source,
            relevance_score=r.score,
            matched_tokens=list(r.matched_tokens),
        )
        for r in retrieval_results
    ]
    doc_assessments = [
        DocumentInspectionSummary(
            doc_id=dr.doc_id,
            assessment=dr.assessment,
            risk_score=dr.risk_score,
            matched_rules=list(dr.matched_rule_ids),
            is_complete=dr.is_complete,
            explanation=dr.explanation,
        )
        for dr in report.document_results
    ]
    findings_summaries = [
        RAGFindingSummary(
            scope=f.scope,
            assessment=f.assessment,
            rule_id=f.rule_id,
            doc_id=f.doc_id,
            chunk_index=f.chunk_index,
            char_start=f.char_start,
            char_end=f.char_end,
            risk_score=f.risk_score,
            explanation=f.explanation,
        )
        for f in report.all_findings
    ]

    return RAGInspectResponse(
        decision=policy_decision.decision,
        decision_source=policy_decision.decision_source,
        risk_score=policy_decision.risk_score,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        is_complete=report.is_complete,
        direct_query_assessment=report.query_result.assessment,
        combined_context_assessment=report.combined_context_result.assessment,
        matched_rules=policy_decision.matched_rule_ids,
        explanation=policy_decision.explanation,
        retrieved_documents=retrieved_summaries,
        document_assessments=doc_assessments,
        findings=findings_summaries,
        ml_assessment=ml_assessment,
    )


@app.post(
    "/api/v1/rag/proxy",
    response_model=RAGProxyResponse,
    dependencies=[
        Depends(authenticate_request),
        Depends(enforce_rate_limit),
        Depends(enforce_concurrency_limit),
    ],
)
def proxy_rag(payload: RAGRequest) -> RAGProxyResponse:
    """Inspect query and retrieved context; forward to mock downstream ONLY if ALLOW."""
    server_request_id = str(uuid.uuid4())

    # 1. Load corpus (fail closed if loading fails)
    try:
        views = generate_ingress_views(payload.query)
        ingress_telemetry = extract_ingress_telemetry(views=views, surface="rag_query")
    except IngressTransformationError as ite:
        ingress_telemetry = extract_ingress_telemetry(error_code=ite.error_code, surface="rag_query")

    try:
        corpus = get_rag_corpus()
    except (CorpusLoadError, FileNotFoundError, OSError) as exc:
        logger.warning("Corpus loading failed: %s", type(exc).__name__)
        log_rag_security_event(
            request_id=server_request_id,
            endpoint="/api/v1/rag/proxy",
            decision="REVIEW",
            matched_rule_ids=[],
            risk_score=1.0,
            forwarded=False,
            query=payload.query,
            retrieved_doc_ids=[],
            num_retrieved=0,
            is_complete=False,
            client_request_id=payload.request_id,
            decision_source="rag_inspection",
            error_code="CORPUS_LOAD_ERROR",
            ingress_telemetry=ingress_telemetry,
        )
        return RAGProxyResponse(
            decision="REVIEW",
            decision_source="rag_inspection",
            risk_score=1.0,
            request_id=server_request_id,
            client_request_id=payload.request_id,
            is_complete=False,
            forwarded=False,
            direct_query_assessment="REVIEW",
            combined_context_assessment="REVIEW",
            matched_rules=[],
            explanation="RAG security inspection could not complete: Document corpus unavailable.",
            retrieved_documents=[],
            document_assessments=[],
            downstream_response=None,
            findings=[],
            ml_assessment=None,
        )

    # 2. Retrieve candidate documents and evaluate security policies
    try:
        retrieval_results = retrieve_documents(
            query=payload.query,
            documents=corpus,
            top_k=payload.top_k,
        )
        retrieved_docs = [r.document for r in retrieval_results]

        # 3. Direct query ML assessment across ingress views
        ml_assessment = evaluate_ingress_prompt_ml(payload.query, classifier=get_classifier())

        # Update telemetry with ML metrics
        if "ingress_transformation_status" in ingress_telemetry and ingress_telemetry["ingress_transformation_status"] == "SUCCESS":
            ingress_telemetry = extract_ingress_telemetry(
                views=views,
                surface="rag_query",
                ml_assessment=ml_assessment,
            )
        else:
            ingress_telemetry["ml_available"] = bool(ml_assessment.available)
            if ml_assessment.available and ml_assessment.score is not None:
                ingress_telemetry["ml_risk_score"] = round(max(0.0, min(float(ml_assessment.score), 1.0)), 4)

        # 4. Multi-level RAG security inspection
        report = inspect_rag_request(
            query=payload.query,
            documents=retrieved_docs,
        )

        # 5. Hybrid policy evaluation
        policy_decision = evaluate_rag_policy(
            report=report,
            ml_assessment=ml_assessment,
        )
    except Exception as exc:
        logger.error("RAG proxy pipeline failure on /api/v1/rag/proxy: %s", type(exc).__name__)
        log_rag_security_event(
            request_id=server_request_id,
            endpoint="/api/v1/rag/proxy",
            decision="REVIEW",
            matched_rule_ids=[],
            risk_score=1.0,
            forwarded=False,
            query=payload.query,
            retrieved_doc_ids=[],
            num_retrieved=0,
            is_complete=False,
            client_request_id=payload.request_id,
            decision_source="rag_inspection",
            error_code="INSPECTION_FAILED",
            ingress_telemetry=ingress_telemetry,
        )
        return RAGProxyResponse(
            decision="REVIEW",
            decision_source="rag_inspection",
            risk_score=1.0,
            request_id=server_request_id,
            client_request_id=payload.request_id,
            is_complete=False,
            forwarded=False,
            direct_query_assessment="REVIEW",
            combined_context_assessment="REVIEW",
            matched_rules=[],
            explanation="RAG security inspection could not complete due to an internal processing error.",
            retrieved_documents=[],
            document_assessments=[],
            downstream_response=None,
            findings=[],
            ml_assessment=None,
        )

    # 6. Build structured retrieval and inspection summaries
    retrieved_summaries = [
        RetrievedDocumentSummary(
            doc_id=r.doc_id,
            title=r.title,
            source=r.source,
            relevance_score=r.score,
            matched_tokens=list(r.matched_tokens),
        )
        for r in retrieval_results
    ]
    doc_assessments = [
        DocumentInspectionSummary(
            doc_id=dr.doc_id,
            assessment=dr.assessment,
            risk_score=dr.risk_score,
            matched_rules=list(dr.matched_rule_ids),
            is_complete=dr.is_complete,
            explanation=dr.explanation,
        )
        for dr in report.document_results
    ]
    findings_summaries = [
        RAGFindingSummary(
            scope=f.scope,
            assessment=f.assessment,
            rule_id=f.rule_id,
            doc_id=f.doc_id,
            chunk_index=f.chunk_index,
            char_start=f.char_start,
            char_end=f.char_end,
            risk_score=f.risk_score,
            explanation=f.explanation,
        )
        for f in report.all_findings
    ]

    # 7. Forward to mock downstream ONLY if final decision is ALLOW
    downstream_result = None
    forwarded = False
    output_enforcement = None

    if policy_decision.decision == "ALLOW":
        rag_prompt = assemble_rag_downstream_prompt(payload.query, retrieved_docs)
        try:
            downstream_result = forward_to_mock_downstream(rag_prompt)
            forwarded = True
        except Exception as exc:
            logger.error("Downstream service call failed on /api/v1/rag/proxy: %s", type(exc).__name__)
            retrieved_doc_ids = [d.doc_id for d in retrieved_docs]
            log_rag_security_event(
                request_id=server_request_id,
                endpoint="/api/v1/rag/proxy",
                decision="BLOCK",
                matched_rule_ids=policy_decision.matched_rule_ids,
                risk_score=1.0,
                forwarded=False,
                query=payload.query,
                retrieved_doc_ids=retrieved_doc_ids,
                num_retrieved=len(retrieved_docs),
                is_complete=False,
                client_request_id=payload.request_id,
                decision_source="deterministic_rule",
                error_code="DOWNSTREAM_CALL_FAILED",
                output_status="scan_error",
                output_findings_count=0,
                output_matched_rule_ids=[],
                output_error_code="DOWNSTREAM_CALL_FAILED",
                ingress_telemetry=ingress_telemetry,
            )
            return RAGProxyResponse(
                decision="BLOCK",
                decision_source="deterministic_rule",
                risk_score=1.0,
                request_id=server_request_id,
                client_request_id=payload.request_id,
                is_complete=False,
                forwarded=False,
                direct_query_assessment=report.query_result.assessment,
                combined_context_assessment=report.combined_context_result.assessment,
                matched_rules=policy_decision.matched_rule_ids,
                explanation="Output blocked by security policy: Downstream response could not be verified by security inspection.",
                retrieved_documents=retrieved_summaries,
                document_assessments=doc_assessments,
                downstream_response=None,
                findings=findings_summaries,
                ml_assessment=ml_assessment,
            )

    # 8. Inspect downstream output for DLP violations
    if forwarded:
        output_enforcement = enforce_output_security(downstream_result)

    if output_enforcement is not None:
        if output_enforcement.effective_decision == "BLOCK":
            effective_decision = "BLOCK"
            effective_downstream = None
            effective_risk = output_enforcement.risk_score
            effective_explanation = output_enforcement.explanation or "Output blocked by security policy."
            effective_decision_source = "deterministic_rule"
            is_complete = report.is_complete and output_enforcement.is_complete
        else:
            effective_decision = "ALLOW"
            effective_downstream = output_enforcement.downstream_response
            effective_risk = policy_decision.risk_score
            effective_explanation = policy_decision.explanation
            effective_decision_source = policy_decision.decision_source
            is_complete = report.is_complete
        all_matched_rules = list(dict.fromkeys(policy_decision.matched_rule_ids + output_enforcement.matched_rule_ids))
        audit_output_status = output_enforcement.audit_outcome
        audit_output_findings_count = output_enforcement.finding_count
        audit_output_matched_rule_ids = output_enforcement.matched_rule_ids
        audit_output_error_code = output_enforcement.error_code
    else:
        effective_decision = policy_decision.decision
        effective_downstream = None
        effective_risk = policy_decision.risk_score
        effective_explanation = policy_decision.explanation
        effective_decision_source = policy_decision.decision_source
        is_complete = report.is_complete
        all_matched_rules = policy_decision.matched_rule_ids
        audit_output_status = None
        audit_output_findings_count = None
        audit_output_matched_rule_ids = None
        audit_output_error_code = None

    # 9. Structured audit logging
    retrieved_doc_ids = [d.doc_id for d in retrieved_docs]
    log_rag_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/rag/proxy",
        decision=effective_decision,
        matched_rule_ids=all_matched_rules,
        risk_score=effective_risk,
        forwarded=forwarded,
        query=payload.query,
        retrieved_doc_ids=retrieved_doc_ids,
        num_retrieved=len(retrieved_docs),
        is_complete=is_complete,
        client_request_id=payload.request_id,
        decision_source=effective_decision_source,
        output_status=audit_output_status,
        output_findings_count=audit_output_findings_count,
        output_matched_rule_ids=audit_output_matched_rule_ids,
        output_error_code=audit_output_error_code,
        ingress_telemetry=ingress_telemetry,
    )

    return RAGProxyResponse(
        decision=effective_decision,
        decision_source=effective_decision_source,
        risk_score=effective_risk,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        is_complete=is_complete,
        forwarded=forwarded,
        direct_query_assessment=report.query_result.assessment,
        combined_context_assessment=report.combined_context_result.assessment,
        matched_rules=all_matched_rules,
        explanation=effective_explanation,
        retrieved_documents=retrieved_summaries,
        document_assessments=doc_assessments,
        downstream_response=effective_downstream,
        findings=findings_summaries,
        ml_assessment=ml_assessment,
    )
