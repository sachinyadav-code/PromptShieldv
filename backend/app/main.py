from contextlib import asynccontextmanager
import logging
import threading
from typing import List, Optional, Sequence, Tuple
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.audit.logger import log_rag_security_event, log_security_event
from app.detection.ml_classifier import get_classifier
from app.detection.rules import get_configured_rules, scan_prompt
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
    logger.error("Unhandled gateway exception on %s: %s", request.url.path, type(exc).__name__, exc_info=True)
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


@app.post("/api/v1/rag/inspect", response_model=RAGInspectResponse)
def inspect_rag(payload: RAGRequest) -> RAGInspectResponse:
    """Inspect user query and retrieved context without forwarding to mock downstream."""
    server_request_id = str(uuid.uuid4())

    # 1. Load corpus (fail closed if loading fails)
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

        # 3. Direct query ML assessment
        ml_assessment = _get_ml_assessment(payload.query)

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


@app.post("/api/v1/rag/proxy", response_model=RAGProxyResponse)
def proxy_rag(payload: RAGRequest) -> RAGProxyResponse:
    """Inspect query and retrieved context; forward to mock downstream ONLY if ALLOW."""
    server_request_id = str(uuid.uuid4())

    # 1. Load corpus (fail closed if loading fails)
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

        # 3. Direct query ML assessment
        ml_assessment = _get_ml_assessment(payload.query)

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

    # 6. Forward to mock downstream ONLY if final decision is ALLOW
    downstream_result = None
    forwarded = False

    if policy_decision.decision == "ALLOW":
        rag_prompt = assemble_rag_downstream_prompt(payload.query, retrieved_docs)
        downstream_result = forward_to_mock_downstream(rag_prompt)
        forwarded = True

    # 7. Structured audit logging
    retrieved_doc_ids = [d.doc_id for d in retrieved_docs]
    log_rag_security_event(
        request_id=server_request_id,
        endpoint="/api/v1/rag/proxy",
        decision=policy_decision.decision,
        matched_rule_ids=policy_decision.matched_rule_ids,
        risk_score=policy_decision.risk_score,
        forwarded=forwarded,
        query=payload.query,
        retrieved_doc_ids=retrieved_doc_ids,
        num_retrieved=len(retrieved_docs),
        is_complete=report.is_complete,
        client_request_id=payload.request_id,
        decision_source=policy_decision.decision_source,
    )

    # 8. Build structured response
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

    return RAGProxyResponse(
        decision=policy_decision.decision,
        decision_source=policy_decision.decision_source,
        risk_score=policy_decision.risk_score,
        request_id=server_request_id,
        client_request_id=payload.request_id,
        is_complete=report.is_complete,
        forwarded=forwarded,
        direct_query_assessment=report.query_result.assessment,
        combined_context_assessment=report.combined_context_result.assessment,
        matched_rules=policy_decision.matched_rule_ids,
        explanation=policy_decision.explanation,
        retrieved_documents=retrieved_summaries,
        document_assessments=doc_assessments,
        downstream_response=downstream_result,
        findings=findings_summaries,
        ml_assessment=ml_assessment,
    )
