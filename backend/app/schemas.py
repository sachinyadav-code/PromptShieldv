"""Pydantic schemas for PromptShield Security Gateway."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class PromptRequest(BaseModel):
    """Incoming prompt payload for inspection or proxying."""

    prompt: str = Field(
        ...,
        description="The user prompt text to inspect or proxy.",
        min_length=1,
        max_length=10000,
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client-provided correlation ID (untrusted).",
    )

    @field_validator("prompt")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        """Ensure prompt is not empty or composed solely of whitespace."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return v


class RuleInfo(BaseModel):
    """Metadata describing a configured deterministic security rule."""

    rule_id: str
    name: str
    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    action: Literal["ALLOW", "FLAG", "BLOCK"]


class MLAssessment(BaseModel):
    """Structured assessment of prompt injection likelihood produced by the ML classifier.

    IMPORTANT SECURITY & STATISTICAL NOTICE:
    - The `score` is an uncalibrated heuristic model estimate derived from a linear
      classifier (TF-IDF + Logistic Regression) trained on synthetic prompt data.
    - It is NOT a Bayesian posterior probability of attack success.
    - It is NOT a safety guarantee or a definitive safety verdict.
    - An unavailable result (available=False) must NEVER be interpreted as benign
      clearance or proof of safety.
    - Deterministic security rules take sovereign precedence in the policy engine.
    - Raw prompt strings, filesystem paths, model internals, and exception traces
      are strictly omitted from this schema.
    """

    available: bool = Field(
        ...,
        strict=True,
        description="True if the ML model is operational and inference succeeded; False otherwise.",
    )
    score: Optional[float] = Field(
        default=None,
        strict=True,
        ge=0.0,
        le=1.0,
        description=(
            "Uncalibrated heuristic model score between 0.0 and 1.0 indicating the "
            "estimated likelihood of prompt injection. This is an uncalibrated model estimate, "
            "not a probability guarantee and not a definitive safety verdict. "
            "Must be None when inference is unavailable."
        ),
    )
    threshold: float = Field(
        ...,
        strict=True,
        ge=0.0,
        le=1.0,
        description="The decision threshold applied for blocking evaluation (e.g., 0.50).",
    )
    predicted_label: Optional[Literal["benign", "injection"]] = Field(
        default=None,
        description=(
            "Predicted classification label ('benign' or 'injection') based on whether "
            "the model score met the threshold. Must be None when inference is unavailable."
        ),
    )
    would_block: bool = Field(
        ...,
        strict=True,
        validation_alias=AliasChoices("would_block", "threshold_met"),
        description="Whether the unrounded model score met or exceeded the decision threshold.",
    )
    model_version: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Identifier of the model version (e.g. 'v1'), or None if unknown.",
    )
    error_code: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Z0-9_]+$",
        description="High-level safe error code if inference is unavailable; None on success.",
    )

    @property
    def threshold_met(self) -> bool:
        """Alias for would_block indicating whether the decision threshold was met."""
        return self.would_block

    @classmethod
    def unavailable(
        cls,
        threshold: float,
        error_code: str = "MODEL_UNAVAILABLE",
        model_version: Optional[str] = None,
    ) -> "MLAssessment":
        """Construct a standardized unavailable ML assessment requiring an explicit threshold."""
        return cls(
            available=False,
            score=None,
            threshold=threshold,
            predicted_label=None,
            would_block=False,
            model_version=model_version,
            error_code=error_code,
        )

    @classmethod
    def from_inference_result(cls, result: Any) -> "MLAssessment":
        """Construct an MLAssessment from an MLInferenceResult or dictionary, preserving original threshold."""
        if hasattr(result, "to_dict"):
            data = dict(result.to_dict())
        elif isinstance(result, dict):
            data = dict(result)
        else:
            data = {
                "available": getattr(result, "available", False),
                "score": getattr(result, "score", None),
                "threshold": getattr(result, "threshold", None),
                "predicted_label": getattr(result, "predicted_label", None),
                "would_block": getattr(result, "would_block", False),
                "model_version": getattr(result, "model_version", None),
                "error_code": getattr(result, "error_code", None),
            }
        return cls(**data)

    @model_validator(mode="after")
    def validate_assessment_consistency(self) -> "MLAssessment":
        """Enforce strict semantic consistency between available status, score, label, and error."""
        if not self.available:
            # 1. Unavailable semantics
            if self.score is not None:
                raise ValueError("ML score must be None when inference is unavailable.")
            if self.predicted_label is not None:
                raise ValueError(
                    "ML predicted_label must be None when inference is unavailable; "
                    "unavailable status must never be represented as benign or safe."
                )
            if self.would_block:
                raise ValueError("ML threshold cannot be met (would_block must be False) when inference is unavailable.")
        else:
            # 2. Available semantics
            if self.score is None:
                raise ValueError("ML score is required when inference is available.")
            if self.predicted_label is None:
                raise ValueError("ML predicted_label is required when inference is available.")
            if self.error_code is not None:
                raise ValueError("ML error_code must be None when inference succeeded (available is True).")

            # 3. Label and threshold/blocking consistency
            if self.predicted_label == "injection" and not self.would_block:
                raise ValueError(
                    "Inconsistent ML assessment: predicted_label is 'injection' but would_block is False."
                )
            if self.predicted_label == "benign" and self.would_block:
                raise ValueError(
                    "Inconsistent ML assessment: predicted_label is 'benign' but would_block is True."
                )

        return self


# Schema aliases for interoperability across modules
MLInferenceAssessment = MLAssessment
MLDetectionResult = MLAssessment
MLAnalysisResult = MLAssessment


class InspectResponse(BaseModel):
    """Response payload for the prompt inspection endpoint."""

    decision: Literal["ALLOW", "BLOCK"]
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heuristic risk score between 0.0 and 1.0 (not a probability).",
    )
    matched_rules: List[str]
    explanation: str
    request_id: str = Field(
        ...,
        description="Authoritative, server-generated unique request identifier.",
    )
    client_request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied correlation ID, if provided.",
    )
    ml_assessment: Optional[MLAssessment] = Field(
        default=None,
        validation_alias=AliasChoices("ml_assessment", "ml_detection", "ml_analysis"),
        description="Optional ML injection assessment result if ML detection was performed.",
    )
    decision_source: Optional[Literal["deterministic_rule", "ml_enforcement", "allow_default"]] = Field(
        default=None,
        description="Origin of the security policy decision: deterministic_rule, ml_enforcement, or allow_default.",
    )

    @property
    def ml_detection(self) -> Optional[MLAssessment]:
        """Alias property for ml_assessment."""
        return self.ml_assessment


class MockDownstreamResponse(BaseModel):
    """Harmless simulated downstream LLM service response."""

    simulated: bool = Field(default=True, description="Flag indicating simulated response.")
    provider: str = Field(default="mock-downstream-service")
    message: str
    received_prompt_length: int


class ProxyResponse(BaseModel):
    """Response payload for the security gateway proxy endpoint."""

    decision: Literal["ALLOW", "BLOCK"]
    request_id: str = Field(
        ...,
        description="Authoritative, server-generated unique request identifier.",
    )
    client_request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied correlation ID, if provided.",
    )
    matched_rules: List[str]
    explanation: str
    downstream_response: Optional[MockDownstreamResponse] = Field(
        default=None,
        description="Downstream response if allowed; omitted/null if blocked.",
    )
    ml_assessment: Optional[MLAssessment] = Field(
        default=None,
        validation_alias=AliasChoices("ml_assessment", "ml_detection", "ml_analysis"),
        description="Optional ML injection assessment result if ML detection was performed.",
    )
    decision_source: Optional[Literal["deterministic_rule", "ml_enforcement", "allow_default"]] = Field(
        default=None,
        description="Origin of the security policy decision: deterministic_rule, ml_enforcement, or allow_default.",
    )

    @property
    def ml_detection(self) -> Optional[MLAssessment]:
        """Alias property for ml_assessment."""
        return self.ml_assessment


class HealthResponse(BaseModel):
    """Gateway health check status."""

    status: str
    service: str
    phase: int


class RAGRequest(BaseModel):
    """Incoming request payload for RAG inspection or proxying."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user search query to inspect and retrieve against.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        strict=True,
        description="Maximum number of candidate documents to retrieve (1-10, default 3).",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client-provided correlation ID (untrusted).",
    )

    @field_validator("query")
    @classmethod
    def validate_query_non_whitespace(cls, v: Any) -> str:
        """Ensure query is a string and not empty or composed solely of whitespace."""
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError("Query must be a string, not a boolean.")
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or contain only whitespace.")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k_not_bool(cls, v: Any) -> int:
        """Ensure top_k is an integer and not passed as a boolean."""
        if isinstance(v, bool):
            raise ValueError("top_k must be an integer, not a boolean.")
        return v


# Schema alias for RAGRequest
RAGPromptRequest = RAGRequest


class RetrievedDocumentSummary(BaseModel):
    """Summary metadata for a retrieved document.

    IMPORTANT SECURITY NOTICE:
    - relevance_score measures keyword overlap ranking and is NOT a security clearance,
      safety verdict, or authorization signal.
    - Full raw document content is strictly omitted by default to prevent sensitive
      data exposure.
    """

    doc_id: str = Field(..., max_length=64, description="Unique document identifier.")
    title: str = Field(..., max_length=256, description="Document title.")
    source: str = Field(..., max_length=256, description="Document origin or source path.")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Keyword overlap relevance score [0.0, 1.0] for ranking only. NOT a security score.",
    )
    matched_tokens: List[str] = Field(
        default_factory=list,
        description="List of query tokens that matched this document.",
    )


class DocumentInspectionSummary(BaseModel):
    """Security inspection summary for an individual retrieved document."""

    doc_id: Optional[str] = Field(default=None, description="Document identifier.")
    assessment: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Security assessment for this document.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heuristic risk score [0.0, 1.0] for this document.",
    )
    matched_rules: List[str] = Field(
        default_factory=list,
        description="Security rules matched in this document.",
    )
    is_complete: bool = Field(
        default=True,
        description="Whether inspection completed fully without exceeding resource limits.",
    )
    explanation: str = Field(
        ...,
        description="Sanitized summary explanation omitting raw payloads.",
    )


class RAGFindingSummary(BaseModel):
    """Granular security finding from multi-level RAG inspection."""

    scope: Literal["direct_query", "document", "chunk", "combined_context"]
    assessment: Literal["ALLOW", "BLOCK", "REVIEW"]
    rule_id: str
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""


class RAGInspectResponse(BaseModel):
    """Response payload for the RAG inspection endpoint."""

    decision: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Authoritative security policy decision: ALLOW, BLOCK, or REVIEW.",
    )
    decision_source: Optional[Literal["deterministic_rule", "ml_enforcement", "rag_inspection", "allow_default"]] = Field(
        default=None,
        description="Origin of the security policy decision.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heuristic security risk score between 0.0 and 1.0 (not a probability).",
    )
    request_id: str = Field(
        ...,
        description="Authoritative, server-generated unique request identifier.",
    )
    client_request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied correlation ID, if provided.",
    )
    is_complete: bool = Field(
        ...,
        description="Whether all inspection tiers completed without exceeding resource limits.",
    )
    direct_query_assessment: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Inspection outcome for the user's direct query.",
    )
    combined_context_assessment: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Inspection outcome for the combined retrieved context.",
    )
    matched_rules: List[str] = Field(
        default_factory=list,
        description="All matched security rule identifiers across all tiers.",
    )
    explanation: str = Field(
        ...,
        description="Sanitized explanation of the inspection outcome.",
    )
    retrieved_documents: List[RetrievedDocumentSummary] = Field(
        default_factory=list,
        description="Summaries of retrieved documents (excluding raw content).",
    )
    document_assessments: List[DocumentInspectionSummary] = Field(
        default_factory=list,
        description="Security assessment summaries for each retrieved document.",
    )
    findings: List[RAGFindingSummary] = Field(
        default_factory=list,
        description="Granular inspection findings.",
    )
    ml_assessment: Optional[MLAssessment] = Field(
        default=None,
        validation_alias=AliasChoices("ml_assessment", "ml_detection", "ml_analysis"),
        description="Optional ML injection assessment of the direct query.",
    )

    @property
    def ml_detection(self) -> Optional[MLAssessment]:
        """Alias property for ml_assessment."""
        return self.ml_assessment


class RAGProxyResponse(BaseModel):
    """Response payload for the RAG proxy endpoint."""

    decision: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Authoritative security policy decision: ALLOW, BLOCK, or REVIEW.",
    )
    decision_source: Optional[Literal["deterministic_rule", "ml_enforcement", "rag_inspection", "allow_default"]] = Field(
        default=None,
        description="Origin of the security policy decision.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heuristic security risk score between 0.0 and 1.0 (not a probability).",
    )
    request_id: str = Field(
        ...,
        description="Authoritative, server-generated unique request identifier.",
    )
    client_request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied correlation ID, if provided.",
    )
    is_complete: bool = Field(
        ...,
        description="Whether all inspection tiers completed without exceeding resource limits.",
    )
    forwarded: bool = Field(
        ...,
        description="True if request was forwarded to downstream mock service; False otherwise.",
    )
    direct_query_assessment: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Inspection outcome for the user's direct query.",
    )
    combined_context_assessment: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Inspection outcome for the combined retrieved context.",
    )
    matched_rules: List[str] = Field(
        default_factory=list,
        description="All matched security rule identifiers across all tiers.",
    )
    explanation: str = Field(
        ...,
        description="Sanitized explanation of the policy decision.",
    )
    retrieved_documents: List[RetrievedDocumentSummary] = Field(
        default_factory=list,
        description="Summaries of retrieved documents (excluding raw content).",
    )
    document_assessments: List[DocumentInspectionSummary] = Field(
        default_factory=list,
        description="Security assessment summaries for each retrieved document.",
    )
    downstream_response: Optional[MockDownstreamResponse] = Field(
        default=None,
        description="Downstream response if allowed and forwarded; null if blocked or held for review.",
    )
    findings: List[RAGFindingSummary] = Field(
        default_factory=list,
        description="Granular inspection findings.",
    )
    ml_assessment: Optional[MLAssessment] = Field(
        default=None,
        validation_alias=AliasChoices("ml_assessment", "ml_detection", "ml_analysis"),
        description="Optional ML injection assessment of the direct query.",
    )

    @property
    def ml_detection(self) -> Optional[MLAssessment]:
        """Alias property for ml_assessment."""
        return self.ml_assessment
