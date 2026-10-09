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
