"""Pydantic schemas for PromptShield Security Gateway."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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


class HealthResponse(BaseModel):
    """Gateway health check status."""

    status: str
    service: str
    phase: int
