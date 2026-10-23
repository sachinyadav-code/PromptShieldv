"""Pydantic schemas and models for PromptShield Phase 3 Tool-Call Guard."""

from typing import Any, Dict, Literal, Optional, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator

ToolCategory = Literal["READ_ONLY", "STATE_MUTATION", "HIGH_IMPACT"]

MAX_TOOL_ARGUMENT_KEYS: int = 50


class ToolDefinition(BaseModel):
    """Specification of an allowed tool in the registry."""

    name: str = Field(..., min_length=1, max_length=64, description="Unique tool identifier.")
    description: str = Field(..., min_length=1, max_length=512, description="Functional summary of the tool.")
    category: ToolCategory = Field(default="READ_ONLY", description="Risk tier for the tool.")
    requires_confirmation: bool = Field(
        default=False,
        description="Whether tool requires explicit out-of-band administrative confirmation.",
    )
    args_schema: Type[BaseModel] = Field(..., description="Pydantic model enforcing argument schema.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class ToolCallRequest(BaseModel):
    """Input payload representing an untrusted tool call request."""

    tool_name: str = Field(..., min_length=1, max_length=64, description="Target tool name.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw proposed arguments dictionary (bounded to 50 top-level keys).",
    )
    call_id: Optional[str] = Field(default=None, description="Optional caller correlation ID.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("arguments")
    @classmethod
    def validate_argument_key_count(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce strict upper bound on argument keys before validation or execution."""
        if not isinstance(v, dict):
            raise ValueError("Tool arguments must be a dictionary of key-value pairs.")
        if len(v) > MAX_TOOL_ARGUMENT_KEYS:
            raise ValueError(
                f"Tool arguments exceed maximum allowed limit of {MAX_TOOL_ARGUMENT_KEYS} keys (got {len(v)})."
            )
        return v


class ToolValidationResult(BaseModel):
    """Evaluation outcome from the Tool-Call Guard policy."""

    decision: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(
        ...,
        description="Authoritative guard decision: ALLOW, BLOCK, or REVIEW.",
    )
    tool_name: str = Field(..., description="Target tool name evaluated.")
    error_code: Optional[str] = Field(
        default=None,
        description="Categorical error identifier if blocked (e.g. TOOL_NOT_ALLOWED, INVALID_ARGUMENTS).",
    )
    explanation: str = Field(..., description="Sanitized rationale for the validation decision.")
    validated_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized and validated arguments if ALLOW; None if blocked or denied.",
    )

    model_config = ConfigDict(extra="forbid")


class ToolExecutionResult(BaseModel):
    """Outcome of attempting to execute a tool call."""

    tool_name: str = Field(..., description="Target tool name executed or denied.")
    success: bool = Field(..., description="Whether execution was allowed and completed successfully.")
    output: Any = Field(default=None, description="Output payload from the tool handler if successful.")
    error: Optional[str] = Field(default=None, description="Error message if execution failed or was denied.")

    model_config = ConfigDict(extra="forbid")


class ToolValidationResponse(BaseModel):
    """Structured response for tool validation gateway endpoint."""

    decision: Literal["ALLOW", "BLOCK", "REVIEW"] = Field(..., description="Authoritative guard decision.")
    tool_name: str = Field(..., description="Target tool name.")
    error_code: Optional[str] = Field(default=None, description="Categorical error identifier if blocked.")
    explanation: str = Field(..., description="Sanitized rationale for the validation decision.")
    request_id: str = Field(..., description="Gateway server request identifier.")
    client_request_id: Optional[str] = Field(default=None, description="Client correlation ID if supplied.")
    validated_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Validated arguments if allowed.")

    model_config = ConfigDict(extra="forbid")


class ToolExecutionResponse(BaseModel):
    """Structured response for tool execution gateway endpoint."""

    success: bool = Field(..., description="Whether execution was allowed and completed successfully.")
    tool_name: str = Field(..., description="Target tool name.")
    request_id: str = Field(..., description="Gateway server request identifier.")
    client_request_id: Optional[str] = Field(default=None, description="Client correlation ID if supplied.")
    output: Any = Field(default=None, description="Output payload from the tool handler if successful.")
    error: Optional[str] = Field(default=None, description="Error explanation if execution failed or was denied.")
    error_code: Optional[str] = Field(default=None, description="Categorical error code if failed or denied.")

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Harmless Demonstration Tool Argument Schemas
# ---------------------------------------------------------------------------


class CalculatorArgs(BaseModel):
    """Strict argument schema for the safe arithmetic calculator tool."""

    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        ...,
        description="Arithmetic operation to perform.",
    )
    a: float = Field(..., ge=-1e9, le=1e9, description="First numeric operand.")
    b: float = Field(..., ge=-1e9, le=1e9, description="Second numeric operand.")

    model_config = ConfigDict(extra="forbid")


class SampleLookupArgs(BaseModel):
    """Strict argument schema for the fixed sample-data lookup tool."""

    record_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Alphanumeric identifier for sample mock record.",
    )

    model_config = ConfigDict(extra="forbid")
