"""Argument validation logic for PromptShield Tool-Call Guard."""

from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, ValidationError

from app.tools.schemas import MAX_TOOL_ARGUMENT_KEYS, ToolDefinition


def validate_tool_arguments(
    tool_def: ToolDefinition,
    arguments: Any,
) -> Tuple[bool, Optional[BaseModel], Optional[str], str]:
    """Validate untrusted candidate arguments against a tool's strict Pydantic schema.

    Enforces:
    1. Arguments must be a JSON/dict object (rejects primitives, lists, null).
    2. Arguments dictionary must not exceed MAX_TOOL_ARGUMENT_KEYS (50).
    3. Missing required fields are rejected.
    4. Unexpected/extra fields are rejected (via extra='forbid').
    5. Type violations and out-of-bound ranges are rejected.

    Args:
        tool_def: Tool specification containing args_schema.
        arguments: Raw untrusted arguments input.

    Returns:
        Tuple of (is_valid, validated_model, error_code, explanation).
    """
    if not isinstance(arguments, dict):
        return (
            False,
            None,
            "INVALID_ARGUMENTS",
            "Tool arguments must be a dictionary of key-value pairs.",
        )

    if len(arguments) > MAX_TOOL_ARGUMENT_KEYS:
        return (
            False,
            None,
            "INVALID_ARGUMENTS",
            f"Tool arguments exceed maximum allowed limit of {MAX_TOOL_ARGUMENT_KEYS} keys (got {len(arguments)}).",
        )

    try:
        validated = tool_def.args_schema.model_validate(arguments)
        return True, validated, None, "Arguments validated successfully."
    except ValidationError as exc:
        # Extract sanitized, structured error summaries without exposing raw internal details
        field_errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            err_type = err.get("type", "value_error")

            if err_type == "extra_forbidden":
                field_errors.append(f"Unexpected extra argument '{loc}' is not permitted.")
            elif err_type == "missing":
                field_errors.append(f"Missing required argument '{loc}'.")
            else:
                field_errors.append(f"Argument '{loc}': {msg}.")

        explanation = "Argument validation failed: " + "; ".join(field_errors)
        return False, None, "INVALID_ARGUMENTS", explanation
    except Exception as exc:
        return (
            False,
            None,
            "INVALID_ARGUMENTS",
            f"Argument parsing failed: {type(exc).__name__}",
        )
