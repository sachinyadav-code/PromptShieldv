"""Tool guard service and policy evaluation for PromptShield Phase 3."""

import logging
from typing import Optional

from app.tools.registry import ToolRegistry, get_tool_registry
from app.tools.schemas import (
    ToolCallRequest,
    ToolExecutionResult,
    ToolValidationResult,
)
from app.tools.validator import validate_tool_arguments

logger = logging.getLogger("promptshield.tools.guard")


class ToolGuard:
    """Security guard verifying that untrusted tool calls conform to strict policies."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """Active tool registry."""
        if self._registry is not None:
            return self._registry
        return get_tool_registry()

    def validate(self, tool_call: ToolCallRequest) -> ToolValidationResult:
        """Evaluate an untrusted tool call request against the tool allowlist and parameter schemas.

        Execution logic:
        1. Allowlist enforcement: Check if tool is in the registry (deny-by-default).
        2. Argument validation: Parse and validate against the tool's strict Pydantic model.
        3. High-impact policy: High-impact actions are blocked from autonomous execution.
           A client-supplied flag cannot independently authorize high-impact actions.

        Args:
            tool_call: Untrusted tool call payload.

        Returns:
            Structured ToolValidationResult with decision (ALLOW, BLOCK, REVIEW) and error codes.
        """
        tool_name = tool_call.tool_name

        # 1. Deny-by-default allowlist check
        tool_def = self.registry.get_tool(tool_name)
        if tool_def is None:
            logger.info("Tool-call guard rejected unregistered tool: '%s'", tool_name)
            return ToolValidationResult(
                decision="BLOCK",
                tool_name=tool_name,
                error_code="TOOL_NOT_ALLOWED",
                explanation=f"Tool '{tool_name}' is not in the allowed registry.",
                validated_arguments=None,
            )

        # 2. Strict argument validation
        is_valid, validated_model, error_code, explanation = validate_tool_arguments(
            tool_def,
            tool_call.arguments,
        )
        if not is_valid or validated_model is None:
            logger.info("Tool-call guard rejected invalid arguments for '%s': %s", tool_name, explanation)
            return ToolValidationResult(
                decision="BLOCK",
                tool_name=tool_name,
                error_code=error_code,
                explanation=explanation,
                validated_arguments=None,
            )

        # 3. High-impact gating
        if tool_def.category == "HIGH_IMPACT" or tool_def.requires_confirmation:
            logger.warning("Tool-call guard blocked high-impact tool without administrative approval: '%s'", tool_name)
            return ToolValidationResult(
                decision="BLOCK",
                tool_name=tool_name,
                error_code="CONFIRMATION_REQUIRED",
                explanation=f"Tool '{tool_name}' is classified as HIGH_IMPACT and requires explicit out-of-band administrative confirmation.",
                validated_arguments=None,
            )

        # 4. Validated and authorized
        return ToolValidationResult(
            decision="ALLOW",
            tool_name=tool_name,
            error_code=None,
            explanation="Tool call authorized and arguments validated.",
            validated_arguments=validated_model.model_dump(),
        )

    def execute(self, tool_call: ToolCallRequest) -> ToolExecutionResult:
        """Safely execute a tool call if and only if validation strictly returns ALLOW.

        Guarantees:
        - Denied calls NEVER reach the underlying execution handler.
        - Tool handlers are isolated from raw unvalidated dictionary inputs.

        Args:
            tool_call: Untrusted tool call payload.

        Returns:
            ToolExecutionResult with success indicator and output or error description.
        """
        validation = self.validate(tool_call)
        if validation.decision != "ALLOW":
            return ToolExecutionResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=f"Execution blocked by tool guard: {validation.explanation}",
            )

        handler = self.registry.get_handler(tool_call.tool_name)
        if handler is None:
            return ToolExecutionResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=f"No execution handler configured for tool '{tool_call.tool_name}'.",
            )

        tool_def = self.registry.get_tool(tool_call.tool_name)
        assert tool_def is not None

        try:
            validated_model = tool_def.args_schema.model_validate(validation.validated_arguments)
            output = handler(validated_model)
            return ToolExecutionResult(
                tool_name=tool_call.tool_name,
                success=True,
                output=output,
                error=None,
            )
        except Exception as exc:
            logger.error("Tool execution failed on '%s': %s", tool_call.tool_name, type(exc).__name__)
            return ToolExecutionResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=f"Tool execution failed: {str(exc)}",
            )


# Global singleton helper
_global_tool_guard = ToolGuard()


def validate_tool_call(tool_call: ToolCallRequest) -> ToolValidationResult:
    """Validate untrusted tool call against default ToolGuard singleton."""
    return _global_tool_guard.validate(tool_call)


def execute_tool_call(tool_call: ToolCallRequest) -> ToolExecutionResult:
    """Validate and execute untrusted tool call against default ToolGuard singleton."""
    return _global_tool_guard.execute(tool_call)
