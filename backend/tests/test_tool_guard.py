"""Unit tests for PromptShield Phase 3.2 Core Tool Guard subsystem.

Verifies:
1. Explicit in-memory tool registry with default harmless demonstration tools.
2. Deny-by-default behavior rejecting unlisted/unknown tools with TOOL_NOT_ALLOWED.
3. Strict schema validation for Calculator (missing args, type mismatch, unexpected keys, bounds).
4. Strict schema validation for Sample Lookup (regex pattern, path traversal injection prevention).
5. Malformed/non-dictionary argument rejection.
6. Separation of validation from execution.
7. Proof that denied or invalid calls CANNOT reach an execution handler.
8. High-impact tools cannot be authorized autonomously and client flags cannot bypass gating.
9. Safe, side-effect-free execution of validated demonstration tools.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import pytest

from app.tools import (
    CalculatorArgs,
    MAX_TOOL_ARGUMENT_KEYS,
    SampleLookupArgs,
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionResult,
    ToolGuard,
    ToolRegistry,
    ToolValidationResult,
    execute_tool_call,
    get_tool_registry,
    reset_tool_registry,
    validate_tool_arguments,
    validate_tool_call,
)


@pytest.fixture(autouse=True)
def clean_tool_registry():
    """Ensure every test runs with a fresh default ToolRegistry."""
    reset_tool_registry()
    yield
    reset_tool_registry()


# ---------------------------------------------------------------------------
# 1. Registry Defaults and Allowlist Deny-by-Default
# ---------------------------------------------------------------------------


def test_registry_defaults_contain_only_demonstration_tools():
    """Verify registry defaults only contain harmless, deterministic demonstration tools."""
    registry = get_tool_registry()
    tools = registry.list_tools()
    tool_names = {t.name for t in tools}

    assert tool_names == {"calculator", "sample_data_lookup"}
    for t in tools:
        assert t.category == "READ_ONLY"
        assert t.requires_confirmation is False


def test_unknown_tool_rejected_by_default():
    """Verify requesting an unregistered tool returns decision=BLOCK and TOOL_NOT_ALLOWED."""
    request = ToolCallRequest(
        tool_name="unregistered_dangerous_action",
        arguments={"cmd": "whoami"},
    )
    result = validate_tool_call(request)

    assert isinstance(result, ToolValidationResult)
    assert result.decision == "BLOCK"
    assert result.tool_name == "unregistered_dangerous_action"
    assert result.error_code == "TOOL_NOT_ALLOWED"
    assert "not in the allowed registry" in result.explanation
    assert result.validated_arguments is None


def test_common_dangerous_tool_names_denied_by_default():
    """Verify common offensive tool patterns are rejected by the allowlist."""
    dangerous_names = [
        "execute_shell",
        "run_bash_command",
        "delete_database",
        "drop_table",
        "send_email",
        "transfer_funds",
        "read_private_file",
        "curl_fetch",
    ]
    for tool_name in dangerous_names:
        res = validate_tool_call(ToolCallRequest(tool_name=tool_name, arguments={}))
        assert res.decision == "BLOCK"
        assert res.error_code == "TOOL_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# 2. Calculator Schema & Argument Validation
# ---------------------------------------------------------------------------


def test_calculator_valid_addition():
    """Verify calculator tool authorizes valid arithmetic arguments."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "add", "a": 12.5, "b": 7.5},
    )
    res = validate_tool_call(req)

    assert res.decision == "ALLOW"
    assert res.error_code is None
    assert res.validated_arguments == {"operation": "add", "a": 12.5, "b": 7.5}


def test_calculator_all_supported_operations():
    """Verify add, subtract, multiply, and divide operations are authorized."""
    for op in ["add", "subtract", "multiply", "divide"]:
        req = ToolCallRequest(
            tool_name="calculator",
            arguments={"operation": op, "a": 10.0, "b": 2.0},
        )
        res = validate_tool_call(req)
        assert res.decision == "ALLOW"


def test_calculator_missing_required_arguments():
    """Verify missing required arguments are rejected with INVALID_ARGUMENTS."""
    # Missing operand 'b'
    req1 = ToolCallRequest(tool_name="calculator", arguments={"operation": "add", "a": 10.0})
    res1 = validate_tool_call(req1)
    assert res1.decision == "BLOCK"
    assert res1.error_code == "INVALID_ARGUMENTS"
    assert "Missing required argument 'b'" in res1.explanation

    # Missing operation
    req2 = ToolCallRequest(tool_name="calculator", arguments={"a": 1.0, "b": 2.0})
    res2 = validate_tool_call(req2)
    assert res2.decision == "BLOCK"
    assert res2.error_code == "INVALID_ARGUMENTS"
    assert "Missing required argument 'operation'" in res2.explanation


def test_calculator_invalid_operation_type():
    """Verify unsupported operations (e.g. modulo, eval, sqrt) are rejected."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "eval", "a": 10.0, "b": 2.0},
    )
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "operation" in res.explanation


def test_calculator_invalid_numeric_types():
    """Verify non-numeric strings or objects are rejected for float operands."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "add", "a": "not-a-number", "b": 5.0},
    )
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "a" in res.explanation


def test_calculator_unexpected_extra_fields_forbidden():
    """Verify unexpected extra arguments trigger extra='forbid' rejection."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={
            "operation": "add",
            "a": 10.0,
            "b": 2.0,
            "injected_parameter": "malicious_payload",
        },
    )
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "Unexpected extra argument 'injected_parameter'" in res.explanation


def test_calculator_out_of_bounds_operands():
    """Verify operand bounds enforcement prevents numeric overflow exploits."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "multiply", "a": 1e12, "b": 2.0},
    )
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "a" in res.explanation


# ---------------------------------------------------------------------------
# 3. Sample Data Lookup Schema & Argument Validation
# ---------------------------------------------------------------------------


def test_sample_lookup_valid_record_id():
    """Verify valid alphanumeric record ID is authorized."""
    req = ToolCallRequest(
        tool_name="sample_data_lookup",
        arguments={"record_id": "item_001"},
    )
    res = validate_tool_call(req)
    assert res.decision == "ALLOW"
    assert res.validated_arguments == {"record_id": "item_001"}


def test_sample_lookup_missing_record_id():
    """Verify missing record_id is rejected."""
    req = ToolCallRequest(tool_name="sample_data_lookup", arguments={})
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "record_id" in res.explanation


def test_sample_lookup_path_traversal_pattern_rejected():
    """Verify path traversal patterns like ../../etc/passwd violate pattern and are blocked."""
    malicious_ids = [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "item/001",
        "item;DROP TABLE users;--",
        "item 001",
    ]
    for bad_id in malicious_ids:
        req = ToolCallRequest(tool_name="sample_data_lookup", arguments={"record_id": bad_id})
        res = validate_tool_call(req)
        assert res.decision == "BLOCK", f"Expected block on bad ID: {bad_id}"
        assert res.error_code == "INVALID_ARGUMENTS"


def test_sample_lookup_unexpected_extra_arguments():
    """Verify extra parameters on sample lookup are rejected."""
    req = ToolCallRequest(
        tool_name="sample_data_lookup",
        arguments={"record_id": "item_001", "filter": "all"},
    )
    res = validate_tool_call(req)
    assert res.decision == "BLOCK"
    assert res.error_code == "INVALID_ARGUMENTS"
    assert "filter" in res.explanation


# ---------------------------------------------------------------------------
# 4. Malformed Input Payloads
# ---------------------------------------------------------------------------


def test_non_dictionary_arguments_rejected():
    """Verify passing non-dict arguments (list, int, str) fails safely."""
    tool_def = get_tool_registry().get_tool("calculator")
    assert tool_def is not None

    for invalid_args in ["a=1,b=2", [1, 2, 3], 42, None]:
        is_valid, model, err_code, exp = validate_tool_arguments(tool_def, invalid_args)
        assert is_valid is False
        assert err_code == "INVALID_ARGUMENTS"
        assert "must be a dictionary" in exp


# ---------------------------------------------------------------------------
# 5. Separation of Validation from Execution
# ---------------------------------------------------------------------------


def test_validation_does_not_execute_handler():
    """Verify validate_tool_call evaluates policy without invoking execution handlers."""
    registry = ToolRegistry(register_defaults=False)
    mock_handler = MagicMock(return_value={"result": 100})
    registry.register_tool(
        ToolDefinition(
            name="calculator",
            description="Math tool",
            category="READ_ONLY",
            requires_confirmation=False,
            args_schema=CalculatorArgs,
        ),
        handler=mock_handler,
    )
    guard = ToolGuard(registry=registry)

    req = ToolCallRequest(tool_name="calculator", arguments={"operation": "add", "a": 1.0, "b": 2.0})
    val_res = guard.validate(req)
    assert val_res.decision == "ALLOW"
    # Handler must NOT have been called during validation
    assert mock_handler.call_count == 0


# ---------------------------------------------------------------------------
# 6. Proof Denied Calls Cannot Reach Handlers
# ---------------------------------------------------------------------------


def test_denied_calls_never_reach_handler():
    """Verify execution handler is NEVER invoked when validation fails or tool is blocked."""
    registry = ToolRegistry(register_defaults=False)
    mock_handler = MagicMock()
    registry.register_tool(
        ToolDefinition(
            name="calculator",
            description="Math tool",
            category="READ_ONLY",
            requires_confirmation=False,
            args_schema=CalculatorArgs,
        ),
        handler=mock_handler,
    )
    guard = ToolGuard(registry=registry)

    # 1. Invalid arguments: missing field
    bad_req_1 = ToolCallRequest(tool_name="calculator", arguments={"operation": "add", "a": 1.0})
    exec_res_1 = guard.execute(bad_req_1)
    assert exec_res_1.success is False
    assert "Execution blocked by tool guard" in exec_res_1.error
    assert mock_handler.call_count == 0

    # 2. Invalid arguments: unexpected extra field
    bad_req_2 = ToolCallRequest(tool_name="calculator", arguments={"operation": "add", "a": 1.0, "b": 2.0, "extra": 1})
    exec_res_2 = guard.execute(bad_req_2)
    assert exec_res_2.success is False
    assert mock_handler.call_count == 0

    # 3. Unregistered tool
    bad_req_3 = ToolCallRequest(tool_name="unknown_tool", arguments={})
    exec_res_3 = guard.execute(bad_req_3)
    assert exec_res_3.success is False
    assert mock_handler.call_count == 0


# ---------------------------------------------------------------------------
# 7. Safe Execution of Allowed Demonstration Tools
# ---------------------------------------------------------------------------


def test_safe_execution_of_calculator():
    """Verify allowed calculator tool executes safely and returns expected math."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "multiply", "a": 6.0, "b": 7.0},
    )
    result = execute_tool_call(req)
    assert result.success is True
    assert result.error is None
    assert result.output == {"operation": "multiply", "a": 6.0, "b": 7.0, "result": 42.0}


def test_calculator_division_by_zero_handled_safely():
    """Verify division by zero raises handled exception without crashing."""
    req = ToolCallRequest(
        tool_name="calculator",
        arguments={"operation": "divide", "a": 10.0, "b": 0.0},
    )
    # Validation passes (b=0 is a valid float within bounds)
    val = validate_tool_call(req)
    assert val.decision == "ALLOW"

    # Execution safely captures runtime division error
    result = execute_tool_call(req)
    assert result.success is False
    assert "Division by zero" in result.error


def test_safe_execution_of_sample_data_lookup():
    """Verify allowed sample data lookup tool returns static mock data."""
    # Existing record
    req1 = ToolCallRequest(tool_name="sample_data_lookup", arguments={"record_id": "item_001"})
    res1 = execute_tool_call(req1)
    assert res1.success is True
    assert res1.output["found"] is True
    assert res1.output["record"]["name"] == "Solar Inverter A1"

    # Non-existent record (safe not-found)
    req2 = ToolCallRequest(tool_name="sample_data_lookup", arguments={"record_id": "non_existent_999"})
    res2 = execute_tool_call(req2)
    assert res2.success is True
    assert res2.output["found"] is False
    assert res2.output["record"] is None


# ---------------------------------------------------------------------------
# 8. High-Impact Tools Cannot Be Authorized Autonomously
# ---------------------------------------------------------------------------


def test_high_impact_tool_blocked_and_cannot_be_bypassed_by_client_flags():
    """Verify that a tool flagged as HIGH_IMPACT is blocked and client flags cannot bypass it."""
    registry = ToolRegistry(register_defaults=False)
    mock_handler = MagicMock()

    class TransferArgs(BaseModel):
        amount: float
        model_config = ConfigDict(extra="forbid")

    registry.register_tool(
        ToolDefinition(
            name="transfer_funds",
            description="Dangerous financial action",
            category="HIGH_IMPACT",
            requires_confirmation=True,
            args_schema=TransferArgs,
        ),
        handler=mock_handler,
    )
    guard = ToolGuard(registry=registry)

    # Attempt A: Standard request
    req1 = ToolCallRequest(tool_name="transfer_funds", arguments={"amount": 100.0})
    res1 = guard.validate(req1)
    assert res1.decision == "BLOCK"
    assert res1.error_code == "CONFIRMATION_REQUIRED"

    # Attempt B: Client tries to inject confirmed=True in arguments
    req2 = ToolCallRequest(tool_name="transfer_funds", arguments={"amount": 100.0, "confirmed": True})
    res2 = guard.validate(req2)
    # Blocked by extra='forbid' on TransferArgs schema
    assert res2.decision == "BLOCK"

    # Attempt C: Execution blocked; handler never called
    exec_res = guard.execute(req1)
    assert exec_res.success is False
    assert mock_handler.call_count == 0


# ---------------------------------------------------------------------------
# 9. Tool Registry Management
# ---------------------------------------------------------------------------


def test_tool_registry_management():
    """Verify programmatic register, unregister, and lookup functions."""
    registry = ToolRegistry(register_defaults=False)
    assert len(registry.list_tools()) == 0

    class DummyArgs(BaseModel):
        val: str
        model_config = ConfigDict(extra="forbid")

    dummy_def = ToolDefinition(
        name="dummy_tool",
        description="Dummy test tool",
        category="READ_ONLY",
        args_schema=DummyArgs,
    )
    registry.register_tool(dummy_def, handler=lambda a: {"val": a.val})
    assert registry.has_tool("dummy_tool") is True
    assert registry.get_tool("dummy_tool") == dummy_def

    # Unregister
    removed = registry.unregister_tool("dummy_tool")
    assert removed is True
    assert registry.has_tool("dummy_tool") is False
    assert registry.get_tool("dummy_tool") is None


# ---------------------------------------------------------------------------
# 10. Argument Dictionary Bounding (MAX_TOOL_ARGUMENT_KEYS = 50)
# ---------------------------------------------------------------------------


def test_tool_call_request_rejects_more_than_50_keys():
    """Verify ToolCallRequest rejects arguments dictionaries exceeding 50 keys."""
    oversized_args = {f"key_{i}": i for i in range(51)}
    with pytest.raises(ValidationError) as exc_info:
        ToolCallRequest(tool_name="calculator", arguments=oversized_args)
    assert "Tool arguments exceed maximum allowed limit of 50 keys" in str(exc_info.value)


def test_tool_call_request_accepts_exactly_50_keys():
    """Verify ToolCallRequest accepts arguments dictionaries with exactly 50 keys."""
    args_50 = {f"key_{i}": i for i in range(50)}
    req = ToolCallRequest(tool_name="test_tool", arguments=args_50)
    assert len(req.arguments) == 50


def test_validator_rejects_more_than_50_argument_keys():
    """Verify validate_tool_arguments rejects dictionaries with more than 50 keys."""
    registry = get_tool_registry()
    calc_def = registry.get_tool("calculator")
    assert calc_def is not None

    oversized_args = {f"key_{i}": i for i in range(51)}
    is_valid, validated_model, error_code, explanation = validate_tool_arguments(
        calc_def,
        oversized_args,
    )
    assert is_valid is False
    assert validated_model is None
    assert error_code == "INVALID_ARGUMENTS"
    assert "exceed maximum allowed limit of 50 keys" in explanation


def test_tool_with_50_arguments_schema_executes_successfully():
    """Verify a tool whose schema declares 50 valid fields succeeds with exactly 50 arguments."""
    from pydantic import create_model

    # Dynamically create a Pydantic model with 50 optional integer fields
    fields = {f"field_{i}": (Optional[int], None) for i in range(50)}
    FiftyFieldArgs = create_model(
        "FiftyFieldArgs",
        __config__=None,
        __base__=BaseModel,
        **fields,
    )
    FiftyFieldArgs.model_config = ConfigDict(extra="forbid")

    mock_handler = MagicMock(return_value={"count": 50, "status": "ok"})
    registry = get_tool_registry()
    registry.register_tool(
        ToolDefinition(
            name="fifty_field_tool",
            description="Tool accepting up to 50 fields",
            category="READ_ONLY",
            args_schema=FiftyFieldArgs,
        ),
        handler=mock_handler,
    )

    guard = ToolGuard(registry=registry)
    args_50 = {f"field_{i}": i for i in range(50)}
    req = ToolCallRequest(tool_name="fifty_field_tool", arguments=args_50)

    val_res = guard.validate(req)
    assert val_res.decision == "ALLOW"
    assert val_res.validated_arguments is not None
    assert len(val_res.validated_arguments) == 50

    exec_res = guard.execute(req)
    assert exec_res.success is True
    assert exec_res.output == {"count": 50, "status": "ok"}
    assert mock_handler.call_count == 1
