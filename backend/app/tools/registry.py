"""In-memory tool registry for PromptShield Phase 3 Tool-Call Guard."""

from typing import Any, Callable, Dict, List, Optional
import threading

from app.tools.schemas import (
    CalculatorArgs,
    SampleLookupArgs,
    ToolDefinition,
)

# ---------------------------------------------------------------------------
# Harmless Demonstration Handlers (Zero Side Effects)
# ---------------------------------------------------------------------------

SAMPLE_RECORDS: Dict[str, Dict[str, Any]] = {
    "item_001": {"name": "Solar Inverter A1", "category": "Hardware", "status": "In Stock"},
    "item_002": {"name": "Thermal Sensor TX", "category": "Sensors", "status": "Low Stock"},
    "item_003": {"name": "Lithium Battery Pack 5kWh", "category": "Storage", "status": "Available"},
}


def calculator_handler(args: CalculatorArgs) -> Dict[str, Any]:
    """Execute safe arithmetic calculation in memory."""
    op = args.operation
    a = args.a
    b = args.b

    if op == "add":
        res = a + b
    elif op == "subtract":
        res = a - b
    elif op == "multiply":
        res = a * b
    elif op == "divide":
        if b == 0:
            raise ValueError("Division by zero is undefined.")
        res = a / b
    else:
        raise ValueError(f"Unsupported operation: {op}")

    return {"operation": op, "a": a, "b": b, "result": res}


def sample_lookup_handler(args: SampleLookupArgs) -> Dict[str, Any]:
    """Execute fixed mock data lookup from static in-memory dictionary."""
    rec_id = args.record_id
    record = SAMPLE_RECORDS.get(rec_id)
    if record is None:
        return {"record_id": rec_id, "found": False, "record": None}
    return {"record_id": rec_id, "found": True, "record": record}


# ---------------------------------------------------------------------------
# Tool Registry Class & Singleton Management
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Thread-safe in-memory tool allowlist registry."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._lock = threading.RLock()
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[[Any], Any]] = {}

        if register_defaults:
            self.register_default_tools()

    def register_default_tools(self) -> None:
        """Register the harmless, deterministic demonstration tools."""
        self.register_tool(
            ToolDefinition(
                name="calculator",
                description="Performs safe mathematical arithmetic (add, subtract, multiply, divide).",
                category="READ_ONLY",
                requires_confirmation=False,
                args_schema=CalculatorArgs,
            ),
            handler=calculator_handler,
        )
        self.register_tool(
            ToolDefinition(
                name="sample_data_lookup",
                description="Look up fixed demonstration hardware inventory records by record ID.",
                category="READ_ONLY",
                requires_confirmation=False,
                args_schema=SampleLookupArgs,
            ),
            handler=sample_lookup_handler,
        )

    def register_tool(
        self,
        definition: ToolDefinition,
        handler: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Register an allowed tool specification and its optional execution handler."""
        with self._lock:
            self._tools[definition.name] = definition
            if handler is not None:
                self._handlers[definition.name] = handler

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        with self._lock:
            removed_tool = self._tools.pop(name, None)
            self._handlers.pop(name, None)
            return removed_tool is not None

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve tool definition if registered."""
        with self._lock:
            return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check whether a tool name is registered in the allowlist."""
        with self._lock:
            return name in self._tools

    def get_handler(self, name: str) -> Optional[Callable[[Any], Any]]:
        """Retrieve execution handler if registered."""
        with self._lock:
            return self._handlers.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Return list of all registered tool definitions."""
        with self._lock:
            return list(self._tools.values())

    def clear(self) -> None:
        """Remove all tools and handlers from the registry."""
        with self._lock:
            self._tools.clear()
            self._handlers.clear()


_registry_lock = threading.RLock()
_default_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Return the global ToolRegistry singleton."""
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = ToolRegistry(register_defaults=True)
        return _default_registry


def reset_tool_registry() -> None:
    """Reset the global ToolRegistry singleton to its fresh default state."""
    global _default_registry
    with _registry_lock:
        _default_registry = ToolRegistry(register_defaults=True)
