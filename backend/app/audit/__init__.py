"""Audit package for structured security event logging."""

from app.audit.logger import (
    clear_audit_events,
    get_audit_events,
    log_output_security_event,
    log_rag_security_event,
    log_security_event,
    log_tool_security_event,
)
from app.audit.storage import (
    AuditStorageError,
    PostgresAuditStore,
    SQLiteAuditStore,
    get_audit_db_path,
    get_audit_store,
    reset_audit_store,
)

__all__ = [
    "AuditStorageError",
    "PostgresAuditStore",
    "SQLiteAuditStore",
    "clear_audit_events",
    "get_audit_db_path",
    "get_audit_events",
    "get_audit_store",
    "log_output_security_event",
    "log_rag_security_event",
    "log_security_event",
    "log_tool_security_event",
    "reset_audit_store",
]
