"""Persistent SQLite storage for PromptShield security audit events.

Features:
- Parameterized SQL execution exclusively; no string interpolation.
- Safe idempotent schema initialization (WAL mode, busy_timeout=5000ms).
- Thread-safe and process-safe writes via connection pooling / thread-local / per-operation connections.
- Deterministic querying with bounded pagination (default=50, max=100).
- Support for filtering by endpoint, decision, event_type, start_time, and end_time.
- Privacy preservation: Never stores or accepts raw prompts, API keys, headers, or secrets.
- Explicit AuditStorageError handling that prevents internal path or SQL leakage.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore
    ConnectionPool = None  # type: ignore

logger = logging.getLogger("promptshield.audit.storage")

ENV_AUDIT_DB_PATH = "PROMPTSHIELD_AUDIT_DB_PATH"
DEFAULT_AUDIT_DB_PATH = "audit.db"

ENV_DATABASE_URL = "PROMPTSHIELD_DATABASE_URL"
FALLBACK_ENV_DATABASE_URL = "DATABASE_URL"

# Pagination ceilings
DEFAULT_QUERY_LIMIT = 50
MAX_QUERY_LIMIT = 100


class AuditStorageError(Exception):
    """Exception raised when audit storage operations fail."""
    pass


def _safe_json_list(val: Any) -> List[str]:
    """Safely convert database representation (str, list, or None) to List[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        trimmed = val.strip()
        if not trimmed:
            return []
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return []
    return []


def get_audit_db_path() -> Path:
    """Retrieve and resolve the configured audit database path from environment."""
    raw_path = os.getenv(ENV_AUDIT_DB_PATH, DEFAULT_AUDIT_DB_PATH).strip()
    if not raw_path:
        raw_path = DEFAULT_AUDIT_DB_PATH
    return Path(raw_path).resolve()


class SQLiteAuditStore:
    """Thread-safe SQLite persistent audit event store."""

    _SCHEMA_TABLE = """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        client_request_id TEXT,
        timestamp TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        decision TEXT,
        decision_source TEXT,
        event_type TEXT NOT NULL,
        matched_rule_ids TEXT,
        heuristic_risk_score REAL,
        forwarded INTEGER,
        prompt_length INTEGER,
        prompt_sha256 TEXT,
        query_length INTEGER,
        query_sha256 TEXT,
        retrieved_doc_ids TEXT,
        num_retrieved INTEGER,
        is_complete INTEGER,
        error_code TEXT,
        outcome TEXT,
        finding_count INTEGER,
        output_status TEXT,
        output_findings_count INTEGER,
        output_matched_rule_ids TEXT,
        output_error_code TEXT,
        ingress_pipeline_version TEXT,
        ingress_surface TEXT,
        ingress_views_evaluated TEXT,
        ingress_views_count INTEGER,
        ingress_transformation_status TEXT,
        ingress_transformation_error_code TEXT,
        ml_available INTEGER,
        ml_risk_score REAL,
        tool_name TEXT,
        argument_count INTEGER,
        argument_keys TEXT,
        argument_sha256 TEXT,
        created_at TEXT NOT NULL
    );
    """

    _SCHEMA_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp DESC, id DESC);
    CREATE INDEX IF NOT EXISTS idx_audit_events_endpoint ON audit_events(endpoint);
    CREATE INDEX IF NOT EXISTS idx_audit_events_decision ON audit_events(decision);
    CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_audit_events_tool_name ON audit_events(tool_name);
    """

    _SCHEMA = _SCHEMA_TABLE + _SCHEMA_INDEXES

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_audit_db_path()
        self._lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new connection configured for safe concurrent access."""
        # Ensure parent directory exists
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, we manage transactions explicitly
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def initialize_schema(self) -> None:
        """Idempotently initialize database schema."""
        with self._lock:
            try:
                conn = self._get_connection()
                try:
                    conn.executescript(self._SCHEMA_TABLE)
                    # Migrate existing tables missing columns
                    existing_cols = {
                        row["name"]
                        for row in conn.execute("PRAGMA table_info(audit_events);").fetchall()
                    }
                    columns_to_add = [
                        ("ingress_pipeline_version", "TEXT"),
                        ("ingress_surface", "TEXT"),
                        ("ingress_views_evaluated", "TEXT"),
                        ("ingress_views_count", "INTEGER"),
                        ("ingress_transformation_status", "TEXT"),
                        ("ingress_transformation_error_code", "TEXT"),
                        ("ml_available", "INTEGER"),
                        ("ml_risk_score", "REAL"),
                        ("tool_name", "TEXT"),
                        ("argument_count", "INTEGER"),
                        ("argument_keys", "TEXT"),
                        ("argument_sha256", "TEXT"),
                    ]
                    for col_name, col_type in columns_to_add:
                        if col_name not in existing_cols:
                            conn.execute(f"ALTER TABLE audit_events ADD COLUMN {col_name} {col_type};")
                    # Create indexes after schema migrations so indexed columns exist
                    conn.executescript(self._SCHEMA_INDEXES)
                finally:
                    conn.close()
                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize audit database schema: %s", type(e).__name__)
                raise AuditStorageError("Failed to initialize persistent audit storage.") from e

    def save_event(self, event: Dict[str, Any]) -> int:
        """Persist a single sanitized audit event. Returns inserted database row ID."""
        if not self._initialized:
            self.initialize_schema()

        # Extract and format fields safely
        event_id = str(event.get("event_id") or "")
        request_id = str(event.get("request_id") or "")
        client_request_id = event.get("client_request_id")
        timestamp = str(event.get("timestamp") or datetime.now(timezone.utc).isoformat())
        endpoint = str(event.get("endpoint") or "")
        decision = event.get("decision")
        decision_source = event.get("decision_source")
        event_type = str(event.get("event_type") or "security_inspection")

        matched_rules_json = (
            json.dumps(event.get("matched_rule_ids"))
            if event.get("matched_rule_ids") is not None
            else None
        )
        heuristic_risk_score = (
            float(event["heuristic_risk_score"])
            if event.get("heuristic_risk_score") is not None
            else None
        )
        forwarded = (
            1 if event.get("forwarded") is True else (0 if event.get("forwarded") is False else None)
        )
        prompt_length = event.get("prompt_length")
        prompt_sha256 = event.get("prompt_sha256")
        query_length = event.get("query_length")
        query_sha256 = event.get("query_sha256")

        retrieved_docs_json = (
            json.dumps(event.get("retrieved_doc_ids"))
            if event.get("retrieved_doc_ids") is not None
            else None
        )
        num_retrieved = event.get("num_retrieved")
        is_complete = (
            1 if event.get("is_complete") is True else (0 if event.get("is_complete") is False else None)
        )
        error_code = event.get("error_code")
        outcome = event.get("outcome")
        finding_count = event.get("finding_count")

        output_status = event.get("output_status")
        output_findings_count = event.get("output_findings_count")
        output_matched_rules_json = (
            json.dumps(event.get("output_matched_rule_ids"))
            if event.get("output_matched_rule_ids") is not None
            else None
        )
        output_error_code = event.get("output_error_code")

        # Ingress telemetry fields
        ingress_pipeline_version = event.get("ingress_pipeline_version")
        ingress_surface = event.get("ingress_surface")
        ingress_views_evaluated_json = (
            json.dumps(event.get("ingress_views_evaluated"))
            if event.get("ingress_views_evaluated") is not None
            else None
        )
        ingress_views_count = event.get("ingress_views_count")
        ingress_transformation_status = event.get("ingress_transformation_status")
        ingress_transformation_error_code = event.get("ingress_transformation_error_code")
        ml_available = (
            1 if event.get("ml_available") is True else (0 if event.get("ml_available") is False else None)
        )
        ml_risk_score = (
            float(event["ml_risk_score"])
            if event.get("ml_risk_score") is not None
            else None
        )
        # Tool security telemetry fields
        tool_name = event.get("tool_name")
        argument_count = (
            int(event["argument_count"])
            if event.get("argument_count") is not None
            else None
        )
        argument_keys_json = (
            json.dumps(event.get("argument_keys"))
            if event.get("argument_keys") is not None
            else None
        )
        argument_sha256 = event.get("argument_sha256")

        created_at = datetime.now(timezone.utc).isoformat()

        insert_sql = """
        INSERT INTO audit_events (
            event_id, request_id, client_request_id, timestamp, endpoint, decision, decision_source,
            event_type, matched_rule_ids, heuristic_risk_score, forwarded, prompt_length, prompt_sha256,
            query_length, query_sha256, retrieved_doc_ids, num_retrieved, is_complete, error_code,
            outcome, finding_count, output_status, output_findings_count, output_matched_rule_ids,
            output_error_code, ingress_pipeline_version, ingress_surface, ingress_views_evaluated,
            ingress_views_count, ingress_transformation_status, ingress_transformation_error_code,
            ml_available, ml_risk_score, tool_name, argument_count, argument_keys, argument_sha256,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?
        );
        """
        params = (
            event_id, request_id, client_request_id, timestamp, endpoint, decision, decision_source,
            event_type, matched_rules_json, heuristic_risk_score, forwarded, prompt_length, prompt_sha256,
            query_length, query_sha256, retrieved_docs_json, num_retrieved, is_complete, error_code,
            outcome, finding_count, output_status, output_findings_count, output_matched_rules_json,
            output_error_code, ingress_pipeline_version, ingress_surface, ingress_views_evaluated_json,
            ingress_views_count, ingress_transformation_status, ingress_transformation_error_code,
            ml_available, ml_risk_score, tool_name, argument_count, argument_keys_json, argument_sha256,
            created_at,
        )

        with self._lock:
            try:
                conn = self._get_connection()
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    cursor = conn.execute(insert_sql, params)
                    row_id = cursor.lastrowid
                    conn.execute("COMMIT;")
                    return row_id
                except Exception as e:
                    conn.execute("ROLLBACK;")
                    raise e
                finally:
                    conn.close()
            except Exception as exc:
                logger.error("Failed to write audit event to database: %s", type(exc).__name__)
                raise AuditStorageError("Persistent audit write failure.") from exc

    def query_events(
        self,
        limit: int = DEFAULT_QUERY_LIMIT,
        offset: int = 0,
        endpoint: Optional[str] = None,
        decision: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query audit events with deterministic ordering and bounded pagination.

        Returns (events, total_count).
        """
        if not self._initialized:
            self.initialize_schema()

        # Validate pagination bounds
        safe_limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
        safe_offset = max(0, int(offset))

        conditions = []
        params = []

        if endpoint:
            conditions.append("endpoint = ?")
            params.append(endpoint.strip())
        if decision:
            conditions.append("decision = ?")
            params.append(decision.strip().upper())
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.strip())
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.strip())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.strip())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM audit_events {where_clause};"
        select_sql = f"""
        SELECT * FROM audit_events
        {where_clause}
        ORDER BY timestamp DESC, id DESC
        LIMIT ? OFFSET ?;
        """

        with self._lock:
            try:
                conn = self._get_connection()
                try:
                    # 1. Total matching count
                    count_cursor = conn.execute(count_sql, tuple(params))
                    total_count = count_cursor.fetchone()[0]

                    # 2. Select items
                    select_params = list(params) + [safe_limit, safe_offset]
                    select_cursor = conn.execute(select_sql, tuple(select_params))
                    rows = select_cursor.fetchall()

                    events = [self._row_to_dict(r) for r in rows]
                    return events, total_count
                finally:
                    conn.close()
            except Exception as exc:
                logger.error("Failed to query audit events: %s", type(exc).__name__)
                raise AuditStorageError("Audit query failure.") from exc

    def clear(self) -> None:
        """Clear all audit events from the database (primarily used by tests)."""
        if not self._initialized:
            self.initialize_schema()
        with self._lock:
            try:
                conn = self._get_connection()
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    conn.execute("DELETE FROM audit_events;")
                    conn.execute("COMMIT;")
                except Exception as e:
                    conn.execute("ROLLBACK;")
                    raise e
                finally:
                    conn.close()
            except Exception as exc:
                logger.error("Failed to clear audit table: %s", type(exc).__name__)
                raise AuditStorageError("Failed to clear audit store.") from exc

    def count(self) -> int:
        """Count total stored events."""
        if not self._initialized:
            self.initialize_schema()
        with self._lock:
            try:
                conn = self._get_connection()
                try:
                    cursor = conn.execute("SELECT COUNT(*) FROM audit_events;")
                    return cursor.fetchone()[0]
                finally:
                    conn.close()
            except Exception as exc:
                logger.error("Failed to count audit records: %s", type(exc).__name__)
                raise AuditStorageError("Failed to count audit records.") from exc

    def close(self) -> None:
        """Close open resources (no-op for SQLiteAuditStore which uses per-operation connections)."""
        self._initialized = False

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        """Convert a database row into a standardized audit event dictionary matching logger.py."""
        d = dict(row)
        result: Dict[str, Any] = {
            "event_id": d["event_id"],
            "request_id": d["request_id"],
            "timestamp": d["timestamp"],
            "endpoint": d["endpoint"],
            "event_type": d["event_type"],
        }
        if d.get("client_request_id") is not None:
            result["client_request_id"] = d["client_request_id"]
        if d.get("decision") is not None:
            result["decision"] = d["decision"]
        if d.get("decision_source") is not None:
            result["decision_source"] = d["decision_source"]
        if d.get("matched_rule_ids") is not None:
            result["matched_rule_ids"] = _safe_json_list(d["matched_rule_ids"])
        if d.get("heuristic_risk_score") is not None:
            result["heuristic_risk_score"] = float(d["heuristic_risk_score"])
        if d.get("forwarded") is not None:
            result["forwarded"] = bool(d["forwarded"])

        if d.get("prompt_length") is not None:
            result["prompt_length"] = d["prompt_length"]
        if d.get("prompt_sha256") is not None:
            result["prompt_sha256"] = d["prompt_sha256"]

        if d.get("query_length") is not None:
            result["query_length"] = d["query_length"]
        if d.get("query_sha256") is not None:
            result["query_sha256"] = d["query_sha256"]
        if d.get("retrieved_doc_ids") is not None:
            result["retrieved_doc_ids"] = _safe_json_list(d["retrieved_doc_ids"])
        if d.get("num_retrieved") is not None:
            result["num_retrieved"] = d["num_retrieved"]
        if d.get("is_complete") is not None:
            result["is_complete"] = bool(d["is_complete"])

        if d.get("error_code") is not None:
            result["error_code"] = d["error_code"]
        if d.get("outcome") is not None:
            result["outcome"] = d["outcome"]
        if d.get("finding_count") is not None:
            result["finding_count"] = d["finding_count"]

        if d.get("output_status") is not None:
            result["output_status"] = d["output_status"]
        if d.get("output_findings_count") is not None:
            result["output_findings_count"] = d["output_findings_count"]
        if d.get("output_matched_rule_ids") is not None:
            result["output_matched_rule_ids"] = _safe_json_list(d["output_matched_rule_ids"])
        if d.get("output_error_code") is not None:
            result["output_error_code"] = d["output_error_code"]

        if d.get("ingress_pipeline_version") is not None:
            result["ingress_pipeline_version"] = d["ingress_pipeline_version"]
        if d.get("ingress_surface") is not None:
            result["ingress_surface"] = d["ingress_surface"]
        if d.get("ingress_views_evaluated") is not None:
            result["ingress_views_evaluated"] = _safe_json_list(d["ingress_views_evaluated"])
        if d.get("ingress_views_count") is not None:
            result["ingress_views_count"] = int(d["ingress_views_count"])
        if d.get("ingress_transformation_status") is not None:
            result["ingress_transformation_status"] = d["ingress_transformation_status"]
        if d.get("ingress_transformation_error_code") is not None:
            result["ingress_transformation_error_code"] = d["ingress_transformation_error_code"]
        if d.get("ml_available") is not None:
            result["ml_available"] = bool(d["ml_available"])
        if d.get("ml_risk_score") is not None:
            result["ml_risk_score"] = float(d["ml_risk_score"])
        if d.get("tool_name") is not None:
            result["tool_name"] = d["tool_name"]
        if d.get("argument_count") is not None:
            result["argument_count"] = int(d["argument_count"])
        if d.get("argument_keys") is not None:
            result["argument_keys"] = _safe_json_list(d["argument_keys"])
        if d.get("argument_sha256") is not None:
            result["argument_sha256"] = d["argument_sha256"]

        return result


class PostgresAuditStore:
    """Thread-safe PostgreSQL persistent audit event store backed by psycopg 3 connection pool."""

    _SCHEMA_TABLE = """
    CREATE TABLE IF NOT EXISTS audit_events (
        id BIGSERIAL PRIMARY KEY,
        event_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        client_request_id TEXT,
        timestamp TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        decision TEXT,
        decision_source TEXT,
        event_type TEXT NOT NULL,
        matched_rule_ids TEXT,
        heuristic_risk_score DOUBLE PRECISION,
        forwarded INTEGER,
        prompt_length INTEGER,
        prompt_sha256 TEXT,
        query_length INTEGER,
        query_sha256 TEXT,
        retrieved_doc_ids TEXT,
        num_retrieved INTEGER,
        is_complete INTEGER,
        error_code TEXT,
        outcome TEXT,
        finding_count INTEGER,
        output_status TEXT,
        output_findings_count INTEGER,
        output_matched_rule_ids TEXT,
        output_error_code TEXT,
        ingress_pipeline_version TEXT,
        ingress_surface TEXT,
        ingress_views_evaluated TEXT,
        ingress_views_count INTEGER,
        ingress_transformation_status TEXT,
        ingress_transformation_error_code TEXT,
        ml_available INTEGER,
        ml_risk_score DOUBLE PRECISION,
        tool_name TEXT,
        argument_count INTEGER,
        argument_keys TEXT,
        argument_sha256 TEXT,
        created_at TEXT NOT NULL
    );
    """

    _SCHEMA_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events (timestamp DESC, id DESC);
    CREATE INDEX IF NOT EXISTS idx_audit_events_endpoint ON audit_events (endpoint);
    CREATE INDEX IF NOT EXISTS idx_audit_events_decision ON audit_events (decision);
    CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events (event_type);
    CREATE INDEX IF NOT EXISTS idx_audit_events_tool_name ON audit_events (tool_name);
    """

    _SCHEMA = _SCHEMA_TABLE + _SCHEMA_INDEXES

    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 5,
        timeout: float = 10.0,
        max_idle: float = 300.0,
    ):
        if not PSYCOPG_AVAILABLE:
            raise AuditStorageError("PostgreSQL driver (psycopg) is not installed.")

        self._dsn = dsn
        self._timeout = timeout
        self._lock = threading.Lock()
        self._initialized = False
        self._pool = ConnectionPool(
            conninfo=self._dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            max_idle=max_idle,
            open=False,
            kwargs={"prepare_threshold": None},
        )

    def initialize_schema(self) -> None:
        """Idempotently initialize database schema."""
        if not PSYCOPG_AVAILABLE:
            raise AuditStorageError("PostgreSQL driver (psycopg) is not installed.")

        with self._lock:
            if self._initialized:
                return
            try:
                self._pool.open()
                self._pool.wait(timeout=self._timeout)
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(self._SCHEMA_TABLE)
                        # Query existing columns from PostgreSQL information_schema
                        cur.execute(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'audit_events';"
                        )
                        existing_cols = {row[0].lower() for row in cur.fetchall()}
                        columns_to_add = [
                            ("ingress_pipeline_version", "TEXT"),
                            ("ingress_surface", "TEXT"),
                            ("ingress_views_evaluated", "TEXT"),
                            ("ingress_views_count", "INTEGER"),
                            ("ingress_transformation_status", "TEXT"),
                            ("ingress_transformation_error_code", "TEXT"),
                            ("ml_available", "INTEGER"),
                            ("ml_risk_score", "DOUBLE PRECISION"),
                            ("tool_name", "TEXT"),
                            ("argument_count", "INTEGER"),
                            ("argument_keys", "TEXT"),
                            ("argument_sha256", "TEXT"),
                        ]
                        for col_name, col_type in columns_to_add:
                            if col_name not in existing_cols:
                                cur.execute(
                                    f"ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
                                )
                        # Create indexes after schema migrations so indexed columns exist
                        cur.execute(self._SCHEMA_INDEXES)
                    conn.commit()
                self._initialized = True
            except Exception as e:
                try:
                    self._pool.close()
                except Exception:
                    pass
                logger.error("Failed to initialize PostgreSQL audit schema: %s", type(e).__name__)
                raise AuditStorageError("Failed to initialize persistent audit storage.") from e

    def close(self) -> None:
        """Close connection pool cleanly."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as exc:
                    logger.error("Error closing PostgreSQL connection pool: %s", type(exc).__name__)
            self._initialized = False

    def save_event(self, event: Dict[str, Any]) -> int:
        """Persist a single sanitized audit event to PostgreSQL. Returns inserted database row ID."""
        if not self._initialized:
            self.initialize_schema()

        event_id = str(event.get("event_id") or "")
        request_id = str(event.get("request_id") or "")
        client_request_id = event.get("client_request_id")
        timestamp = str(event.get("timestamp") or datetime.now(timezone.utc).isoformat())
        endpoint = str(event.get("endpoint") or "")
        decision = event.get("decision")
        decision_source = event.get("decision_source")
        event_type = str(event.get("event_type") or "security_inspection")

        matched_rules_json = (
            json.dumps(event.get("matched_rule_ids"))
            if event.get("matched_rule_ids") is not None
            else None
        )
        heuristic_risk_score = (
            float(event["heuristic_risk_score"])
            if event.get("heuristic_risk_score") is not None
            else None
        )
        forwarded = (
            1 if event.get("forwarded") is True else (0 if event.get("forwarded") is False else None)
        )
        prompt_length = event.get("prompt_length")
        prompt_sha256 = event.get("prompt_sha256")
        query_length = event.get("query_length")
        query_sha256 = event.get("query_sha256")

        retrieved_docs_json = (
            json.dumps(event.get("retrieved_doc_ids"))
            if event.get("retrieved_doc_ids") is not None
            else None
        )
        num_retrieved = event.get("num_retrieved")
        is_complete = (
            1 if event.get("is_complete") is True else (0 if event.get("is_complete") is False else None)
        )
        error_code = event.get("error_code")
        outcome = event.get("outcome")
        finding_count = event.get("finding_count")

        output_status = event.get("output_status")
        output_findings_count = event.get("output_findings_count")
        output_matched_rules_json = (
            json.dumps(event.get("output_matched_rule_ids"))
            if event.get("output_matched_rule_ids") is not None
            else None
        )
        output_error_code = event.get("output_error_code")

        ingress_pipeline_version = event.get("ingress_pipeline_version")
        ingress_surface = event.get("ingress_surface")
        ingress_views_evaluated_json = (
            json.dumps(event.get("ingress_views_evaluated"))
            if event.get("ingress_views_evaluated") is not None
            else None
        )
        ingress_views_count = event.get("ingress_views_count")
        ingress_transformation_status = event.get("ingress_transformation_status")
        ingress_transformation_error_code = event.get("ingress_transformation_error_code")
        ml_available = (
            1 if event.get("ml_available") is True else (0 if event.get("ml_available") is False else None)
        )
        ml_risk_score = (
            float(event["ml_risk_score"])
            if event.get("ml_risk_score") is not None
            else None
        )
        # Tool security telemetry fields
        tool_name = event.get("tool_name")
        argument_count = (
            int(event["argument_count"])
            if event.get("argument_count") is not None
            else None
        )
        argument_keys_json = (
            json.dumps(event.get("argument_keys"))
            if event.get("argument_keys") is not None
            else None
        )
        argument_sha256 = event.get("argument_sha256")

        created_at = datetime.now(timezone.utc).isoformat()

        insert_sql = """
        INSERT INTO audit_events (
            event_id, request_id, client_request_id, timestamp, endpoint, decision, decision_source,
            event_type, matched_rule_ids, heuristic_risk_score, forwarded, prompt_length, prompt_sha256,
            query_length, query_sha256, retrieved_doc_ids, num_retrieved, is_complete, error_code,
            outcome, finding_count, output_status, output_findings_count, output_matched_rule_ids,
            output_error_code, ingress_pipeline_version, ingress_surface, ingress_views_evaluated,
            ingress_views_count, ingress_transformation_status, ingress_transformation_error_code,
            ml_available, ml_risk_score, tool_name, argument_count, argument_keys, argument_sha256,
            created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s
        ) RETURNING id;
        """
        params = (
            event_id, request_id, client_request_id, timestamp, endpoint, decision, decision_source,
            event_type, matched_rules_json, heuristic_risk_score, forwarded, prompt_length, prompt_sha256,
            query_length, query_sha256, retrieved_docs_json, num_retrieved, is_complete, error_code,
            outcome, finding_count, output_status, output_findings_count, output_matched_rules_json,
            output_error_code, ingress_pipeline_version, ingress_surface, ingress_views_evaluated_json,
            ingress_views_count, ingress_transformation_status, ingress_transformation_error_code,
            ml_available, ml_risk_score, tool_name, argument_count, argument_keys_json, argument_sha256,
            created_at,
        )

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, params)
                    row_id = cur.fetchone()[0]
                conn.commit()
                return int(row_id)
        except Exception as exc:
            logger.error("Failed to write audit event to PostgreSQL: %s", type(exc).__name__)
            raise AuditStorageError("Persistent audit write failure.") from exc

    def query_events(
        self,
        limit: int = DEFAULT_QUERY_LIMIT,
        offset: int = 0,
        endpoint: Optional[str] = None,
        decision: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query audit events from PostgreSQL with deterministic ordering and bounded pagination."""
        if not self._initialized:
            self.initialize_schema()

        safe_limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
        safe_offset = max(0, int(offset))

        conditions = []
        params = []

        if endpoint:
            conditions.append("endpoint = %s")
            params.append(endpoint.strip())
        if decision:
            conditions.append("decision = %s")
            params.append(decision.strip().upper())
        if event_type:
            conditions.append("event_type = %s")
            params.append(event_type.strip())
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time.strip())
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time.strip())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM audit_events {where_clause};"
        select_sql = f"""
        SELECT * FROM audit_events
        {where_clause}
        ORDER BY timestamp DESC, id DESC
        LIMIT %s OFFSET %s;
        """

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(count_sql, tuple(params))
                    total_count = int(cur.fetchone()[0])

                select_params = list(params) + [safe_limit, safe_offset]
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(select_sql, tuple(select_params))
                    rows = cur.fetchall()

                events = [self._row_to_dict(r) for r in rows]
                return events, total_count
        except Exception as exc:
            logger.error("Failed to query audit events from PostgreSQL: %s", type(exc).__name__)
            raise AuditStorageError("Audit query failure.") from exc

    def count(self) -> int:
        """Count total stored events in PostgreSQL."""
        if not self._initialized:
            self.initialize_schema()
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM audit_events;")
                    return int(cur.fetchone()[0])
        except Exception as exc:
            logger.error("Failed to count audit records in PostgreSQL: %s", type(exc).__name__)
            raise AuditStorageError("Failed to count audit records.") from exc

    def clear(self) -> None:
        """Clear all audit events from the PostgreSQL table (used by tests)."""
        if not self._initialized:
            self.initialize_schema()
        with self._lock:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM audit_events;")
                    conn.commit()
            except Exception as exc:
                logger.error("Failed to clear PostgreSQL audit table: %s", type(exc).__name__)
                raise AuditStorageError("Failed to clear audit store.") from exc

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        """Convert a database row into a standardized audit event dictionary matching logger.py."""
        return SQLiteAuditStore._row_to_dict(row)


_store_instance: Optional[Union[SQLiteAuditStore, PostgresAuditStore]] = None
_store_lock = threading.Lock()


def get_audit_store() -> Union[SQLiteAuditStore, PostgresAuditStore]:
    """Get or create singleton audit store (PostgresAuditStore or SQLiteAuditStore)."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                pg_url = os.getenv(ENV_DATABASE_URL) or os.getenv(FALLBACK_ENV_DATABASE_URL)
                if pg_url and pg_url.strip():
                    _store_instance = PostgresAuditStore(pg_url.strip())
                else:
                    _store_instance = SQLiteAuditStore()
    return _store_instance


def reset_audit_store() -> None:
    """Reset the singleton instance (used by test fixtures and shutdown). Closes active connections."""
    global _store_instance
    with _store_lock:
        if _store_instance is not None:
            try:
                _store_instance.close()
            except Exception:
                pass
            _store_instance = None
