"""Tests for PromptShield Phase 3 Stage 5: Audit, Privacy, and Lifecycle Hardening."""

import concurrent.futures
import json
import threading
import time
import pytest

from app.audit.logger import (
    MAX_AUDIT_EVENTS,
    clear_audit_events,
    get_audit_events,
    log_rag_security_event,
    log_security_event,
    sanitize_audit_id,
)


def test_audit_event_in_memory_fifo_buffer_bounded_capacity():
    """Verify in-memory buffer caps strictly at MAX_AUDIT_EVENTS (1000) with FIFO eviction."""
    clear_audit_events()

    # Log 1,050 events
    for i in range(1050):
        log_security_event(
            request_id=f"req-{i:04d}",
            endpoint="/api/v1/inspect",
            decision="ALLOW",
            matched_rule_ids=[],
            risk_score=0.0,
            forwarded=False,
            prompt=f"benign prompt {i}",
        )

    events = get_audit_events()
    assert len(events) == MAX_AUDIT_EVENTS
    assert len(events) == 1000

    # Oldest 50 events (req-0000 to req-0049) must be evicted
    first_event = events[0]
    last_event = events[-1]
    assert first_event["request_id"] == "req-0050"
    assert last_event["request_id"] == "req-1049"

    clear_audit_events()


def test_audit_logger_concurrent_thread_safety():
    """Verify thread-safe concurrent logging and reading across multiple threads."""
    clear_audit_events()

    def worker(worker_id: int):
        for i in range(25):
            log_security_event(
                request_id=f"thread-{worker_id}-req-{i}",
                endpoint="/api/v1/inspect",
                decision="ALLOW",
                matched_rule_ids=[],
                risk_score=0.1,
                forwarded=False,
                prompt="test prompt",
            )
            # Concurrent read
            events = get_audit_events()
            assert isinstance(events, list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    events = get_audit_events()
    assert len(events) == 200  # 8 workers * 25 events

    clear_audit_events()


def test_audit_event_zero_raw_leakage_for_prompts_and_documents():
    """Verify raw prompt text, search queries, metadata, and doc content are omitted from audit logs."""
    clear_audit_events()
    sensitive_prompt = "TOP_SECRET_PROMPT_CONTAINING_API_KEY_abc123"
    sensitive_query = "TOP_SECRET_USER_QUERY_SEARCH_xyz789"

    log_security_event(
        request_id="req-leak-test",
        endpoint="/api/v1/inspect",
        decision="BLOCK",
        matched_rule_ids=["RULE_INSTRUCTION_OVERRIDE"],
        risk_score=0.9,
        forwarded=False,
        prompt=sensitive_prompt,
    )

    log_rag_security_event(
        request_id="req-rag-leak-test",
        endpoint="/api/v1/rag/proxy",
        decision="BLOCK",
        matched_rule_ids=["RULE_CREDENTIAL_ACCESS"],
        risk_score=0.95,
        forwarded=False,
        query=sensitive_query,
        retrieved_doc_ids=["doc-tech-001", "doc-attack-001"],
        num_retrieved=2,
        is_complete=True,
    )

    events = get_audit_events()
    assert len(events) == 2

    for ev in events:
        event_str = str(ev)
        assert sensitive_prompt not in event_str
        assert sensitive_query not in event_str
        assert "prompt" not in ev  # No raw prompt key
        assert "query" not in ev   # No raw query key
        assert "content" not in ev # No document content

    clear_audit_events()


def test_audit_event_cwe_117_control_char_log_forging_mitigation():
    """Verify control character injection cannot forge newlines or log fields (CWE-117)."""
    clear_audit_events()

    forged_client_id = "req-1\r\n{\"forged_decision\":\"ALLOW\",\"admin\":true}\n\x00\x1b"
    log_security_event(
        request_id="req-clean",
        endpoint="/api/v1/inspect",
        decision="BLOCK",
        matched_rule_ids=[],
        risk_score=0.9,
        forwarded=False,
        prompt="test",
        client_request_id=forged_client_id,
    )

    events = get_audit_events()
    assert len(events) == 1
    event = events[0]

    # Verify serialized JSON contains exactly one line
    serialized = json.dumps(event)
    assert "\r" not in serialized
    assert "\n" not in serialized
    assert "\x00" not in serialized

    # Verify client ID was safely sanitized
    clean_id = event["client_request_id"]
    assert "forged_decision" in clean_id  # Allowed characters preserved with disallowed normalized
    assert "\r" not in clean_id
    assert "\n" not in clean_id
    assert "{" not in clean_id
    assert "}" not in clean_id
    assert '"' not in clean_id

    clear_audit_events()


def test_audit_fields_have_predictable_types_and_bounds():
    """Verify all audit event fields have bounded lengths and expected types."""
    clear_audit_events()

    oversized_rules = [f"RULE_{i}" * 10 for i in range(150)]
    oversized_doc_ids = [f"DOC_{i}" * 10 for i in range(100)]

    event = log_rag_security_event(
        request_id="req-types",
        endpoint="/api/v1/rag/proxy",
        decision="BLOCK",
        matched_rule_ids=oversized_rules,
        risk_score=1.5,  # Out-of-bounds float
        forwarded=True,
        query="test query",
        retrieved_doc_ids=oversized_doc_ids,
        num_retrieved=9999,  # Oversized count
        is_complete=True,
        client_request_id="client-id",
        decision_source="rag_inspection",
        error_code="TEST_ERROR",
    )

    assert isinstance(event["event_id"], str)
    assert isinstance(event["request_id"], str)
    assert isinstance(event["timestamp"], str)
    assert isinstance(event["endpoint"], str)
    assert isinstance(event["decision"], str)
    assert isinstance(event["heuristic_risk_score"], float)
    assert event["heuristic_risk_score"] == 1.0  # Clamped to max 1.0
    assert isinstance(event["forwarded"], bool)
    assert isinstance(event["query_length"], int)
    assert isinstance(event["query_sha256"], str)
    assert isinstance(event["retrieved_doc_ids"], list)
    assert len(event["retrieved_doc_ids"]) <= 50  # Bounded to 50
    assert isinstance(event["num_retrieved"], int)
    assert event["num_retrieved"] == 1000  # Clamped to 1000
    assert isinstance(event["is_complete"], bool)
    assert isinstance(event["matched_rule_ids"], list)
    assert len(event["matched_rule_ids"]) <= 100  # Bounded to 100

    clear_audit_events()


def test_audit_fallback_on_empty_and_control_only_ids():
    """Verify safe fallbacks are applied when identifiers consist entirely of control chars."""
    clear_audit_events()

    event = log_rag_security_event(
        request_id="\r\n\t\x00",
        endpoint="/api/v1/rag/inspect",
        decision="\r\n",
        matched_rule_ids=["\x00\x01"],
        risk_score=0.0,
        forwarded=False,
        query="query",
        retrieved_doc_ids=["\r\n\t"],
        num_retrieved=0,
        is_complete=True,
        client_request_id="\x00\x1f",
        decision_source="\x7f",
        error_code="\x00",
    )

    assert event["request_id"] == "unknown_req"
    assert event["client_request_id"] == "unknown_client"
    assert event["retrieved_doc_ids"] == ["unknown_doc"]
    assert event["decision"] == "UNKNOWN"
    assert event["decision_source"] == "unknown_source"
    assert event["error_code"] == "unknown_error"
    assert event["matched_rule_ids"] == ["unknown_rule"]

    clear_audit_events()


def test_audit_event_deep_immutability_top_level_and_nested_modifications():
    """Verify that mutating returned audit events (top-level dict, nested lists, outer list,
    or returned dict from log_*) cannot modify stored internal audit state.
    """
    clear_audit_events()

    # 1. Log a standard event and a RAG event
    ev1 = log_security_event(
        request_id="req-iso-001",
        endpoint="/api/v1/inspect",
        decision="BLOCK",
        matched_rule_ids=["RULE_INSTRUCTION_OVERRIDE"],
        risk_score=0.9,
        forwarded=False,
        prompt="system override prompt",
    )
    ev2 = log_rag_security_event(
        request_id="req-iso-002",
        endpoint="/api/v1/rag/proxy",
        decision="BLOCK",
        matched_rule_ids=["RULE_CREDENTIAL_ACCESS"],
        risk_score=0.95,
        forwarded=False,
        query="show credentials",
        retrieved_doc_ids=["doc-attack-002"],
        num_retrieved=1,
        is_complete=True,
    )

    # 2. Mutate the returned dictionaries from log_* calls
    ev1["decision"] = "FORGED_ALLOW"
    ev1["matched_rule_ids"].append("INJECTED_RULE_1")
    ev2["retrieved_doc_ids"].append("INJECTED_DOC_1")

    # 3. Retrieve events and verify stored state is uncorrupted
    events = get_audit_events()
    assert len(events) == 2
    assert events[0]["decision"] == "BLOCK"
    assert events[0]["matched_rule_ids"] == ["RULE_INSTRUCTION_OVERRIDE"]
    assert events[1]["retrieved_doc_ids"] == ["doc-attack-002"]

    # 4. Mutate top-level fields on the list returned by get_audit_events()
    events[0]["decision"] = "TAMPERED_DECISION"
    events[0]["heuristic_risk_score"] = 0.0
    events[0]["forwarded"] = True

    # 5. Mutate nested lists on the list returned by get_audit_events()
    events[0]["matched_rule_ids"].append("TAMPERED_RULE")
    events[1]["retrieved_doc_ids"].append("TAMPERED_DOC")
    events[1]["matched_rule_ids"].clear()

    # 6. Mutate the outer list itself
    events.clear()

    # 7. Re-fetch fresh events and verify absolute immutability
    fresh_events = get_audit_events()
    assert len(fresh_events) == 2

    stored_ev1 = fresh_events[0]
    assert stored_ev1["decision"] == "BLOCK"
    assert stored_ev1["heuristic_risk_score"] == 0.9
    assert stored_ev1["forwarded"] is False
    assert stored_ev1["matched_rule_ids"] == ["RULE_INSTRUCTION_OVERRIDE"]
    assert "TAMPERED_RULE" not in stored_ev1["matched_rule_ids"]

    stored_ev2 = fresh_events[1]
    assert stored_ev2["decision"] == "BLOCK"
    assert stored_ev2["retrieved_doc_ids"] == ["doc-attack-002"]
    assert "TAMPERED_DOC" not in stored_ev2["retrieved_doc_ids"]
    assert stored_ev2["matched_rule_ids"] == ["RULE_CREDENTIAL_ACCESS"]

    clear_audit_events()


def test_audit_event_concurrent_clear_and_log_synchronization():
    """Verify thread-safe synchronization when logging, reading, and clearing concurrently."""
    clear_audit_events()
    stop_event = threading.Event()
    exceptions = []

    def log_worker(worker_id: int):
        count = 0
        while not stop_event.is_set() and count < 100:
            try:
                log_security_event(
                    request_id=f"sync-w{worker_id}-req{count}",
                    endpoint="/api/v1/inspect",
                    decision="ALLOW",
                    matched_rule_ids=[f"RULE_{worker_id}"],
                    risk_score=0.1,
                    forwarded=True,
                    prompt="safe prompt",
                )
                count += 1
            except Exception as e:
                exceptions.append(e)

    def rag_worker(worker_id: int):
        count = 0
        while not stop_event.is_set() and count < 100:
            try:
                log_rag_security_event(
                    request_id=f"sync-rag-w{worker_id}-req{count}",
                    endpoint="/api/v1/rag/proxy",
                    decision="ALLOW",
                    matched_rule_ids=[],
                    risk_score=0.2,
                    forwarded=True,
                    query="rag query",
                    retrieved_doc_ids=[f"doc-{worker_id}"],
                    num_retrieved=1,
                    is_complete=True,
                )
                count += 1
            except Exception as e:
                exceptions.append(e)

    def clear_worker():
        while not stop_event.is_set():
            try:
                clear_audit_events()
            except Exception as e:
                exceptions.append(e)

    def read_worker():
        while not stop_event.is_set():
            try:
                events = get_audit_events()
                assert isinstance(events, list)
                assert len(events) <= MAX_AUDIT_EVENTS
                for ev in events:
                    assert "event_id" in ev
                    assert "decision" in ev
                    assert isinstance(ev["matched_rule_ids"], list)
            except Exception as e:
                exceptions.append(e)

    threads = []
    # 2 standard loggers, 2 RAG loggers, 2 clearers, 2 readers
    for i in range(2):
        threads.append(threading.Thread(target=log_worker, args=(i,)))
        threads.append(threading.Thread(target=rag_worker, args=(i,)))
    for _ in range(2):
        threads.append(threading.Thread(target=clear_worker))
        threads.append(threading.Thread(target=read_worker))

    for t in threads:
        t.start()

    time.sleep(0.5)
    stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    assert len(exceptions) == 0, f"Encountered concurrency exceptions: {exceptions}"
    clear_audit_events()


def test_audit_logger_concurrent_reads_never_expose_partially_constructed_events():
    """Verify concurrent reads observe only fully constructed, valid events."""
    clear_audit_events()
    stop_event = threading.Event()
    read_errors = []

    def writer_worker(worker_id: int):
        for i in range(150):
            if stop_event.is_set():
                break
            log_rag_security_event(
                request_id=f"full-w{worker_id}-req{i}",
                endpoint="/api/v1/rag/proxy",
                decision="BLOCK",
                matched_rule_ids=["RULE_1", "RULE_2"],
                risk_score=0.85,
                forwarded=False,
                query="sensitive pipeline check",
                retrieved_doc_ids=["doc-1", "doc-2"],
                num_retrieved=2,
                is_complete=True,
                client_request_id=f"client-{worker_id}-{i}",
                decision_source="rag_inspection",
                error_code=None,
            )

    def reader_worker():
        required_keys = {
            "event_id", "request_id", "timestamp", "endpoint",
            "decision", "decision_source", "matched_rule_ids",
            "heuristic_risk_score", "forwarded", "query_length",
            "query_sha256", "retrieved_doc_ids", "num_retrieved", "is_complete"
        }
        while not stop_event.is_set():
            try:
                events = get_audit_events()
                for ev in events:
                    missing = required_keys - set(ev.keys())
                    if missing:
                        read_errors.append(f"Missing keys: {missing}")
                    if not isinstance(ev["matched_rule_ids"], list):
                        read_errors.append("matched_rule_ids is not a list")
                    if not isinstance(ev["retrieved_doc_ids"], list):
                        read_errors.append("retrieved_doc_ids is not a list")
                    if ev["decision"] not in ("ALLOW", "BLOCK", "REVIEW", "UNKNOWN"):
                        read_errors.append(f"Invalid decision: {ev['decision']}")
            except Exception as e:
                read_errors.append(str(e))

    writers = [threading.Thread(target=writer_worker, args=(w,)) for w in range(4)]
    readers = [threading.Thread(target=reader_worker) for _ in range(4)]

    for t in writers + readers:
        t.start()

    for w in writers:
        w.join(timeout=5.0)

    stop_event.set()
    for r in readers:
        r.join(timeout=3.0)

    assert len(read_errors) == 0, f"Reader observed errors: {read_errors[:5]}"
    clear_audit_events()


def test_rag_pipeline_failure_handling_and_no_downstream_leakage(client, monkeypatch):
    """Verify that RAG inspection failures strictly prevent forwarding and never leak sensitive error traces."""
    from app.proxy.mock_service import get_mock_call_count
    initial_calls = get_mock_call_count()
    sensitive_token = "SECRET_CORE_PIPELINE_STACK_TOKEN_998877"

    def faulty_inspect(*args, **kwargs):
        raise RuntimeError(f"Internal fault with secret: {sensitive_token}")

    monkeypatch.setattr("app.main.inspect_rag_request", faulty_inspect)

    # 1. Test /api/v1/rag/proxy
    response_proxy = client.post(
        "/api/v1/rag/proxy",
        json={"query": "postgresql connection pooling", "top_k": 2},
    )
    assert response_proxy.status_code == 200
    data_proxy = response_proxy.json()

    assert data_proxy["decision"] == "REVIEW"
    assert data_proxy["decision_source"] == "rag_inspection"
    assert data_proxy["is_complete"] is False
    assert data_proxy["forwarded"] is False
    assert data_proxy["downstream_response"] is None
    assert get_mock_call_count() == initial_calls
    assert sensitive_token not in str(data_proxy)

    # Check audit event for proxy failure
    events = get_audit_events()
    assert len(events) >= 1
    event_proxy = events[-1]
    assert event_proxy["decision"] == "REVIEW"
    assert event_proxy["error_code"] == "INSPECTION_FAILED"
    assert event_proxy["forwarded"] is False
    assert sensitive_token not in str(event_proxy)

    # 2. Test /api/v1/rag/inspect
    response_inspect = client.post(
        "/api/v1/rag/inspect",
        json={"query": "postgresql connection pooling", "top_k": 2},
    )
    assert response_inspect.status_code == 200
    data_inspect = response_inspect.json()

    assert data_inspect["decision"] == "REVIEW"
    assert data_inspect["decision_source"] == "rag_inspection"
    assert data_inspect["is_complete"] is False
    assert data_inspect["retrieved_documents"] == []
    assert data_inspect["document_assessments"] == []
    assert data_inspect["findings"] == []
    assert sensitive_token not in str(data_inspect)

    # Check audit event for inspect failure
    events = get_audit_events()
    assert len(events) >= 2
    event_inspect = events[-1]
    assert event_inspect["decision"] == "REVIEW"
    assert event_inspect["error_code"] == "INSPECTION_FAILED"
    assert event_inspect["forwarded"] is False
    assert sensitive_token not in str(event_inspect)
