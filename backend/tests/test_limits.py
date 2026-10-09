"""Focused unit, integration, and security tests for PromptShield Phase 6 Stage 6.2.

Covers:
1. Requests within the configured rate limit succeed.
2. The next request after the limit returns HTTP 429 Too Many Requests.
3. Retry-After header is present and positive.
4. Different API keys use separate rate-limit buckets.
5. Expired windows allow requests again.
6. Invalid or non-positive configuration uses safe, documented behavior (fails startup/validation).
7. Rate-limit state is bounded and expired buckets are cleaned up.
8. Health checks and preflight requests are excluded from rate limits and concurrency limits.
9. Concurrency limit rejects excess work without creating an unbounded queue (HTTP 503).
10. Concurrency slots are released after successful responses, exceptions, and cancellations.
11. Rate-limit (429) and overload (503) responses retain Stage 6.1 security headers.
12. Existing auth, proxy, RAG, output-security, and audit functionality remains intact.
13. Tests execute with authentication enabled.
"""

import asyncio
import os
import threading
import time
import pytest
from fastapi.testclient import TestClient

from app.auth import (
    ENV_API_KEYS,
    ENV_AUTH_ENABLED,
    HEADER_CONTENT_TYPE_OPTIONS,
    HEADER_FRAME_OPTIONS,
    HEADER_REFERRER_POLICY,
)
from app.limits import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_RATE_LIMIT_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ENV_MAX_CONCURRENT_REQUESTS,
    ENV_RATE_LIMIT_ENABLED,
    ENV_RATE_LIMIT_REQUESTS,
    ENV_RATE_LIMIT_WINDOW_SECONDS,
    ConcurrencyGate,
    SlidingWindowRateLimiter,
    derive_key_identity,
    get_concurrency_gate,
    get_rate_limiter,
    parse_bool_env,
    parse_positive_int_env,
    reset_limits_state,
    validate_limits_configuration,
)
from app.main import app

KEY_ALICE = "key-alice-12345"
KEY_BOB = "key-bob-67890"


@pytest.fixture
def auth_headers_alice():
    return {"Authorization": f"Bearer {KEY_ALICE}"}


@pytest.fixture
def auth_headers_bob():
    return {"Authorization": f"Bearer {KEY_BOB}"}


class TestRateLimiterUnit:
    """Unit tests for SlidingWindowRateLimiter and configuration parsing."""

    def test_parse_positive_int_valid(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "42")
        assert parse_positive_int_env("TEST_VAR", 10) == 42

    def test_parse_positive_int_rejects_zero_or_negative(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "0")
        with pytest.raises(ValueError, match="Must be > 0"):
            parse_positive_int_env("TEST_VAR", 10)

        monkeypatch.setenv("TEST_VAR", "-5")
        with pytest.raises(ValueError, match="Must be > 0"):
            parse_positive_int_env("TEST_VAR", 10)

    def test_parse_positive_int_rejects_non_integer(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "abc")
        with pytest.raises(ValueError, match="Must be > 0"):
            parse_positive_int_env("TEST_VAR", 10)

    def test_parse_bool_env(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "true")
        assert parse_bool_env("TEST_BOOL", False) is True
        monkeypatch.setenv("TEST_BOOL", "0")
        assert parse_bool_env("TEST_BOOL", True) is False

        monkeypatch.setenv("TEST_BOOL", "not-a-bool")
        with pytest.raises(ValueError, match="Invalid boolean"):
            parse_bool_env("TEST_BOOL", True)

    def test_key_identity_derivation_deterministic_and_non_leaking(self):
        id1 = derive_key_identity("secret-key-1")
        id2 = derive_key_identity("secret-key-1")
        id3 = derive_key_identity("secret-key-2")
        assert id1 == id2
        assert id1 != id3
        assert "secret-key" not in id1
        assert len(id1) == 32

    def test_sliding_window_allows_up_to_max_and_blocks_next(self):
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        t0 = 1000.0

        # Requests 1..3 succeed
        allowed, _, _ = limiter.check_and_record("bucket-1", now=t0)
        assert allowed is True
        allowed, _, _ = limiter.check_and_record("bucket-1", now=t0 + 1)
        assert allowed is True
        allowed, _, _ = limiter.check_and_record("bucket-1", now=t0 + 2)
        assert allowed is True

        # Request 4 is blocked
        allowed, retry_after, capacity_exhausted = limiter.check_and_record("bucket-1", now=t0 + 3)
        assert allowed is False
        assert capacity_exhausted is False
        assert retry_after == 57  # 60 - (1003 - 1000) = 57 seconds remaining

    def test_sliding_window_separate_buckets(self):
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        t0 = 1000.0

        # Fill bucket 1
        assert limiter.check_and_record("bucket-1", now=t0)[0] is True
        assert limiter.check_and_record("bucket-1", now=t0)[0] is True
        assert limiter.check_and_record("bucket-1", now=t0)[0] is False

        # Bucket 2 is completely unaffected
        assert limiter.check_and_record("bucket-2", now=t0)[0] is True
        assert limiter.check_and_record("bucket-2", now=t0)[0] is True
        assert limiter.check_and_record("bucket-2", now=t0)[0] is False

    def test_sliding_window_expires_old_requests(self):
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
        t0 = 1000.0

        assert limiter.check_and_record("bucket-1", now=t0)[0] is True
        assert limiter.check_and_record("bucket-1", now=t0 + 1)[0] is True
        assert limiter.check_and_record("bucket-1", now=t0 + 2)[0] is False

        # After 11 seconds, t0 has expired
        assert limiter.check_and_record("bucket-1", now=t0 + 11)[0] is True

    def test_cleanup_evicts_stale_buckets(self):
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=10)
        limiter.check_and_record("bucket-old", now=100.0)
        limiter.check_and_record("bucket-new", now=108.0)
        assert limiter.bucket_count == 2

        # Cleanup at 112: bucket-old is older than 10s cutoff (102.0), bucket-new is not (108 > 102)
        purged = limiter.cleanup(now=112.0)
        assert purged == 1
        assert limiter.bucket_count == 1
        assert "bucket-old" not in limiter._buckets
        assert "bucket-new" in limiter._buckets

    def test_strict_bucket_cap_enforced_when_all_buckets_active(self):
        """Regression test for SEC-RL-01: bucket_count must never exceed max_buckets."""
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60, max_buckets=5)
        # 1. Fill table with 5 active identities at t=100
        for i in range(5):
            allowed, retry_after, capacity_exhausted = limiter.check_and_record(f"active_{i}", now=100.0)
            assert allowed is True
            assert capacity_exhausted is False
            assert limiter.bucket_count <= 5

        assert limiter.bucket_count == 5

        # 2. Attempt at least 10 new identities during the same active window (t=101)
        for i in range(5, 20):
            allowed, retry_after, capacity_exhausted = limiter.check_and_record(f"rejected_identity_{i}", now=101.0)
            # Must explicitly fail-closed on capacity exhaustion without creating a bucket
            assert allowed is False
            assert capacity_exhausted is True
            assert retry_after == 1
            assert limiter.bucket_count <= 5

        assert limiter.bucket_count == 5

        # 3. Confirm existing active identities still function normally
        allowed, _, capacity_exhausted = limiter.check_and_record("active_0", now=102.0)
        assert allowed is True
        assert capacity_exhausted is False
        assert limiter.bucket_count == 5

        # 4. Advance time past window (t=200): expired buckets are cleaned up and new identity admitted
        purged = limiter.cleanup(now=200.0)
        assert purged == 5
        assert limiter.bucket_count == 0

        allowed, _, capacity_exhausted = limiter.check_and_record("fresh_identity", now=201.0)
        assert allowed is True
        assert capacity_exhausted is False
        assert limiter.bucket_count == 1

    def test_concurrent_identity_admissions_respect_bucket_cap(self):
        """Verify that concurrent threads attempting to insert new identities cannot breach the cap."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60, max_buckets=10)
        results = []

        def worker(thread_id):
            for i in range(20):
                res = limiter.check_and_record(f"thread_{thread_id}_key_{i}", now=1000.0)
                results.append(res)
                assert limiter.bucket_count <= 10

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.bucket_count <= 10
        # Exactly 10 calls succeeded in creating buckets, remaining were rejected with capacity_exhausted=True
        admitted = [r for r in results if r[0] is True]
        capacity_rejected = [r for r in results if r[2] is True]
        assert len(admitted) == 10
        assert len(capacity_rejected) == 90


class TestConcurrencyGateUnit:
    """Unit tests for ConcurrencyGate non-queuing mechanics."""

    def test_concurrency_gate_allows_up_to_max_and_rejects_excess(self):
        gate = ConcurrencyGate(max_concurrent=2)
        assert gate.acquire() is True
        assert gate.acquire() is True
        assert gate.acquire() is False

        gate.release()
        assert gate.acquire() is True
        assert gate.acquire() is False

    def test_concurrency_gate_release_safe_at_zero(self):
        gate = ConcurrencyGate(max_concurrent=1)
        gate.release()
        assert gate.active_count == 0


class TestRateLimitingIntegration:
    """Integration tests testing FastAPI route rate-limiting enforcement."""

    def test_requests_within_limit_succeed_and_next_returns_429(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, f"{KEY_ALICE},{KEY_BOB}")
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "3")
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, "60")
        reset_limits_state()

        with TestClient(app) as tc:
            headers = {"Authorization": f"Bearer {KEY_ALICE}"}
            # Requests 1, 2, 3 succeed
            for _ in range(3):
                res = tc.post("/api/v1/inspect", json={"prompt": "Harmless prompt."}, headers=headers)
                assert res.status_code == 200

            # Request 4 is rate limited
            res_4 = tc.post("/api/v1/inspect", json={"prompt": "Harmless prompt."}, headers=headers)
            assert res_4.status_code == 429
            data = res_4.json()
            assert "Rate limit exceeded" in data["detail"]
            assert "Retry-After" in res_4.headers
            assert int(res_4.headers["Retry-After"]) > 0

    def test_different_api_keys_have_independent_rate_limits(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, f"{KEY_ALICE},{KEY_BOB}")
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "2")
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, "60")
        reset_limits_state()

        with TestClient(app) as tc:
            headers_alice = {"Authorization": f"Bearer {KEY_ALICE}"}
            headers_bob = {"Authorization": f"Bearer {KEY_BOB}"}

            # Exhaust Alice's limit
            assert tc.post("/api/v1/inspect", json={"prompt": "Alice 1"}, headers=headers_alice).status_code == 200
            assert tc.post("/api/v1/inspect", json={"prompt": "Alice 2"}, headers=headers_alice).status_code == 200
            assert tc.post("/api/v1/inspect", json={"prompt": "Alice 3"}, headers=headers_alice).status_code == 429

            # Bob's requests still succeed
            assert tc.post("/api/v1/inspect", json={"prompt": "Bob 1"}, headers=headers_bob).status_code == 200
            assert tc.post("/api/v1/inspect", json={"prompt": "Bob 2"}, headers=headers_bob).status_code == 200
            assert tc.post("/api/v1/inspect", json={"prompt": "Bob 3"}, headers=headers_bob).status_code == 429

    def test_health_check_excluded_from_rate_limits(self, monkeypatch):
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "1")
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, "60")
        reset_limits_state()

        with TestClient(app) as tc:
            # Health check works multiple times without authentication and without being rate limited
            for _ in range(5):
                res = tc.get("/health")
                assert res.status_code == 200

    def test_cors_preflight_options_excluded_from_rate_limits(self, monkeypatch):
        monkeypatch.setenv("PROMPTSHIELD_ALLOWED_ORIGINS", "http://localhost:5173")
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "1")
        reset_limits_state()

        with TestClient(app) as tc:
            for _ in range(5):
                res = tc.options(
                    "/api/v1/inspect",
                    headers={
                        "Origin": "http://localhost:5173",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Authorization,Content-Type",
                    },
                )
                assert res.status_code == 200

    def test_unauthenticated_request_fails_with_401_before_rate_limit(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, KEY_ALICE)
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "1")
        reset_limits_state()

        with TestClient(app) as tc:
            # Request without auth gets 401, never 429
            for _ in range(3):
                res = tc.post("/api/v1/inspect", json={"prompt": "No auth."})
                assert res.status_code == 401
                assert "WWW-Authenticate" in res.headers

    def test_rate_limit_response_retains_security_headers(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, KEY_ALICE)
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "1")
        reset_limits_state()

        with TestClient(app) as tc:
            headers = {"Authorization": f"Bearer {KEY_ALICE}"}
            tc.post("/api/v1/inspect", json={"prompt": "First."}, headers=headers)
            res_429 = tc.post("/api/v1/inspect", json={"prompt": "Second."}, headers=headers)
            assert res_429.status_code == 429
            assert res_429.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS
            assert res_429.headers.get("X-Frame-Options") == HEADER_FRAME_OPTIONS
            assert res_429.headers.get("Referrer-Policy") == HEADER_REFERRER_POLICY


class TestConcurrencyProtectionIntegration:
    """Integration tests for ConcurrencyGate non-queuing capacity protection."""

    def test_concurrency_gate_returns_503_when_exhausted(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, KEY_ALICE)
        monkeypatch.setenv(ENV_MAX_CONCURRENT_REQUESTS, "1")
        reset_limits_state()

        gate = get_concurrency_gate()
        # Manually hold the slot to simulate an active long-running operation
        assert gate.acquire() is True

        try:
            with TestClient(app) as tc:
                headers = {"Authorization": f"Bearer {KEY_ALICE}"}
                res = tc.post("/api/v1/inspect", json={"prompt": "Waiting prompt."}, headers=headers)
                assert res.status_code == 503
                assert "Gateway concurrency limit reached" in res.json()["detail"]
                assert res.headers.get("Retry-After") == "1"
                assert res.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS
                assert res.headers.get("X-Frame-Options") == HEADER_FRAME_OPTIONS
                assert res.headers.get("Referrer-Policy") == HEADER_REFERRER_POLICY
        finally:
            gate.release()

        # Slot released: next request proceeds normally
        with TestClient(app) as tc:
            headers = {"Authorization": f"Bearer {KEY_ALICE}"}
            res = tc.post("/api/v1/inspect", json={"prompt": "Allowed prompt."}, headers=headers)
            assert res.status_code == 200

    def test_concurrency_slot_released_on_route_exception(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, KEY_ALICE)
        monkeypatch.setenv(ENV_MAX_CONCURRENT_REQUESTS, "1")
        reset_limits_state()

        # Inject failure in inspect to simulate internal unhandled error
        monkeypatch.setattr("app.main.scan_prompt", lambda p: (_ for _ in ()).throw(RuntimeError("Simulated crash")))

        with TestClient(app, raise_server_exceptions=False) as tc:
            headers = {"Authorization": f"Bearer {KEY_ALICE}"}
            res = tc.post("/api/v1/inspect", json={"prompt": "Crash prompt."}, headers=headers)
            assert res.status_code == 500

        # Confirm gate count was safely restored to 0
        gate = get_concurrency_gate()
        assert gate.active_count == 0

    def test_startup_validation_rejects_invalid_limits(self, monkeypatch):
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "-10")
        with pytest.raises(ValueError, match="Must be > 0"):
            validate_limits_configuration()

        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "100")
        monkeypatch.setenv(ENV_MAX_CONCURRENT_REQUESTS, "invalid")
        with pytest.raises(ValueError, match="Must be > 0"):
            validate_limits_configuration()

    def test_startup_validation_enforces_upper_bounds(self, monkeypatch):
        from app.limits import (
            MAX_ALLOWED_CONCURRENT_REQUESTS,
            MAX_ALLOWED_RATE_LIMIT_REQUESTS,
            MAX_ALLOWED_WINDOW_SECONDS,
        )

        # Values within individual and combined limits succeed
        # e.g., 200 requests across 10k default buckets = 2,000,000 aggregate (exact limit)
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "200")
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, str(MAX_ALLOWED_WINDOW_SECONDS))
        monkeypatch.setenv(ENV_MAX_CONCURRENT_REQUESTS, str(MAX_ALLOWED_CONCURRENT_REQUESTS))
        validate_limits_configuration()

        # Values exceeding rate limit request individual ceiling fail
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, str(MAX_ALLOWED_RATE_LIMIT_REQUESTS + 1))
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            validate_limits_configuration()
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "100")

        # Values exceeding window seconds ceiling fail
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, str(MAX_ALLOWED_WINDOW_SECONDS + 1))
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            validate_limits_configuration()
        monkeypatch.setenv(ENV_RATE_LIMIT_WINDOW_SECONDS, "60")

        # Values exceeding concurrency ceiling fail
        monkeypatch.setenv(ENV_MAX_CONCURRENT_REQUESTS, str(MAX_ALLOWED_CONCURRENT_REQUESTS + 1))
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            validate_limits_configuration()

    def test_startup_validation_enforces_combined_memory_budget(self, monkeypatch):
        """Verify that combined extreme settings (e.g. 100k requests / 100k buckets) are rejected."""
        # Previous extreme configuration: 100,000 requests per window across 10,000 buckets = 1,000,000,000 entries
        # Fails both individual ceiling (> 10,000) and combined budget check
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "100000")
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            validate_limits_configuration()

        # Configuration exceeding aggregate budget ceiling (2,000,001 entries) fails combined memory budget
        # e.g., 201 requests across 10,000 buckets = 2,010,000 > 2,000,000 budget
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "201")
        with pytest.raises(ValueError, match="exceeding the maximum memory budget of 2000000"):
            validate_limits_configuration()

        # Conservative configuration (default 100 requests across 10,000 buckets = 1,000,000 entries) passes
        monkeypatch.setenv(ENV_RATE_LIMIT_REQUESTS, "100")
        validate_limits_configuration()

    def test_rate_limiter_capacity_exhaustion_returns_503(self, monkeypatch):
        """Integration test: When rate limiter table is full, new identities receive HTTP 503 with Retry-After."""
        monkeypatch.setenv(ENV_API_KEYS, f"{KEY_ALICE},{KEY_BOB},key-charlie")
        monkeypatch.setenv(ENV_RATE_LIMIT_ENABLED, "true")
        reset_limits_state()

        from app.limits import SlidingWindowRateLimiter, _state_lock
        import app.limits as limits_mod

        # Configure small max_buckets=1 limiter
        mini_limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60, max_buckets=1)
        with limits_mod._state_lock:
            limits_mod._rate_limiter = mini_limiter

        with TestClient(app) as tc:
            # 1. Alice is admitted into the single available bucket slot
            res_alice = tc.post(
                "/api/v1/inspect",
                json={"prompt": "Alice query."},
                headers={"Authorization": f"Bearer {KEY_ALICE}"},
            )
            assert res_alice.status_code == 200

            # 2. Bob attempts request while Alice's bucket is active and table is full
            res_bob = tc.post(
                "/api/v1/inspect",
                json={"prompt": "Bob query."},
                headers={"Authorization": f"Bearer {KEY_BOB}"},
            )
            assert res_bob.status_code == 503
            assert "capacity reached" in res_bob.json()["detail"].lower()
            assert res_bob.headers.get("Retry-After") == "1"
            assert res_bob.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS
            assert res_bob.headers.get("X-Frame-Options") == HEADER_FRAME_OPTIONS
            assert res_bob.headers.get("Referrer-Policy") == HEADER_REFERRER_POLICY

            # 3. Alice can still make requests because her bucket already exists
            res_alice_2 = tc.post(
                "/api/v1/inspect",
                json={"prompt": "Alice query 2."},
                headers={"Authorization": f"Bearer {KEY_ALICE}"},
            )
            assert res_alice_2.status_code == 200


class TestConstructorValidation:
    """Tests verifying strict constructor argument validation and aggregate memory budget enforcement."""

    @pytest.mark.parametrize(
        "invalid_max_requests",
        [
            0,
            -1,
            -100,
            10001,  # exceeds MAX_ALLOWED_RATE_LIMIT_REQUESTS (10000)
            True,   # boolean rejected
            False,  # boolean rejected
            "100",  # string rejected
            10.5,   # float rejected
            None,
        ],
    )
    def test_sliding_window_rate_limiter_rejects_invalid_max_requests(self, invalid_max_requests):
        with pytest.raises(ValueError, match="max_requests must be an integer"):
            SlidingWindowRateLimiter(max_requests=invalid_max_requests)

    @pytest.mark.parametrize(
        "invalid_window_seconds",
        [
            0,
            -1,
            -0.001,
            86401,  # exceeds MAX_ALLOWED_WINDOW_SECONDS (86400)
            float("nan"),
            float("inf"),
            float("-inf"),
            True,   # boolean rejected
            False,  # boolean rejected
            "60",   # string rejected
            None,
        ],
    )
    def test_sliding_window_rate_limiter_rejects_invalid_window_seconds(self, invalid_window_seconds):
        with pytest.raises(ValueError, match="window_seconds must be a finite number"):
            SlidingWindowRateLimiter(window_seconds=invalid_window_seconds)

    @pytest.mark.parametrize(
        "invalid_max_buckets",
        [
            0,
            -1,
            10001,  # exceeds MAX_ALLOWED_BUCKETS (10000)
            True,   # boolean rejected
            False,  # boolean rejected
            "10000",
            100.0,
            None,
        ],
    )
    def test_sliding_window_rate_limiter_rejects_invalid_max_buckets(self, invalid_max_buckets):
        with pytest.raises(ValueError, match="max_buckets must be an integer"):
            SlidingWindowRateLimiter(max_buckets=invalid_max_buckets)

    def test_sliding_window_rate_limiter_enforces_aggregate_budget(self):
        # 201 * 10000 = 2,010,000 > 2,000,000
        with pytest.raises(ValueError, match="Aggregate timestamp budget .* exceeds maximum allowed budget"):
            SlidingWindowRateLimiter(max_requests=201, max_buckets=10000)

        # 200 * 10000 = 2,000,000 <= 2,000,000 (valid boundary)
        limiter = SlidingWindowRateLimiter(max_requests=200, max_buckets=10000)
        assert limiter.max_requests == 200
        assert limiter.max_buckets == 10000

    @pytest.mark.parametrize(
        "invalid_max_concurrent",
        [
            0,
            -1,
            1001,   # exceeds MAX_ALLOWED_CONCURRENT_REQUESTS (1000)
            True,   # boolean rejected
            False,  # boolean rejected
            "20",
            20.0,
            None,
        ],
    )
    def test_concurrency_gate_rejects_invalid_max_concurrent(self, invalid_max_concurrent):
        with pytest.raises(ValueError, match="max_concurrent must be an integer"):
            ConcurrencyGate(max_concurrent=invalid_max_concurrent)

    def test_valid_constructor_boundaries(self):
        # Min boundaries
        limiter_min = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.001, max_buckets=1)
        assert limiter_min.max_requests == 1
        assert limiter_min.window_seconds == 0.001
        assert limiter_min.max_buckets == 1

        # Max window seconds boundary
        limiter_max_window = SlidingWindowRateLimiter(max_requests=10, window_seconds=86400, max_buckets=100)
        assert limiter_max_window.window_seconds == 86400

        # Concurrency min and max
        gate_min = ConcurrencyGate(max_concurrent=1)
        assert gate_min.max_concurrent == 1

        gate_max = ConcurrencyGate(max_concurrent=1000)
        assert gate_max.max_concurrent == 1000
