"""Rate limiting and concurrency control for PromptShield Security Gateway.

Provides:
- In-memory bounded-sliding-window rate limiting per authenticated API key identity.
- Safe key-bucket derivation via HMAC-SHA256 with an ephemeral process salt.
- Automatic table cleanup of expired buckets to enforce strict memory bounds.
- HTTP 429 Too Many Requests response with standard Retry-After header.
- Concurrency gate for expensive endpoints (/inspect, /proxy, /rag/*).
- HTTP 503 Service Unavailable with Retry-After when processing capacity is exhausted.
- Public route (/health) and CORS preflight (OPTIONS) exemption.
- Thread-safe and async-safe process-local execution.
"""

from collections import deque
import hashlib
import hmac
import logging
import math
import os
import threading
import time
from typing import AsyncGenerator, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status

from app.auth import authenticate_request

logger = logging.getLogger("promptshield.limits")

ENV_RATE_LIMIT_ENABLED = "PROMPTSHIELD_RATE_LIMIT_ENABLED"
ENV_RATE_LIMIT_REQUESTS = "PROMPTSHIELD_RATE_LIMIT_REQUESTS"
ENV_RATE_LIMIT_WINDOW_SECONDS = "PROMPTSHIELD_RATE_LIMIT_WINDOW_SECONDS"
ENV_MAX_CONCURRENT_REQUESTS = "PROMPTSHIELD_MAX_CONCURRENT_REQUESTS"

# Documented Production Defaults
DEFAULT_RATE_LIMIT_ENABLED = True
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_MAX_CONCURRENT_REQUESTS = 20
DEFAULT_MAX_BUCKETS = 10000

# Documented Defensive Upper Bounds (protects against accidental extreme env settings)
MAX_ALLOWED_RATE_LIMIT_REQUESTS = 10000  # 10k requests/window individual ceiling
MAX_ALLOWED_WINDOW_SECONDS = 86400  # 24 hours ceiling to avoid unbounded timestamp retention
MAX_ALLOWED_CONCURRENT_REQUESTS = 1000  # 1,000 concurrent processing ceiling to protect worker threads
MAX_ALLOWED_BUCKETS = 10000  # 10k identities ceiling to prevent process heap exhaustion

# Combined Memory Budget
# Total potential retained timestamps across all active buckets must not exceed 2,000,000.
# At ~8 bytes per float timestamp + deque node overhead, 2M entries consumes < 40 MB heap,
# strictly bounding memory on resource-constrained deployments.
MAX_ALLOWED_AGGREGATE_TIMESTAMPS = 2000000

# Process-level salt for key hashing so derived bucket IDs cannot be inverted across restarts
_KEY_SALT = os.urandom(32)


class RateLimitExceeded(HTTPException):
    """HTTP 429 Too Many Requests exception with Retry-After header."""

    def __init__(self, retry_after: int, detail: str = "Rate limit exceeded. Please retry later."):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )


class CapacityLimitExceeded(HTTPException):
    """HTTP 503 Service Unavailable exception when rate-limiter bucket capacity is exhausted."""

    def __init__(self, retry_after: int = 1, detail: str = "Gateway rate-limiter capacity reached. Please retry later."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )


class ConcurrencyLimitExceeded(HTTPException):
    """HTTP 503 Service Unavailable exception when gateway capacity is exhausted."""

    def __init__(self, retry_after: int = 1, detail: str = "Gateway concurrency limit reached. Please retry later."):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )


def parse_bool_env(env_var: str, default: bool) -> bool:
    """Parse boolean environment variable. Fails closed on invalid non-empty values."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("true", "1", "yes", "enabled"):
        return True
    if val in ("false", "0", "no", "disabled"):
        return False
    raise ValueError(f"Invalid boolean setting for {env_var}: '{raw}'")


def parse_positive_int_env(env_var: str, default: int, max_val: Optional[int] = None) -> int:
    """Parse positive integer environment variable with optional upper bound.

    Rejects non-positive, non-integer, or out-of-range values.
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    val_str = raw.strip()
    try:
        val = int(val_str)
        if val <= 0:
            raise ValueError(f"Invalid positive integer setting for {env_var}: '{raw}'. Must be > 0.")
        if max_val is not None and val > max_val:
            raise ValueError(
                f"Setting for {env_var} exceeds maximum allowed limit of {max_val}: got {val}."
            )
        return val
    except ValueError as e:
        if "exceeds maximum allowed limit" in str(e) or "Must be > 0" in str(e):
            raise
        raise ValueError(f"Invalid positive integer setting for {env_var}: '{raw}'. Must be > 0.")


def derive_key_identity(api_key: str) -> str:
    """Derive a safe bucket identifier from raw API key without storing or leaking the key.

    Uses HMAC-SHA256 with an ephemeral process salt.
    """
    if not api_key:
        return "anonymous"
    h = hmac.new(_KEY_SALT, api_key.encode("utf-8"), hashlib.sha256)
    return h.hexdigest()[:32]


class SlidingWindowRateLimiter:
    """Thread-safe bounded sliding window rate limiter.

    Tracks timestamps per key identity using deques and prunes expired timestamps.
    Automatically evicts expired buckets when table size approaches max_buckets.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        max_buckets: int = DEFAULT_MAX_BUCKETS,
    ):
        if type(max_requests) is not int or max_requests < 1 or max_requests > MAX_ALLOWED_RATE_LIMIT_REQUESTS:
            raise ValueError(
                f"max_requests must be an integer between 1 and {MAX_ALLOWED_RATE_LIMIT_REQUESTS}, got {max_requests!r}"
            )

        if (
            type(window_seconds) not in (int, float)
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
            or window_seconds > MAX_ALLOWED_WINDOW_SECONDS
        ):
            raise ValueError(
                f"window_seconds must be a finite number in (0, {MAX_ALLOWED_WINDOW_SECONDS}], got {window_seconds!r}"
            )

        if type(max_buckets) is not int or max_buckets < 1 or max_buckets > MAX_ALLOWED_BUCKETS:
            raise ValueError(
                f"max_buckets must be an integer between 1 and {MAX_ALLOWED_BUCKETS}, got {max_buckets!r}"
            )

        aggregate = max_requests * max_buckets
        if aggregate > MAX_ALLOWED_AGGREGATE_TIMESTAMPS:
            raise ValueError(
                f"Aggregate timestamp budget {aggregate} (max_requests={max_requests} * max_buckets={max_buckets}) "
                f"exceeds maximum allowed budget of {MAX_ALLOWED_AGGREGATE_TIMESTAMPS}"
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_buckets = max_buckets
        self._buckets: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def check_and_record(
        self, identity: str, now: Optional[float] = None
    ) -> Tuple[bool, int, bool]:
        """Check if request for identity is allowed.

        Returns (allowed: bool, retry_after: int, capacity_exhausted: bool).
        - If allowed: (True, 0, False)
        - If rate limited (per-key limit exceeded): (False, retry_after, False)
        - If capacity exhausted (bucket cap reached and cannot admit new identity): (False, retry_after, True)
        """
        current_time = time.time() if now is None else now
        cutoff = current_time - self.window_seconds

        with self._lock:
            # First attempt cleanup of expired buckets if at or over capacity
            if len(self._buckets) >= self.max_buckets:
                self._cleanup_locked(cutoff)

            timestamps = self._buckets.get(identity)
            if timestamps is None:
                # If still at or over capacity, reject new identity (fail-closed)
                if len(self._buckets) >= self.max_buckets:
                    return False, 1, True
                timestamps = deque()
                self._buckets[identity] = timestamps
            else:
                while timestamps and timestamps[0] <= cutoff:
                    timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                oldest = timestamps[0]
                retry_after = max(1, int(self.window_seconds - (current_time - oldest) + 0.999))
                return False, retry_after, False

            timestamps.append(current_time)
            return True, 0, False

    def cleanup(self, now: Optional[float] = None) -> int:
        """Explicitly purge expired buckets from the table. Returns number of purged buckets."""
        current_time = time.time() if now is None else now
        cutoff = current_time - self.window_seconds
        with self._lock:
            return self._cleanup_locked(cutoff)

    def _cleanup_locked(self, cutoff: float) -> int:
        expired_keys = [
            k for k, q in self._buckets.items()
            if not q or q[-1] <= cutoff
        ]
        for k in expired_keys:
            del self._buckets[k]
        return len(expired_keys)

    def reset(self):
        """Reset internal bucket state."""
        with self._lock:
            self._buckets.clear()

    @property
    def bucket_count(self) -> int:
        with self._lock:
            return len(self._buckets)


class ConcurrencyGate:
    """Thread-safe non-queuing concurrency gate.

    Immediately rejects requests when active concurrent operations reach max_concurrent,
    preventing unbounded waiting queues from building up and exhausting memory.
    """

    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS):
        if type(max_concurrent) is not int or max_concurrent < 1 or max_concurrent > MAX_ALLOWED_CONCURRENT_REQUESTS:
            raise ValueError(
                f"max_concurrent must be an integer between 1 and {MAX_ALLOWED_CONCURRENT_REQUESTS}, got {max_concurrent!r}"
            )

        self.max_concurrent = max_concurrent
        self.active_count = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a concurrency slot without blocking.

        Returns True if acquired, False if at capacity.
        """
        with self._lock:
            if self.active_count >= self.max_concurrent:
                return False
            self.active_count += 1
            return True

    def release(self):
        """Release an acquired concurrency slot."""
        with self._lock:
            if self.active_count > 0:
                self.active_count -= 1

    def reset(self):
        """Reset active count."""
        with self._lock:
            self.active_count = 0


_rate_limiter: Optional[SlidingWindowRateLimiter] = None
_concurrency_gate: Optional[ConcurrencyGate] = None
_state_lock = threading.Lock()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get or create singleton SlidingWindowRateLimiter based on current configuration."""
    global _rate_limiter
    if _rate_limiter is None:
        with _state_lock:
            if _rate_limiter is None:
                max_req = parse_positive_int_env(
                    ENV_RATE_LIMIT_REQUESTS,
                    DEFAULT_RATE_LIMIT_REQUESTS,
                    max_val=MAX_ALLOWED_RATE_LIMIT_REQUESTS,
                )
                window_sec = parse_positive_int_env(
                    ENV_RATE_LIMIT_WINDOW_SECONDS,
                    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
                    max_val=MAX_ALLOWED_WINDOW_SECONDS,
                )
                _rate_limiter = SlidingWindowRateLimiter(
                    max_requests=max_req,
                    window_seconds=window_sec,
                    max_buckets=DEFAULT_MAX_BUCKETS,
                )
    return _rate_limiter


def get_concurrency_gate() -> ConcurrencyGate:
    """Get or create singleton ConcurrencyGate based on current configuration."""
    global _concurrency_gate
    if _concurrency_gate is None:
        with _state_lock:
            if _concurrency_gate is None:
                max_conc = parse_positive_int_env(
                    ENV_MAX_CONCURRENT_REQUESTS,
                    DEFAULT_MAX_CONCURRENT_REQUESTS,
                    max_val=MAX_ALLOWED_CONCURRENT_REQUESTS,
                )
                _concurrency_gate = ConcurrencyGate(max_concurrent=max_conc)
    return _concurrency_gate


def validate_limits_configuration():
    """Validate all limits configuration variables and combined memory budget.

    Called during application lifespan startup.
    Fails startup with ValueError if any setting is non-positive or exceeds individual
    or combined memory ceilings.
    """
    parse_bool_env(ENV_RATE_LIMIT_ENABLED, DEFAULT_RATE_LIMIT_ENABLED)
    max_req = parse_positive_int_env(
        ENV_RATE_LIMIT_REQUESTS,
        DEFAULT_RATE_LIMIT_REQUESTS,
        max_val=MAX_ALLOWED_RATE_LIMIT_REQUESTS,
    )
    parse_positive_int_env(
        ENV_RATE_LIMIT_WINDOW_SECONDS,
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        max_val=MAX_ALLOWED_WINDOW_SECONDS,
    )
    parse_positive_int_env(
        ENV_MAX_CONCURRENT_REQUESTS,
        DEFAULT_MAX_CONCURRENT_REQUESTS,
        max_val=MAX_ALLOWED_CONCURRENT_REQUESTS,
    )

    # Combined Memory Budget Check:
    # Under full capacity saturation, total potential timestamps across all active buckets
    # must not exceed MAX_ALLOWED_AGGREGATE_TIMESTAMPS.
    aggregate_capacity = max_req * DEFAULT_MAX_BUCKETS
    if aggregate_capacity > MAX_ALLOWED_AGGREGATE_TIMESTAMPS:
        raise ValueError(
            f"Configured rate limit requests ({max_req}) across maximum bucket capacity "
            f"({DEFAULT_MAX_BUCKETS}) yields an aggregate potential timestamp retention of "
            f"{aggregate_capacity}, exceeding the maximum memory budget of {MAX_ALLOWED_AGGREGATE_TIMESTAMPS} entries."
        )


def reset_limits_state():
    """Reset rate limiter and concurrency gate singletons and internal states (used for testing)."""
    global _rate_limiter, _concurrency_gate
    with _state_lock:
        if _rate_limiter:
            _rate_limiter.reset()
        if _concurrency_gate:
            _concurrency_gate.reset()
        _rate_limiter = None
        _concurrency_gate = None


async def enforce_rate_limit(
    request: Request,
    api_key: Optional[str] = Depends(authenticate_request),
) -> None:
    """FastAPI dependency that enforces per-key rate limiting.

    Dependencies:
    - Runs AFTER authenticate_request so unauthenticated requests are rejected with 401 first.
    - If rate limiting is disabled, passes through.
    - If bucket table capacity is exhausted and cannot admit a new identity, raises CapacityLimitExceeded (HTTP 503).
    - If per-key limit is exceeded, raises RateLimitExceeded (HTTP 429).
    """
    if not parse_bool_env(ENV_RATE_LIMIT_ENABLED, DEFAULT_RATE_LIMIT_ENABLED):
        return

    identity = derive_key_identity(api_key or "unauthenticated")
    limiter = get_rate_limiter()
    allowed, retry_after, capacity_exhausted = limiter.check_and_record(identity)
    if not allowed:
        if capacity_exhausted:
            raise CapacityLimitExceeded(retry_after=retry_after)
        raise RateLimitExceeded(retry_after=retry_after)


async def enforce_concurrency_limit() -> AsyncGenerator[None, None]:
    """FastAPI dependency generator that enforces concurrency protection.

    Non-queuing: immediately returns 503 if capacity is exhausted.
    Guarantees slot release on success, exception, or client disconnect via try/finally.
    """
    gate = get_concurrency_gate()
    if not gate.acquire():
        raise ConcurrencyLimitExceeded(retry_after=1)
    try:
        yield
    finally:
        gate.release()
