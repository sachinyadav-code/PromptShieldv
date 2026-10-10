"""Authentication and security configuration for PromptShield Security Gateway.

Provides:
- Configurable API key authentication via Authorization: Bearer or X-API-Key.
- Constant-time secret comparison via secrets.compare_digest.
- Fail-closed credential validation rejecting missing, malformed, or invalid keys.
- Explicit production vs development mode configuration (PROMPTSHIELD_AUTH_ENABLED).
- Public endpoint exemptions (e.g. /health) with all sensitive routes protected.
"""

import os
import re
import secrets
from typing import Optional, Set

from fastapi import HTTPException, Request, status

ENV_AUTH_ENABLED = "PROMPTSHIELD_AUTH_ENABLED"
ENV_API_KEYS = "PROMPTSHIELD_API_KEYS"
ENV_ALLOWED_ORIGINS = "PROMPTSHIELD_ALLOWED_ORIGINS"

HEADER_CONTENT_TYPE_OPTIONS = "nosniff"
HEADER_FRAME_OPTIONS = "DENY"
HEADER_REFERRER_POLICY = "strict-origin-when-cross-origin"

PUBLIC_EXEMPT_PATHS: Set[str] = {
    "/health",
}

DOCS_PATHS: Set[str] = {
    "/docs",
    "/openapi.json",
    "/redoc",
}


class AuthenticationError(HTTPException):
    """HTTP 401 Unauthorized exception with WWW-Authenticate header."""

    def __init__(self, detail: str = "Invalid or missing API key."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def is_auth_enabled() -> bool:
    """Check if gateway authentication is enabled.

    Enabled by default (fail-closed secure default).
    Can be explicitly disabled only when PROMPTSHIELD_AUTH_ENABLED is set to false or 0.
    """
    raw = os.getenv(ENV_AUTH_ENABLED, "true").strip().lower()
    return raw not in ("false", "0", "no", "disabled")


def get_configured_api_keys() -> Set[str]:
    """Retrieve the set of valid API keys configured for this gateway.

    Parses comma- or whitespace-separated keys from PROMPTSHIELD_API_KEYS.
    Keys are stripped of surrounding whitespace. Empty entries are discarded.
    """
    raw = os.getenv(ENV_API_KEYS, "")
    if not raw or not raw.strip():
        return set()
    tokens = re.split(r"[\s,]+", raw.strip())
    return {t for t in tokens if t}



def validate_api_key(supplied_key: Optional[str]) -> bool:
    """Validate a candidate API key against configured keys using constant-time comparison.

    Never leaks whether a partial match exists.
    """
    if not supplied_key or not isinstance(supplied_key, str):
        return False

    valid_keys = get_configured_api_keys()
    if not valid_keys:
        return False

    matched = False
    for expected_key in valid_keys:
        if secrets.compare_digest(supplied_key, expected_key):
            matched = True
    return matched


async def authenticate_request(request: Request) -> Optional[str]:
    """Dependency that enforces API key authentication on protected endpoints.

    Checks credentials in:
    1. Authorization: Bearer <key>
    2. X-API-Key: <key>

    Rejects missing, malformed, invalid, or conflicting credentials with HTTP 401.
    Returns the validated key or None if auth is explicitly disabled.
    """
    if not is_auth_enabled():
        return None

    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_EXEMPT_PATHS or path in DOCS_PATHS:
        return None

    bearer_creds = None
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            bearer_creds = parts[1]
        else:
            raise AuthenticationError("Malformed Authorization header. Expected Bearer <key>.")

    x_api_key = request.headers.get("X-API-Key")
    if x_api_key:
        x_api_key = x_api_key.strip()

    if bearer_creds and x_api_key:
        if not secrets.compare_digest(bearer_creds, x_api_key):
            raise AuthenticationError("Conflicting credentials provided in Authorization and X-API-Key headers.")
        candidate_key = bearer_creds
    elif bearer_creds:
        candidate_key = bearer_creds
    elif x_api_key:
        candidate_key = x_api_key
    else:
        raise AuthenticationError("API key required. Provide Authorization: Bearer <key> or X-API-Key.")

    if not validate_api_key(candidate_key):
        raise AuthenticationError("Invalid API key.")

    return candidate_key


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_allowed_origins() -> list[str]:
    """Parse configured CORS allowed origins from PROMPTSHIELD_ALLOWED_ORIGINS.

    Defaults to local development frontend origins:
    http://localhost:5173, http://127.0.0.1:5173.
    """
    raw = os.getenv(ENV_ALLOWED_ORIGINS)
    if raw is None:
        return list(DEFAULT_ALLOWED_ORIGINS)
    raw = raw.strip()
    if not raw:
        return []
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins
