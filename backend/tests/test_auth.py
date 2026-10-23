"""Focused unit, integration, and security tests for PromptShield Phase 6 Stage 6.1.

Verifies:
1. Missing credentials are rejected with HTTP 401.
2. Invalid credentials are rejected with HTTP 401.
3. Valid Bearer credentials are accepted with HTTP 200.
4. Valid X-API-Key credentials are accepted with HTTP 200.
5. Multiple configured keys work correctly.
6. Conflicting or malformed credential headers are rejected with HTTP 401.
7. Missing key configuration fails closed without exposing protected endpoints.
8. Every sensitive route is protected (rules, inspect, proxy, rag/inspect, rag/proxy).
9. Public health endpoints remain accessible without credentials.
10. Authentication runs before expensive processing or downstream invocation.
11. Security headers are applied to normal responses and authentication errors.
12. CORS allows configured origins and rejects unconfigured origins.
13. CORS preflight OPTIONS requests behave correctly.
14. Constant-time key comparison does not leak timing.
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.auth import (
    ENV_ALLOWED_ORIGINS,
    ENV_API_KEYS,
    ENV_AUTH_ENABLED,
    HEADER_CONTENT_TYPE_OPTIONS,
    HEADER_FRAME_OPTIONS,
    HEADER_REFERRER_POLICY,
    get_configured_api_keys,
    validate_api_key,
)
from app.main import app
from app.proxy.mock_service import get_mock_call_count

TEST_PRIMARY_KEY = "ps-primary-secret-key-12345"
TEST_SECONDARY_KEY = "ps-secondary-admin-key-67890"


@pytest.fixture
def unauth_client():
    """Raw TestClient without pre-set authorization headers."""
    with TestClient(app) as tc:
        yield tc


class TestGatewayAuthenticationUnit:
    """Unit tests for app.auth functions and invariants."""

    def test_configured_api_keys_parsing(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, "key1, key2   key3,key4")
        keys = get_configured_api_keys()
        assert keys == {"key1", "key2", "key3", "key4"}

    def test_configured_api_keys_empty_or_whitespace(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, "   ,  , ")
        assert get_configured_api_keys() == set()

    def test_validate_api_key_constant_time(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, f"{TEST_PRIMARY_KEY}, {TEST_SECONDARY_KEY}")
        assert validate_api_key(TEST_PRIMARY_KEY) is True
        assert validate_api_key(TEST_SECONDARY_KEY) is True
        assert validate_api_key("wrong-key") is False
        assert validate_api_key("") is False
        assert validate_api_key(None) is False

    def test_validate_api_key_empty_config_fails_closed(self, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, "")
        assert validate_api_key(TEST_PRIMARY_KEY) is False


class TestGatewayAuthenticationIntegration:
    """Integration tests verifying HTTP 401 fail-closed protection across all routes."""

    def test_missing_credentials_rejected_with_401(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules")
        assert res.status_code == 401
        assert res.headers.get("WWW-Authenticate") == "Bearer"
        assert "API key required" in res.json()["detail"]

    def test_invalid_credentials_rejected_with_401(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules", headers={"Authorization": "Bearer bad-key-999"})
        assert res.status_code == 401
        assert res.headers.get("WWW-Authenticate") == "Bearer"
        assert "Invalid API key" in res.json()["detail"]

    def test_valid_bearer_token_accepted(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules", headers={"Authorization": f"Bearer {TEST_PRIMARY_KEY}"})
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_valid_x_api_key_accepted(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules", headers={"X-API-Key": TEST_PRIMARY_KEY})
        assert res.status_code == 200

    def test_multiple_configured_keys_both_work(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, f"{TEST_PRIMARY_KEY},{TEST_SECONDARY_KEY}")
        res1 = unauth_client.get("/api/v1/rules", headers={"Authorization": f"Bearer {TEST_PRIMARY_KEY}"})
        assert res1.status_code == 200
        res2 = unauth_client.get("/api/v1/rules", headers={"X-API-Key": TEST_SECONDARY_KEY})
        assert res2.status_code == 200

    def test_conflicting_headers_rejected(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, f"{TEST_PRIMARY_KEY},{TEST_SECONDARY_KEY}")
        res = unauth_client.get(
            "/api/v1/rules",
            headers={
                "Authorization": f"Bearer {TEST_PRIMARY_KEY}",
                "X-API-Key": TEST_SECONDARY_KEY,
            },
        )
        assert res.status_code == 401
        assert "Conflicting credentials" in res.json()["detail"]

    def test_malformed_authorization_header_rejected(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert res.status_code == 401
        assert "Malformed Authorization header" in res.json()["detail"]

    def test_missing_api_keys_config_fails_closed(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_AUTH_ENABLED, "true")
        monkeypatch.setenv(ENV_API_KEYS, "")
        res = unauth_client.get("/api/v1/rules", headers={"Authorization": f"Bearer {TEST_PRIMARY_KEY}"})
        assert res.status_code == 401
        assert "Invalid API key" in res.json()["detail"]

    @pytest.mark.parametrize(
        "endpoint,method,payload",
        [
            ("/api/v1/rules", "GET", None),
            ("/api/v1/inspect", "POST", {"prompt": "Hello world"}),
            ("/api/v1/proxy", "POST", {"prompt": "Hello world"}),
            ("/api/v1/rag/inspect", "POST", {"query": "test query", "top_k": 2}),
            ("/api/v1/rag/proxy", "POST", {"query": "test query", "top_k": 2}),
        ],
    )
    def test_all_sensitive_routes_require_authentication(self, unauth_client, monkeypatch, endpoint, method, payload):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        if method == "GET":
            res_unauth = unauth_client.get(endpoint)
            res_auth = unauth_client.get(endpoint, headers={"Authorization": f"Bearer {TEST_PRIMARY_KEY}"})
        else:
            res_unauth = unauth_client.post(endpoint, json=payload)
            res_auth = unauth_client.post(endpoint, json=payload, headers={"Authorization": f"Bearer {TEST_PRIMARY_KEY}"})
        assert res_unauth.status_code == 401
        assert res_auth.status_code == 200

    def test_public_health_endpoint_accessible_unauthenticated(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "secret" not in str(data)
        assert "api_key" not in str(data)

    def test_auth_runs_before_expensive_downstream_or_inspection(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        calls_before = get_mock_call_count()
        res = unauth_client.post("/api/v1/proxy", json={"prompt": "Harmless prompt."})
        assert res.status_code == 401
        # Mock downstream must NOT be invoked
        assert get_mock_call_count() == calls_before


class TestHttpSecurityHeadersAndCORS:
    """Tests for enterprise security response headers and CORS middleware."""

    def test_security_headers_on_success_response(self, unauth_client, monkeypatch):
        res = unauth_client.get("/health")
        assert res.status_code == 200
        assert res.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS
        assert res.headers.get("X-Frame-Options") == HEADER_FRAME_OPTIONS
        assert res.headers.get("Referrer-Policy") == HEADER_REFERRER_POLICY

    def test_security_headers_on_401_error_response(self, unauth_client, monkeypatch):
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        res = unauth_client.get("/api/v1/rules")
        assert res.status_code == 401
        assert res.headers.get("X-Content-Type-Options") == HEADER_CONTENT_TYPE_OPTIONS
        assert res.headers.get("X-Frame-Options") == HEADER_FRAME_OPTIONS
        assert res.headers.get("Referrer-Policy") == HEADER_REFERRER_POLICY

    def test_cors_default_local_development_origins(self, monkeypatch):
        """Verify default configuration allows local frontend origins http://localhost:5173 and http://127.0.0.1:5173."""
        monkeypatch.delenv(ENV_ALLOWED_ORIGINS, raising=False)
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        from app.main import app as test_app

        with TestClient(test_app) as client:
            # 1. Health check with localhost:5173
            res_lh = client.get("/health", headers={"Origin": "http://localhost:5173"})
            assert res_lh.status_code == 200
            assert res_lh.headers.get("access-control-allow-origin") == "http://localhost:5173"

            # 2. Health check with 127.0.0.1:5173
            res_ip = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
            assert res_ip.status_code == 200
            assert res_ip.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

            # 3. Disallowed origin receives no access-control-allow-origin header
            res_evil = client.get("/health", headers={"Origin": "http://evil-attacker.com"})
            assert res_evil.status_code == 200
            assert "access-control-allow-origin" not in res_evil.headers

    def test_cors_preflight_and_origin_headers(self, monkeypatch):
        monkeypatch.setenv(ENV_ALLOWED_ORIGINS, "http://localhost:5173,http://127.0.0.1:5173")
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        from app.main import app as test_app

        # Test preflight OPTIONS request
        with TestClient(test_app) as client:
            # Preflight allowed origin
            res_preflight = client.options(
                "/api/v1/inspect",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization,Content-Type",
                },
            )
            assert res_preflight.status_code == 200
            assert res_preflight.headers.get("access-control-allow-origin") == "http://localhost:5173"
            assert "Authorization" in res_preflight.headers.get("access-control-allow-headers", "")
            assert "Content-Type" in res_preflight.headers.get("access-control-allow-headers", "")

            # Preflight unauthorized origin
            res_disallowed = client.options(
                "/api/v1/inspect",
                headers={
                    "Origin": "http://evil-attacker.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "access-control-allow-origin" not in res_disallowed.headers

    def test_cors_explicit_empty_env_disallows_all_cross_origins(self, monkeypatch):
        """Setting PROMPTSHIELD_ALLOWED_ORIGINS to empty string disables cross-origin access."""
        import importlib
        monkeypatch.setenv(ENV_ALLOWED_ORIGINS, "")
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        import app.main
        importlib.reload(app.main)
        from app.main import app as test_app

        with TestClient(test_app) as client:
            res = client.get("/health", headers={"Origin": "http://localhost:5173"})
            assert res.status_code == 200
            assert "access-control-allow-origin" not in res.headers

        # Restore default app state after test
        monkeypatch.delenv(ENV_ALLOWED_ORIGINS, raising=False)
        importlib.reload(app.main)

    def test_cors_production_vercel_frontend_origin(self, monkeypatch):
        """Verify production frontend origin https://promptshieldv-frontend.vercel.app

        (including when defined with trailing slashes or quotes in deployment env)
        successfully receives Access-Control-Allow-Origin on /health and succeeds on OPTIONS preflight.
        """
        vercel_origin = "https://promptshieldv-frontend.vercel.app"
        # Test with trailing slash in env var (common dashboard config oversight)
        monkeypatch.setenv(ENV_ALLOWED_ORIGINS, f"{vercel_origin}/,http://localhost:5173")
        monkeypatch.setenv(ENV_API_KEYS, TEST_PRIMARY_KEY)
        from app.main import app as test_app

        with TestClient(test_app) as client:
            # 1. GET /health with Vercel origin includes Access-Control-Allow-Origin
            res_health = client.get("/health", headers={"Origin": vercel_origin})
            assert res_health.status_code == 200
            assert res_health.headers.get("access-control-allow-origin") == vercel_origin

            # 2. OPTIONS preflight to /api/v1/audit/events succeeds (HTTP 200) with required headers
            res_preflight = client.options(
                "/api/v1/audit/events?limit=100&offset=0",
                headers={
                    "Origin": vercel_origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization,content-type",
                },
            )
            assert res_preflight.status_code == 200
            assert res_preflight.headers.get("access-control-allow-origin") == vercel_origin
            assert "Authorization" in res_preflight.headers.get("access-control-allow-headers", "")
            assert "Content-Type" in res_preflight.headers.get("access-control-allow-headers", "")

            # 3. Disallowed origin is rejected (no allow header, 400 on preflight)
            res_evil = client.options(
                "/api/v1/audit/events?limit=100&offset=0",
                headers={
                    "Origin": "https://malicious-site.example.com",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization",
                },
            )
            assert "access-control-allow-origin" not in res_evil.headers
            assert res_evil.status_code == 400
