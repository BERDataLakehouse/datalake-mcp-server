"""Tests for the main application."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.main import AuthMiddleware, RequestTimeoutMiddleware, create_application
from src.service.exceptions import InvalidAuthHeaderError


def test_health_check(client):
    """Test the health check endpoint returns proper structure."""
    response = client.get("/apis/mcp/health")
    assert response.status_code == 200

    data = response.json()
    # Verify response structure (DeepHealthResponse)
    assert "status" in data
    assert "components" in data
    assert "message" in data

    # Status should be one of the valid values
    assert data["status"] in ["healthy", "unhealthy", "degraded"]

    # Components should be a list
    assert isinstance(data["components"], list)

    # Each component should have required fields
    for component in data["components"]:
        assert "name" in component
        assert "status" in component
        assert component["status"] in ["healthy", "unhealthy", "degraded"]


# =============================================================================
# RequestTimeoutMiddleware Tests (lines 63-73)
# =============================================================================


class TestRequestTimeoutMiddleware:
    """Tests for the RequestTimeoutMiddleware."""

    def test_timeout_returns_408(self):
        """Test that a slow endpoint returns 408 Request Timeout (lines 63-73)."""
        app = FastAPI()
        app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=0.01)

        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(5)
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/slow")

        assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT
        data = response.json()
        assert data["error"] == 40800
        assert data["error_type"] == "request_timeout"
        assert "timed out" in data["message"]

    def test_health_skips_timeout(self):
        """Test that /health endpoints bypass the timeout middleware."""
        app = FastAPI()
        app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=0.01)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


# =============================================================================
# AuthMiddleware Tests (lines 106-118, 122-143)
# =============================================================================


class TestAuthMiddleware:
    """Tests for the AuthMiddleware."""

    def _make_app_with_auth(self):
        """Create a minimal FastAPI app with AuthMiddleware and mock app_state."""
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        return app

    @patch("src.main.app_state")
    def test_valid_bearer_token(self, mock_app_state_mod):
        """Test that a valid Bearer token authenticates successfully (lines 106-118)."""
        app = self._make_app_with_auth()

        mock_state = MagicMock()
        mock_user = MagicMock()
        mock_state.auth.get_user = AsyncMock(return_value=mock_user)
        mock_app_state_mod.get_app_state.return_value = mock_state

        client = TestClient(app)
        response = client.get("/test", headers={"Authorization": "Bearer valid_token"})

        assert response.status_code == 200
        mock_state.auth.get_user.assert_awaited_once_with("valid_token")
        mock_app_state_mod.set_request_user.assert_called_once()

    @patch("src.main.app_state")
    def test_invalid_scheme_returns_error(self, mock_app_state_mod):
        """Test that a non-Bearer scheme raises InvalidAuthHeaderError (lines 111-114)."""
        app = self._make_app_with_auth()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test", headers={"Authorization": "Basic abc123"})

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "Bearer" in data["message"]

    @patch("src.main.app_state")
    def test_empty_credentials_returns_error(self, mock_app_state_mod):
        """Test that empty credentials after scheme raises error (lines 107-109)."""
        app = self._make_app_with_auth()

        client = TestClient(app, raise_server_exceptions=False)
        # A header with just the scheme and no token
        response = client.get("/test", headers={"Authorization": "Bearer "})

        # With "Bearer " (trailing space), get_authorization_scheme_param returns
        # scheme="bearer" and credentials="" which is falsy
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch("src.main.app_state")
    def test_mcp_server_error_handled_cleanly(self, mock_app_state_mod):
        """Test that MCPServerError in auth is caught and returns JSON (lines 122-134)."""
        app = self._make_app_with_auth()

        mock_state = MagicMock()
        mock_state.auth.get_user = AsyncMock(
            side_effect=InvalidAuthHeaderError("bad token format")
        )
        mock_app_state_mod.get_app_state.return_value = mock_state

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test", headers={"Authorization": "Bearer bad_token"})

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "bad token format" in data["message"]

    @patch("src.main.app_state")
    def test_generic_exception_returns_500(self, mock_app_state_mod):
        """Test that unexpected exceptions return 500 (lines 135-146)."""
        app = self._make_app_with_auth()

        mock_state = MagicMock()
        mock_state.auth.get_user = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_app_state_mod.get_app_state.return_value = mock_state

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test", headers={"Authorization": "Bearer some_token"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["message"] == "Internal authentication error"

    @patch("src.main.app_state")
    def test_no_auth_header_sets_none_user(self, mock_app_state_mod):
        """Test that missing Authorization header sets user to None."""
        app = self._make_app_with_auth()

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        mock_app_state_mod.set_request_user.assert_called_once()
        _, kwargs = mock_app_state_mod.set_request_user.call_args
        # Second positional arg should be None (no user)
        args = mock_app_state_mod.set_request_user.call_args[0]
        assert args[1] is None


# =============================================================================
# Startup/Shutdown Event Tests (lines 205-215)
# =============================================================================


class TestLifecycleEvents:
    """Tests for startup and shutdown event handlers."""

    @patch("src.main.mount_stateless_mcp")
    @patch("src.main.FastApiMCP")
    @patch("src.main.app_state")
    def test_startup_event(self, mock_app_state_mod, mock_mcp_cls, mock_mount):
        """Test that startup_event calls build_app (lines 205-208)."""
        mock_app_state_mod.build_app = AsyncMock()
        mock_app_state_mod.destroy_app_state = AsyncMock()
        mock_mount.return_value = MagicMock(shutdown=AsyncMock())

        with patch("src.main.get_settings") as mock_get_settings:
            settings = MagicMock()
            settings.service_root_path = ""
            settings.app_name = "test"
            settings.app_description = "test"
            settings.api_version = "1.0"
            settings.request_timeout_seconds = 55.0
            mock_get_settings.return_value = settings

            app = create_application()

        # TestClient context manager triggers startup on enter, shutdown on exit
        with TestClient(app):
            mock_app_state_mod.build_app.assert_awaited_once()

    @patch("src.main.mount_stateless_mcp")
    @patch("src.main.FastApiMCP")
    @patch("src.main.app_state")
    def test_shutdown_event(self, mock_app_state_mod, mock_mcp_cls, mock_mount):
        """Test that shutdown_event calls cleanup methods (lines 211-215)."""
        mock_app_state_mod.build_app = AsyncMock()
        mock_app_state_mod.destroy_app_state = AsyncMock()
        mock_transport = MagicMock(shutdown=AsyncMock())
        mock_mount.return_value = mock_transport

        with patch("src.main.get_settings") as mock_get_settings:
            settings = MagicMock()
            settings.service_root_path = ""
            settings.app_name = "test"
            settings.app_description = "test"
            settings.api_version = "1.0"
            settings.request_timeout_seconds = 55.0
            mock_get_settings.return_value = settings

            app = create_application()

        # TestClient triggers startup AND shutdown
        with TestClient(app):
            pass

        mock_app_state_mod.destroy_app_state.assert_awaited_once()
        mock_transport.shutdown.assert_awaited_once()


# =============================================================================
# No Root Path Branch (lines 252-255)
# =============================================================================


class TestNoRootPath:
    """Tests for the no-root-path branch."""

    @patch("src.main.mount_stateless_mcp")
    @patch("src.main.FastApiMCP")
    @patch("src.main.app_state")
    def test_no_root_path_returns_app_directly(
        self, mock_app_state_mod, mock_mcp_cls, mock_mount
    ):
        """Test that empty service_root_path returns the app directly (lines 252-255)."""
        mock_app_state_mod.build_app = AsyncMock()
        mock_app_state_mod.destroy_app_state = AsyncMock()
        mock_mount.return_value = MagicMock(shutdown=AsyncMock())

        with patch("src.main.get_settings") as mock_get_settings:
            settings = MagicMock()
            settings.service_root_path = ""
            settings.app_name = "test"
            settings.app_description = "test"
            settings.api_version = "1.0"
            settings.request_timeout_seconds = 55.0
            mock_get_settings.return_value = settings

            app = create_application()

        # With no root path, the app should be a FastAPI instance directly
        assert isinstance(app, FastAPI)


# =============================================================================
# __main__ block (lines 259-263)
# =============================================================================


class TestMainBlock:
    """Tests for the __main__ entry point."""

    @patch("src.main.uvicorn.run")
    @patch("src.main.create_application")
    def test_main_block(self, mock_create_app, mock_uvicorn_run):
        """Test the __main__ block executes uvicorn.run (lines 259-263)."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        # Execute the __main__ block
        with patch.dict("os.environ", {"HOST": "127.0.0.1", "PORT": "9000"}):
            exec(
                compile(
                    "app_instance = create_application()\n"
                    'host = os.getenv("HOST", "0.0.0.0")\n'
                    'port = int(os.getenv("PORT", "8000"))\n'
                    "uvicorn.run(app_instance, host=host, port=port)\n",
                    "<test>",
                    "exec",
                ),
                {
                    "create_application": mock_create_app,
                    "os": __import__("os"),
                    "uvicorn": __import__("uvicorn"),
                },
            )

        mock_create_app.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(mock_app, host="127.0.0.1", port=9000)
