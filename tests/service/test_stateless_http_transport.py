"""
Tests for the stateless HTTP transport module.

Tests cover:
- StatelessHttpTransport class initialization and configuration
- Session manager lifecycle (lazy startup, shutdown)
- Request handling and response conversion
- Error handling scenarios
- mount_stateless_mcp function for FastAPI integration
- Path normalization
- Concurrent request handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI, HTTPException

from src.service.stateless_http_transport import (
    StatelessHttpTransport,
    mount_stateless_mcp,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP Server instance."""
    server = MagicMock()
    server.name = "TestMCPServer"
    return server


@pytest.fixture
def mock_event_store():
    """Create a mock EventStore instance."""
    return MagicMock()


@pytest.fixture
def mock_session_manager():
    """Create a mock StreamableHTTPSessionManager."""
    manager = MagicMock()

    # Mock the async context manager for run()
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=None)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    manager.run.return_value = async_cm

    # Mock handle_request
    manager.handle_request = AsyncMock()

    return manager


@pytest.fixture
def mock_fastapi_mcp():
    """Create a mock FastApiMCP instance."""
    mcp = MagicMock()
    mcp.fastapi = FastAPI()
    mcp.server = MagicMock()
    mcp.server.name = "TestMCPServer"
    return mcp


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.method = "POST"
    request.url.path = "/mcp"
    request.scope = {"type": "http", "method": "POST", "path": "/mcp"}
    request.receive = AsyncMock()
    return request


@pytest.fixture
async def transport_with_cleanup(mock_mcp_server):
    """Create a transport that will be properly cleaned up after test."""
    transports = []

    def factory(**kwargs):
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server, **kwargs)
        transports.append(transport)
        return transport

    yield factory

    # Cleanup all created transports
    for transport in transports:
        await transport.shutdown()


# =============================================================================
# Test StatelessHttpTransport Initialization
# =============================================================================


class TestStatelessHttpTransportInit:
    """Tests for StatelessHttpTransport initialization."""

    def test_init_with_defaults(self, mock_mcp_server):
        """Test initialization with default parameters."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        assert transport.mcp_server is mock_mcp_server
        assert transport.event_store is None
        assert transport.json_response is True
        assert transport._session_manager is None
        assert transport._manager_task is None
        assert transport._manager_started is False

    def test_init_with_custom_event_store(self, mock_mcp_server, mock_event_store):
        """Test initialization with custom event store."""
        transport = StatelessHttpTransport(
            mcp_server=mock_mcp_server,
            event_store=mock_event_store,
        )

        assert transport.event_store is mock_event_store

    def test_init_with_json_response_false(self, mock_mcp_server):
        """Test initialization with json_response=False."""
        transport = StatelessHttpTransport(
            mcp_server=mock_mcp_server,
            json_response=False,
        )

        assert transport.json_response is False

    def test_init_creates_startup_lock(self, mock_mcp_server):
        """Test that initialization creates an asyncio lock."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        assert isinstance(transport._startup_lock, asyncio.Lock)


# =============================================================================
# Test Session Manager Startup
# =============================================================================


class TestEnsureSessionManagerStarted:
    """Tests for _ensure_session_manager_started method."""

    @pytest.mark.asyncio
    async def test_starts_session_manager_on_first_call(self, mock_mcp_server):
        """Test that session manager is started on first call."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            # Setup mock
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            await transport._ensure_session_manager_started()

            # Verify session manager was created with stateless=True
            MockManager.assert_called_once_with(
                app=mock_mcp_server,
                event_store=None,
                json_response=True,
                stateless=True,
            )
            assert transport._manager_started is True
            assert transport._session_manager is mock_manager

    @pytest.mark.asyncio
    async def test_idempotent_when_already_started(self, mock_mcp_server):
        """Test that repeated calls don't restart the manager."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            # Call multiple times
            await transport._ensure_session_manager_started()
            await transport._ensure_session_manager_started()
            await transport._ensure_session_manager_started()

            # Should only be created once
            MockManager.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_event_store_to_manager(
        self, mock_mcp_server, mock_event_store
    ):
        """Test that event store is passed to session manager."""
        transport = StatelessHttpTransport(
            mcp_server=mock_mcp_server,
            event_store=mock_event_store,
        )

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            await transport._ensure_session_manager_started()

            MockManager.assert_called_once_with(
                app=mock_mcp_server,
                event_store=mock_event_store,
                json_response=True,
                stateless=True,
            )

    @pytest.mark.asyncio
    async def test_passes_json_response_false_to_manager(self, mock_mcp_server):
        """Test that json_response=False is passed to session manager."""
        transport = StatelessHttpTransport(
            mcp_server=mock_mcp_server,
            json_response=False,
        )

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            await transport._ensure_session_manager_started()

            MockManager.assert_called_once_with(
                app=mock_mcp_server,
                event_store=None,
                json_response=False,
                stateless=True,
            )


# =============================================================================
# Test Request Handling
# =============================================================================


class TestHandleRequest:
    """Tests for handle_request method."""

    @pytest.mark.asyncio
    async def test_handle_request_success(self, mock_mcp_server, mock_request):
        """Test successful request handling."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            # Mock handle_request to simulate ASGI response
            async def mock_handle_request(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"result": "success"}',
                    }
                )

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.status_code == 200
            assert response.body == b'{"result": "success"}'
            assert response.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_handle_request_with_404_status(self, mock_mcp_server, mock_request):
        """Test request handling with non-200 status code."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"Not Found",
                    }
                )

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.status_code == 404
            assert response.body == b"Not Found"

    @pytest.mark.asyncio
    async def test_handle_request_multiple_body_chunks(
        self, mock_mcp_server, mock_request
    ):
        """Test request handling with multiple body chunks."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [],
                    }
                )
                # Send body in multiple chunks
                await send({"type": "http.response.body", "body": b"chunk1"})
                await send({"type": "http.response.body", "body": b"chunk2"})
                await send({"type": "http.response.body", "body": b"chunk3"})

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.body == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_handle_request_session_manager_not_initialized(
        self, mock_mcp_server, mock_request
    ):
        """Test that HTTPException is raised if session manager not initialized."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)
        transport._manager_started = True  # Pretend started but no manager
        transport._session_manager = None

        with pytest.raises(HTTPException) as exc_info:
            await transport.handle_request(mock_request)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Session manager not initialized"

    @pytest.mark.asyncio
    async def test_handle_request_exception_raises_http_exception(
        self, mock_mcp_server, mock_request
    ):
        """Test that exceptions in session manager raise HTTPException."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                raise RuntimeError("Internal error")

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            with pytest.raises(HTTPException) as exc_info:
                await transport.handle_request(mock_request)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Internal server error"

    @pytest.mark.asyncio
    async def test_handle_request_empty_headers(self, mock_mcp_server, mock_request):
        """Test handling response with no headers."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        # No headers key
                    }
                )
                await send({"type": "http.response.body", "body": b"data"})

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.status_code == 200
            assert response.body == b"data"


# =============================================================================
# Test Shutdown
# =============================================================================


class TestShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_running_task(self, mock_mcp_server):
        """Test that shutdown cancels the running manager task."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            await transport._ensure_session_manager_started()
            assert transport._manager_started is True

            await transport.shutdown()

            assert transport._manager_started is False

    @pytest.mark.asyncio
    async def test_shutdown_when_not_started(self, mock_mcp_server):
        """Test that shutdown is safe when not started."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        # Should not raise
        await transport.shutdown()

        assert transport._manager_started is False

    @pytest.mark.asyncio
    async def test_shutdown_when_task_already_done(self, mock_mcp_server):
        """Test shutdown when task has already completed."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        # Create a completed task
        async def noop():
            pass

        transport._manager_task = asyncio.create_task(noop())
        await transport._manager_task  # Let it complete

        # Should not raise
        await transport.shutdown()


# =============================================================================
# Test mount_stateless_mcp Function
# =============================================================================


class TestMountStatelessMcp:
    """Tests for mount_stateless_mcp function."""

    def test_mount_to_default_fastapi_app(self, mock_fastapi_mcp):
        """Test mounting to default FastAPI app."""
        transport = mount_stateless_mcp(mock_fastapi_mcp)

        assert isinstance(transport, StatelessHttpTransport)
        assert transport.mcp_server is mock_fastapi_mcp.server

        # Verify route was added
        routes = [route.path for route in mock_fastapi_mcp.fastapi.routes]
        assert "/mcp" in routes

    def test_mount_with_custom_path(self, mock_fastapi_mcp):
        """Test mounting with custom path."""
        transport = mount_stateless_mcp(mock_fastapi_mcp, mount_path="/custom-mcp")

        assert isinstance(transport, StatelessHttpTransport)

        routes = [route.path for route in mock_fastapi_mcp.fastapi.routes]
        assert "/custom-mcp" in routes

    def test_mount_path_normalization_adds_leading_slash(self, mock_fastapi_mcp):
        """Test that mount path without leading slash is normalized."""
        mount_stateless_mcp(mock_fastapi_mcp, mount_path="mcp")

        routes = [route.path for route in mock_fastapi_mcp.fastapi.routes]
        assert "/mcp" in routes

    def test_mount_path_normalization_removes_trailing_slash(self, mock_fastapi_mcp):
        """Test that mount path with trailing slash is normalized."""
        mount_stateless_mcp(mock_fastapi_mcp, mount_path="/mcp/")

        routes = [route.path for route in mock_fastapi_mcp.fastapi.routes]
        assert "/mcp" in routes
        assert "/mcp/" not in routes

    def test_mount_to_custom_router(self, mock_fastapi_mcp):
        """Test mounting to a custom APIRouter."""
        custom_router = APIRouter(prefix="/api/v1")

        transport = mount_stateless_mcp(
            mock_fastapi_mcp,
            router=custom_router,
            mount_path="/mcp",
        )

        assert isinstance(transport, StatelessHttpTransport)

    def test_mount_to_custom_fastapi_app(self, mock_fastapi_mcp):
        """Test mounting to a different FastAPI app."""
        custom_app = FastAPI()

        transport = mount_stateless_mcp(
            mock_fastapi_mcp,
            router=custom_app,
            mount_path="/mcp",
        )

        assert isinstance(transport, StatelessHttpTransport)

        routes = [route.path for route in custom_app.routes]
        assert "/mcp" in routes

    def test_mount_registers_correct_methods(self, mock_fastapi_mcp):
        """Test that GET, POST, DELETE methods are registered."""
        mount_stateless_mcp(mock_fastapi_mcp)

        # Find the MCP route
        mcp_route = None
        for route in mock_fastapi_mcp.fastapi.routes:
            if hasattr(route, "path") and route.path == "/mcp":
                mcp_route = route
                break

        assert mcp_route is not None
        assert "GET" in mcp_route.methods
        assert "POST" in mcp_route.methods
        assert "DELETE" in mcp_route.methods

    def test_mount_route_not_in_schema(self, mock_fastapi_mcp):
        """Test that mounted route is not included in OpenAPI schema."""
        mount_stateless_mcp(mock_fastapi_mcp)

        # Find the MCP route
        for route in mock_fastapi_mcp.fastapi.routes:
            if hasattr(route, "path") and route.path == "/mcp":
                assert route.include_in_schema is False
                break


# =============================================================================
# Test Concurrency
# =============================================================================


class TestConcurrency:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_startup_only_creates_one_manager(self, mock_mcp_server):
        """Test that concurrent startups only create one session manager."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)
        call_count = 0

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:

            def create_manager(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_manager = MagicMock()
                async_cm = MagicMock()
                async_cm.__aenter__ = AsyncMock(return_value=None)
                async_cm.__aexit__ = AsyncMock(return_value=None)
                mock_manager.run.return_value = async_cm
                return mock_manager

            MockManager.side_effect = create_manager

            # Start multiple concurrent startup calls
            await asyncio.gather(
                transport._ensure_session_manager_started(),
                transport._ensure_session_manager_started(),
                transport._ensure_session_manager_started(),
                transport._ensure_session_manager_started(),
                transport._ensure_session_manager_started(),
            )

            # Should only be created once due to locking
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_after_startup(
        self, mock_mcp_server, mock_request
    ):
        """Test handling multiple concurrent requests."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)
        request_count = 0

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                nonlocal request_count
                request_count += 1
                await asyncio.sleep(0.01)  # Simulate some work
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"request_{request_count}".encode(),
                    }
                )

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            # Send multiple concurrent requests
            results = await asyncio.gather(
                transport.handle_request(mock_request),
                transport.handle_request(mock_request),
                transport.handle_request(mock_request),
            )

            assert len(results) == 3
            assert all(r.status_code == 200 for r in results)


# =============================================================================
# Test Stateless Mode Verification
# =============================================================================


class TestStatelessModeVerification:
    """Tests to verify stateless mode is correctly enabled."""

    @pytest.mark.asyncio
    async def test_stateless_true_is_passed_to_manager(self, mock_mcp_server):
        """Verify that stateless=True is always passed to StreamableHTTPSessionManager."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm
            MockManager.return_value = mock_manager

            await transport._ensure_session_manager_started()

            # This is the key assertion - stateless=True enables horizontal scaling
            call_kwargs = MockManager.call_args.kwargs
            assert call_kwargs["stateless"] is True


# =============================================================================
# Test Logging
# =============================================================================


class TestLogging:
    """Tests for logging behavior."""

    @pytest.mark.asyncio
    async def test_logs_debug_message_on_startup(self, mock_mcp_server, caplog):
        """Test that startup logs debug message."""
        import logging

        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with caplog.at_level(logging.DEBUG):
            with patch(
                "src.service.stateless_http_transport.StreamableHTTPSessionManager"
            ) as MockManager:
                mock_manager = MagicMock()
                async_cm = MagicMock()
                async_cm.__aenter__ = AsyncMock(return_value=None)
                async_cm.__aexit__ = AsyncMock(return_value=None)
                mock_manager.run.return_value = async_cm
                MockManager.return_value = mock_manager

                await transport._ensure_session_manager_started()

                # Check debug log for stateless session manager startup
                log_messages = [record.message for record in caplog.records]
                assert any(
                    "stateless" in msg.lower() and "session manager" in msg.lower()
                    for msg in log_messages
                )

                # Cleanup
                await transport.shutdown()

    def test_mount_logs_message(self, mock_fastapi_mcp, caplog):
        """Test that mount logs appropriate message."""
        import logging

        with caplog.at_level(logging.INFO):
            mount_stateless_mcp(mock_fastapi_mcp)

        log_messages = [record.message for record in caplog.records]
        assert any("horizontal scaling enabled" in msg for msg in log_messages)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handle_request_with_empty_body(self, mock_mcp_server, mock_request):
        """Test handling response with empty body."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                await send(
                    {"type": "http.response.start", "status": 204, "headers": []}
                )
                await send({"type": "http.response.body"})  # No body key

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.status_code == 204
            assert response.body == b""

    @pytest.mark.asyncio
    async def test_handle_request_preserves_multiple_headers(
        self, mock_mcp_server, mock_request
    ):
        """Test that multiple headers are preserved in response."""
        transport = StatelessHttpTransport(mcp_server=mock_mcp_server)

        with patch(
            "src.service.stateless_http_transport.StreamableHTTPSessionManager"
        ) as MockManager:
            mock_manager = MagicMock()
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=None)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_manager.run.return_value = async_cm

            async def mock_handle_request(scope, receive, send):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"x-custom-header", b"custom-value"),
                            (b"cache-control", b"no-cache"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b"{}"})

            mock_manager.handle_request = mock_handle_request
            MockManager.return_value = mock_manager

            response = await transport.handle_request(mock_request)

            assert response.headers["content-type"] == "application/json"
            assert response.headers["x-custom-header"] == "custom-value"
            assert response.headers["cache-control"] == "no-cache"

    def test_mount_with_empty_path(self, mock_fastapi_mcp):
        """Test mounting with empty path defaults correctly."""
        # Empty string should be normalized to "/"
        mount_stateless_mcp(mock_fastapi_mcp, mount_path="")

        routes = [route.path for route in mock_fastapi_mcp.fastapi.routes]
        # Empty becomes "/" after normalization
        assert "/" in routes or "" in routes
