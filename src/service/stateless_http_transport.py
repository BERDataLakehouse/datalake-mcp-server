"""
Stateless HTTP transport for MCP server enabling horizontal scaling.

This module provides a stateless HTTP transport wrapper around the MCP
StreamableHTTPSessionManager. By setting stateless=True, each request is
independent and does not require session affinity, enabling true horizontal
pod scaling in Kubernetes.

Key benefits:
- No session state stored in memory
- Requests can be handled by any pod
- Works with standard load balancers (no sticky sessions needed)
- Ideal for Kubernetes HPA (Horizontal Pod Autoscaler)
"""

import asyncio
import logging

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from mcp.server.lowlevel.server import Server
from mcp.server.streamable_http_manager import EventStore, StreamableHTTPSessionManager

logger = logging.getLogger(__name__)


class StatelessHttpTransport:
    """
    Stateless HTTP transport for MCP that enables horizontal scaling.

    Unlike the default fastapi-mcp HTTP transport which uses stateless=False,
    this transport sets stateless=True on the StreamableHTTPSessionManager,
    eliminating per-pod session storage and enabling true horizontal scaling.
    """

    def __init__(
        self,
        mcp_server: Server,
        event_store: EventStore | None = None,
        json_response: bool = True,
    ):
        """
        Initialize the stateless HTTP transport.

        Args:
            mcp_server: The MCP server instance to handle requests
            event_store: Optional event store for advanced resumability or observability use
                cases. In most stateless deployments this can be omitted; if provided, it
                will be passed through to the underlying StreamableHTTPSessionManager.
            json_response: Whether to use JSON responses (default True for HTTP)
        """
        self.mcp_server = mcp_server
        if event_store is not None:
            logger.warning(
                "An 'event_store' was provided to StatelessHttpTransport. "
                "Stateless mode typically does not require an event store; "
                "ensure this configuration is intentional for your use case."
            )
        self.event_store = event_store
        self.json_response = json_response
        self._session_manager: StreamableHTTPSessionManager | None = None
        self._manager_task: asyncio.Task | None = None
        self._manager_started = False
        self._startup_lock = asyncio.Lock()

    async def _ensure_session_manager_started(self) -> None:
        """
        Ensure the session manager is started.

        This is called lazily on the first request to start the session manager
        if it hasn't been started yet.
        """
        if self._manager_started:
            return

        async with self._startup_lock:
            if self._manager_started:
                return

            logger.debug("Starting stateless StreamableHTTP session manager")

            # Create the session manager with stateless=True for horizontal scaling
            # This is the key difference from fastapi-mcp's default implementation
            self._session_manager = StreamableHTTPSessionManager(
                app=self.mcp_server,
                event_store=self.event_store,
                json_response=self.json_response,
                stateless=True,  # Enable stateless mode for horizontal scaling
            )

            # Start the session manager in a background task
            async def run_session_manager():
                try:
                    async with self._session_manager.run():
                        logger.info(
                            "Stateless StreamableHTTP session manager is running "
                            "(horizontal scaling enabled)"
                        )
                        # Keep running until cancelled
                        await asyncio.Event().wait()
                except asyncio.CancelledError:
                    logger.info(
                        "Stateless StreamableHTTP session manager is shutting down"
                    )
                    raise
                except Exception:
                    logger.exception(
                        "Error in stateless StreamableHTTP session manager"
                    )
                    raise

            self._manager_task = asyncio.create_task(run_session_manager())
            self._manager_started = True

            # Give the session manager a moment to initialize
            await asyncio.sleep(0.1)

    async def handle_request(self, request: Request) -> Response:
        """
        Handle a FastAPI request by delegating to the stateless session manager.

        This converts FastAPI's Request/Response to ASGI scope/receive/send
        and then converts the result back to a FastAPI Response.
        """
        await self._ensure_session_manager_started()

        if not self._session_manager:
            raise HTTPException(
                status_code=500, detail="Session manager not initialized"
            )

        logger.debug(
            f"Handling stateless MCP request: {request.method} {request.url.path}"
        )

        # Capture the response from the session manager
        response_started = False
        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []
        response_body = b""

        async def send_callback(message: dict) -> None:
            nonlocal response_started, response_status, response_headers, response_body

            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")

        try:
            # Delegate to the session manager's handle_request method
            await self._session_manager.handle_request(
                request.scope, request.receive, send_callback
            )

            # Convert the captured ASGI response to a FastAPI Response
            headers_dict = {
                name.decode(): value.decode() for name, value in response_headers
            }

            return Response(
                content=response_body,
                status_code=response_status,
                headers=headers_dict,
            )

        except HTTPException as http_exc:
            # Preserve existing HTTPExceptions but log with request context for debugging
            logger.exception(
                "HTTPException in stateless StreamableHTTPSessionManager for %s %s",
                request.method,
                request.url.path,
            )
            raise http_exc
        except Exception as exc:
            # Log unexpected errors with request context and propagate as 500 while
            # preserving the original exception as the cause for debugging.
            logger.exception(
                "Unexpected error in stateless StreamableHTTPSessionManager for %s %s",
                request.method,
                request.url.path,
            )
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    async def shutdown(self) -> None:
        """Clean up the session manager and background task."""
        if self._manager_task and not self._manager_task.done():
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                # Expected during shutdown when cancelling the manager task.
                logger.debug("Stateless HTTP manager task cancelled during shutdown")
        self._manager_started = False


def mount_stateless_mcp(
    mcp: "FastApiMCP",  # noqa: F821
    router: FastAPI | APIRouter | None = None,
    mount_path: str = "/mcp",
) -> StatelessHttpTransport:
    """
    Mount the MCP server with stateless HTTP transport for horizontal scaling.

    This is a drop-in replacement for mcp.mount_http() that enables stateless mode,
    allowing the MCP server to scale horizontally without sticky sessions.

    Args:
        mcp: The FastApiMCP instance
        router: The FastAPI app or APIRouter to mount to (defaults to mcp.fastapi)
        mount_path: Path where the MCP endpoint will be mounted (default: "/mcp")

    Returns:
        The StatelessHttpTransport instance for lifecycle management

    Example:
        mcp = FastApiMCP(app, name="MyMCP")
        transport = mount_stateless_mcp(mcp)

        # In shutdown handler:
        await transport.shutdown()
    """
    # Normalize mount path
    if not mount_path.startswith("/"):
        mount_path = f"/{mount_path}"
    if mount_path.endswith("/"):
        mount_path = mount_path[:-1]

    if router is None:
        router = mcp.fastapi

    # Create the stateless transport
    transport = StatelessHttpTransport(mcp_server=mcp.server)

    # Register the endpoint
    @router.api_route(
        mount_path,
        methods=["GET", "POST", "DELETE"],
        include_in_schema=False,
        operation_id="mcp_stateless_http",
    )
    async def handle_mcp_stateless_http(request: Request) -> Response:
        return await transport.handle_request(request)

    # Re-include router if it's an APIRouter (same hack as fastapi-mcp)
    if isinstance(router, APIRouter):
        mcp.fastapi.include_router(router)

    logger.info(
        f"MCP stateless HTTP server listening at {mount_path} (horizontal scaling enabled)"
    )

    return transport
