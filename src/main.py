"""
Main application module for the Spark Manager API.
"""

import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_mcp import FastApiMCP
from starlette.middleware.base import BaseHTTPMiddleware

from src.routes import delta, health
from src.service import app_state
from src.service.config import configure_logging, get_settings
from src.service.stateless_http_transport import mount_stateless_mcp
from src.service.exception_handlers import universal_error_handler
from src.service.exceptions import InvalidAuthHeaderError
from src.service.models import ErrorResponse

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Middleware constants
_SCHEME = "Bearer"


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce HTTP request timeouts.

    Returns a clean 408 Request Timeout response before upstream proxies/gateways
    return 504 Gateway Timeout with HTML error pages. This ensures clients receive
    a user-friendly JSON error message instead of raw HTML.
    """

    def __init__(self, app, timeout_seconds: float = 55.0):
        """
        Initialize the timeout middleware.

        Args:
            app: The FastAPI/Starlette application.
            timeout_seconds: Maximum request processing time in seconds.
                Should be set lower than your proxy/gateway timeout.
        """
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip timeout for health checks to ensure they respond quickly
        if request.url.path in ("/health", "/health/ready", "/health/live"):
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout after {self.timeout_seconds}s: "
                f"{request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content={
                    "error": 40800,
                    "error_type": "request_timeout",
                    "message": (
                        f"Request timed out after {self.timeout_seconds} seconds. "
                        "The operation took too long to complete. "
                        "Consider using pagination, reducing query scope, "
                        "or breaking up large operations into smaller requests."
                    ),
                },
            )


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate users and set them in the request state."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_user = None
        auth_header = request.headers.get("Authorization")

        if auth_header:
            scheme, credentials = get_authorization_scheme_param(auth_header)
            if not (scheme and credentials):
                raise InvalidAuthHeaderError(
                    f"Authorization header requires {_SCHEME} scheme followed by token"
                )
            if scheme.lower() != _SCHEME.lower():
                # don't put the received scheme in the error message, might be a token
                raise InvalidAuthHeaderError(
                    f"Authorization header requires {_SCHEME} scheme"
                )

            app_state_obj = app_state.get_app_state(request)
            request_user = await app_state_obj.auth.get_user(credentials)

        app_state.set_request_user(request, request_user)

        return await call_next(request)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.api_version,
        responses={
            "4XX": {"model": ErrorResponse},
            "5XX": {"model": ErrorResponse},
        },
    )

    # Add exception handlers
    app.add_exception_handler(Exception, universal_error_handler)

    # Add middleware (order matters - outermost middleware is added last)
    # 1. GZip compresses responses
    # 2. RequestTimeout ensures we return 408 before proxy returns 504
    # 3. Auth handles authentication
    app.add_middleware(GZipMiddleware)
    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=settings.request_timeout_seconds,
    )
    app.add_middleware(AuthMiddleware)

    # Include routers
    app.include_router(health.router)
    app.include_router(delta.router)

    # MCP Server Integration with STATELESS HTTP transport for horizontal scaling
    # This enables true horizontal pod scaling in Kubernetes without sticky sessions
    logger.info("Setting up MCP server with stateless HTTP transport...")
    mcp = FastApiMCP(
        app,
        name="DeltaLakeMCP",
        description="MCP Server for interacting with Delta Lake tables via Spark",
        include_tags=["Delta Lake"],  # Only include endpoints tagged with "Delta Lake"
    )
    # Use stateless HTTP transport instead of default SSE for horizontal scaling
    mcp_transport = mount_stateless_mcp(mcp)
    logger.info(
        "MCP server mounted with stateless HTTP transport (horizontal scaling enabled)"
    )

    # Define startup and shutdown event handlers
    async def startup_event():
        logger.info("Starting application")
        await app_state.build_app(app)
        logger.info("Application started")

    async def shutdown_event():
        logger.info("Shutting down application")
        await mcp_transport.shutdown()
        await app_state.destroy_app_state(app)
        logger.info("Application shut down")

    # Handle service root path mounting for proper URL routing
    # This is critical for preventing double path prefixes in MCP client requests
    if settings.service_root_path:
        # Create a root FastAPI application to handle path mounting
        # This prevents the MCP client from incorrectly constructing URLs
        root_app = FastAPI()

        # Mount the main app at the specified root path (e.g., "/apis/mcp")
        # This creates the following URL structure:
        # - Root app handles: /
        # - Main app handles: /apis/mcp/*
        # - MCP endpoint becomes: /apis/mcp/mcp
        #
        # WHY THIS WORKS:
        # Without this mounting structure, when the MCP client discovers the server
        # at "https://cdmhub.ci.kbase.us/apis/mcp/mcp", it incorrectly assumes
        # the base URL is "https://cdmhub.ci.kbase.us" and then tries to construct
        # tool calls by appending the discovered path again, resulting in:
        # "https://cdmhub.ci.kbase.us/apis/mcp/apis/mcp/mcp" (double /apis/mcp)
        #
        # With proper mounting:
        # 1. The root app serves at the domain root
        # 2. The main app is mounted at /apis/mcp
        # 3. MCP endpoint is accessible at /apis/mcp/mcp
        # 4. MCP client correctly identifies the base as the mounted path
        # 5. Tool calls are made to /apis/mcp/tools/call (correct path)
        root_app.mount(settings.service_root_path, app)

        # Event handlers must be attached to the root app since it's what gets served
        root_app.add_event_handler("startup", startup_event)
        root_app.add_event_handler("shutdown", shutdown_event)

        return root_app
    else:
        # No root path mounting needed - serve the app directly
        app.add_event_handler("startup", startup_event)
        app.add_event_handler("shutdown", shutdown_event)

    return app


if __name__ == "__main__":
    app_instance = create_application()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # Validate PostgreSQL is configured for readonly access
    if os.getenv("POSTGRES_USER") != "readonly_user":
        raise ValueError("POSTGRES_USER must be set to readonly_user")

    # MinIO credentials are read dynamically from each user's home directory
    # No validation needed here - credentials are loaded per-request from
    # /home/{username}/.berdl_minio_credentials

    uvicorn.run(app_instance, host=host, port=port)
