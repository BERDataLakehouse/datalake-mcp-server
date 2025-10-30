"""
Main application module for the Spark Manager API.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_mcp import FastApiMCP
from starlette.middleware.base import BaseHTTPMiddleware

from src.routes import delta, health
from src.service import app_state
from src.service.config import configure_logging, get_settings
from src.service.exception_handlers import universal_error_handler
from src.service.exceptions import InvalidAuthHeaderError
from src.service.models import ErrorResponse

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Middleware constants
_SCHEME = "Bearer"


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

    # Add middleware
    app.add_middleware(GZipMiddleware)
    app.add_middleware(AuthMiddleware)

    # Include routers
    app.include_router(health.router)
    app.include_router(delta.router)

    # MCP Server Integration
    logger.info("Setting up MCP server...")
    mcp = FastApiMCP(
        app,
        name="DeltaLakeMCP",
        description="MCP Server for interacting with Delta Lake tables via Spark",
        include_tags=["Delta Lake"], # Only include endpoints tagged with "Delta Lake"
    )
    mcp.mount()
    logger.info("MCP server mounted")

    # Define startup and shutdown event handlers
    async def startup_event():
        logger.info("Starting application")
        await app_state.build_app(app)
        logger.info("Application started")

    async def shutdown_event():
        logger.info("Shutting down application")
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
