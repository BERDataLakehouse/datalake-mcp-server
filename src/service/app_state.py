"""
Application state information and retrieval functions.

All functions assume that the application state has been initialized via
calling the build_app() method.
"""

import asyncio
import logging
import os
from typing import NamedTuple

from fastapi import FastAPI, Request

from src.service.kb_auth import KBaseAuth, KBaseUser

logger = logging.getLogger(__name__)


class AppState(NamedTuple):
    """Holds application state."""

    auth: KBaseAuth


class RequestState(NamedTuple):
    """Holds request specific state."""

    user: KBaseUser | None


async def build_app(app: FastAPI) -> None:
    """
    Build the application state.

    Args:
        app: The FastAPI app.
    """
    logger.info("Initializing application state...")

    # Initialize auth with KBase auth URL and admin roles from environment variables
    auth_url = os.environ.get("KBASE_AUTH_URL", "https://ci.kbase.us/services/auth/")
    admin_roles = os.environ.get("KBASE_ADMIN_ROLES", "KBASE_ADMIN").split(",")
    required_roles = os.environ.get("KBASE_REQUIRED_ROLES", "BERDL_USER").split(",")

    logger.info("Connecting to KBase auth service...")
    auth = await KBaseAuth.create(
        auth_url, required_roles=required_roles, full_admin_roles=admin_roles
    )
    logger.info("KBase auth service connected")

    # Store components in app state
    app.state._auth = auth
    app.state._spark_state = AppState(auth=auth)
    logger.info("Application state initialized")


async def destroy_app_state(app: FastAPI) -> None:
    """
    Destroy the application state, shutting down services and releasing resources.

    Args:
        app: The FastAPI app.
    """
    # Currently no resources need to be explicitly cleaned up
    # https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown
    await asyncio.sleep(0.250)
    logger.info("Application state destroyed")


def get_app_state(request: Request) -> AppState:
    """
    Get the application state from a request.

    Args:
        request: The FastAPI request.

    Returns:
        The application state.

    Raises:
        ValueError: If app state has not been initialized.
    """
    return _get_app_state_from_app(request.app)


def _get_app_state_from_app(app: FastAPI) -> AppState:
    """
    Get the application state from a FastAPI app.

    Args:
        app: The FastAPI app.

    Returns:
        The application state.

    Raises:
        ValueError: If app state has not been initialized.
    """
    if not hasattr(app.state, "_spark_state") or not app.state._spark_state:
        raise ValueError("App state has not been initialized")
    return app.state._spark_state


def set_request_user(request: Request, user: KBaseUser | None) -> None:
    """
    Set the user for the current request.

    Args:
        request: The FastAPI request.
        user: The KBase user.
    """
    request.state._request_state = RequestState(user=user)


def get_request_user(request: Request) -> KBaseUser | None:
    """
    Get the user for a request.

    Args:
        request: The FastAPI request.

    Returns:
        The authenticated KBaseUser if available, otherwise None.
    """
    if not hasattr(request.state, "_request_state") or not request.state._request_state:
        return None
    return request.state._request_state.user
