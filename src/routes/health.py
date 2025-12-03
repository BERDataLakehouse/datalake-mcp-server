"""
Health check routes for the API.

Provides health checks that verify connectivity to all backend services.
"""

import logging
import time
from typing import Callable

from fastapi import APIRouter

from src.service.models import ComponentHealth, DeepHealthResponse

logger = logging.getLogger(__name__)

# Create a router for health endpoints
router = APIRouter(tags=["health"])


def _timed_check(name: str, check_fn: Callable[[], bool | str]) -> ComponentHealth:
    """
    Run a health check function and measure its latency.

    Args:
        name: Name of the component being checked
        check_fn: Function that returns True for healthy, False for unhealthy,
                  or a string message for degraded/error states

    Returns:
        ComponentHealth with status and latency
    """
    start = time.time()
    try:
        result = check_fn()
        latency_ms = (time.time() - start) * 1000

        if result is True:
            return ComponentHealth(
                name=name,
                status="healthy",
                latency_ms=round(latency_ms, 2),
            )
        elif result is False:
            return ComponentHealth(
                name=name,
                status="unhealthy",
                message="Health check failed",
                latency_ms=round(latency_ms, 2),
            )
        else:
            # String message indicates degraded or specific error
            return ComponentHealth(
                name=name,
                status="degraded",
                message=str(result),
                latency_ms=round(latency_ms, 2),
            )
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.warning(f"Health check failed for {name}: {e}")
        return ComponentHealth(
            name=name,
            status="unhealthy",
            message=str(e)[:200],  # Truncate long error messages
            latency_ms=round(latency_ms, 2),
        )


def _check_redis() -> bool | str:
    """Check Redis connectivity."""
    from src.cache.redis_cache import _get_redis_client

    client = _get_redis_client()
    if client is None:
        return "Redis client not initialized"

    # Simple ping check
    response = client.ping()
    return response is True


def _check_postgresql() -> bool | str:
    """Check PostgreSQL connectivity."""
    import os

    # Skip check if PostgreSQL is not configured
    if not os.environ.get("POSTGRES_URL"):
        return "PostgreSQL not configured (optional)"

    from src.postgres.connection import get_postgres_connection

    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS health_check")
            result = cur.fetchone()
            # Result is a dict due to dict_row factory from connection.py
            if result is None:
                return False
            # Access as dict (dict_row returns dict-like objects)
            return result["health_check"] == 1  # type: ignore[index]
    finally:
        conn.close()


def _check_hive_metastore() -> bool | str:
    """Check Hive Metastore connectivity."""
    from src.delta_lake.hive_metastore import get_hive_metastore_client
    from src.settings import get_settings

    settings = get_settings()
    client = get_hive_metastore_client(settings)

    try:
        client.open()
        # Simple check - get list of databases (should always have at least 'default')
        databases = client.get_databases("*")
        return len(databases) >= 0  # Even empty list is valid
    finally:
        client.close()


@router.get(
    "/health",
    response_model=DeepHealthResponse,
    summary="Health check",
    description=(
        "Returns detailed health status of all backend services including "
        "Redis, PostgreSQL (if configured), and Hive Metastore Thrift connection."
    ),
)
async def health_check():
    """
    Health check that verifies connectivity to all backend services.

    Checks:
    - Redis (caching)
    - PostgreSQL (if configured, for Hive Metastore queries)
    - Hive Metastore (Thrift connection)
    """
    components = [
        _timed_check("redis", _check_redis),
        _timed_check("postgresql", _check_postgresql),
        _timed_check("hive_metastore", _check_hive_metastore),
    ]

    # Determine overall status
    unhealthy_count = sum(1 for c in components if c.status == "unhealthy")
    degraded_count = sum(1 for c in components if c.status == "degraded")

    if unhealthy_count > 0:
        overall_status = "unhealthy"
        message = f"{unhealthy_count} component(s) unhealthy"
    elif degraded_count > 0:
        overall_status = "degraded"
        message = f"{degraded_count} component(s) degraded"
    else:
        overall_status = "healthy"
        message = "All components healthy"

    return DeepHealthResponse(
        status=overall_status,
        components=components,
        message=message,
    )
