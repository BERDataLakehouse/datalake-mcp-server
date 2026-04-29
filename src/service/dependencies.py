"""
Dependencies for FastAPI dependency injection.
"""

import logging
import os
import random
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Generator
from urllib.parse import quote, urlparse

import grpc
import httpx
import trino
from fastapi import Depends, Request
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import AnyUrl
from pyspark.sql import SparkSession

# Use MCP server's local copy of spark session utilities
# (copied from berdl_notebook_utils but adapted for shared multi-user service)
from src.delta_lake.setup_spark_session import (
    get_spark_session as _get_spark_session,
    get_spark_session_with_retry as _get_spark_session_with_retry,
)
from src.service import app_state
from src.service.exceptions import MissingTokenError, TrinoConnectionError
from src.service.http_bearer import KBaseHTTPBearer
from src.service.models import QueryEngine
from src.service.spark_session_pool import STANDALONE_POOL_SIZE
from src.settings import BERDLSettings, get_settings
from src.trino_engine.trino_connection import create_trino_connection

# Initialize the KBase auth dependency for use in routes
auth = KBaseHTTPBearer()

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SparkContext:
    """
    Execution context for Spark operations.

    For Spark Connect mode:
        - spark is set to the active SparkSession
        - is_standalone_subprocess is False
        - settings_dict contains picklable settings (for async query subprocess reuse)

    For Standalone mode:
        - spark is None (session created in subprocess)
        - is_standalone_subprocess is True
        - settings_dict contains picklable settings for subprocess

    Routes should check is_standalone_subprocess and either:
    1. Use spark directly for queries (Connect mode)
    2. Call run_in_spark_process() with the operation (Standalone mode)
    """

    spark: SparkSession | None = None
    is_standalone_subprocess: bool = False
    settings_dict: dict = field(default_factory=dict)
    app_name: str = ""
    username: str = ""
    auth_token: str | None = None


@dataclass
class TrinoContext:
    """
    Execution context for Trino operations.

    Each request creates its own independent ``trino.dbapi.Connection`` with
    the requesting user's credentials — no state is shared between requests.
    """

    connection: trino.dbapi.Connection
    username: str = ""
    auth_token: str | None = None
    settings_dict: dict = field(default_factory=dict)


def resolve_engine(requested: QueryEngine | None = None) -> QueryEngine:
    """
    Determine which query engine to use.

    Priority: per-request override > ``QUERY_ENGINE`` env var > default ``spark``.
    """
    if requested is not None:
        return requested
    env_val = os.getenv("QUERY_ENGINE", "spark").lower()
    try:
        return QueryEngine(env_val)
    except ValueError:
        logger.warning(f"Unknown QUERY_ENGINE value '{env_val}', defaulting to spark")
        return QueryEngine.SPARK


def sanitize_k8s_name(name: str) -> str:
    """
    Sanitize a string to be Kubernetes DNS-1123 subdomain compliant.

    Kubernetes resource names must:
    - Consist of lowercase alphanumeric characters, '-', or '.'
    - Start and end with an alphanumeric character
    - Be at most 253 characters long

    Args:
        name: The string to sanitize (e.g., username with underscores)

    Returns:
        A DNS-1123 compliant string (replaces underscores with hyphens)
    """
    # Replace underscores and other invalid characters with hyphens
    sanitized = re.sub(r"[^a-z0-9.-]", "-", name.lower())

    # Ensure it starts and ends with alphanumeric
    sanitized = re.sub(r"^[^a-z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-z0-9]+$", "", sanitized)

    # Collapse multiple consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Truncate to 253 characters (K8s limit)
    return sanitized[:253]


DEFAULT_SPARK_POOL = "default"
SPARK_CONNECT_PORT = "15002"

# =============================================================================
# CONCURRENCY ARCHITECTURE
# =============================================================================
# Full concurrency is enabled for both Spark Connect and Standalone modes:
#
# SPARK CONNECT MODE:
#   - No lock needed - Connect sessions are client-only gRPC connections
#   - Each request gets its own channel to the user's remote Spark cluster
#   - Fully concurrent - limited only by the remote cluster capacity
#
# STANDALONE MODE:
#   - Uses ProcessPoolExecutor for process isolation
#   - Each request runs in a separate process with its own JVM
#   - Concurrent up to STANDALONE_POOL_SIZE workers
#
# This architecture replaces the previous _session_mode_lock which serialized
# all requests to prevent JVM mode conflicts.
# =============================================================================


def fetch_user_minio_credentials(
    governance_api_url: str, auth_token: str
) -> tuple[str, str]:
    """
    Fetch user's MinIO credentials from the Governance API.

    Calls GET /credentials/ on the governance API with the user's Bearer token.
    The API returns cached credentials (idempotent) or creates them on first call.

    Args:
        governance_api_url: Base URL of the governance API (e.g. "http://minio-service:8000")
        auth_token: KBase auth token for the user

    Returns:
        Tuple of (access_key, secret_key)

    Raises:
        Exception: If the API call fails or returns invalid data
    """
    url = f"{str(governance_api_url).rstrip('/')}/credentials/"
    logger.debug(f"Fetching MinIO credentials from governance API: {url}")

    response = httpx.get(
        url, headers={"Authorization": f"Bearer {auth_token}"}, timeout=10.0
    )
    response.raise_for_status()

    data = response.json()
    access_key = data.get("access_key")
    secret_key = data.get("secret_key")

    if not access_key or not secret_key:
        raise ValueError(
            "Invalid credentials response from governance API: missing access_key or secret_key"
        )

    logger.info(
        f"Successfully fetched MinIO credentials for user: {data.get('username')}"
    )
    return access_key, secret_key


# Per-user in-process cache for Polaris credentials.
#
# POST /polaris/user_provision/{username} is a provisioning call: idempotent but
# multi-step (DB + Polaris HTTP) even on a hit. Calling it on every request
# forces every datalake-mcp-server request to wait on MMS, which caps throughput
# at well under 1k RPS and wedges the threadpool when MMS is slow. Polaris
# credentials are stable per-user until explicit rotation, so we cache the
# response in-process with a TTL and coalesce concurrent fetches per user.
_POLARIS_CRED_CACHE_TTL = int(os.getenv("POLARIS_CRED_CACHE_TTL", "900"))
_polaris_cred_cache: dict[str, tuple[float, dict[str, str] | None]] = {}
_polaris_cred_cache_lock = threading.Lock()
_polaris_per_user_locks: dict[str, threading.Lock] = {}


def _get_polaris_user_lock(username: str) -> threading.Lock:
    with _polaris_cred_cache_lock:
        lock = _polaris_per_user_locks.get(username)
        if lock is None:
            lock = threading.Lock()
            _polaris_per_user_locks[username] = lock
        return lock


def invalidate_polaris_credentials(username: str) -> None:
    """Drop a user's cached Polaris credentials (e.g. after explicit rotation)."""
    with _polaris_cred_cache_lock:
        _polaris_cred_cache.pop(username, None)


def fetch_user_polaris_credentials(
    governance_api_url: str,
    username: str,
    auth_token: str,
) -> dict[str, str] | None:
    """
    Provision and fetch the user's Polaris credentials from the Governance API.

    Mirrors spark_notebook's startup flow by calling
    POST /polaris/user_provision/{username}. The Governance API reuses cached
    credentials server-side and returns the personal and tenant Polaris catalogs
    the user can access.

    Results are cached in-process per-user for ``POLARIS_CRED_CACHE_TTL`` seconds
    to keep this off the per-request hot path.
    """
    cached = _polaris_cred_cache.get(username)
    if cached and (time.monotonic() - cached[0]) < _POLARIS_CRED_CACHE_TTL:
        return cached[1]

    user_lock = _get_polaris_user_lock(username)
    with user_lock:
        cached = _polaris_cred_cache.get(username)
        if cached and (time.monotonic() - cached[0]) < _POLARIS_CRED_CACHE_TTL:
            return cached[1]

        result = _fetch_user_polaris_credentials_uncached(
            governance_api_url, username, auth_token
        )
        _polaris_cred_cache[username] = (time.monotonic(), result)
        return result


def _fetch_user_polaris_credentials_uncached(
    governance_api_url: str,
    username: str,
    auth_token: str,
) -> dict[str, str] | None:
    safe_username = quote(username, safe="")
    url = (
        f"{str(governance_api_url).rstrip('/')}/polaris/user_provision/{safe_username}"
    )
    logger.debug(f"Fetching Polaris credentials from governance API: {url}")

    response = httpx.post(
        url, headers={"Authorization": f"Bearer {auth_token}"}, timeout=30.0
    )
    response.raise_for_status()

    data = response.json()
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    personal_catalog = data.get("personal_catalog")
    tenant_catalogs = data.get("tenant_catalogs") or []

    if not client_id or not client_secret or not personal_catalog:
        raise ValueError(
            "Invalid Polaris response from governance API: missing "
            "client_id, client_secret, or personal_catalog"
        )

    if isinstance(tenant_catalogs, str):
        tenant_catalogs_value = tenant_catalogs
    else:
        tenant_catalogs_value = ",".join(str(c) for c in tenant_catalogs if c)

    logger.info(f"Successfully fetched Polaris credentials for user: {username}")
    return {
        "POLARIS_CREDENTIAL": f"{client_id}:{client_secret}",
        "POLARIS_PERSONAL_CATALOG": str(personal_catalog),
        "POLARIS_TENANT_CATALOGS": tenant_catalogs_value,
    }


def _get_effective_polaris_settings(
    settings: BERDLSettings,
    username: str,
    auth_token: str,
) -> dict[str, str | None]:
    """Build per-user Polaris settings, fetching fresh credentials when configured."""
    polaris_settings: dict[str, str | None] = {
        "POLARIS_CATALOG_URI": str(settings.POLARIS_CATALOG_URI)
        if settings.POLARIS_CATALOG_URI
        else None,
        "POLARIS_CREDENTIAL": settings.POLARIS_CREDENTIAL,
        "POLARIS_PERSONAL_CATALOG": settings.POLARIS_PERSONAL_CATALOG,
        "POLARIS_TENANT_CATALOGS": settings.POLARIS_TENANT_CATALOGS,
    }

    if not settings.POLARIS_CATALOG_URI:
        return polaris_settings

    try:
        fetched_settings = fetch_user_polaris_credentials(
            settings.GOVERNANCE_API_URL,
            username,
            auth_token,
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch Polaris credentials for %s: %s: %s. "
            "Continuing with configured Polaris settings.",
            username,
            type(e).__name__,
            e,
            exc_info=True,
        )
        return polaris_settings

    if fetched_settings:
        polaris_settings.update(fetched_settings)
    return polaris_settings


def _build_user_settings_dict(
    settings: BERDLSettings,
    username: str,
    auth_token: str,
    minio_access_key: str,
    minio_secret_key: str,
) -> dict[str, str | int | bool | None]:
    """Build picklable per-user settings for Spark, Trino, and async subprocesses."""
    return {
        "KBASE_AUTH_TOKEN": auth_token,
        "USER": username,
        "MINIO_ACCESS_KEY": minio_access_key,
        "MINIO_SECRET_KEY": minio_secret_key,
        "MINIO_ENDPOINT_URL": settings.MINIO_ENDPOINT_URL,
        "MINIO_SECURE": settings.MINIO_SECURE,
        "SPARK_HOME": settings.SPARK_HOME,
        "SPARK_MASTER_URL": str(settings.SPARK_MASTER_URL)
        if settings.SPARK_MASTER_URL
        else None,
        "BERDL_HIVE_METASTORE_URI": str(settings.BERDL_HIVE_METASTORE_URI),
        "SPARK_WORKER_COUNT": settings.SPARK_WORKER_COUNT,
        "SPARK_WORKER_CORES": settings.SPARK_WORKER_CORES,
        "SPARK_WORKER_MEMORY": settings.SPARK_WORKER_MEMORY,
        "SPARK_MASTER_CORES": settings.SPARK_MASTER_CORES,
        "SPARK_MASTER_MEMORY": settings.SPARK_MASTER_MEMORY,
        "GOVERNANCE_API_URL": str(settings.GOVERNANCE_API_URL),
        "BERDL_POD_IP": settings.BERDL_POD_IP,
        **_get_effective_polaris_settings(settings, username, auth_token),
    }


def get_user_from_request(request: Request) -> str:
    """
    Extract the authenticated user from the request state.

    The user is set by the AuthMiddleware after validating the Bearer token.

    Args:
        request: FastAPI request object

    Returns:
        Username of the authenticated user

    Raises:
        Exception: If user is not authenticated
    """
    user = app_state.get_request_user(request)
    if user is None:
        raise MissingTokenError(
            "User not authenticated. Authorization header required."
        )
    return user.user


def get_token_from_request(request: Request) -> str | None:
    """
    Extract the Bearer token from the request Authorization header.

    Uses the same parser as the auth middleware (get_authorization_scheme_param)
    to handle case-insensitive schemes and extra whitespace consistently.

    Args:
        request: FastAPI request object

    Returns:
        Token string or None if not present or not a Bearer scheme
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None
    scheme, credentials = get_authorization_scheme_param(auth_header)
    if scheme.lower() != "bearer" or not credentials:
        return None
    return credentials


def construct_user_spark_connect_url(username: str) -> str:
    """
    Construct the Spark Connect URL for a specific user's notebook pod.

    In BERDL, each user has their own notebook pod with a Spark Connect server.
    The URL pattern differs between environments:
    - Docker Compose (local dev): sc://spark-notebook:15002 or sc://spark-notebook-{username}:15002
    - Kubernetes (prod/stage/dev): sc://jupyter-{sanitized-username}.jupyterhub-{env}:15002

    Args:
        username: KBase username (may contain underscores or special characters)

    Returns:
        User-specific Spark Connect URL with DNS-safe username

    Notes:
        For docker-compose local development, service names don't follow the username pattern.
        Use the SPARK_CONNECT_URL_TEMPLATE environment variable to override the default pattern.

        For Kubernetes, the MCP server is in namespace (dev/prod/stage) and notebooks are in
        namespace (jupyterhub-dev/jupyterhub-prod/jupyterhub-stage), so we need cross-namespace DNS.

        IMPORTANT: The username is sanitized to be DNS-1123 compliant (underscores → hyphens)
        to match the Kubernetes Service name created by JupyterHub.
    """
    # Check if there's a custom template (useful for docker-compose)
    template = os.getenv("SPARK_CONNECT_URL_TEMPLATE")

    if template:
        # Template should contain {username} placeholder
        # Example: "sc://spark-notebook:15002" (no placeholder = shared)
        # Example: "sc://spark-notebook-{username}:15002"
        # Note: For templates, we use the sanitized username to match Kubernetes Service names
        sanitized_username = sanitize_k8s_name(username)
        url = template.format(username=sanitized_username)
        logger.info(
            f"Using custom Spark Connect URL template: {url} (username: {username} → {sanitized_username})"
        )
        return url

    # For Kubernetes: need to determine the environment and construct cross-namespace DNS
    # Environment can be dev, prod, or stage
    k8s_env = os.getenv("K8S_ENVIRONMENT", "dev")  # Default to dev if not specified

    # Sanitize username for DNS-1123 compliance (e.g., tian_gu_test → tian-gu-test)
    # This must match the Service name created by JupyterHub's spark_connect_service.py
    sanitized_username = sanitize_k8s_name(username)

    # Cross-namespace DNS pattern: {service}.{namespace}.svc.cluster.local
    # But short form works too: {service}.{namespace}
    notebook_namespace = f"jupyterhub-{k8s_env}"
    url = f"sc://jupyter-{sanitized_username}.{notebook_namespace}:{SPARK_CONNECT_PORT}"
    logger.info(
        f"Using Kubernetes cross-namespace Spark Connect URL: {url} (username: {username} → {sanitized_username})"
    )
    return url


def is_spark_connect_reachable(spark_connect_url: str, timeout: float = 2.0) -> bool:
    """
    Check if Spark Connect server is reachable via gRPC health check.

    This performs a proper gRPC connection attempt rather than just a TCP port check,
    which ensures the Spark Connect server is actually ready to accept connections.
    A simple TCP check can pass when:
    - A load balancer/service mesh fronts the port
    - Spark Connect is starting up but not ready
    - Network issues cause intermittent connectivity

    Args:
        spark_connect_url: Spark Connect URL (e.g., "sc://jupyter-user.namespace:15002")
        timeout: Connection timeout in seconds (default: 2.0)

    Returns:
        True if Spark Connect server is reachable and responding, False otherwise
    """
    try:
        # Parse URL to extract host and port
        # Format: sc://host:port
        url_str = spark_connect_url.replace("sc://", "tcp://")
        parsed = urlparse(url_str)
        host = parsed.hostname
        port = parsed.port or 15002  # Default Spark Connect port

        if not host:
            logger.info(f"Failed to parse hostname from URL: {spark_connect_url}")
            return False

        # First do a quick TCP check to avoid slow gRPC timeout on unreachable hosts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)  # Quick TCP check
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            logger.info(f"TCP port check failed for {spark_connect_url}")
            return False

        # TCP port is open, now verify with a lightweight gRPC connection attempt
        channel = grpc.insecure_channel(
            f"{host}:{port}",
            options=[
                ("grpc.connect_timeout_ms", int(timeout * 1000)),
                ("grpc.initial_reconnect_backoff_ms", 100),
                ("grpc.max_reconnect_backoff_ms", 500),
            ],
        )

        try:
            # Wait for channel to be ready with timeout
            grpc.channel_ready_future(channel).result(timeout=timeout)
            logger.info(f"gRPC channel ready for {spark_connect_url}")
            return True
        except grpc.FutureTimeoutError:
            logger.info(f"gRPC channel timeout for {spark_connect_url}")
            return False
        except Exception as e:
            logger.info(f"gRPC channel check failed for {spark_connect_url}: {e}")
            return False
        finally:
            channel.close()

    except Exception as e:
        logger.info(f"Health check failed for {spark_connect_url}: {e}")
        return False


def get_spark_context(
    request: Request,
    settings: Annotated[BERDLSettings, Depends(get_settings)],
) -> Generator[SparkContext, None, None]:
    """
    Get a SparkContext for executing Spark operations.

    This function determines the execution mode and returns an appropriate context:

    For Spark Connect mode:
        - Returns SparkContext with active SparkSession
        - Operations execute directly via the session

    For Standalone mode:
        - Returns SparkContext with subprocess info (no session created here)
        - Operations should be dispatched to ProcessPoolExecutor

    Connection Strategy:
    1. Try user's Spark Connect server (sc://jupyter-{username}:15002)
    2. Fall back to shared Spark cluster via ProcessPoolExecutor

    Usage in endpoints:
        @app.get("/query")
        def query_table(ctx: Annotated[SparkContext, Depends(get_spark_context)]):
            if ctx.is_standalone_subprocess:
                # Dispatch to process pool
                result = run_in_spark_process(
                    query_table_subprocess,
                    ctx.settings_dict,
                    query,
                    username=ctx.username
                )
            else:
                # Use Spark Connect session directly
                result = delta_service.query_delta_table(
                    spark=ctx.spark,
                    query=query,
                )
            return result

    Args:
        request: FastAPI request object (used to extract authenticated user)
        settings: BERDL settings from environment variables

    Yields:
        SparkContext with either a SparkSession (Connect) or subprocess info (Standalone)

    Raises:
        Exception: If user is not authenticated or credentials are missing
    """
    # Get authenticated user from request
    username = get_user_from_request(request)

    # Try to get auth token for operations that need it
    auth_token = None
    try:
        auth_token = get_token_from_request(request)
    except Exception:
        # Token not required for all operations
        pass

    logger.info(f"Creating Spark context for user: {username}")

    # Fetch user's MinIO credentials from the governance API
    if not auth_token:
        raise Exception(
            f"Cannot create Spark context: no auth token available for user {username}"
        )
    try:
        minio_access_key, minio_secret_key = fetch_user_minio_credentials(
            settings.GOVERNANCE_API_URL, auth_token
        )
        logger.debug(f"Fetched MinIO credentials for user {username}")
    except Exception as e:
        logger.error(
            f"Failed to fetch MinIO credentials for {username}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise Exception(
            f"Cannot create Spark context: failed to fetch MinIO credentials for user {username}: "
            f"{type(e).__name__}: {e}"
        )

    base_user_settings_dict = _build_user_settings_dict(
        settings=settings,
        username=username,
        auth_token=auth_token,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    )

    # Try Spark Connect first with gRPC health check
    spark_connect_url = construct_user_spark_connect_url(username)
    logger.info(f"Checking Spark Connect availability: {spark_connect_url}")

    # gRPC health check to see if Spark Connect server is ready
    use_spark_connect = is_spark_connect_reachable(spark_connect_url, timeout=2.0)
    spark_connect_failed = False

    if use_spark_connect:
        # =======================================================================
        # SPARK CONNECT MODE (FULLY CONCURRENT - NO LOCK)
        # =======================================================================
        # Connect sessions are client-only gRPC connections that don't conflict
        # with each other. Each request gets its own channel to the user's
        # remote Spark cluster, enabling full concurrency.
        # =======================================================================
        logger.info(
            "Spark Connect server reachable, creating session (no lock required)"
        )

        # Session creation phase - exceptions here trigger fallback to Standalone
        spark = None
        app_name = f"datalake_mcp_server_{username}"
        try:
            user_settings = BERDLSettings(
                SPARK_CONNECT_URL=AnyUrl(spark_connect_url),
                **{
                    k: v
                    for k, v in base_user_settings_dict.items()
                    if k != "SPARK_MASTER_URL"
                },
                SPARK_MASTER_URL=AnyUrl(base_user_settings_dict["SPARK_MASTER_URL"])
                if base_user_settings_dict.get("SPARK_MASTER_URL")
                else None,
            )

            # Use retry wrapper to handle INVALID_HANDLE.SESSION_CLOSED errors
            # that occur when the Spark Connect server has restarted
            spark = _get_spark_session_with_retry(
                app_name=app_name,
                settings=user_settings,
                use_spark_connect=True,
                max_retries=2,
            )

            # Validate session works before returning by making a quick server round-trip
            # This catches any remaining edge cases where session appears valid but isn't
            try:
                _ = spark.version
            except Exception as validation_error:
                logger.warning(
                    f"Session validation failed for {username}, session may be unusable: "
                    f"{type(validation_error).__name__}: {validation_error}"
                )
                raise

            logger.info(f"✅ Connected via Spark Connect for user {username}")

        except Exception as e:
            # Spark Connect session creation failed despite health check passing
            # This can happen due to race conditions or transient network issues
            logger.error(
                f"Spark Connect session creation failed for {username}, "
                f"falling back to shared cluster: {type(e).__name__}: {e}",
                exc_info=True,
            )
            spark_connect_failed = True

        # Yield phase - OUTSIDE the try/except so route execution errors propagate
        # correctly instead of triggering a fallback (which would yield twice)
        if spark is not None and not spark_connect_failed:
            # Populate settings_dict even in Connect mode so async query
            # endpoints can reuse get_spark_context and still have the
            # picklable settings needed for subprocess execution.
            connect_settings_dict = base_user_settings_dict.copy()
            if not connect_settings_dict.get("BERDL_POD_IP"):
                connect_settings_dict["BERDL_POD_IP"] = "0.0.0.0"
            connect_settings_dict["SPARK_CONNECT_URL"] = spark_connect_url

            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict=connect_settings_dict,
                app_name=app_name,
                username=username,
                auth_token=auth_token,
            )
            try:
                yield ctx
            finally:
                # IMPORTANT: Do NOT call spark.stop() here.
                # The yielded SparkContext may outlive this request when used
                # by async query background tasks (AsyncQueryExecutor). The
                # background task holds a reference to ctx after the HTTP 202
                # response is sent and this cleanup runs. Stopping the session
                # here would break in-flight background queries.
                # Connect sessions belong to the user's notebook pod and are
                # cleaned up when the pod terminates.
                logger.debug(
                    "Skipping spark.stop() for Spark Connect (cluster owned by user's notebook)"
                )
            return  # Successfully completed with Spark Connect

    if not use_spark_connect or spark_connect_failed:
        # =======================================================================
        # STANDALONE MODE (CONCURRENT VIA PROCESS POOL)
        # =======================================================================
        # Instead of creating a SparkSession here (which would cause JVM conflicts),
        # we return context info for subprocess execution. The route will dispatch
        # the operation to the ProcessPoolExecutor, where each worker creates its
        # own isolated SparkSession.
        # =======================================================================
        reason = (
            "Spark Connect session creation failed"
            if spark_connect_failed
            else "Spark Connect server unreachable"
        )
        logger.info(
            f"{reason}, returning Standalone subprocess context "
            f"(pool size: {STANDALONE_POOL_SIZE})"
        )

        # Use shared cluster master URL(s) - supports comma-separated list for load balancing
        # Example: "spark://master1:7077,spark://master2:7077,spark://master3:7077"
        master_urls_env = os.getenv(
            "SHARED_SPARK_MASTER_URL",
            "spark://sharedsparkclustermaster.prod:7077",
        )
        # Parse comma-separated list and randomly select one for load balancing
        master_urls = [url.strip() for url in master_urls_env.split(",") if url.strip()]
        if not master_urls:
            master_urls = ["spark://sharedsparkclustermaster.prod:7077"]

        shared_master_url = random.choice(master_urls)
        logger.debug(
            f"Selected Spark master: {shared_master_url} (from {len(master_urls)} available)"
        )

        # Prepare settings dict for subprocess (all values must be picklable)
        standalone_settings_dict = base_user_settings_dict.copy()
        if not standalone_settings_dict.get("BERDL_POD_IP"):
            standalone_settings_dict["BERDL_POD_IP"] = "0.0.0.0"
        standalone_settings_dict["SPARK_MASTER_URL"] = shared_master_url
        standalone_settings_dict["SPARK_CONNECT_URL"] = (
            "sc://localhost:15002"  # Placeholder
        )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        app_name = f"datalake_mcp_server_{username}_{timestamp}"

        # Yield SparkContext for subprocess execution - NO SESSION CREATED HERE
        ctx = SparkContext(
            spark=None,  # No session in main process
            is_standalone_subprocess=True,
            settings_dict=standalone_settings_dict,
            app_name=app_name,
            username=username,
            auth_token=auth_token,
        )
        try:
            yield ctx
        finally:
            # No cleanup needed - subprocess handles its own session lifecycle
            pass


# Keep get_spark_session for backward compatibility with existing routes
def get_spark_session(
    request: Request,
    settings: Annotated[BERDLSettings, Depends(get_settings)],
) -> Generator[SparkSession, None, None]:
    """
    [DEPRECATED] Get a SparkSession instance.

    This function is maintained for backward compatibility with routes that
    haven't been migrated to use get_spark_context yet.

    WARNING: This function creates a SparkSession in the main process for
    Standalone mode, which limits concurrency. New routes should use
    get_spark_context instead.

    Args:
        request: FastAPI request object (used to extract authenticated user)
        settings: BERDL settings from environment variables

    Yields:
        SparkSession configured for the user
    """
    username = get_user_from_request(request)
    auth_token = get_token_from_request(request)
    logger.info(f"Creating Spark session for user: {username}")

    # Fetch user's MinIO credentials from the governance API
    if not auth_token:
        raise Exception(
            f"Cannot create Spark session: no auth token available for user {username}"
        )
    try:
        minio_access_key, minio_secret_key = fetch_user_minio_credentials(
            settings.GOVERNANCE_API_URL, auth_token
        )
    except Exception as e:
        raise Exception(
            f"Cannot create Spark session: failed to fetch MinIO credentials for user {username}: "
            f"{type(e).__name__}: {e}"
        )

    base_user_settings = _build_user_settings_dict(
        settings=settings,
        username=username,
        auth_token=auth_token,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    )

    spark_connect_url = construct_user_spark_connect_url(username)
    use_spark_connect = is_spark_connect_reachable(spark_connect_url, timeout=2.0)
    spark_connect_failed = False

    if use_spark_connect:
        try:
            user_settings = BERDLSettings(
                SPARK_CONNECT_URL=AnyUrl(spark_connect_url),
                **base_user_settings,
            )
            spark = _get_spark_session(
                app_name=f"datalake_mcp_server_{username}",
                settings=user_settings,
                use_spark_connect=True,
            )
            logger.info(f"✅ Connected via Spark Connect for user {username}")
            yield spark
            return

        except Exception as e:
            logger.error(
                f"Spark Connect failed for {username}, falling back: {e}",
                exc_info=True,
            )
            spark_connect_failed = True

    if not use_spark_connect or spark_connect_failed:
        # Fallback to Standalone mode (in-process)
        master_urls_env = os.getenv(
            "SHARED_SPARK_MASTER_URL",
            "spark://sharedsparkclustermaster.prod:7077",
        )
        master_urls = [url.strip() for url in master_urls_env.split(",") if url.strip()]
        if not master_urls:
            master_urls = ["spark://sharedsparkclustermaster.prod:7077"]

        shared_master_url = random.choice(master_urls)

        fallback_settings = base_user_settings.copy()
        if not fallback_settings.get("BERDL_POD_IP"):
            fallback_settings["BERDL_POD_IP"] = "0.0.0.0"
        fallback_settings["SPARK_MASTER_URL"] = AnyUrl(shared_master_url)
        fallback_settings["SPARK_CONNECT_URL"] = AnyUrl("sc://localhost:15002")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        app_name = f"datalake_mcp_server_{username}_{timestamp}"

        spark = _get_spark_session(
            app_name=app_name,
            settings=BERDLSettings(**fallback_settings),
            use_spark_connect=False,
        )

        logger.info(f"✅ Connected via shared Spark cluster for user {username}")

        try:
            yield spark
        finally:
            try:
                hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem
                hadoop_fs.closeAll()
                spark.stop()
            except Exception as e:
                logger.error(f"Error stopping Spark session: {e}")


def get_trino_context(
    request: Request,
    settings: Annotated[BERDLSettings, Depends(get_settings)],
) -> Generator[TrinoContext, None, None]:
    """
    FastAPI dependency that creates a per-request Trino connection.

    Each request gets its own ``trino.dbapi.Connection`` with the
    authenticated user's credentials.  The connection is closed at the
    end of the request.
    """
    username = get_user_from_request(request)
    auth_token = get_token_from_request(request)

    if not auth_token:
        raise MissingTokenError(
            f"Cannot create Trino context: no auth token available for user {username}"
        )

    logger.info(f"Creating Trino context for user: {username}")

    try:
        minio_access_key, minio_secret_key = fetch_user_minio_credentials(
            settings.GOVERNANCE_API_URL, auth_token
        )
    except Exception as e:
        raise TrinoConnectionError(
            f"Cannot create Trino context: failed to fetch MinIO credentials "
            f"for user {username}: {type(e).__name__}: {e}"
        ) from e

    user_settings_dict = _build_user_settings_dict(
        settings=settings,
        username=username,
        auth_token=auth_token,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    )

    conn = create_trino_connection(
        username=username,
        auth_token=auth_token,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        trino_host=settings.TRINO_HOST,
        trino_port=settings.TRINO_PORT,
        hive_metastore_uri=str(settings.BERDL_HIVE_METASTORE_URI),
        minio_endpoint_url=settings.MINIO_ENDPOINT_URL,
        minio_secure=settings.MINIO_SECURE,
        polaris_catalog_uri=user_settings_dict.get("POLARIS_CATALOG_URI"),
        polaris_credential=user_settings_dict.get("POLARIS_CREDENTIAL"),
        polaris_personal_catalog=user_settings_dict.get("POLARIS_PERSONAL_CATALOG"),
        polaris_tenant_catalogs=user_settings_dict.get("POLARIS_TENANT_CATALOGS"),
    )

    settings_dict = {
        **user_settings_dict,
        "TRINO_HOST": settings.TRINO_HOST,
        "TRINO_PORT": settings.TRINO_PORT,
    }

    ctx = TrinoContext(
        connection=conn,
        username=username,
        auth_token=auth_token,
        settings_dict=settings_dict,
    )

    try:
        yield ctx
    finally:
        try:
            conn.close()
            logger.debug(f"Trino connection closed for user {username}")
        except Exception as e:
            logger.warning(f"Error closing Trino connection: {e}")
