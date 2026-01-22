"""
Dependencies for FastAPI dependency injection.
"""

import json
import logging
import os
import random
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Generator
from urllib.parse import urlparse

import grpc
from fastapi import Depends, Request
from pydantic import AnyUrl
from pyspark.sql import SparkSession

# Use MCP server's local copy of spark session utilities
# (copied from berdl_notebook_utils but adapted for shared multi-user service)
from src.delta_lake.setup_spark_session import get_spark_session as _get_spark_session
from src.service import app_state
from src.service.exceptions import MissingTokenError
from src.service.http_bearer import KBaseHTTPBearer
from src.service.spark_session_pool import STANDALONE_POOL_SIZE
from src.settings import BERDLSettings, get_settings

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
        - settings_dict is None

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


def read_user_minio_credentials(username: str) -> tuple[str, str]:
    """
    Read user's MinIO credentials from their home directory.

    Each user has a .berdl_minio_credentials file in their home directory with format:
    {"username": "user", "access_key": "key", "secret_key": "secret"}

    Args:
        username: KBase username

    Returns:
        Tuple of (access_key, secret_key)

    Raises:
        FileNotFoundError: If credentials file doesn't exist
        ValueError: If credentials file is malformed
    """
    # Construct path to credentials file
    creds_path = Path(f"/home/{username}/.berdl_minio_credentials")

    logger.debug(f"Reading MinIO credentials from: {creds_path}")

    if not creds_path.exists():
        raise FileNotFoundError(
            f"MinIO credentials file not found at {creds_path}. "
            f"User {username} must have .berdl_minio_credentials in their home directory."
        )

    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)

        access_key = creds.get("access_key")
        secret_key = creds.get("secret_key")

        if not access_key or not secret_key:
            raise ValueError(
                f"Invalid credentials format in {creds_path}. "
                f'Expected: {{"username": "user", "access_key": "key", "secret_key": "secret"}}'
            )

        logger.info(f"Successfully loaded MinIO credentials for user: {username}")
        return access_key, secret_key

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse MinIO credentials file {creds_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading MinIO credentials for {username}: {e}")
        raise


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

    Args:
        request: FastAPI request object

    Returns:
        Token string (without 'Bearer ' prefix) or None if not present
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    return None


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

    # Read user's MinIO credentials from their home directory
    try:
        minio_access_key, minio_secret_key = read_user_minio_credentials(username)
        logger.debug(f"Loaded MinIO credentials for user {username}")
    except FileNotFoundError as e:
        logger.error(f"MinIO credentials file not found for {username}: {e}")
        raise Exception(
            f"Cannot create Spark context: MinIO credentials file not found for user {username}. "
            f"Ensure .berdl_minio_credentials exists in user's home directory at /home/{username}/.berdl_minio_credentials"
        )
    except Exception as e:
        logger.error(
            f"Failed to load MinIO credentials for {username}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise Exception(
            f"Cannot create Spark context: Error reading MinIO credentials for user {username}: {type(e).__name__}: {e}"
        )

    # Build base user-specific settings as picklable dict
    # Note: URLs are converted to strings for pickling
    base_user_settings_dict = {
        "USER": username,
        "MINIO_ACCESS_KEY": minio_access_key,
        "MINIO_SECRET_KEY": minio_secret_key,
        "MINIO_ENDPOINT_URL": settings.MINIO_ENDPOINT_URL,
        "MINIO_SECURE": settings.MINIO_SECURE,
        "SPARK_HOME": settings.SPARK_HOME,
        "SPARK_MASTER_URL": str(settings.SPARK_MASTER_URL)
        if settings.SPARK_MASTER_URL
        else None,
        "BERDL_HIVE_METASTORE_URI": settings.BERDL_HIVE_METASTORE_URI,
        "SPARK_WORKER_COUNT": settings.SPARK_WORKER_COUNT,
        "SPARK_WORKER_CORES": settings.SPARK_WORKER_CORES,
        "SPARK_WORKER_MEMORY": settings.SPARK_WORKER_MEMORY,
        "SPARK_MASTER_CORES": settings.SPARK_MASTER_CORES,
        "SPARK_MASTER_MEMORY": settings.SPARK_MASTER_MEMORY,
        "GOVERNANCE_API_URL": settings.GOVERNANCE_API_URL,
        "BERDL_POD_IP": settings.BERDL_POD_IP,
    }

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

            app_name = f"datalake_mcp_server_{username}"
            spark = _get_spark_session(
                app_name=app_name,
                settings=user_settings,
                use_spark_connect=True,
            )
            logger.info(f"✅ Connected via Spark Connect for user {username}")

            # Yield SparkContext with active session
            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict={},  # Not needed for Connect mode
                app_name=app_name,
                username=username,
                auth_token=auth_token,
            )
            yield ctx

            # No cleanup: Connect sessions belong to user's notebook pod
            logger.debug(
                "Skipping spark.stop() for Spark Connect (cluster owned by user's notebook)"
            )
            return  # Successfully completed with Spark Connect

        except Exception as e:
            # Spark Connect session creation failed despite health check passing
            # This can happen due to race conditions or transient network issues
            logger.error(
                f"Spark Connect session creation failed for {username}, "
                f"falling back to shared cluster: {type(e).__name__}: {e}",
                exc_info=True,
            )
            spark_connect_failed = True

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
        yield ctx
        # No cleanup needed - subprocess handles its own session lifecycle


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
    logger.info(f"Creating Spark session for user: {username}")

    # Read user's MinIO credentials
    try:
        minio_access_key, minio_secret_key = read_user_minio_credentials(username)
    except FileNotFoundError:
        raise Exception(
            f"Cannot create Spark session: MinIO credentials file not found for user {username}. "
            f"Ensure .berdl_minio_credentials exists at /home/{username}/.berdl_minio_credentials"
        )
    except Exception as e:
        raise Exception(
            f"Cannot create Spark session: Error reading MinIO credentials for user {username}: {type(e).__name__}: {e}"
        )

    base_user_settings = {
        "USER": username,
        "MINIO_ACCESS_KEY": minio_access_key,
        "MINIO_SECRET_KEY": minio_secret_key,
        "MINIO_ENDPOINT_URL": settings.MINIO_ENDPOINT_URL,
        "MINIO_SECURE": settings.MINIO_SECURE,
        "SPARK_HOME": settings.SPARK_HOME,
        "SPARK_MASTER_URL": settings.SPARK_MASTER_URL,
        "BERDL_HIVE_METASTORE_URI": settings.BERDL_HIVE_METASTORE_URI,
        "SPARK_WORKER_COUNT": settings.SPARK_WORKER_COUNT,
        "SPARK_WORKER_CORES": settings.SPARK_WORKER_CORES,
        "SPARK_WORKER_MEMORY": settings.SPARK_WORKER_MEMORY,
        "SPARK_MASTER_CORES": settings.SPARK_MASTER_CORES,
        "SPARK_MASTER_MEMORY": settings.SPARK_MASTER_MEMORY,
        "GOVERNANCE_API_URL": settings.GOVERNANCE_API_URL,
        "BERDL_POD_IP": settings.BERDL_POD_IP,
    }

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
