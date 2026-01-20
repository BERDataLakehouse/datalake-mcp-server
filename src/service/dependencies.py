"""
Dependencies for FastAPI dependency injection.
"""

import json
import logging
import os
import re
import socket
import threading
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
from src.settings import BERDLSettings, get_settings

# Initialize the KBase auth dependency for use in routes
auth = KBaseHTTPBearer()

# Configure logging
logger = logging.getLogger(__name__)


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
# SESSION MODE LOCK
# =============================================================================
# PySpark's JVM can only have one session type (Connect or Regular) active at
# a time. This lock serializes ALL session creation to prevent mode conflicts.
# - Connect: Holds lock during session CREATION only, then releases
# - Standalone: Holds lock for ENTIRE request duration (create + query + cleanup)
#
# IMPORTANT: We use Semaphore(1) instead of RLock because FastAPI runs generator
# cleanup in a different thread than where the lock was acquired. RLock has
# thread affinity (only acquiring thread can release), causing "cannot release
# un-acquired lock" errors. Semaphore can be released by any thread.
# =============================================================================
_session_mode_lock = threading.Semaphore(1)


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


def get_spark_session(
    request: Request,
    settings: Annotated[BERDLSettings, Depends(get_settings)],
) -> Generator[SparkSession, None, None]:
    """
    Get a SparkSession instance configured for the authenticated user with automatic cleanup.

    This function tries to connect to the user's personal Spark Connect server first.
    If unavailable, it falls back to a shared Spark cluster. The session is created
    fresh for each request with the user's MinIO credentials, ensuring proper isolation.

    The session is automatically stopped after the request completes via generator cleanup.

    Connection Strategy:
    1. Try user's Spark Connect server (sc://jupyter-{username}:15002)
    2. Fall back to shared Spark cluster (spark://sharedsparkclustermaster.prod:7077)

    Session Mode Locking:
    - ALL session creation is serialized via _session_mode_lock
    - Connect: Lock held during creation only; queries run concurrently after
    - Standalone: Lock held for entire request (creation + query + cleanup)
    - This prevents mode conflicts (PySpark JVM only supports one session type)

    Usage in endpoints:
        @app.get("/databases")
        def get_databases(spark: Annotated[SparkSession, Depends(get_spark_session)]):
            # Use spark here
            databases = spark.sql("SHOW DATABASES").collect()
            return {"databases": [db.databaseName for db in databases]}
            # spark.stop() is automatically called after return

    Args:
        request: FastAPI request object (used to extract authenticated user)
        settings: BERDL settings from environment variables

    Yields:
        SparkSession configured for the user (either via Connect or direct cluster)

    Raises:
        Exception: If user is not authenticated or both connection methods fail
    """
    # Get authenticated user from request
    username = get_user_from_request(request)

    logger.info(f"Creating Spark session for user: {username}")

    # Read user's MinIO credentials from their home directory
    try:
        minio_access_key, minio_secret_key = read_user_minio_credentials(username)
        logger.debug(f"Loaded MinIO credentials for user {username}")
    except FileNotFoundError as e:
        logger.error(f"MinIO credentials file not found for {username}: {e}")
        raise Exception(
            f"Cannot create Spark session: MinIO credentials file not found for user {username}. "
            f"Ensure .berdl_minio_credentials exists in user's home directory at /home/{username}/.berdl_minio_credentials"
        )
    except Exception as e:
        logger.error(
            f"Failed to load MinIO credentials for {username}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise Exception(
            f"Cannot create Spark session: Error reading MinIO credentials for user {username}: {type(e).__name__}: {e}"
        )

    # Build base user-specific settings
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

    # Try Spark Connect first with gRPC health check
    spark_connect_url = construct_user_spark_connect_url(username)
    logger.info(f"Checking Spark Connect availability: {spark_connect_url}")

    # gRPC health check to see if Spark Connect server is ready
    use_spark_connect = is_spark_connect_reachable(spark_connect_url, timeout=2.0)
    spark_connect_failed = False

    if use_spark_connect:
        # =======================================================================
        # SPARK CONNECT MODE
        # Must acquire lock to prevent creating Connect session while Standalone
        # session is active (JVM can only have one session type at a time).
        # Lock is only held during session CREATION, not the entire request.
        # =======================================================================
        logger.info(
            "Spark Connect server reachable, acquiring session lock for Connect mode..."
        )

        try:
            with _session_mode_lock:
                logger.info("Session lock acquired for Connect mode")
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

            logger.info("Session lock released for Connect mode")

            # Yield session OUTSIDE the lock - Connect sessions can run concurrently
            # once created (they use the user's dedicated cluster)
            yield spark

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
        # STANDALONE MODE (hold lock for ENTIRE request to prevent mode conflicts)
        # =======================================================================
        reason = (
            "Spark Connect session creation failed"
            if spark_connect_failed
            else "Spark Connect server unreachable"
        )
        logger.info(
            f"{reason}, using shared Spark cluster. "
            "Acquiring standalone request lock..."
        )

        # CRITICAL: Hold the lock for the ENTIRE request lifecycle
        # This serializes standalone requests but prevents mode conflicts
        with _session_mode_lock:
            logger.info("Standalone request lock acquired")

            # Use shared cluster master URL
            shared_master_url = os.getenv(
                "SHARED_SPARK_MASTER_URL",
                "spark://sharedsparkclustermaster.prod:7077",
            )

            # Create fallback settings with updated SPARK_MASTER_URL
            fallback_settings_dict = base_user_settings.copy()
            fallback_settings_dict["SPARK_MASTER_URL"] = AnyUrl(shared_master_url)
            # Use a dummy connect URL to satisfy Pydantic validation
            fallback_settings_dict["SPARK_CONNECT_URL"] = AnyUrl("sc://localhost:15002")
            # Ensure BERDL_POD_IP is set for legacy mode
            if not fallback_settings_dict.get("BERDL_POD_IP"):
                fallback_settings_dict["BERDL_POD_IP"] = (
                    "0.0.0.0"  # Let Spark auto-detect
                )

            fallback_settings = BERDLSettings(**fallback_settings_dict)

            # Note: SPARK_REMOTE env var handling is done within _get_spark_session
            # when use_spark_connect=False to ensure it's cleared at the right time
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            spark = _get_spark_session(
                app_name=f"datalake_mcp_server_{username}_{timestamp}",
                settings=fallback_settings,
                use_spark_connect=False,
            )

            logger.info(
                f"✅ Connected via shared Spark cluster for user {username} at {shared_master_url} "
                f"(MinIO access key: {minio_access_key[:10]}...)"
            )

            try:
                # Yield session while holding the lock
                yield spark
            finally:
                # Cleanup: Must stop session to release cluster resources
                try:
                    logger.info("Stopping Spark session (shared cluster cleanup)")
                    # CRITICAL: Clear Hadoop FileSystem cache to prevent credential leakage
                    try:
                        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem
                        hadoop_fs.closeAll()
                        logger.info("Cleared Hadoop FileSystem cache")
                    except Exception as fs_err:
                        logger.info(f"Could not clear FileSystem cache: {fs_err}")

                    spark.stop()
                    logger.info("Spark session stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping Spark session: {e}", exc_info=True)

            logger.info("Standalone request lock released")
