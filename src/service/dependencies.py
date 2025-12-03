"""
Dependencies for FastAPI dependency injection.
"""

import json
import logging
import os
import socket
from pathlib import Path
from typing import Annotated, Generator
from urllib.parse import urlparse

from fastapi import Depends, Request
from pydantic import AnyUrl
from pyspark.sql import SparkSession

# Use MCP server's local copy of spark session utilities
# (copied from berdl_notebook_utils but adapted for shared multi-user service)
from src.delta_lake.setup_spark_session import get_spark_session as _get_spark_session
from src.service import app_state
from src.service.http_bearer import KBaseHTTPBearer
from src.settings import BERDLSettings, get_settings

# Initialize the KBase auth dependency for use in routes
auth = KBaseHTTPBearer()

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_SPARK_POOL = "default"
SPARK_CONNECT_PORT = "15002"


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
        raise Exception("User not authenticated. Authorization header required.")
    return user.user


def construct_user_spark_connect_url(username: str) -> str:
    """
    Construct the Spark Connect URL for a specific user's notebook pod.

    In BERDL, each user has their own notebook pod with a Spark Connect server.
    The URL pattern differs between environments:
    - Docker Compose (local dev): sc://spark-notebook:15002 or sc://spark-notebook-{username}:15002
    - Kubernetes (prod/stage/dev): sc://jupyter-{username}.jupyterhub-{env}:15002

    Args:
        username: KBase username

    Returns:
        User-specific Spark Connect URL

    Notes:
        For docker-compose local development, service names don't follow the username pattern.
        Use the SPARK_CONNECT_URL_TEMPLATE environment variable to override the default pattern.

        For Kubernetes, the MCP server is in namespace (dev/prod/stage) and notebooks are in
        namespace (jupyterhub-dev/jupyterhub-prod/jupyterhub-stage), so we need cross-namespace DNS.
    """
    # Check if there's a custom template (useful for docker-compose)
    template = os.getenv("SPARK_CONNECT_URL_TEMPLATE")

    if template:
        # Template should contain {username} placeholder
        # Example: "sc://spark-notebook:15002" (no placeholder = shared)
        # Example: "sc://spark-notebook-{username}:15002"
        url = template.format(username=username)
        logger.info(f"Using custom Spark Connect URL template: {url}")
        return url

    # For Kubernetes: need to determine the environment and construct cross-namespace DNS
    # Environment can be dev, prod, or stage
    k8s_env = os.getenv("K8S_ENVIRONMENT", "dev")  # Default to dev if not specified

    # Cross-namespace DNS pattern: {service}.{namespace}.svc.cluster.local
    # But short form works too: {service}.{namespace}
    notebook_namespace = f"jupyterhub-{k8s_env}"
    url = f"sc://jupyter-{username}.{notebook_namespace}:{SPARK_CONNECT_PORT}"
    logger.info(f"Using Kubernetes cross-namespace Spark Connect URL: {url}")
    return url


def is_spark_connect_reachable(spark_connect_url: str, timeout: float = 1.0) -> bool:
    """
    Quick TCP check if Spark Connect server is reachable.

    Args:
        spark_connect_url: Spark Connect URL (e.g., "sc://jupyter-user.namespace:15002")
        timeout: Connection timeout in seconds (default: 1.0)

    Returns:
        True if port is reachable, False otherwise
    """
    try:
        # Parse URL to extract host and port
        # Format: sc://host:port
        url_str = spark_connect_url.replace("sc://", "tcp://")
        parsed = urlparse(url_str)
        host = parsed.hostname
        port = parsed.port or 15002  # Default Spark Connect port

        if not host:
            logger.debug(f"Failed to parse hostname from URL: {spark_connect_url}")
            return False

        # Attempt TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        return result == 0
    except Exception as e:
        logger.debug(f"TCP check failed for {spark_connect_url}: {e}")
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
    spark = None
    try:
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

        # Try Spark Connect first with TCP pre-flight check
        spark_connect_url = construct_user_spark_connect_url(username)
        logger.info(f"Checking Spark Connect availability: {spark_connect_url}")

        # Quick TCP check to see if Spark Connect port is reachable
        if is_spark_connect_reachable(spark_connect_url, timeout=1.0):
            logger.info(
                f"Spark Connect port reachable, attempting connection: {spark_connect_url}"
            )
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
            except Exception as e:
                logger.warning(f"Spark Connect session creation failed: {e}")
                spark = None  # Trigger fallback
        else:
            logger.info("Spark Connect port unreachable, skipping to shared cluster")
            spark = None

        # Fall back to shared Spark cluster if needed
        if spark is None:
            logger.info("Falling back to shared Spark cluster...")
            try:
                # Use shared cluster master URL
                shared_master_url = os.getenv(
                    "SHARED_SPARK_MASTER_URL",
                    "spark://sharedsparkclustermaster.prod:7077",
                )

                # Create fallback settings with updated SPARK_MASTER_URL
                fallback_settings_dict = base_user_settings.copy()
                fallback_settings_dict["SPARK_MASTER_URL"] = AnyUrl(shared_master_url)

                fallback_settings = BERDLSettings(**fallback_settings_dict)

                spark = _get_spark_session(
                    app_name=f"datalake_mcp_server_{username}_shared",
                    settings=fallback_settings,
                    use_spark_connect=False,
                )

                logger.info(
                    f"✅ Connected via shared Spark cluster for user {username} at {shared_master_url}"
                )

            except Exception as cluster_error:
                logger.error(
                    f"Both Spark Connect and shared cluster failed for user {username}"
                )
                logger.error(f"Shared cluster error: {cluster_error}")
                raise Exception(
                    f"Unable to create Spark session for user {username}. "
                    f"Spark Connect unavailable, shared cluster failed: {cluster_error}. "
                    f"Please contact a BERDL administrator."
                )

        # Yield the spark session to the endpoint
        logger.debug("Spark session created, yielding to endpoint")
        yield spark

    finally:
        # Always stop the session, even if an exception occurred
        if spark is not None:
            try:
                logger.info("Stopping Spark session (cleanup)")
                spark.stop()
                logger.debug("Spark session stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Spark session: {e}", exc_info=True)
