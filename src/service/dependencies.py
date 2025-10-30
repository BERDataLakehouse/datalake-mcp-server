"""
Dependencies for FastAPI dependency injection.
"""

import json
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request
from pydantic import AnyUrl
from pyspark.sql import SparkSession

# Use shared Spark utilities from berdl_notebook_utils
from berdl_notebook_utils.setup_spark_session import (
    get_spark_session as _get_spark_session,
)
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


def get_spark_session(
    request: Request,
    settings: Annotated[BERDLSettings, Depends(get_settings)],
) -> SparkSession:
    """
    Get a SparkSession instance configured for the authenticated user.

    This function connects to the user's personal Spark Connect server running
    in their notebook pod. Each user has a dedicated Spark cluster (master + workers)
    and a Spark Connect server in their notebook pod.

    Args:
        request: FastAPI request object (used to extract authenticated user)
        settings: BERDL settings from environment variables

    Returns:
        SparkSession configured to connect to the user's Spark Connect server

    Raises:
        Exception: If user is not authenticated or Spark Connect server is unreachable
    """
    # Get authenticated user from request
    username = get_user_from_request(request)

    logger.info(f"Creating Spark session for user: {username}")

    # Construct user-specific Spark Connect URL
    spark_connect_url = construct_user_spark_connect_url(username)

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

    # Build user-specific settings with dynamic Spark Connect URL and MinIO credentials
    # Use MCP server's BERDLSettings (compatible with notebook's get_spark_session)
    user_settings = BERDLSettings(
        USER=username,
        SPARK_CONNECT_URL=AnyUrl(spark_connect_url),
        MINIO_ACCESS_KEY=minio_access_key,
        MINIO_SECRET_KEY=minio_secret_key,
        MINIO_ENDPOINT_URL=settings.MINIO_ENDPOINT_URL,
        MINIO_SECURE=settings.MINIO_SECURE,
        SPARK_HOME=settings.SPARK_HOME,
        SPARK_MASTER_URL=settings.SPARK_MASTER_URL,
        BERDL_HIVE_METASTORE_URI=settings.BERDL_HIVE_METASTORE_URI,
        SPARK_WORKER_COUNT=settings.SPARK_WORKER_COUNT,
        SPARK_WORKER_CORES=settings.SPARK_WORKER_CORES,
        SPARK_WORKER_MEMORY=settings.SPARK_WORKER_MEMORY,
        SPARK_MASTER_CORES=settings.SPARK_MASTER_CORES,
        SPARK_MASTER_MEMORY=settings.SPARK_MASTER_MEMORY,
        GOVERNANCE_API_URL=settings.GOVERNANCE_API_URL,
        BERDL_POD_IP=settings.BERDL_POD_IP,
    )

    try:
        return _get_spark_session(
            app_name=f"datalake_mcp_server_{username}",
            use_spark_connect=True,  # Connect to user's Spark Connect server
            scheduler_pool=str(os.getenv("SPARK_POOL", DEFAULT_SPARK_POOL)),
            settings=user_settings,
        )
    except Exception as e:
        logger.error(
            f"Failed to connect to Spark Connect server for user {username}: {e}"
        )
        raise Exception(
            f"Unable to connect to Spark session for user {username}. "
            f"Please ensure you are logged into BERDL JupyterHub and your notebook is running. "
            f"Error: {str(e)}"
        )
