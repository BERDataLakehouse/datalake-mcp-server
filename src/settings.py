"""
BERDL environment configuration for datalake-mcp-server.

This module provides Pydantic-based settings validation for the MCP server,
compatible with the spark_notebook environment.
"""

import logging
from functools import lru_cache

from pydantic import AnyHttpUrl, AnyUrl, Field
from pydantic_settings import BaseSettings

# Configure logging
logger = logging.getLogger(__name__)


class BERDLSettings(BaseSettings):
    """
    BERDL environment configuration using Pydantic Settings.

    This configuration is for the shared MCP server that dynamically connects
    to user-specific Spark Connect servers based on authenticated requests.

    Most fields have defaults since user-specific values come from
    their individual notebook environments.
    """

    # Core authentication (set dynamically per request)
    KBASE_AUTH_TOKEN: str = Field(
        default="", description="KBase auth token (set per request)"
    )
    USER: str = Field(
        default="", description="KBase username (set dynamically per request)"
    )

    # MinIO configuration (used for constructing S3 paths, actual creds from user's Spark)
    MINIO_ENDPOINT_URL: str = Field(
        default="minio:9002", description="MinIO endpoint (hostname:port)"
    )
    MINIO_ACCESS_KEY: str = Field(
        default="", description="MinIO access key (from user's Spark)"
    )
    MINIO_SECRET_KEY: str = Field(
        default="", description="MinIO secret key (from user's Spark)"
    )
    MINIO_SECURE: bool = Field(
        default=False, description="Use secure connection (True/False)"
    )

    # Spark configuration
    SPARK_HOME: str = Field(
        default="/usr/local/spark", description="Spark installation directory"
    )
    SPARK_MASTER_URL: AnyUrl | None = Field(
        default=None,
        description="Spark Master URL (spark://host:port) - not used in Connect mode",
    )
    SPARK_CONNECT_URL: AnyUrl = Field(
        default=AnyUrl("sc://localhost:15002"),
        description=(
            "Spark Connect URL - CONSTRUCTED DYNAMICALLY per user. "
            "In Kubernetes: sc://jupyter-{username}.jupyterhub-{env}:15002 (cross-namespace DNS). "
            "Override pattern with SPARK_CONNECT_URL_TEMPLATE env var for docker-compose. "
            "Set K8S_ENVIRONMENT (dev/prod/stage) to control namespace."
        ),
    )

    # Hive configuration
    BERDL_HIVE_METASTORE_URI: AnyUrl = Field(
        default=AnyUrl("thrift://hive-metastore:9083"),
        description="Hive Metastore Thrift endpoint",
    )

    # Profile-specific Spark configuration (defaults for session creation)
    SPARK_WORKER_COUNT: int = Field(
        default=1, description="Number of Spark workers from profile"
    )
    SPARK_WORKER_CORES: int = Field(
        default=1, description="Cores per Spark worker from profile"
    )
    SPARK_WORKER_MEMORY: str = Field(
        default="2GiB",
        pattern=r"^\d+[kmgKMGT]i?[bB]?$",
        description="Memory per Spark worker from profile",
    )
    SPARK_MASTER_CORES: int = Field(
        default=1, description="Cores for Spark master from profile"
    )
    SPARK_MASTER_MEMORY: str = Field(
        default="1GiB",
        pattern=r"^\d+[kmgKMGT]i?[bB]?$",
        description="Memory for Spark master from profile",
    )

    # Data Governance API configuration
    GOVERNANCE_API_URL: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://minio-manager-service:8000"),
        description="Data governance API endpoint",
    )

    # Optional: Pod IP for legacy mode (not used in MCP server)
    BERDL_POD_IP: str | None = Field(
        default=None, description="Pod IP for legacy Spark mode"
    )


@lru_cache(maxsize=1)
def get_settings() -> BERDLSettings:
    """
    Get cached BERDLSettings instance. Only creates the object once.

    Returns:
        BERDLSettings: Cached settings instance

    Raises:
        ValidationError: If environment variables are missing or invalid
    """
    return BERDLSettings()
