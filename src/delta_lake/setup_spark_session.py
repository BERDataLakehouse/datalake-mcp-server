"""
Spark session setup for the Datalake MCP Server.

This is a simplified version of berdl_notebook_utils.setup_spark_session.py
that works in a shared multi-user service context. It only supports Spark Connect mode
and doesn't rely on environment variables (to avoid race conditions).

MAINTENANCE NOTE: This file is copied from:
/spark_notebook/notebook_utils/berdl_notebook_utils/setup_spark_session.py

When updating, copy the relevant Spark Connect logic from that file.
"""

import warnings
from datetime import datetime

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from src.settings import BERDLSettings

# Suppress Protobuf version warnings from PySpark Spark Connect
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# Suppress CANNOT_MODIFY_CONFIG warnings for Hive metastore settings in Spark Connect
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pyspark.sql.connect.conf"
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Fair scheduler configuration
SPARK_DEFAULT_POOL = "default"
SPARK_POOLS = [SPARK_DEFAULT_POOL, "highPriority"]

# Memory overhead percentages for Spark components
EXECUTOR_MEMORY_OVERHEAD = (
    0.1  # 10% overhead for executors (accounts for JVM + system overhead)
)
DRIVER_MEMORY_OVERHEAD = 0.05  # 5% overhead for driver (typically less memory pressure)

# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================


def convert_memory_format(memory_str: str, overhead_percentage: float = 0.1) -> str:
    """
    Convert memory format from profile format to Spark format with overhead adjustment.

    Args:
        memory_str: Memory string in profile format (supports B, KiB, MiB, GiB, TiB)
        overhead_percentage: Percentage of memory to reserve for system overhead (default: 0.1 = 10%)

    Returns:
        Memory string in Spark format with overhead accounted for
    """
    import re

    # Extract number and unit from memory string
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([kmgtKMGT]i?[bB]?)$", memory_str)
    if not match:
        raise ValueError(f"Invalid memory format: {memory_str}")

    value, unit = match.groups()
    value = float(value)

    # Convert to bytes for calculation
    unit_lower = unit.lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }

    # Remove trailing 'b' if present for lookup
    unit_key = (
        unit_lower.rstrip("b") + "b" if unit_lower.endswith("b") else unit_lower + "b"
    )
    if unit_key not in multipliers:
        unit_key = unit_lower

    bytes_value = value * multipliers.get(unit_key, multipliers["b"])

    # Apply overhead reduction (reserve percentage for system)
    adjusted_bytes = bytes_value * (1 - overhead_percentage)

    # Convert back to appropriate Spark unit (prefer GiB for larger values)
    if adjusted_bytes >= 1024**3:
        adjusted_value = adjusted_bytes / (1024**3)
        spark_unit = "g"
    elif adjusted_bytes >= 1024**2:
        adjusted_value = adjusted_bytes / (1024**2)
        spark_unit = "m"
    elif adjusted_bytes >= 1024:
        adjusted_value = adjusted_bytes / 1024
        spark_unit = "k"
    else:
        adjusted_value = adjusted_bytes
        spark_unit = ""

    # Format as integer to ensure Spark compatibility
    return f"{int(round(adjusted_value))}{spark_unit}"


def _get_executor_config(settings: BERDLSettings) -> dict[str, str]:
    """
    Get Spark executor and driver configuration based on profile settings.

    Args:
        settings: BERDLSettings instance with profile-specific configuration

    Returns:
        Dictionary of Spark executor and driver configuration
    """
    # Convert memory formats from profile to Spark format with overhead adjustment
    executor_memory = convert_memory_format(
        settings.SPARK_WORKER_MEMORY, EXECUTOR_MEMORY_OVERHEAD
    )
    driver_memory = convert_memory_format(
        settings.SPARK_MASTER_MEMORY, DRIVER_MEMORY_OVERHEAD
    )

    config = {
        # Driver configuration (critical for remote cluster connections)
        "spark.driver.memory": driver_memory,
        "spark.driver.cores": str(settings.SPARK_MASTER_CORES),
        # Executor configuration
        "spark.executor.instances": str(settings.SPARK_WORKER_COUNT),
        "spark.executor.cores": str(settings.SPARK_WORKER_CORES),
        "spark.executor.memory": executor_memory,
        # Disable dynamic allocation since we're setting explicit instances
        "spark.dynamicAllocation.enabled": "false",
        "spark.dynamicAllocation.shuffleTracking.enabled": "false",
    }

    return config


def _get_spark_defaults_conf() -> dict[str, str]:
    """
    Get Spark defaults configuration.
    """
    return {
        # Decommissioning
        "spark.decommission.enabled": "true",
        "spark.storage.decommission.rddBlocks.enabled": "true",
        # Broadcast join configurations
        "spark.sql.autoBroadcastJoinThreshold": "52428800",  # 50MB (default is 10MB)
        # Shuffle and compression configurations
        "spark.reducer.maxSizeInFlight": "96m",  # 96MB (default is 48MB)
        "spark.shuffle.file.buffer": "1m",  # 1MB (default is 32KB)
        # Delta Lake optimizations
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
    }


def _get_s3_conf(settings: BERDLSettings) -> dict[str, str]:
    """
    Get S3 configuration for MinIO.

    Args:
        settings: BERDLSettings instance with configuration

    Returns:
        Dictionary of S3/MinIO Spark configuration properties
    """
    # Use user's SQL warehouse
    warehouse_dir = f"s3a://cdm-lake/users-sql-warehouse/{settings.USER}/"
    event_log_dir = f"s3a://cdm-spark-job-logs/spark-job-logs/{settings.USER}/"

    config = {
        "spark.hadoop.fs.s3a.endpoint": settings.MINIO_ENDPOINT_URL,
        "spark.hadoop.fs.s3a.access.key": settings.MINIO_ACCESS_KEY,
        "spark.hadoop.fs.s3a.secret.key": settings.MINIO_SECRET_KEY,
        "spark.hadoop.fs.s3a.connection.ssl.enabled": str(
            settings.MINIO_SECURE
        ).lower(),
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.sql.warehouse.dir": warehouse_dir,
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": event_log_dir,
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.databricks.delta.retentionDurationCheck.enabled": "false",
    }

    return config


def _filter_immutable_spark_connect_configs(config: dict[str, str]) -> dict[str, str]:
    """
    Filter out configurations that cannot be modified in Spark Connect mode.

    These configs must be set server-side when the Spark Connect server starts.
    Attempting to set them from the client results in CANNOT_MODIFY_CONFIG warnings.

    Args:
        config: Dictionary of Spark configurations

    Returns:
        Filtered configuration dictionary with only mutable configs
    """
    immutable_configs = {
        # Cluster-level settings (must be set at master startup)
        "spark.decommission.enabled",
        "spark.storage.decommission.rddBlocks.enabled",
        "spark.reducer.maxSizeInFlight",
        "spark.shuffle.file.buffer",
        # Driver and executor resource configs (locked at server startup)
        "spark.driver.memory",
        "spark.driver.cores",
        "spark.executor.instances",
        "spark.executor.cores",
        "spark.executor.memory",
        "spark.dynamicAllocation.enabled",
        "spark.dynamicAllocation.shuffleTracking.enabled",
        # Event logging (locked at server startup)
        "spark.eventLog.enabled",
        "spark.eventLog.dir",
        # SQL extensions (must be loaded at startup)
        "spark.sql.extensions",
        "spark.sql.catalog.spark_catalog",
        # Hive catalog (locked at startup)
        "spark.sql.catalogImplementation",
        # Warehouse directory (locked at server startup)
        "spark.sql.warehouse.dir",
    }

    return {k: v for k, v in config.items() if k not in immutable_configs}


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def get_spark_session(
    app_name: str | None = None,
    settings: BERDLSettings | None = None,
) -> SparkSession:
    """
    Create and configure a Spark session with BERDL-specific settings for MCP server.

    This is a simplified version that only supports Spark Connect mode and works
    in a multi-user shared service context (no environment variable dependencies).

    Args:
        app_name: Application name. If None, generates a timestamp-based name
        settings: BERDLSettings instance with user-specific configuration

    Returns:
        Configured SparkSession instance connected via Spark Connect

    Raises:
        ValueError: If settings is None or required fields are missing

    Example:
        >>> user_settings = BERDLSettings(USER="username", SPARK_CONNECT_URL="sc://host:15002", ...)
        >>> spark = get_spark_session("MyApp", settings=user_settings)
    """
    if settings is None:
        raise ValueError("settings parameter is required for MCP server")

    # Generate app name if not provided
    if app_name is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        app_name = f"datalake_mcp_server_{timestamp}"

    # Build common configuration dictionary
    config: dict[str, str] = {"spark.app.name": app_name}

    # Add default Spark configurations
    config.update(_get_spark_defaults_conf())

    # Add profile-specific executor and driver configuration
    config.update(_get_executor_config(settings))

    # Configure S3 and Delta Lake
    config.update(_get_s3_conf(settings))

    # Configure Hive metastore
    config["hive.metastore.uris"] = str(settings.BERDL_HIVE_METASTORE_URI)
    config["spark.sql.catalogImplementation"] = "hive"
    config["spark.sql.hive.metastore.version"] = "4.0.0"
    config["spark.sql.hive.metastore.jars"] = "path"
    config["spark.sql.hive.metastore.jars.path"] = "/usr/local/spark/jars/*"

    # Spark Connect: filter out immutable configs and use remote URL
    config = _filter_immutable_spark_connect_configs(config)
    config["spark.remote"] = str(settings.SPARK_CONNECT_URL)

    # Create and configure Spark session
    spark_conf = SparkConf().setAll(list(config.items()))
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

    return spark
