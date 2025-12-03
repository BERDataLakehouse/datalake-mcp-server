"""
Spark session setup for the Datalake MCP Server.

This is copied from berdl_notebook_utils.setup_spark_session.py and adapted
for the MCP server context (multi-user shared service, no environment variables).

MAINTENANCE NOTE: This file is copied from:
/spark_notebook/notebook_utils/berdl_notebook_utils/setup_spark_session.py

When updating, copy the file and adapt the imports and warehouse configuration.
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Any

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from src.settings import BERDLSettings, get_settings

# Configure logging
logger = logging.getLogger(__name__)

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
    # Some Spark versions don't accept fractional memory values
    return f"{int(round(adjusted_value))}{spark_unit}"


def _get_executor_conf(
    settings: BERDLSettings, use_spark_connect: bool
) -> dict[str, str]:
    """
    Get Spark executor and driver configuration based on profile settings.

    Args:
        settings: BERDLSettings instance with profile-specific configuration
        use_spark_connect: bool indicating whether or not spark connect is to be used

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

    if use_spark_connect:
        conf_base = {"spark.remote": str(settings.SPARK_CONNECT_URL)}
    else:
        # Legacy mode: add driver/executor configs
        conf_base = {
            "spark.driver.host": str(os.environ.get("SPARK_DRIVER_HOST")),
            "spark.master": str(settings.SPARK_MASTER_URL),
        }

    return {
        **conf_base,
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
    }


def _get_delta_conf() -> dict[str, str]:
    return {
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.databricks.delta.retentionDurationCheck.enabled": "false",
        # Delta Lake optimizations
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
    }


def _get_hive_conf(settings: BERDLSettings) -> dict[str, str]:
    return {
        "hive.metastore.uris": str(settings.BERDL_HIVE_METASTORE_URI),
        "spark.sql.catalogImplementation": "hive",
        "spark.sql.hive.metastore.version": "4.0.0",
        "spark.sql.hive.metastore.jars": "path",
        "spark.sql.hive.metastore.jars.path": "/usr/local/spark/jars/*",
    }


def _get_s3_conf(
    settings: BERDLSettings, tenant_name: str | None = None
) -> dict[str, str]:
    """
    Get S3 configuration for MinIO.

    Args:
        settings: BERDLSettings instance with configuration
        tenant_name: Tenant/group name to use for SQL warehouse. If provided,
                    configures Spark to write tables to the tenant's SQL warehouse.
                    If None, uses the user's personal SQL warehouse.

    Returns:
        Dictionary of S3/MinIO Spark configuration properties

    """
    # Construct warehouse path directly (MCP server doesn't have access to governance API)
    if tenant_name:
        # Tenant warehouse: s3a://cdm-lake/tenant-sql-warehouse/{tenant_name}/
        warehouse_dir = f"s3a://cdm-lake/tenant-sql-warehouse/{tenant_name}/"
    else:
        # User warehouse: s3a://cdm-lake/users-sql-warehouse/{username}/
        warehouse_dir = f"s3a://cdm-lake/users-sql-warehouse/{settings.USER}/"

    event_log_dir = f"s3a://cdm-spark-job-logs/spark-job-logs/{settings.USER}/"

    return {
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
    }


IMMUTABLE_CONFIGS = {
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
    return {k: v for k, v in config.items() if k not in IMMUTABLE_CONFIGS}


def _set_scheduler_pool(spark: SparkSession, scheduler_pool: str) -> None:
    """Set the scheduler pool for the Spark session."""
    if scheduler_pool not in SPARK_POOLS:
        print(
            f"Warning: Scheduler pool '{scheduler_pool}' not in available pools: {SPARK_POOLS}. "
            f"Defaulting to '{SPARK_DEFAULT_POOL}'"
        )
        scheduler_pool = SPARK_DEFAULT_POOL

    spark.sparkContext.setLocalProperty("spark.scheduler.pool", scheduler_pool)


def generate_spark_conf(
    app_name: str | None = None,
    local: bool = False,
    use_delta_lake: bool = True,
    use_s3: bool = True,
    use_hive: bool = True,
    settings: BERDLSettings | None = None,
    tenant_name: str | None = None,
    use_spark_connect: bool = True,
) -> dict[str, str]:
    """Generate a spark session configuration dictionary from a set of input variables."""
    # Generate app name if not provided
    if app_name is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        app_name = f"kbase_spark_session_{timestamp}"

    # Build common configuration dictionary
    config: dict[str, str] = {"spark.app.name": app_name}

    if use_delta_lake:
        config.update(_get_delta_conf())

    if not local:
        # Add default Spark configurations
        config.update(_get_spark_defaults_conf())

        if settings is None:
            get_settings.cache_clear()
            settings = get_settings()

        # Add profile-specific executor and driver configuration
        config.update(_get_executor_conf(settings, use_spark_connect))

        if use_s3:
            config.update(_get_s3_conf(settings, tenant_name))

        if use_hive:
            config.update(_get_hive_conf(settings))

        if use_spark_connect:
            # Spark Connect: filter out immutable configs that cannot be modified from the client
            config = _filter_immutable_spark_connect_configs(config)

    return config


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def get_spark_session(
    app_name: str | None = None,
    local: bool = False,
    # TODO: switch to `use_delta_lake` for consistency with s3 / hive
    delta_lake: bool = True,
    scheduler_pool: str = SPARK_DEFAULT_POOL,
    use_s3: bool = True,
    use_hive: bool = True,
    settings: BERDLSettings | None = None,
    tenant_name: str | None = None,
    use_spark_connect: bool = True,
    override: dict[str, Any] | None = None,
) -> SparkSession:
    """
    Create and configure a Spark session with BERDL-specific settings.

    This function creates a Spark session configured for the BERDL environment,
    including support for Delta Lake, MinIO S3 storage, and tenant-aware warehouses.

    Args:
        app_name: Application name. If None, generates a timestamp-based name
        local: If True, creates a local Spark session; the only other allowable option is `delta_lake`
        delta_lake: If True, enables Delta Lake support with required JARs
        scheduler_pool: Fair scheduler pool name (default: "default")
        use_s3: if True, enables reading from and writing to s3
        use_hive: If True, enables Hive metastore integration
        settings: BERDLSettings instance. If None, creates new instance from env vars
        tenant_name: Tenant/group name to use for SQL warehouse location. If specified,
                     tables will be written to the tenant's SQL warehouse instead
                     of the user's personal warehouse.
        use_spark_connect: If True, uses Spark Connect instead of legacy mode
        override: dictionary of tag-value pairs to replace the values in the generated spark conf (e.g. for testing)

    Returns:
        Configured SparkSession instance

    Raises:
        EnvironmentError: If required environment variables are missing
        ValueError: If user is not a member of the specified tenant

    Example:
        >>> # Basic usage (user's personal warehouse)
        >>> spark = get_spark_session("MyApp")

        >>> # Using tenant warehouse (writes to tenant's SQL directory)
        >>> spark = get_spark_session("MyApp", tenant_name="research_team")

        >>> # With custom scheduler pool
        >>> spark = get_spark_session("MyApp", scheduler_pool="highPriority")

        >>> # Local development
        >>> spark = get_spark_session("TestApp", local=True)
    """
    config = generate_spark_conf(
        app_name,
        local,
        delta_lake,
        use_s3,
        use_hive,
        settings,
        tenant_name,
        use_spark_connect,
    )
    if override:
        config.update(override)

    # For legacy (non-Connect) mode, we must ensure SPARK_REMOTE is not set
    # PySpark 3.5+ checks this env var and will try to use Connect mode if set,
    # even when spark.remote is not in the config
    saved_spark_remote = None
    if not use_spark_connect:
        saved_spark_remote = os.environ.pop("SPARK_REMOTE", None)
        if saved_spark_remote:
            logger.info(
                f"Temporarily cleared SPARK_REMOTE env var (was: {saved_spark_remote})"
            )
        # Also ensure spark.remote is explicitly NOT in the config
        config.pop("spark.remote", None)

    try:
        spark_conf = SparkConf().setAll(list(config.items()))
        spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

        # Post-creation configuration (only for legacy mode with SparkContext)
        if not local and not use_spark_connect:
            spark.sparkContext.setLogLevel("DEBUG")
            _set_scheduler_pool(spark, scheduler_pool)

        return spark
    finally:
        # Restore SPARK_REMOTE if it was set
        if saved_spark_remote is not None:
            os.environ["SPARK_REMOTE"] = saved_spark_remote
            logger.info(f"Restored SPARK_REMOTE env var to: {saved_spark_remote}")
