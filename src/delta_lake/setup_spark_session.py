"""
Spark session setup for the Datalake MCP Server.

This is copied from berdl_notebook_utils.setup_spark_session.py and adapted
for the MCP server context (multi-user shared service, no environment variables).

MAINTENANCE NOTE: This file is copied from:
/spark_notebook/notebook_utils/berdl_notebook_utils/setup_spark_session.py

When updating, copy the file and adapt the imports and warehouse configuration.
"""

import contextlib
import logging
import os
import re
import socket
import threading
import time
import warnings
from datetime import datetime
from typing import Any

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.connect.session import SparkSession as RemoteSparkSession

from src.settings import BERDLSettings, get_settings

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# THREAD SAFETY
# =============================================================================
# Lock for Standalone mode session creation only.
# PySpark's SparkSession.builder.getOrCreate() is NOT thread-safe and manipulates
# global state (environment variables, builder._options). Without this lock,
# concurrent requests can cause undefined behavior and deadlocks.
#
# NOTE: Spark Connect mode does NOT need this lock because Connect sessions
# are client-only gRPC connections that don't share JVM state. Each Connect
# session can be created concurrently without conflicts.
_standalone_session_lock = threading.Lock()

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

# Session errors that indicate a stale/closed session requiring recreation
# These errors occur when the Spark Connect server has restarted or the session has timed out
SESSION_CLOSED_ERRORS = [
    "INVALID_HANDLE.SESSION_CLOSED",
    "INVALID_HANDLE.SESSION_CHANGED",
    "INVALID_HANDLE.SESSION_NOT_FOUND",
]

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
        driver_host = socket.gethostbyname(socket.gethostname())

        conf_base = {
            "spark.driver.host": driver_host,
            "spark.driver.bindAddress": "0.0.0.0",  # Bind to all interfaces
            "spark.master": str(settings.SPARK_MASTER_URL),
        }
        logger.info(f"Legacy mode: driver.host={driver_host}, bindAddress=0.0.0.0")

    return {
        **conf_base,
        # Driver configuration (critical for remote cluster connections)
        "spark.driver.memory": driver_memory,
        "spark.driver.cores": str(settings.SPARK_MASTER_CORES),
        # Executor configuration
        "spark.executor.instances": str(settings.SPARK_WORKER_COUNT),
        "spark.executor.cores": str(settings.SPARK_WORKER_CORES),
        "spark.executor.memory": executor_memory,
        # Limit total cores for standalone mode (without this, Spark allocates all available)
        "spark.cores.max": str(
            settings.SPARK_WORKER_CORES * settings.SPARK_WORKER_COUNT
        ),
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
        # CRITICAL: Disable filesystem cache to prevent credential leakage between users
        # Hadoop caches S3AFileSystem instances at the JVM level keyed by URI scheme.
        # Without this, the first user's credentials are cached and reused for all
        # subsequent users, causing 403 Access Denied errors in shared cluster mode.
        "spark.hadoop.fs.s3a.impl.disable.cache": "true",
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


def _clear_spark_env_for_mode_switch(use_spark_connect: bool) -> None:
    """
    Clear PySpark environment variables to allow clean mode switching.

    PySpark 3.4+ uses several environment variables to determine whether to use
    Spark Connect or classic mode. These persist across sessions and can cause
    conflicts when switching modes within the same process.

    Environment variables managed:
    - SPARK_CONNECT_MODE_ENABLED: Set to "1" when Connect mode is used
    - SPARK_REMOTE: Spark Connect URL
    - SPARK_LOCAL_REMOTE: Set when using local Connect server
    - MASTER: Spark master URL (classic mode)
    - SPARK_API_MODE: Can be "classic" or "connect"

    Args:
        use_spark_connect: If True, clears legacy mode vars; if False, clears Connect vars
    """
    if use_spark_connect:
        # Switching TO Spark Connect: clear legacy mode variables
        env_vars_to_clear = ["MASTER"]
        logger.debug(
            f"Clearing legacy mode env vars for Spark Connect: {env_vars_to_clear}"
        )
    else:
        # Switching TO legacy mode: clear ALL Spark Connect related variables
        env_vars_to_clear = [
            "SPARK_CONNECT_MODE_ENABLED",  # Critical: forces Connect mode if present
            "SPARK_REMOTE",  # Connect URL
            "SPARK_LOCAL_REMOTE",  # Local Connect server flag
        ]
        logger.debug(
            f"Clearing Connect mode env vars for legacy mode: {env_vars_to_clear}"
        )

    for var in env_vars_to_clear:
        if var in os.environ:
            logger.info(f"Clearing environment variable: {var}={os.environ[var]}")
            del os.environ[var]


def _clear_stale_spark_connect_sessions() -> None:
    """
    Clear cached Spark Connect sessions to prevent INVALID_HANDLE.SESSION_CLOSED errors.

    When a Spark Connect server restarts or a session times out, the client-side
    cached session handle becomes stale. PySpark's getOrCreate() checks for cached
    sessions first (via _active_session and _default_session), and if found, tries
    to reuse them. When the handle is stale, this causes:

        [INVALID_HANDLE.SESSION_CLOSED] The handle ... is invalid. Session was closed.

    This function clears the thread-local _active_session and global _default_session
    to force getOrCreate() to create a fresh session via builder.create().

    This is safe for Spark Connect mode because:
    1. Connect sessions are client-side gRPC connections (no JVM state)
    2. Each Connect session has its own session_id on the server
    3. Clearing the cache just means we'll create a new session_id

    This is NOT needed for Standalone mode where sessions share JVM state.
    """
    logger.debug("Clearing stale Spark Connect sessions...")
    cleared_active = False
    cleared_default = False

    try:
        with RemoteSparkSession._lock:
            # Clear thread-local active session
            if hasattr(RemoteSparkSession._active_session, "session"):
                old_session = getattr(
                    RemoteSparkSession._active_session, "session", None
                )
                if old_session is not None:
                    session_id = getattr(old_session, "session_id", "unknown")
                    logger.info(
                        f"Clearing stale Spark Connect active session: "
                        f"session_id={session_id}"
                    )
                    # Best-effort cleanup: try to stop the old session to release
                    # gRPC channels/resources before clearing the reference
                    try:
                        old_session.stop()
                    except Exception:
                        # Session may already be invalid - that's expected
                        logger.debug(
                            f"Could not stop stale active session {session_id} "
                            f"(may already be closed)"
                        )
                    RemoteSparkSession._active_session.session = None
                    cleared_active = True

            # Clear global default session
            if RemoteSparkSession._default_session is not None:
                session_id = getattr(
                    RemoteSparkSession._default_session, "session_id", "unknown"
                )
                logger.info(
                    f"Clearing stale Spark Connect default session: "
                    f"session_id={session_id}"
                )
                # Best-effort cleanup: try to stop the old session
                try:
                    RemoteSparkSession._default_session.stop()
                except Exception:
                    # Session may already be invalid - that's expected
                    logger.debug(
                        f"Could not stop stale default session {session_id} "
                        f"(may already be closed)"
                    )
                RemoteSparkSession._default_session = None
                cleared_default = True

        # Log summary of what was cleared
        if cleared_active or cleared_default:
            logger.info(
                f"Cleared stale sessions: active={cleared_active}, default={cleared_default}"
            )
        else:
            logger.debug("No stale sessions found to clear")

    except Exception as e:
        # Don't fail the session creation just because cache clearing failed
        # Include traceback for debugging (per Copilot feedback)
        logger.warning(
            f"Failed to clear stale Spark Connect sessions: {e}", exc_info=True
        )


def _is_session_error(error: Exception) -> bool:
    """
    Check if an error indicates a stale/closed session that requires recreation.

    These errors occur when the Spark Connect server has restarted or the session
    has timed out, making the client's cached session_id invalid.

    Args:
        error: The exception to check

    Returns:
        True if the error is a session-related error that can be retried
    """
    error_str = str(error)
    return any(err in error_str for err in SESSION_CLOSED_ERRORS)


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

    # ==========================================================================
    # CRITICAL: Thread-safe session creation (Standalone mode only)
    # ==========================================================================
    # PySpark's SparkSession.builder is NOT thread-safe. The following operations
    # must be performed atomically to prevent race conditions:
    #
    # 1. Clearing environment variables (os.environ modifications)
    # 2. Clearing builder._options
    # 3. Creating SparkConf and calling getOrCreate()
    #
    # For Spark Connect mode: No lock needed - Connect sessions are client-only
    # gRPC connections that don't share JVM builder state. Each Connect session
    # can be created concurrently without conflicts.
    #
    # For Standalone mode: Lock required - sessions share the same JVM and
    # builder state. Without this lock, concurrent requests can cause:
    # - Environment variable corruption between threads
    # - Builder options being modified mid-creation
    # - Undefined behavior leading to service hangs
    # ==========================================================================

    # Use nullcontext for Spark Connect (no lock), real lock for Standalone
    session_lock = (
        contextlib.nullcontext() if use_spark_connect else _standalone_session_lock
    )

    with session_lock:
        if not use_spark_connect:
            logger.info("Acquired Standalone session creation lock")
        else:
            logger.debug("Spark Connect mode - no lock required")

        # Clean environment before creating session
        # PySpark 3.4+ uses environment variables to determine mode
        _clear_spark_env_for_mode_switch(use_spark_connect)

        # For Spark Connect: clear any stale cached sessions to prevent
        # INVALID_HANDLE.SESSION_CLOSED errors when server has restarted
        if use_spark_connect:
            _clear_stale_spark_connect_sessions()

        # Clear builder's cached options to prevent conflicts
        builder = SparkSession.builder
        if hasattr(builder, "_options"):
            builder._options.clear()

        # Use loadDefaults=False to prevent SparkConf from inheriting configuration
        # from any existing JVM (e.g., spark.master from a previous session).
        spark_conf = SparkConf(loadDefaults=False).setAll(list(config.items()))

        # Use the same builder instance that we cleared
        if use_spark_connect and not local:
            # CRITICAL: Use create() instead of getOrCreate() for remote Spark Connect mode.
            # getOrCreate() checks cached sessions first and may reuse stale session_ids
            # from sessions that the server has already closed (e.g., after server restart).
            # create() always generates a fresh session_id via uuid.uuid4(), ensuring
            # we never send a stale session_id to the server.
            # Note: This only applies to remote Spark Connect, not local sessions.
            logger.info(
                f"Creating Spark Connect session using create() "
                f"(app_name={app_name}, remote={config.get('spark.remote', 'N/A')})"
            )
            spark = builder.config(conf=spark_conf).create()

            # Log the session_id to verify it's a fresh UUID
            session_id = getattr(spark, "session_id", "unknown")
            logger.info(
                f"Spark Connect session created successfully: session_id={session_id}"
            )
        else:
            # Standalone mode or local mode: getOrCreate() is safe because:
            # - Standalone: sessions share the same JVM, no server-side session registry
            # - Local: no remote server, session state is all in-process
            mode = "local" if local else "standalone"
            logger.info(f"Creating {mode} session using getOrCreate()")
            spark = builder.config(conf=spark_conf).getOrCreate()

        if not use_spark_connect:
            logger.info("Standalone session creation complete, releasing lock")

    # Post-creation configuration (only for legacy mode with SparkContext)
    # This can be done outside the lock as it operates on the session instance
    if not local and not use_spark_connect:
        _set_scheduler_pool(spark, scheduler_pool)

    return spark


def get_spark_session_with_retry(
    app_name: str | None = None,
    local: bool = False,
    delta_lake: bool = True,
    scheduler_pool: str = SPARK_DEFAULT_POOL,
    use_s3: bool = True,
    use_hive: bool = True,
    settings: BERDLSettings | None = None,
    tenant_name: str | None = None,
    use_spark_connect: bool = True,
    override: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> SparkSession:
    """
    Get a Spark session with automatic retry on stale session errors.

    This wrapper around get_spark_session() handles INVALID_HANDLE.SESSION_CLOSED
    errors by automatically retrying with a fresh session. This is particularly
    useful for Spark Connect mode where the server may have restarted since
    the last request.

    Args:
        app_name: Application name. If None, generates a timestamp-based name
        local: If True, creates a local Spark session
        delta_lake: If True, enables Delta Lake support
        scheduler_pool: Fair scheduler pool name (default: "default")
        use_s3: If True, enables S3/MinIO integration
        use_hive: If True, enables Hive metastore integration
        settings: BERDLSettings instance. If None, creates from env vars
        tenant_name: Tenant/group name for SQL warehouse location
        use_spark_connect: If True, uses Spark Connect instead of legacy mode
        override: Dictionary of config overrides
        max_retries: Maximum number of retry attempts (default: 2)

    Returns:
        Configured SparkSession instance

    Raises:
        Exception: If session creation fails after all retries

    Example:
        >>> spark = get_spark_session_with_retry(
        ...     app_name="MyApp",
        ...     use_spark_connect=True,
        ...     max_retries=2,
        ... )
    """
    last_error: Exception | None = None
    username = settings.USER if settings else "unknown"

    logger.info(
        f"get_spark_session_with_retry called for user={username}, "
        f"use_spark_connect={use_spark_connect}, max_retries={max_retries}"
    )

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                backoff_ms = int(0.1 * (2**attempt) * 1000)
                logger.info(
                    f"[{username}] Session retry attempt {attempt}/{max_retries} "
                    f"after stale session error (backoff={backoff_ms}ms)"
                )
                # Clear stale sessions before retry
                _clear_stale_spark_connect_sessions()
                # Exponential backoff: 100ms, 200ms, 400ms...
                time.sleep(0.1 * (2**attempt))
            else:
                logger.info(f"[{username}] First session creation attempt")

            session = get_spark_session(
                app_name=app_name,
                local=local,
                delta_lake=delta_lake,
                scheduler_pool=scheduler_pool,
                use_s3=use_s3,
                use_hive=use_hive,
                settings=settings,
                tenant_name=tenant_name,
                use_spark_connect=use_spark_connect,
                override=override,
            )

            logger.info(
                f"[{username}] Session created successfully on attempt {attempt + 1}"
            )
            return session

        except Exception as e:
            is_session_err = _is_session_error(e)
            logger.warning(
                f"[{username}] Session creation failed on attempt {attempt + 1}: "
                f"is_session_error={is_session_err}, "
                f"error={type(e).__name__}: {e}"
            )

            if is_session_err and attempt < max_retries:
                logger.info(
                    f"[{username}] Will retry (attempt {attempt + 1} of {max_retries + 1})"
                )
                last_error = e
                continue

            # Either not a session error, or we've exhausted retries
            if is_session_err:
                logger.error(
                    f"[{username}] Exhausted all {max_retries + 1} retry attempts "
                    f"for session error: {e}"
                )
            raise

    # This should not be reached, but just in case
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Session creation failed after {max_retries + 1} attempts")
