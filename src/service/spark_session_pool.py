"""
Process pool for running Spark sessions in isolated processes.

PySpark's JVM driver is single-threaded for session management and cannot run
multiple concurrent sessions in the same process. This pool runs each Spark
operation in a separate process with its own JVM, enabling true concurrency
for Standalone mode.

For Spark Connect mode, no pool is needed - each session is a client-only
gRPC connection to a remote Spark cluster.

Configuration:
    STANDALONE_SPARK_POOL_SIZE: Number of worker processes (default: 4)
    STANDALONE_POOL_TIMEOUT: Timeout for pool operations in seconds (default: 300)
"""

import atexit
import logging
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

from src.service.exceptions import SparkTimeoutError

logger = logging.getLogger(__name__)

# Configurable pool size (default: 4 workers)
STANDALONE_POOL_SIZE = int(os.getenv("STANDALONE_SPARK_POOL_SIZE", "4"))

# Timeout for pool operations (default: 10 minutes)
STANDALONE_POOL_TIMEOUT = float(os.getenv("STANDALONE_POOL_TIMEOUT", "600"))

# Global process pool (lazy initialized)
_spark_process_pool: ProcessPoolExecutor | None = None

T = TypeVar("T")


def _get_pool() -> ProcessPoolExecutor:
    """
    Get or create the Spark process pool.

    The pool is lazily initialized on first use and shared across all requests.
    Each worker process will load its own JVM when it executes its first Spark operation.

    Returns:
        ProcessPoolExecutor configured for Spark operations
    """
    global _spark_process_pool
    if _spark_process_pool is None:
        logger.info(f"Creating Spark process pool with {STANDALONE_POOL_SIZE} workers")
        _spark_process_pool = ProcessPoolExecutor(
            max_workers=STANDALONE_POOL_SIZE,
            # Use default multiprocessing context (fork on Unix, spawn on Windows)
            # Fork is faster but can cause issues with some JVM configurations
            mp_context=None,
        )
    return _spark_process_pool


def _shutdown_pool() -> None:
    """
    Shutdown the process pool on exit.

    Called automatically via atexit to clean up worker processes.
    """
    global _spark_process_pool
    if _spark_process_pool is not None:
        logger.info("Shutting down Spark process pool")
        try:
            _spark_process_pool.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            logger.warning(f"Error shutting down Spark process pool: {e}")
        _spark_process_pool = None


# Register cleanup on process exit
atexit.register(_shutdown_pool)


def run_in_spark_process(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = None,
    operation_name: str = "spark_operation",
    **kwargs: Any,
) -> T:
    """
    Run a function in a separate process with its own SparkSession.

    The function should create its own SparkSession, execute the operation,
    and return serializable results. This enables concurrent Standalone Spark
    operations without JVM conflicts.

    Args:
        func: Function to execute (must be picklable - no lambdas or closures)
        *args: Positional arguments for the function
        timeout: Timeout in seconds (default: STANDALONE_POOL_TIMEOUT)
        operation_name: Name for error messages
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function (must be serializable via pickle)

    Raises:
        SparkTimeoutError: If the operation times out
        Exception: Any exception raised by the function (re-raised)

    Example:
        def count_table_in_process(db: str, table: str, settings_dict: dict) -> int:
            from src.delta_lake.setup_spark_session import get_spark_session
            from src.settings import BERDLSettings

            settings = BERDLSettings(**settings_dict)
            spark = get_spark_session(settings=settings, use_spark_connect=False)
            try:
                return spark.table(f"{db}.{table}").count()
            finally:
                spark.stop()

        count = run_in_spark_process(count_table_in_process, "mydb", "users", settings_dict)
    """
    if timeout is None:
        timeout = STANDALONE_POOL_TIMEOUT

    pool = _get_pool()
    logger.debug(f"Submitting '{operation_name}' to Spark process pool")

    future = pool.submit(func, *args, **kwargs)

    try:
        result = future.result(timeout=timeout)
        logger.debug(f"'{operation_name}' completed successfully in process pool")
        return result
    except FuturesTimeoutError:
        # Cancel the future (won't stop already running work, but prevents pickup)
        future.cancel()
        logger.error(
            f"Operation '{operation_name}' timed out after {timeout}s in process pool"
        )
        raise SparkTimeoutError(
            operation=operation_name,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Operation '{operation_name}' failed in process pool: {e}")
        raise


def get_pool_status() -> dict[str, Any]:
    """
    Get status information about the Spark process pool.

    Useful for health checks and debugging.

    Returns:
        Dictionary with pool status information
    """
    global _spark_process_pool

    if _spark_process_pool is None:
        return {
            "initialized": False,
            "max_workers": STANDALONE_POOL_SIZE,
            "timeout_seconds": STANDALONE_POOL_TIMEOUT,
        }

    return {
        "initialized": True,
        "max_workers": STANDALONE_POOL_SIZE,
        "timeout_seconds": STANDALONE_POOL_TIMEOUT,
        # ProcessPoolExecutor doesn't expose active worker count directly
        # but we can check if it's still running
        "shutdown": getattr(_spark_process_pool, "_shutdown", False),
    }
