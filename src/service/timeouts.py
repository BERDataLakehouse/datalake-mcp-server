"""
Timeout utilities for long-running operations.

Provides timeout wrappers for Spark operations to prevent the service
from becoming unresponsive due to slow or stuck queries.
"""

import atexit
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

from src.service.exceptions import SparkTimeoutError

logger = logging.getLogger(__name__)

# Default timeout values (in seconds)
DEFAULT_SPARK_QUERY_TIMEOUT = int(os.getenv("SPARK_QUERY_TIMEOUT", "300"))  # 5 minutes
DEFAULT_SPARK_COLLECT_TIMEOUT = int(
    os.getenv("SPARK_COLLECT_TIMEOUT", "120")
)  # 2 minutes

# Thread pool for timeout execution
# Using a modest pool size since Spark operations are already parallelized
_timeout_executor = ThreadPoolExecutor(
    max_workers=10, thread_name_prefix="spark_timeout"
)


def _shutdown_timeout_executor() -> None:
    """Shutdown the timeout executor thread pool on process exit."""
    logger.info("Shutting down timeout executor thread pool")
    _timeout_executor.shutdown(wait=False)


# Register cleanup on process exit
atexit.register(_shutdown_timeout_executor)


T = TypeVar("T")


def with_timeout(
    timeout_seconds: float | None = None,
    operation_name: str = "spark_operation",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add timeout to a function.

    Args:
        timeout_seconds: Maximum execution time in seconds. If None, uses DEFAULT_SPARK_QUERY_TIMEOUT
        operation_name: Name of the operation for error messages

    Returns:
        Decorated function that will raise SparkTimeoutError if timeout exceeded

    Example:
        @with_timeout(timeout_seconds=60, operation_name="query_delta_table")
        def my_query():
            return spark.sql("SELECT * FROM large_table").collect()
    """
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_SPARK_QUERY_TIMEOUT
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            future = _timeout_executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FuturesTimeoutError:
                # Note: cancel() will NOT stop the underlying Spark job.
                # Once the timeout occurs, the task is already running and cannot
                # be interrupted. The Spark job continues in the background.
                future.cancel()
                logger.error(
                    f"Operation '{operation_name}' timed out after {timeout_seconds}s"
                )
                raise SparkTimeoutError(
                    operation=operation_name,
                    timeout=timeout_seconds,
                )

        return wrapper

    return decorator


def run_with_timeout(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: float | None = None,
    operation_name: str = "spark_operation",
    spark_session: Any | None = None,
) -> T:
    """
    Run a function with a timeout.

    This is a functional alternative to the decorator for one-off usage.

    Args:
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout_seconds: Maximum execution time in seconds
        operation_name: Name of the operation for error messages
        spark_session: Optional SparkSession to stop on timeout. If provided,
                      spark.stop() will be called immediately on timeout to
                      interrupt any running Spark operations.

    Returns:
        Result of the function

    Raises:
        SparkTimeoutError: If the operation exceeds the timeout

    Example:
        results = run_with_timeout(
            lambda: spark.sql(query).collect(),
            timeout_seconds=60,
            operation_name="execute_query",
            spark_session=spark,  # Will be stopped on timeout
        )
    """
    if kwargs is None:
        kwargs = {}
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_SPARK_QUERY_TIMEOUT
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    future = _timeout_executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_seconds)
        return result
    except FuturesTimeoutError:
        # Note: cancel() will NOT stop the underlying Spark job.
        # Once the timeout occurs, the task is already running and cannot
        # be interrupted. The Spark job continues in the background.
        future.cancel()
        logger.error(f"Operation '{operation_name}' timed out after {timeout_seconds}s")

        # If a SparkSession was provided, stop it to interrupt the operation
        if spark_session is not None:
            try:
                logger.info(
                    f"Stopping Spark session due to timeout in '{operation_name}'"
                )
                spark_session.stop()
            except Exception as e:
                logger.warning(f"Error stopping Spark session on timeout: {e}")

        raise SparkTimeoutError(
            operation=operation_name,
            timeout=timeout_seconds,
        )


@contextmanager
def spark_operation_timeout(
    timeout_seconds: float | None = None,
    operation_name: str = "spark_operation",
):
    """
    Context manager for timing Spark operations with a warning on slow operations.

    Note: This does NOT actually enforce a timeout - it only logs warnings.
    For hard timeouts, use run_with_timeout() or @with_timeout decorator.

    Args:
        timeout_seconds: Threshold for warning (defaults to DEFAULT_SPARK_QUERY_TIMEOUT)
        operation_name: Name of the operation for logging

    Example:
        with spark_operation_timeout(60, "count_table"):
            count = spark.table("large_table").count()
    """
    import time

    if timeout_seconds is None:
        timeout_seconds = DEFAULT_SPARK_QUERY_TIMEOUT

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            logger.warning(
                f"Spark operation '{operation_name}' took {elapsed:.1f}s "
                f"(threshold: {timeout_seconds}s)"
            )
        elif elapsed > timeout_seconds * 0.8:
            # Warn if operation is approaching timeout
            logger.info(
                f"Spark operation '{operation_name}' completed in {elapsed:.1f}s "
                f"(approaching {timeout_seconds}s threshold)"
            )
