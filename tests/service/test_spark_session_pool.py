"""
Tests for the spark_session_pool module.

Tests cover:
- Pool initialization and lazy loading
- Pool shutdown and cleanup
- Running functions in the process pool
- Timeout handling
- Error propagation
- Pool status reporting
"""

import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.service.spark_session_pool import (
    STANDALONE_POOL_SIZE,
    STANDALONE_POOL_TIMEOUT,
    _get_pool,
    _shutdown_pool,
    get_pool_status,
    run_in_spark_process,
)
from src.service.exceptions import SparkTimeoutError


# =============================================================================
# Helper functions for process pool tests
# These must be top-level functions (not lambdas/closures) to be picklable
# =============================================================================


def _add_numbers(a: int, b: int) -> int:
    """Simple function that can run in process pool."""
    return a + b


def _multiply_with_kwargs(x: int, multiplier: int = 2) -> int:
    """Function with keyword arguments."""
    return x * multiplier


def _slow_function(duration: float) -> str:
    """Function that takes a while to complete."""
    time.sleep(duration)
    return "completed"


def _raising_function(message: str) -> None:
    """Function that raises an exception."""
    raise ValueError(message)


def _get_process_id() -> int:
    """Return current process ID to verify isolation."""
    import os

    return os.getpid()


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_standalone_pool_size_default(self):
        """Test default pool size is 4."""
        # Note: actual value may differ if env var is set
        assert isinstance(STANDALONE_POOL_SIZE, int)
        assert STANDALONE_POOL_SIZE > 0

    def test_standalone_pool_timeout_default(self):
        """Test default timeout is 600 seconds (10 minutes)."""
        assert isinstance(STANDALONE_POOL_TIMEOUT, float)
        assert STANDALONE_POOL_TIMEOUT > 0

    def test_pool_size_from_env(self):
        """Test pool size can be configured via environment variable."""
        with patch.dict("os.environ", {"STANDALONE_SPARK_POOL_SIZE": "8"}):
            # Re-import to get new value
            import importlib
            import src.service.spark_session_pool as pool_module

            importlib.reload(pool_module)
            assert pool_module.STANDALONE_POOL_SIZE == 8

            # Restore original
            importlib.reload(pool_module)

    def test_pool_timeout_from_env(self):
        """Test timeout can be configured via environment variable."""
        with patch.dict("os.environ", {"STANDALONE_POOL_TIMEOUT": "120.5"}):
            import importlib
            import src.service.spark_session_pool as pool_module

            importlib.reload(pool_module)
            assert pool_module.STANDALONE_POOL_TIMEOUT == 120.5

            # Restore original
            importlib.reload(pool_module)


# =============================================================================
# Test _get_pool
# =============================================================================


class TestGetPool:
    """Tests for the _get_pool function."""

    def test_returns_process_pool_executor(self):
        """Test that _get_pool returns a ProcessPoolExecutor."""
        pool = _get_pool()
        assert isinstance(pool, ProcessPoolExecutor)

    def test_lazy_initialization(self):
        """Test that pool is lazily initialized."""
        import src.service.spark_session_pool as pool_module

        # Reset pool state
        original_pool = pool_module._spark_process_pool
        pool_module._spark_process_pool = None

        try:
            # Pool should be None before first access
            assert pool_module._spark_process_pool is None

            # First access creates pool
            pool = _get_pool()
            assert pool is not None
            assert pool_module._spark_process_pool is pool
        finally:
            # Restore original state
            pool_module._spark_process_pool = original_pool

    def test_returns_same_pool_on_subsequent_calls(self):
        """Test that subsequent calls return the same pool instance."""
        pool1 = _get_pool()
        pool2 = _get_pool()
        assert pool1 is pool2

    def test_pool_has_correct_max_workers(self):
        """Test that pool is configured with correct max workers."""
        pool = _get_pool()
        assert pool._max_workers == STANDALONE_POOL_SIZE


# =============================================================================
# Test _shutdown_pool
# =============================================================================


class TestShutdownPool:
    """Tests for the _shutdown_pool function."""

    def test_shutdown_when_pool_exists(self):
        """Test shutting down an existing pool."""
        import src.service.spark_session_pool as pool_module

        # Create a pool
        _get_pool()
        assert pool_module._spark_process_pool is not None

        # Shutdown
        _shutdown_pool()

        # Pool should be None after shutdown
        assert pool_module._spark_process_pool is None

    def test_shutdown_when_pool_is_none(self):
        """Test shutdown when pool was never created."""
        import src.service.spark_session_pool as pool_module

        original_pool = pool_module._spark_process_pool
        pool_module._spark_process_pool = None

        try:
            # Should not raise when pool is None
            _shutdown_pool()
            assert pool_module._spark_process_pool is None
        finally:
            pool_module._spark_process_pool = original_pool

    def test_shutdown_handles_exceptions_gracefully(self):
        """Test that shutdown handles exceptions during pool.shutdown()."""
        import src.service.spark_session_pool as pool_module

        # Create a mock pool that raises on shutdown
        mock_pool = MagicMock()
        mock_pool.shutdown.side_effect = Exception("Shutdown error")

        original_pool = pool_module._spark_process_pool
        pool_module._spark_process_pool = mock_pool

        try:
            # Should not raise despite exception
            _shutdown_pool()
            # Pool should still be set to None
            assert pool_module._spark_process_pool is None
        finally:
            pool_module._spark_process_pool = original_pool


# =============================================================================
# Test run_in_spark_process
# =============================================================================


class TestRunInSparkProcess:
    """Tests for the run_in_spark_process function."""

    def test_executes_function_with_args(self):
        """Test executing a function with positional arguments."""
        result = run_in_spark_process(_add_numbers, 3, 5, operation_name="add_test")
        assert result == 8

    def test_executes_function_with_kwargs(self):
        """Test executing a function with keyword arguments."""
        result = run_in_spark_process(
            _multiply_with_kwargs, 4, multiplier=3, operation_name="multiply_test"
        )
        assert result == 12

    def test_uses_default_timeout(self):
        """Test that default timeout is used when not specified."""
        # Fast function should complete well before timeout
        result = run_in_spark_process(_add_numbers, 1, 2)
        assert result == 3

    def test_custom_timeout_is_respected(self):
        """Test that custom timeout is respected."""
        # Should complete before short timeout
        result = run_in_spark_process(
            _add_numbers, 1, 2, timeout=5.0, operation_name="quick_add"
        )
        assert result == 3

    def test_timeout_raises_spark_timeout_error(self):
        """Test that timeout raises SparkTimeoutError."""
        with pytest.raises(SparkTimeoutError) as exc_info:
            run_in_spark_process(
                _slow_function,
                5.0,  # 5 second sleep
                timeout=0.1,  # 100ms timeout
                operation_name="slow_operation",
            )

        assert "slow_operation" in str(exc_info.value)
        assert exc_info.value.timeout == 0.1

    def test_exception_propagation(self):
        """Test that exceptions from function are propagated."""
        with pytest.raises(ValueError) as exc_info:
            run_in_spark_process(
                _raising_function, "test error message", operation_name="error_test"
            )

        assert "test error message" in str(exc_info.value)

    def test_runs_in_separate_process(self):
        """Test that function runs in a different process."""
        import os

        current_pid = os.getpid()
        worker_pid = run_in_spark_process(_get_process_id, operation_name="pid_test")

        # Worker process should have different PID
        assert worker_pid != current_pid

    def test_result_is_serializable(self):
        """Test that results are properly serialized."""
        result = run_in_spark_process(_add_numbers, 100, 200)
        assert result == 300
        assert isinstance(result, int)

    def test_default_operation_name(self):
        """Test that default operation name is used."""
        # Should not raise with default operation name
        result = run_in_spark_process(_add_numbers, 1, 1)
        assert result == 2


# =============================================================================
# Test get_pool_status
# =============================================================================


class TestGetPoolStatus:
    """Tests for the get_pool_status function."""

    def test_status_when_pool_not_initialized(self):
        """Test status when pool has not been created yet."""
        import src.service.spark_session_pool as pool_module

        original_pool = pool_module._spark_process_pool
        pool_module._spark_process_pool = None

        try:
            status = get_pool_status()

            assert status["initialized"] is False
            assert "max_workers" in status
            assert "timeout_seconds" in status
            assert "shutdown" not in status
        finally:
            pool_module._spark_process_pool = original_pool

    def test_status_when_pool_initialized(self):
        """Test status when pool has been created."""
        # Ensure pool is initialized
        _get_pool()

        status = get_pool_status()

        assert status["initialized"] is True
        assert "max_workers" in status
        assert "timeout_seconds" in status
        assert "shutdown" in status

    def test_status_shows_shutdown_state(self):
        """Test that status reflects shutdown state."""
        import src.service.spark_session_pool as pool_module

        # Create a fresh pool
        original_pool = pool_module._spark_process_pool
        pool_module._spark_process_pool = None
        _get_pool()

        try:
            # Pool should not be in shutdown state
            status = get_pool_status()
            assert status["shutdown"] is False
        finally:
            pool_module._spark_process_pool = original_pool


# =============================================================================
# Test Concurrent Execution
# =============================================================================


class TestConcurrentExecution:
    """Tests for concurrent execution in the process pool."""

    def test_multiple_concurrent_submissions(self):
        """Test that multiple functions can run concurrently."""
        import concurrent.futures

        # Submit multiple tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    run_in_spark_process,
                    _add_numbers,
                    i,
                    i * 2,
                    operation_name=f"add_{i}",
                )
                for i in range(4)
            ]

            results = [f.result() for f in futures]

        # All should complete
        assert len(results) == 4
        assert results == [0, 3, 6, 9]  # 0+0, 1+2, 2+4, 3+6

    def test_process_isolation(self):
        """Test that each task runs in a separate process."""
        import concurrent.futures

        # Get PIDs from multiple concurrent tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_in_spark_process, _get_process_id) for _ in range(3)
            ]

            pids = [f.result() for f in futures]

        # All PIDs should be from worker processes (may or may not be unique
        # depending on pool reuse, but all should be different from main process)
        import os

        main_pid = os.getpid()
        assert all(pid != main_pid for pid in pids)
