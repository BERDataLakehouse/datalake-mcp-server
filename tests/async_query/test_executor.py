"""
Tests for the async query executor module.

Tests cover:
- Query submission and task tracking
- Successful execution flow (PENDING -> RUNNING -> SUCCEEDED)
- Failed execution flow (PENDING -> RUNNING -> FAILED)
- S3 result upload
- Graceful shutdown
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.async_query.executor import AsyncQueryExecutor
from src.service.dependencies import SparkContext
from src.service.models import (
    JobStatus,
    PaginationInfo,
    TableQueryResponse,
)
from src.service.query_executor import execute_query


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def executor():
    """Create an AsyncQueryExecutor."""
    return AsyncQueryExecutor()


@pytest.fixture
def ctx():
    """SparkContext for tests."""
    return SparkContext(
        spark=None,
        is_standalone_subprocess=True,
        settings_dict={"MINIO_ACCESS_KEY": "key"},
        app_name="mcp_test",
        username="testuser",
    )


@pytest.fixture
def submit_kwargs(ctx):
    """Default kwargs for submit_query / _execute_query calls."""
    return dict(
        job_id="job-1",
        user="testuser",
        query="SELECT 1",
        limit=1000,
        offset=0,
        ctx=ctx,
        minio_endpoint="localhost:9002",
        minio_access_key="key",
        minio_secret_key="secret",
        minio_secure=False,
    )


@pytest.fixture
def mock_query_response():
    """A TableQueryResponse for mocking execute_query."""
    return TableQueryResponse(
        result=[{"id": 1}, {"id": 2}],
        pagination=PaginationInfo(limit=1000, offset=0, total_count=50, has_more=True),
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for executor initialization."""

    def test_no_process_pool(self):
        """Executor does not create a ProcessPoolExecutor."""
        exe = AsyncQueryExecutor()
        assert not hasattr(exe, "_pool")

    def test_empty_active_tasks(self, executor):
        """Executor starts with no active tasks."""
        assert len(executor._active_tasks) == 0


# =============================================================================
# submit_query Tests
# =============================================================================


class TestSubmitQuery:
    """Tests for query submission."""

    @pytest.mark.asyncio
    async def test_submit_creates_task(self, executor, submit_kwargs):
        """submit_query creates an asyncio task and tracks it."""

        async def mock_execute(*args, **kwargs):
            pass

        with patch.object(executor, "_execute_query", side_effect=mock_execute):
            await executor.submit_query(**submit_kwargs)

            assert "job-1" in executor._active_tasks
            # Clean up
            executor._active_tasks["job-1"].cancel()
            try:
                await executor._active_tasks["job-1"]
            except (asyncio.CancelledError, Exception):
                pass


# =============================================================================
# _execute_query Tests
# =============================================================================


class TestExecuteQuery:
    """Tests for the background execution orchestration."""

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, executor, submit_kwargs, mock_query_response
    ):
        """Successful query updates status to RUNNING then SUCCEEDED."""
        with (
            patch(
                "src.async_query.executor.execute_query",
                return_value=mock_query_response,
            ),
            patch("src.async_query.executor.job_store") as mock_job_store,
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_s3.build_result_path.return_value = "prefix/"
            mock_s3.build_query_result_root_path.return_value = "root/"
            mock_client = MagicMock()
            mock_s3.create_s3_client.return_value = mock_client
            await executor._execute_query(**submit_kwargs)

            # Verify status transitions
            calls = mock_job_store.update_job_status.call_args_list
            assert len(calls) >= 2

            # First call: update_job_status(client, job_id, RUNNING, user=...)
            assert calls[0].args[0] is mock_client  # S3 client
            assert calls[0].args[2] == JobStatus.RUNNING
            assert calls[0].kwargs.get("user") == "testuser"

            # Second call: SUCCEEDED with result metadata and user
            assert calls[1].args[0] is mock_client
            assert calls[1].args[2] == JobStatus.SUCCEEDED
            assert calls[1].kwargs.get("user") == "testuser"
            assert calls[1].kwargs.get("row_count") == 2
            assert calls[1].kwargs.get("total_count") == 50
            assert calls[1].kwargs.get("has_more") is True

            # Verify .s3keep creation
            assert mock_s3.create_s3keep.call_count == 2
            mock_s3.create_s3keep.assert_any_call(mock_client, "cdm-lake", "prefix/")
            mock_s3.create_s3keep.assert_any_call(mock_client, "cdm-lake", "root/")

    @pytest.mark.asyncio
    async def test_execute_query_called_via_to_thread(
        self, executor, submit_kwargs, mock_query_response
    ):
        """execute_query is called via asyncio.to_thread."""
        with (
            patch(
                "src.async_query.executor.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread,
            patch("src.async_query.executor.job_store"),
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_to_thread.return_value = mock_query_response
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_s3.build_result_path.return_value = "prefix/"
            mock_s3.build_query_result_root_path.return_value = "root/"
            mock_s3.create_s3_client.return_value = MagicMock()

            await executor._execute_query(**submit_kwargs)

            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args.args[0] is execute_query

    @pytest.mark.asyncio
    async def test_result_uploaded_to_s3(
        self, executor, submit_kwargs, mock_query_response
    ):
        """Query result is uploaded to S3 via s3_client.upload_result."""
        with (
            patch(
                "src.async_query.executor.execute_query",
                return_value=mock_query_response,
            ),
            patch("src.async_query.executor.job_store"),
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_s3.build_result_path.return_value = "prefix/"
            mock_s3.build_query_result_root_path.return_value = "root/"
            mock_client = MagicMock()
            mock_s3.create_s3_client.return_value = mock_client

            await executor._execute_query(**submit_kwargs)

            mock_s3.upload_result.assert_called_once()
            call_args = mock_s3.upload_result.call_args
            assert call_args.args[0] is mock_client
            assert call_args.args[1] == "cdm-lake"
            assert call_args.args[2] == "prefix/"

    @pytest.mark.asyncio
    async def test_failed_execution(self, executor, submit_kwargs):
        """Failed query updates status to FAILED with error message."""
        with (
            patch(
                "src.async_query.executor.execute_query",
                side_effect=RuntimeError("Spark error"),
            ),
            patch("src.async_query.executor.job_store") as mock_job_store,
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_client = MagicMock()
            mock_s3.create_s3_client.return_value = mock_client

            await executor._execute_query(**submit_kwargs)

            # Verify FAILED status with user param
            calls = mock_job_store.update_job_status.call_args_list
            last_call = calls[-1]
            assert last_call.args[0] is mock_client  # S3 client
            assert last_call.args[2] == JobStatus.FAILED
            assert last_call.kwargs.get("user") == "testuser"
            assert "Spark error" in last_call.kwargs.get("error_message", "")

    @pytest.mark.asyncio
    async def test_task_removed_from_active(
        self, executor, submit_kwargs, mock_query_response
    ):
        """Task is removed from _active_tasks after completion."""
        with (
            patch(
                "src.async_query.executor.execute_query",
                return_value=mock_query_response,
            ),
            patch("src.async_query.executor.job_store"),
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_s3.build_result_path.return_value = "prefix/"
            mock_s3.build_query_result_root_path.return_value = "root/"
            mock_s3.create_s3_client.return_value = MagicMock()

            # Pre-populate active_tasks
            executor._active_tasks["job-1"] = MagicMock()

            await executor._execute_query(**submit_kwargs)

            assert "job-1" not in executor._active_tasks

    @pytest.mark.asyncio
    async def test_task_removed_on_failure(self, executor, submit_kwargs):
        """Task is removed from _active_tasks even after failure."""
        with (
            patch(
                "src.async_query.executor.execute_query",
                side_effect=RuntimeError("boom"),
            ),
            patch("src.async_query.executor.job_store"),
            patch("src.async_query.executor.s3_client") as mock_s3,
        ):
            mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
            mock_s3.create_s3_client.return_value = MagicMock()

            executor._active_tasks["job-1"] = MagicMock()

            await executor._execute_query(**submit_kwargs)

            assert "job-1" not in executor._active_tasks


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self, executor):
        """Shutdown cancels all active tasks."""
        task1 = MagicMock()
        task2 = MagicMock()
        executor._active_tasks = {"job-1": task1, "job-2": task2}

        await executor.shutdown()

        task1.cancel.assert_called_once()
        task2.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_pool(self, executor):
        """Shutdown does not attempt to close a process pool (none exists)."""
        assert not hasattr(executor, "_pool")
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_empty_tasks(self, executor):
        """Shutdown works with no active tasks."""
        await executor.shutdown()
        assert len(executor._active_tasks) == 0
