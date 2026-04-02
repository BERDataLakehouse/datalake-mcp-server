"""
Tests for the shared query executor module.

Tests cover:
- Standalone mode dispatch to run_in_spark_process
- Connect mode dispatch to delta_service.query_delta_table
- max_rows and operation_name parameters passed through correctly
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.dependencies import SparkContext
from src.service.models import PaginationInfo, TableQueryResponse
from src.service.query_executor import execute_query, execute_query_trino


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def standalone_ctx():
    """SparkContext for standalone mode."""
    return SparkContext(
        spark=None,
        is_standalone_subprocess=True,
        settings_dict={"MINIO_ACCESS_KEY": "key"},
        app_name="mcp_test",
        username="testuser",
    )


@pytest.fixture
def connect_ctx():
    """SparkContext for Spark Connect mode."""
    return SparkContext(
        spark=MagicMock(),
        is_standalone_subprocess=False,
        settings_dict={"MINIO_ACCESS_KEY": "key"},
        app_name="mcp_test",
        username="testuser",
    )


# =============================================================================
# Standalone Mode Tests
# =============================================================================


class TestStandaloneMode:
    """Tests for standalone mode dispatch."""

    def test_calls_run_in_spark_process(self, standalone_ctx):
        """Standalone mode dispatches via run_in_spark_process."""
        mock_result = {
            "result": [{"id": 1}],
            "pagination": {
                "limit": 100,
                "offset": 0,
                "total_count": 1,
                "has_more": False,
            },
        }

        with patch(
            "src.service.query_executor.run_in_spark_process",
            return_value=mock_result,
        ) as mock_run:
            response = execute_query(standalone_ctx, "SELECT 1", 100, 0, "testuser")

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["query"] == "SELECT 1"
            assert call_kwargs["limit"] == 100
            assert call_kwargs["offset"] == 0
            assert call_kwargs["username"] == "testuser"
            assert call_kwargs["app_name"] == "mcp_test"

            assert response.result == [{"id": 1}]
            assert response.pagination.total_count == 1
            assert response.pagination.has_more is False

    def test_returns_table_query_response(self, standalone_ctx):
        """Result is converted to TableQueryResponse."""
        mock_result = {
            "result": [{"a": 1}, {"a": 2}],
            "pagination": {
                "limit": 50,
                "offset": 10,
                "total_count": 100,
                "has_more": True,
            },
        }

        with patch(
            "src.service.query_executor.run_in_spark_process",
            return_value=mock_result,
        ):
            response = execute_query(
                standalone_ctx, "SELECT a FROM t", 50, 10, "testuser"
            )

            assert isinstance(response, TableQueryResponse)
            assert len(response.result) == 2
            assert response.pagination.limit == 50
            assert response.pagination.offset == 10
            assert response.pagination.total_count == 100
            assert response.pagination.has_more is True

    def test_max_rows_passed_through(self, standalone_ctx):
        """max_rows parameter is forwarded to subprocess."""
        mock_result = {
            "result": [],
            "pagination": {
                "limit": 100,
                "offset": 0,
                "total_count": 0,
                "has_more": False,
            },
        }

        with patch(
            "src.service.query_executor.run_in_spark_process",
            return_value=mock_result,
        ) as mock_run:
            execute_query(standalone_ctx, "SELECT 1", 100, 0, "testuser", max_rows=5000)

            assert mock_run.call_args.kwargs["max_rows"] == 5000

    def test_operation_name_passed_through(self, standalone_ctx):
        """operation_name parameter is forwarded."""
        mock_result = {
            "result": [],
            "pagination": {
                "limit": 100,
                "offset": 0,
                "total_count": 0,
                "has_more": False,
            },
        }

        with patch(
            "src.service.query_executor.run_in_spark_process",
            return_value=mock_result,
        ) as mock_run:
            execute_query(
                standalone_ctx,
                "SELECT 1",
                100,
                0,
                "testuser",
                operation_name="async_query",
            )

            assert mock_run.call_args.kwargs["operation_name"] == "async_query"


# =============================================================================
# Connect Mode Tests
# =============================================================================


class TestConnectMode:
    """Tests for Spark Connect mode dispatch."""

    def test_calls_query_delta_table(self, connect_ctx):
        """Connect mode calls delta_service.query_delta_table."""
        mock_response = TableQueryResponse(
            result=[{"id": 1}],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=1, has_more=False
            ),
        )

        with patch(
            "src.service.query_executor.delta_service.query_delta_table",
            return_value=mock_response,
        ) as mock_query:
            response = execute_query(connect_ctx, "SELECT 1", 100, 0, "testuser")

            mock_query.assert_called_once_with(
                spark=connect_ctx.spark,
                query="SELECT 1",
                limit=100,
                offset=0,
                username="testuser",
                max_rows=1000,
            )

            assert response.result == [{"id": 1}]
            assert response.pagination.total_count == 1

    def test_max_rows_passed_through(self, connect_ctx):
        """max_rows parameter is forwarded to query_delta_table."""
        mock_response = TableQueryResponse(
            result=[],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=0, has_more=False
            ),
        )

        with patch(
            "src.service.query_executor.delta_service.query_delta_table",
            return_value=mock_response,
        ) as mock_query:
            execute_query(connect_ctx, "SELECT 1", 100, 0, "testuser", max_rows=5000)

            assert mock_query.call_args.kwargs["max_rows"] == 5000

    def test_does_not_call_run_in_spark_process(self, connect_ctx):
        """Connect mode does not use run_in_spark_process."""
        mock_response = TableQueryResponse(
            result=[],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=0, has_more=False
            ),
        )

        with (
            patch(
                "src.service.query_executor.delta_service.query_delta_table",
                return_value=mock_response,
            ),
            patch(
                "src.service.query_executor.run_in_spark_process",
            ) as mock_run,
        ):
            execute_query(connect_ctx, "SELECT 1", 100, 0, "testuser")

            mock_run.assert_not_called()


# =============================================================================
# Trino Mode Tests
# =============================================================================


class TestTrinoMode:
    """Tests for Trino execution via execute_query_trino."""

    def test_delegates_to_trino_service(self):
        """execute_query_trino delegates to trino_service.query_via_trino."""
        mock_conn = MagicMock()
        mock_response = TableQueryResponse(
            result=[{"id": 1}],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=1, has_more=False
            ),
        )

        with patch(
            "src.service.query_executor.trino_service.query_via_trino",
            return_value=mock_response,
        ) as mock_query:
            response = execute_query_trino(mock_conn, "SELECT 1", 100, 0, "testuser")

            mock_query.assert_called_once_with(
                conn=mock_conn,
                query="SELECT 1",
                limit=100,
                offset=0,
                username="testuser",
                max_rows=1000,
            )
            assert response.result == [{"id": 1}]

    def test_max_rows_passed_through(self):
        """max_rows parameter is forwarded."""
        mock_conn = MagicMock()
        mock_response = TableQueryResponse(
            result=[],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=0, has_more=False
            ),
        )

        with patch(
            "src.service.query_executor.trino_service.query_via_trino",
            return_value=mock_response,
        ) as mock_query:
            execute_query_trino(
                mock_conn, "SELECT 1", 100, 0, "testuser", max_rows=5000
            )

            assert mock_query.call_args.kwargs["max_rows"] == 5000
