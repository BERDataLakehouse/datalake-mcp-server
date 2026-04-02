"""
Tests for the Trino query execution service module.

Tests cover:
- query_via_trino: paginated query execution, caching, metadata queries
- count_via_trino: row counting
- sample_via_trino: row sampling
- select_via_trino: structured SELECT execution
- _cursor_to_dicts: cursor result conversion
- _generate_cache_key: cache key includes engine marker
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.exceptions import TrinoOperationError, TrinoQueryError
from src.service.models import TableQueryResponse, TableSelectResponse
from src.trino_engine.trino_service import (
    _cursor_to_dicts,
    _generate_cache_key,
    _store_in_cache,
    _validate_trino_identifier,
    count_via_trino,
    query_via_trino,
    sample_via_trino,
    select_via_trino,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_conn():
    """Create a mock Trino connection with cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# =============================================================================
# _cursor_to_dicts Tests
# =============================================================================


class TestCursorToDicts:
    """Tests for cursor result conversion."""

    def test_empty_results(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        assert _cursor_to_dicts(cursor) == []

    def test_converts_tuples_to_dicts(self):
        cursor = MagicMock()
        cursor.description = [("id",), ("name",)]
        cursor.fetchall.return_value = [(1, "alice"), (2, "bob")]
        result = _cursor_to_dicts(cursor)
        assert result == [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]


# =============================================================================
# _generate_cache_key Tests
# =============================================================================


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_includes_trino_engine_marker(self):
        key = _generate_cache_key({"query": "SELECT 1"})
        key_with_user = _generate_cache_key({"query": "SELECT 1"}, username="alice")
        # Keys should be different (user scoped)
        assert key != key_with_user

    def test_deterministic(self):
        key1 = _generate_cache_key({"query": "SELECT 1"}, username="alice")
        key2 = _generate_cache_key({"query": "SELECT 1"}, username="alice")
        assert key1 == key2

    def test_different_params_different_keys(self):
        key1 = _generate_cache_key({"query": "SELECT 1"})
        key2 = _generate_cache_key({"query": "SELECT 2"})
        assert key1 != key2


# =============================================================================
# _store_in_cache Tests
# =============================================================================


class TestStoreInCache:
    """Tests for _store_in_cache function."""

    @patch("src.trino_engine.trino_service.set_cached_value")
    def test_delegates_to_set_cached_value(self, mock_set):
        """Line 76: _store_in_cache delegates to set_cached_value."""
        _store_in_cache("ns", "key123", [{"data": 1}], ttl=300)
        mock_set.assert_called_once_with(
            namespace="ns", cache_key="key123", data=[{"data": 1}], ttl=300
        )


# =============================================================================
# _validate_trino_identifier Tests
# =============================================================================


class TestValidateTrinoIdentifier:
    """Tests for _validate_trino_identifier wrapper."""

    def test_valid_identifiers_pass(self):
        _validate_trino_identifier("mydb", "database")
        _validate_trino_identifier("my_table", "table")
        _validate_trino_identifier("_private", "column")
        _validate_trino_identifier("Col123", "column")

    def test_invalid_identifier_raises_trino_query_error(self):
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            _validate_trino_identifier('"; DROP TABLE --', "database")

    def test_empty_identifier_raises(self):
        with pytest.raises(TrinoQueryError, match="Invalid table"):
            _validate_trino_identifier("", "table")

    def test_identifier_with_spaces_raises(self):
        with pytest.raises(TrinoQueryError, match="Invalid column"):
            _validate_trino_identifier("col name", "column")

    def test_identifier_with_special_chars_raises(self):
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            _validate_trino_identifier("my-db", "database")


# =============================================================================
# query_via_trino Tests
# =============================================================================


class TestQueryViaTrino:
    """Tests for query_via_trino function."""

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_paginated_query(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        conn, cursor = mock_conn
        cursor.description = [("id",), ("name",)]
        # First fetchall for paginated query, second for count query
        cursor.fetchall.return_value = [(1, "a")]
        cursor.fetchone.return_value = (5,)

        result = query_via_trino(conn, "SELECT * FROM t", limit=1, offset=0)

        assert isinstance(result, TableQueryResponse)
        assert result.result == [{"id": 1, "name": "a"}]
        assert result.pagination.total_count == 5

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_no_more_pages_when_fewer_rows(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,)]
        # COUNT query returns total_count=1
        cursor.fetchone.return_value = (1,)

        result = query_via_trino(conn, "SELECT id FROM t", limit=10, offset=0)

        assert result.pagination.has_more is False
        assert result.pagination.total_count == 1

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_offset_past_end_returns_correct_total_count(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        """Offset beyond total rows should still return accurate total_count."""
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        # No rows returned (offset=100, but only 10 rows exist)
        cursor.fetchall.return_value = []
        # COUNT query returns the true total
        cursor.fetchone.return_value = (10,)

        result = query_via_trino(conn, "SELECT id FROM t", limit=10, offset=100)

        assert result.pagination.total_count == 10
        assert result.pagination.has_more is False
        assert result.result == []

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_backtick_query_translated_to_double_quotes(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        """Spark-style backtick identifiers should be converted to double quotes."""
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,)]
        cursor.fetchone.return_value = (1,)

        query_via_trino(conn, "SELECT * FROM `mydb`.`mytable`", limit=10, offset=0)

        # Verify the executed SQL uses double quotes, not backticks
        executed_sql = cursor.execute.call_args_list[0].args[0]
        assert "`" not in executed_sql
        assert '"mydb"."mytable"' in executed_sql

    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_limit_exceeds_max_rows_raises(self, mock_validate, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="exceeds maximum"):
            query_via_trino(conn, "SELECT 1", limit=999999, offset=0, max_rows=1000)

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_cache_hit_returns_cached(self, mock_validate, mock_cache_get, mock_conn):
        conn, _ = mock_conn
        cached = [
            {
                "result": [{"id": 1}],
                "pagination": {
                    "limit": 10,
                    "offset": 0,
                    "total_count": 1,
                    "has_more": False,
                },
            }
        ]
        mock_cache_get.return_value = cached

        result = query_via_trino(conn, "SELECT 1", limit=10, offset=0)
        assert result.result == [{"id": 1}]

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_metadata_query_dispatches(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        """DESCRIBE/SHOW queries route through metadata path."""
        conn, cursor = mock_conn
        cursor.description = [("col_name",), ("data_type",)]
        cursor.fetchall.return_value = [("id", "bigint")]

        result = query_via_trino(conn, "DESCRIBE mydb.table1", limit=10, offset=0)

        assert isinstance(result, TableQueryResponse)
        assert result.result == [{"col_name": "id", "data_type": "bigint"}]

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_metadata_query_cache_hit(self, mock_validate, mock_cache_get, mock_conn):
        """Lines 98-99: metadata query returns from cache."""
        conn, _ = mock_conn
        cached = [
            {
                "result": [{"col_name": "id", "data_type": "bigint"}],
                "pagination": {
                    "limit": 1,
                    "offset": 0,
                    "total_count": 1,
                    "has_more": False,
                },
            }
        ]
        mock_cache_get.return_value = cached

        result = query_via_trino(conn, "DESCRIBE mydb.table1", limit=10, offset=0)
        assert result.result == [{"col_name": "id", "data_type": "bigint"}]
        assert result.pagination.has_more is False

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_metadata_query_error_raises(
        self, mock_validate, mock_cache_get, mock_conn
    ):
        """Lines 127-129: metadata query exception raises TrinoOperationError."""
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("Trino down")

        with pytest.raises(
            TrinoOperationError, match="Failed to execute metadata query"
        ):
            query_via_trino(conn, "DESCRIBE mydb.table1", limit=10, offset=0)

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_offset_without_order_by_warns(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        """Line 166: warning logged when offset > 0 without ORDER BY."""
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,)]
        cursor.fetchone.return_value = (5,)

        with patch("src.trino_engine.trino_service.logger") as mock_logger:
            query_via_trino(conn, "SELECT id FROM t", limit=1, offset=1)
            mock_logger.warning.assert_called_once()
            assert "ORDER BY" in mock_logger.warning.call_args.args[0]

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_cached_count_hit(
        self, mock_validate, mock_store, mock_cache_get, mock_conn
    ):
        """Line 203: cached count used when page is full."""
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        # Return exactly `limit` rows so the count path is triggered
        cursor.fetchall.return_value = [(i,) for i in range(10)]

        # First call: page cache miss; second call: count cache hit
        mock_cache_get.side_effect = [None, [{"count": 50}]]

        result = query_via_trino(conn, "SELECT id FROM t", limit=10, offset=0)
        assert result.pagination.total_count == 50
        assert result.pagination.has_more is True

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_generic_exception_raises_trino_operation_error(
        self, mock_validate, mock_cache_get, mock_conn
    ):
        """Lines 228-232: generic exception in paginated query raises TrinoOperationError."""
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("connection lost")

        with pytest.raises(TrinoOperationError, match="Failed to execute query"):
            query_via_trino(conn, "SELECT 1 FROM t", limit=10, offset=0)

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_trino_query_error_reraised(self, mock_validate, mock_cache_get, mock_conn):
        """Lines 228-229: TrinoQueryError is re-raised directly."""
        conn, cursor = mock_conn
        cursor.execute.side_effect = TrinoQueryError("bad query")

        with pytest.raises(TrinoQueryError, match="bad query"):
            query_via_trino(conn, "SELECT 1 FROM t", limit=10, offset=0)


# =============================================================================
# count_via_trino Tests
# =============================================================================


class TestCountViaTrino:
    """Tests for count_via_trino function."""

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    def test_returns_count(self, mock_exists, mock_store, mock_cache, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = (42,)

        result = count_via_trino(conn, "mydb", "mytable")
        assert result == 42

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    def test_cache_hit(self, mock_exists, mock_cache, mock_conn):
        conn, _ = mock_conn
        mock_cache.return_value = [{"count": 100}]

        result = count_via_trino(conn, "mydb", "mytable")
        assert result == 100

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_exists")
    def test_query_error_raises_trino_error(self, mock_exists, mock_cache, mock_conn):
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("Trino down")

        with pytest.raises(TrinoOperationError, match="Failed to count"):
            count_via_trino(conn, "mydb", "mytable")

    def test_invalid_database_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            count_via_trino(conn, "my;db", "mytable")

    def test_invalid_table_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid table"):
            count_via_trino(conn, "mydb", "my table")


# =============================================================================
# sample_via_trino Tests
# =============================================================================


class TestSampleViaTrino:
    """Tests for sample_via_trino function."""

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    def test_basic_sample(self, mock_exists, mock_store, mock_cache, mock_conn):
        conn, cursor = mock_conn
        cursor.description = [("id",), ("name",)]
        cursor.fetchall.return_value = [(1, "a"), (2, "b")]

        result = sample_via_trino(conn, "mydb", "mytable", limit=2)
        assert len(result) == 2
        assert result[0] == {"id": 1, "name": "a"}

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    @patch("src.trino_engine.trino_service._check_query_is_valid")
    def test_with_where_clause(
        self, mock_validate, mock_exists, mock_store, mock_cache, mock_conn
    ):
        conn, cursor = mock_conn
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,)]

        result = sample_via_trino(
            conn, "mydb", "mytable", limit=5, where_clause="id > 0"
        )
        assert len(result) == 1

        # Verify WHERE clause in the executed SQL
        executed_sql = cursor.execute.call_args.args[0]
        assert "WHERE id > 0" in executed_sql

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    def test_cache_hit(self, mock_exists, mock_cache, mock_conn):
        """Lines 293-294: sample returns cached data."""
        conn, _ = mock_conn
        cached_data = [{"id": 1, "name": "cached"}]
        mock_cache.return_value = cached_data

        result = sample_via_trino(conn, "mydb", "mytable", limit=5)
        assert result == cached_data
        mock_exists.assert_not_called()

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_exists")
    def test_query_error_raises(self, mock_exists, mock_cache, mock_conn):
        """Lines 322-324: sample query exception raises TrinoOperationError."""
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("Trino error")

        with pytest.raises(TrinoOperationError, match="Failed to sample rows"):
            sample_via_trino(conn, "mydb", "mytable", limit=5)

    def test_limit_out_of_range_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(ValueError, match="Limit must be"):
            sample_via_trino(conn, "mydb", "mytable", limit=0)

    def test_invalid_database_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            sample_via_trino(conn, "bad;db", "mytable", limit=5)

    def test_invalid_table_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid table"):
            sample_via_trino(conn, "mydb", "DROP TABLE", limit=5)

    def test_invalid_column_raises(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid column"):
            sample_via_trino(
                conn, "mydb", "mytable", limit=5, columns=["id", "1; DROP"]
            )


# =============================================================================
# select_via_trino Tests
# =============================================================================


class TestSelectViaTrino:
    """Tests for select_via_trino function."""

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    @patch("src.trino_engine.trino_service.build_select_query_trino")
    def test_basic_select(
        self, mock_build, mock_exists, mock_store, mock_cache, mock_conn
    ):
        conn, cursor = mock_conn
        # Count query result
        cursor.fetchone.return_value = (10,)
        # Main query result
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,), (2,)]

        mock_build.side_effect = [
            "SELECT COUNT(*) FROM t",  # count query
            "SELECT id FROM t LIMIT 100 OFFSET 0",  # main query
        ]

        from src.service.models import TableSelectRequest

        request = TableSelectRequest(database="mydb", table="mytable")
        result = select_via_trino(conn, request)

        assert isinstance(result, TableSelectResponse)
        assert len(result.data) == 2
        assert result.pagination.total_count == 10

    @patch("src.trino_engine.trino_service._get_from_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    def test_cache_hit(self, mock_exists, mock_cache, mock_conn):
        """Lines 343-344: select returns from cache."""
        conn, _ = mock_conn
        cached = [
            {
                "data": [{"id": 1}],
                "pagination": {
                    "limit": 100,
                    "offset": 0,
                    "total_count": 1,
                    "has_more": False,
                },
            }
        ]
        mock_cache.return_value = cached

        from src.service.models import TableSelectRequest

        request = TableSelectRequest(database="mydb", table="mytable")
        result = select_via_trino(conn, request)
        assert result.data == [{"id": 1}]
        assert result.pagination.total_count == 1

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._check_exists")
    @patch("src.trino_engine.trino_service.build_select_query_trino")
    def test_count_query_error_raises(
        self, mock_build, mock_exists, mock_cache, mock_conn
    ):
        """Lines 362-364: exception during count query raises TrinoOperationError."""
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("count failed")
        mock_build.return_value = "SELECT COUNT(*) FROM t"

        from src.service.models import TableSelectRequest

        request = TableSelectRequest(database="mydb", table="mytable")
        with pytest.raises(TrinoOperationError, match="Failed to execute count query"):
            select_via_trino(conn, request)

    @patch("src.trino_engine.trino_service._get_from_cache", return_value=None)
    @patch("src.trino_engine.trino_service._store_in_cache")
    @patch("src.trino_engine.trino_service._check_exists")
    @patch("src.trino_engine.trino_service.build_select_query_trino")
    def test_main_query_error_raises(
        self, mock_build, mock_exists, mock_store, mock_cache, mock_conn
    ):
        """Lines 393-395: exception during main select query raises TrinoOperationError."""
        conn, cursor = mock_conn
        # Count query succeeds
        cursor.fetchone.return_value = (10,)
        # Main query fails
        call_count = 0

        def side_effect(sql):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("main query failed")

        cursor.execute.side_effect = side_effect
        mock_build.side_effect = [
            "SELECT COUNT(*) FROM t",
            "SELECT id FROM t LIMIT 100 OFFSET 0",
        ]

        from src.service.models import TableSelectRequest

        request = TableSelectRequest(database="mydb", table="mytable")
        with pytest.raises(TrinoOperationError, match="Failed to execute select query"):
            select_via_trino(conn, request)
