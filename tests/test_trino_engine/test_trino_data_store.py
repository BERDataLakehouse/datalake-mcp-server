"""
Tests for the Trino metadata operations module.

Tests cover:
- get_databases_trino: HMS vs Trino SQL path, namespace filtering
- get_tables_trino: HMS vs Trino SQL path
- get_table_schema_trino: column name retrieval
- get_db_structure_trino: full structure with/without schema
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.exceptions import TrinoOperationError, TrinoQueryError
from src.trino_engine.trino_data_store import (
    get_databases_trino,
    get_db_structure_trino,
    get_table_schema_trino,
    get_tables_trino,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_conn():
    """Create a mock Trino connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


@pytest.fixture
def mock_settings():
    """Minimal mock settings for HMS functions."""
    settings = MagicMock()
    settings.BERDL_HIVE_METASTORE_URI = "thrift://hms:9083"
    return settings


# =============================================================================
# get_databases_trino Tests
# =============================================================================


class TestGetDatabasesTrino:
    """Tests for the get_databases_trino function."""

    @patch("src.trino_engine.trino_data_store.hive_metastore.get_databases")
    def test_hms_mode(self, mock_hms, mock_conn, mock_settings):
        mock_hms.return_value = ["db1", "db2"]
        conn, _ = mock_conn

        result = get_databases_trino(conn, use_hms=True, settings=mock_settings)

        assert result == ["db1", "db2"]
        mock_hms.assert_called_once_with(settings=mock_settings)

    @patch("src.trino_engine.trino_data_store.get_settings")
    @patch("src.trino_engine.trino_data_store.hive_metastore.get_databases")
    def test_settings_default_from_get_settings(
        self, mock_hms, mock_get_settings, mock_conn
    ):
        """When settings=None, get_settings() is called (line 39)."""
        fallback_settings = MagicMock()
        mock_get_settings.return_value = fallback_settings
        mock_hms.return_value = ["db1"]
        conn, _ = mock_conn

        result = get_databases_trino(conn, use_hms=True, settings=None)

        mock_get_settings.assert_called_once()
        mock_hms.assert_called_once_with(settings=fallback_settings)
        assert result == ["db1"]

    def test_trino_sql_mode(self, mock_conn, mock_settings):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [("default",), ("mydb",)]

        result = get_databases_trino(conn, use_hms=False, settings=mock_settings)

        assert result == ["default", "mydb"]
        cursor.execute.assert_called_with("SHOW SCHEMAS")

    def test_trino_sql_error_raises(self, mock_conn, mock_settings):
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("Trino error")

        with pytest.raises(TrinoOperationError, match="Failed to list databases"):
            get_databases_trino(conn, use_hms=False, settings=mock_settings)

    def test_filter_by_namespace_requires_token(self, mock_conn, mock_settings):
        conn, _ = mock_conn
        with pytest.raises(ValueError, match="auth_token is required"):
            get_databases_trino(
                conn,
                use_hms=True,
                filter_by_namespace=True,
                auth_token=None,
                settings=mock_settings,
            )

    @patch("src.trino_engine.trino_data_store._get_user_namespace_prefixes")
    @patch("src.trino_engine.trino_data_store.hive_metastore.get_databases")
    def test_filter_by_namespace_error_propagates(
        self,
        mock_hms,
        mock_prefixes,
        mock_conn,
        mock_settings,
    ):
        """Lines 63-65: exception during namespace filtering is re-raised."""
        mock_hms.return_value = ["db1"]
        mock_prefixes.side_effect = Exception("namespace lookup failed")
        conn, _ = mock_conn

        with pytest.raises(Exception, match="namespace lookup failed"):
            get_databases_trino(
                conn,
                use_hms=True,
                filter_by_namespace=True,
                auth_token="tok",
                settings=mock_settings,
            )

    @patch("src.trino_engine.trino_data_store._extract_databases_from_paths")
    @patch("src.trino_engine.trino_data_store._get_accessible_paths")
    @patch("src.trino_engine.trino_data_store._get_user_namespace_prefixes")
    @patch("src.trino_engine.trino_data_store.hive_metastore.get_databases")
    def test_filter_by_namespace(
        self,
        mock_hms,
        mock_prefixes,
        mock_paths,
        mock_extract,
        mock_conn,
        mock_settings,
    ):
        mock_hms.return_value = ["u_alice__data", "u_bob__data", "shared_db"]
        mock_prefixes.return_value = ["u_alice__"]
        mock_paths.return_value = ["/shared_db/"]
        mock_extract.return_value = ["shared_db"]
        conn, _ = mock_conn

        result = get_databases_trino(
            conn,
            use_hms=True,
            filter_by_namespace=True,
            auth_token="tok",
            settings=mock_settings,
        )

        assert "u_alice__data" in result
        assert "shared_db" in result
        assert "u_bob__data" not in result


# =============================================================================
# get_tables_trino Tests
# =============================================================================


class TestGetTablesTrino:
    """Tests for the get_tables_trino function."""

    @patch("src.trino_engine.trino_data_store.hive_metastore.get_tables")
    def test_hms_mode(self, mock_hms, mock_conn, mock_settings):
        mock_hms.return_value = ["table1", "table2"]
        conn, _ = mock_conn

        result = get_tables_trino(conn, "mydb", use_hms=True, settings=mock_settings)

        assert result == ["table1", "table2"]

    def test_trino_sql_mode(self, mock_conn, mock_settings):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [("orders",), ("users",)]

        result = get_tables_trino(conn, "mydb", use_hms=False, settings=mock_settings)

        assert result == ["orders", "users"]
        cursor.execute.assert_called_with('SHOW TABLES FROM "mydb"')

    @patch("src.trino_engine.trino_data_store.get_settings")
    @patch("src.trino_engine.trino_data_store.hive_metastore.get_tables")
    def test_settings_default_from_get_settings(
        self, mock_hms, mock_get_settings, mock_conn
    ):
        """Line 78: when settings=None, get_settings() is called."""
        fallback_settings = MagicMock()
        mock_get_settings.return_value = fallback_settings
        mock_hms.return_value = ["t1"]
        conn, _ = mock_conn

        result = get_tables_trino(conn, "mydb", use_hms=True, settings=None)

        mock_get_settings.assert_called_once()
        mock_hms.assert_called_once_with(database="mydb", settings=fallback_settings)
        assert result == ["t1"]

    def test_trino_sql_error_raises(self, mock_conn, mock_settings):
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("fail")

        with pytest.raises(TrinoOperationError, match="Failed to list tables"):
            get_tables_trino(conn, "mydb", use_hms=False, settings=mock_settings)

    def test_invalid_database_rejects_injection(self, mock_conn, mock_settings):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            get_tables_trino(
                conn, '"; DROP TABLE --', use_hms=False, settings=mock_settings
            )


# =============================================================================
# get_table_schema_trino Tests
# =============================================================================


class TestGetTableSchemaTrino:
    """Tests for the get_table_schema_trino function."""

    def test_returns_column_names(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [
            ("id", "bigint", "", ""),
            ("name", "varchar", "", ""),
        ]

        result = get_table_schema_trino(conn, "mydb", "users")

        assert result == ["id", "name"]
        cursor.execute.assert_called_with('SHOW COLUMNS FROM "mydb"."users"')

    def test_error_raises(self, mock_conn):
        conn, cursor = mock_conn
        cursor.execute.side_effect = Exception("fail")

        with pytest.raises(TrinoOperationError, match="Failed to get schema"):
            get_table_schema_trino(conn, "mydb", "users")

    def test_invalid_database_rejects_injection(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid database"):
            get_table_schema_trino(conn, "bad;db", "users")

    def test_invalid_table_rejects_injection(self, mock_conn):
        conn, _ = mock_conn
        with pytest.raises(TrinoQueryError, match="Invalid table"):
            get_table_schema_trino(conn, "mydb", "users; DROP TABLE")


# =============================================================================
# get_db_structure_trino Tests
# =============================================================================


class TestGetDbStructureTrino:
    """Tests for the get_db_structure_trino function."""

    @patch("src.trino_engine.trino_data_store.get_settings")
    @patch("src.trino_engine.trino_data_store.get_table_schema_trino")
    @patch("src.trino_engine.trino_data_store.get_tables_trino")
    @patch("src.trino_engine.trino_data_store.get_databases_trino")
    def test_settings_default_from_get_settings(
        self, mock_dbs, mock_tables, mock_schema, mock_get_settings, mock_conn
    ):
        """Line 117: when settings=None, get_settings() is called."""
        fallback_settings = MagicMock()
        mock_get_settings.return_value = fallback_settings
        mock_dbs.return_value = ["db1"]
        mock_tables.return_value = ["t1"]
        conn, _ = mock_conn

        get_db_structure_trino(conn, with_schema=False, filter_by_namespace=False, settings=None)

        mock_get_settings.assert_called_once()
        mock_dbs.assert_called_once_with(
            conn, use_hms=True, filter_by_namespace=False, auth_token=None, settings=fallback_settings
        )

    @patch("src.trino_engine.trino_data_store.get_table_schema_trino")
    @patch("src.trino_engine.trino_data_store.get_tables_trino")
    @patch("src.trino_engine.trino_data_store.get_databases_trino")
    def test_without_schema(
        self, mock_dbs, mock_tables, mock_schema, mock_conn, mock_settings
    ):
        conn, _ = mock_conn
        mock_dbs.return_value = ["db1"]
        mock_tables.return_value = ["t1", "t2"]

        result = get_db_structure_trino(conn, with_schema=False, filter_by_namespace=False, settings=mock_settings)

        assert result == {"db1": ["t1", "t2"]}
        mock_schema.assert_not_called()

    @patch("src.trino_engine.trino_data_store.get_table_schema_trino")
    @patch("src.trino_engine.trino_data_store.get_tables_trino")
    @patch("src.trino_engine.trino_data_store.get_databases_trino")
    def test_with_schema(
        self, mock_dbs, mock_tables, mock_schema, mock_conn, mock_settings
    ):
        conn, _ = mock_conn
        mock_dbs.return_value = ["db1"]
        mock_tables.return_value = ["t1"]
        mock_schema.return_value = ["id", "name"]

        result = get_db_structure_trino(conn, with_schema=True, filter_by_namespace=False, settings=mock_settings)

        assert result == {"db1": {"t1": ["id", "name"]}}
