"""Tests for the Hive metastore module."""

from unittest.mock import MagicMock, patch

import pytest

from src.delta_lake.hive_metastore import (
    get_databases,
    get_hive_metastore_client,
    get_tables,
)


class TestGetHiveMetastoreClient:
    """Tests for get_hive_metastore_client function."""

    def test_valid_thrift_uri_with_port(self):
        """Test parsing a valid thrift URI with host and port."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        with patch("src.delta_lake.hive_metastore.HMSClient") as mock_hms:
            mock_client = MagicMock()
            mock_hms.return_value = mock_client

            result = get_hive_metastore_client(settings)

            mock_hms.assert_called_once_with(host="hive-metastore", port=9083)
            assert result == mock_client

    def test_valid_thrift_uri_without_port(self):
        """Test parsing a valid thrift URI without port (uses default 9083)."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore"

        with patch("src.delta_lake.hive_metastore.HMSClient") as mock_hms:
            mock_client = MagicMock()
            mock_hms.return_value = mock_client

            result = get_hive_metastore_client(settings)

            mock_hms.assert_called_once_with(host="hive-metastore", port=9083)
            assert result == mock_client

    def test_invalid_uri_format(self):
        """Test that invalid URI format raises ValueError."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "http://hive-metastore:9083"

        with pytest.raises(ValueError) as exc_info:
            get_hive_metastore_client(settings)

        assert "Invalid HMS URI format" in str(exc_info.value)
        assert "Expected thrift://host:port" in str(exc_info.value)

    def test_uri_with_custom_port(self):
        """Test parsing a thrift URI with custom port."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://custom-host:19083"

        with patch("src.delta_lake.hive_metastore.HMSClient") as mock_hms:
            mock_client = MagicMock()
            mock_hms.return_value = mock_client

            result = get_hive_metastore_client(settings)

            mock_hms.assert_called_once_with(host="custom-host", port=19083)
            assert result == mock_client


class TestGetDatabases:
    """Tests for get_databases function."""

    def test_get_databases_success(self):
        """Test successful retrieval of databases."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        mock_client = MagicMock()
        mock_client.get_databases.return_value = ["default", "my_database", "test_db"]

        with patch("src.delta_lake.hive_metastore.HMSClient", return_value=mock_client):
            result = get_databases(settings)

            mock_client.open.assert_called_once()
            mock_client.get_databases.assert_called_once_with("*")
            mock_client.close.assert_called_once()
            assert result == ["default", "my_database", "test_db"]

    def test_get_databases_exception_still_closes(self):
        """Test that client is closed even when exception occurs."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        mock_client = MagicMock()
        mock_client.get_databases.side_effect = Exception("HMS connection failed")

        with patch("src.delta_lake.hive_metastore.HMSClient", return_value=mock_client):
            with pytest.raises(Exception) as exc_info:
                get_databases(settings)

            assert "HMS connection failed" in str(exc_info.value)
            mock_client.open.assert_called_once()
            mock_client.close.assert_called_once()


class TestGetTables:
    """Tests for get_tables function."""

    def test_get_tables_success(self):
        """Test successful retrieval of tables."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        mock_client = MagicMock()
        mock_client.get_tables.return_value = ["table1", "table2", "table3"]

        with patch("src.delta_lake.hive_metastore.HMSClient", return_value=mock_client):
            result = get_tables("my_database", settings)

            mock_client.open.assert_called_once()
            mock_client.get_tables.assert_called_once_with("my_database", "*")
            mock_client.close.assert_called_once()
            assert result == ["table1", "table2", "table3"]

    def test_get_tables_exception_still_closes(self):
        """Test that client is closed even when exception occurs."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        mock_client = MagicMock()
        mock_client.get_tables.side_effect = Exception("Database not found")

        with patch("src.delta_lake.hive_metastore.HMSClient", return_value=mock_client):
            with pytest.raises(Exception) as exc_info:
                get_tables("nonexistent_db", settings)

            assert "Database not found" in str(exc_info.value)
            mock_client.open.assert_called_once()
            mock_client.close.assert_called_once()

    def test_get_tables_empty_result(self):
        """Test retrieval of empty table list."""
        settings = MagicMock()
        settings.BERDL_HIVE_METASTORE_URI = "thrift://hive-metastore:9083"

        mock_client = MagicMock()
        mock_client.get_tables.return_value = []

        with patch("src.delta_lake.hive_metastore.HMSClient", return_value=mock_client):
            result = get_tables("empty_db", settings)

            assert result == []
            mock_client.close.assert_called_once()
