"""
Tests for the delta_lake data store module.

Tests cover:
- get_databases() - with/without namespace filtering
- get_tables() - HMS and Spark modes
- get_db_structure() - database structure retrieval
- database_exists() / table_exists() - validation
- Governance API interactions with mocked httpx client
- Namespace prefix extraction
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.delta_lake import data_store


# =============================================================================
# Test _format_output Helper
# =============================================================================


class TestFormatOutput:
    """Tests for the _format_output helper function."""

    def test_format_as_json(self):
        """Test formatting data as JSON string."""
        data = ["db1", "db2"]
        result = data_store._format_output(data, return_json=True)
        assert result == json.dumps(data)

    def test_format_as_raw(self):
        """Test returning raw data."""
        data = ["db1", "db2"]
        result = data_store._format_output(data, return_json=False)
        assert result == data

    def test_format_complex_data_as_json(self):
        """Test formatting complex data as JSON."""
        data = {"db1": ["table1", "table2"], "db2": ["table3"]}
        result = data_store._format_output(data, return_json=True)
        assert result == json.dumps(data)


# =============================================================================
# Test _extract_databases_from_paths
# =============================================================================


class TestExtractDatabasesFromPaths:
    """Tests for the _extract_databases_from_paths function."""

    def test_extract_from_user_warehouse_paths(self):
        """Test extracting databases from user warehouse paths."""
        paths = [
            "s3a://cdm-lake/users-sql-warehouse/user1/u_user1__test.db/table1/",
            "s3a://cdm-lake/users-sql-warehouse/user1/u_user1__other.db/table2/",
        ]

        result = data_store._extract_databases_from_paths(paths)

        assert "u_user1__test" in result
        assert "u_user1__other" in result

    def test_extract_from_tenant_warehouse_paths(self):
        """Test extracting databases from tenant warehouse paths."""
        paths = [
            "s3a://cdm-lake/tenant-sql-warehouse/mygroup/mygroup__shared.db/table1/",
        ]

        result = data_store._extract_databases_from_paths(paths)

        assert "mygroup__shared" in result

    def test_ignores_non_sql_warehouse_paths(self):
        """Test that non-SQL warehouse paths are ignored."""
        paths = [
            "s3a://cdm-lake/general-warehouse/user1/data/",
            "s3a://cdm-lake/logs/spark-job-123/",
        ]

        result = data_store._extract_databases_from_paths(paths)

        assert len(result) == 0

    def test_returns_sorted_unique_databases(self):
        """Test that results are sorted and unique."""
        paths = [
            "s3a://cdm-lake/users-sql-warehouse/user1/db_z.db/t1/",
            "s3a://cdm-lake/users-sql-warehouse/user1/db_a.db/t2/",
            "s3a://cdm-lake/users-sql-warehouse/user1/db_a.db/t3/",
        ]

        result = data_store._extract_databases_from_paths(paths)

        assert result == ["db_a", "db_z"]

    def test_handles_empty_paths(self):
        """Test handling of empty paths list."""
        result = data_store._extract_databases_from_paths([])
        assert result == []


# =============================================================================
# Test _get_user_namespace_prefixes
# =============================================================================


class TestGetUserNamespacePrefixes:
    """Tests for the _get_user_namespace_prefixes function."""

    def test_returns_user_and_group_prefixes(self, mock_httpx_client, mock_settings):
        """Test that both user and group prefixes are returned."""
        client = mock_httpx_client(
            {
                "http://localhost:8000/workspaces/me/namespace-prefix": {
                    "user_namespace_prefix": "u_testuser__"
                },
                "http://localhost:8000/workspaces/me/groups": {"groups": ["group1"]},
                "http://localhost:8000/workspaces/me/namespace-prefix?tenant=group1": {
                    "tenant_namespace_prefix": "group1__"
                },
            }
        )

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                prefixes = data_store._get_user_namespace_prefixes("test_token")

        assert "u_testuser__" in prefixes
        assert "group1__" in prefixes

    def test_handles_no_groups(self, mock_httpx_client, mock_settings):
        """Test handling when user has no groups."""
        client = mock_httpx_client(
            {
                "http://localhost:8000/workspaces/me/namespace-prefix": {
                    "user_namespace_prefix": "u_solo__"
                },
                "http://localhost:8000/workspaces/me/groups": {"groups": []},
            }
        )

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                prefixes = data_store._get_user_namespace_prefixes("test_token")

        assert prefixes == ["u_solo__"]

    def test_handles_api_error(self, mock_httpx_client, mock_settings):
        """Test handling of API errors."""
        client = mock_httpx_client({})
        client.get.side_effect = Exception("API error")

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                with pytest.raises(Exception, match="Could not filter"):
                    data_store._get_user_namespace_prefixes("test_token")


# =============================================================================
# Test _get_accessible_paths
# =============================================================================


class TestGetAccessiblePaths:
    """Tests for the _get_accessible_paths function."""

    def test_returns_accessible_paths(self, mock_httpx_client, mock_settings):
        """Test that accessible paths are returned."""
        paths = [
            "s3a://cdm-lake/users-sql-warehouse/user1/db.db/",
            "s3a://cdm-lake/tenant-sql-warehouse/group/db.db/",
        ]
        client = mock_httpx_client(
            {
                "http://localhost:8000/workspaces/me/accessible-paths": {
                    "accessible_paths": paths
                }
            }
        )

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store._get_accessible_paths("test_token")

        assert result == paths

    def test_handles_empty_paths(self, mock_httpx_client, mock_settings):
        """Test handling of empty accessible paths."""
        client = mock_httpx_client(
            {
                "http://localhost:8000/workspaces/me/accessible-paths": {
                    "accessible_paths": []
                }
            }
        )

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store._get_accessible_paths("test_token")

        assert result == []


# =============================================================================
# Test get_databases
# =============================================================================


class TestGetDatabases:
    """Tests for the get_databases function."""

    def test_get_databases_via_hms(self, mock_settings):
        """Test getting databases via Hive Metastore."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_databases",
            return_value=["db1", "db2"],
        ):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store.get_databases(
                    use_hms=True, return_json=False, settings=mock_settings
                )

        assert result == ["db1", "db2"]

    def test_get_databases_via_spark(self, mock_spark_session, mock_settings):
        """Test getting databases via Spark."""
        spark = mock_spark_session(databases=["spark_db1", "spark_db2"])

        result = data_store.get_databases(
            spark=spark, use_hms=False, return_json=False, settings=mock_settings
        )

        assert "spark_db1" in result
        assert "spark_db2" in result

    def test_get_databases_returns_json(self, mock_settings):
        """Test that get_databases can return JSON."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_databases",
            return_value=["db1"],
        ):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store.get_databases(
                    use_hms=True, return_json=True, settings=mock_settings
                )

        assert result == '["db1"]'

    def test_get_databases_with_namespace_filter(
        self, mock_httpx_client, mock_settings
    ):
        """Test namespace filtering of databases."""
        client = mock_httpx_client(
            {
                "http://localhost:8000/workspaces/me/namespace-prefix": {
                    "user_namespace_prefix": "u_user__"
                },
                "http://localhost:8000/workspaces/me/groups": {"groups": []},
                "http://localhost:8000/workspaces/me/accessible-paths": {
                    "accessible_paths": []
                },
            }
        )

        with patch("src.delta_lake.data_store._get_http_client", return_value=client):
            with patch(
                "src.delta_lake.data_store.hive_metastore.get_databases",
                return_value=["u_user__db1", "u_other__db2", "shared_db"],
            ):
                with patch(
                    "src.delta_lake.data_store.get_settings", return_value=mock_settings
                ):
                    result = data_store.get_databases(
                        use_hms=True,
                        return_json=False,
                        filter_by_namespace=True,
                        auth_token="test_token",
                        settings=mock_settings,
                    )

        # Only user's databases should be returned
        assert "u_user__db1" in result
        assert "u_other__db2" not in result

    def test_get_databases_filter_requires_token(self, mock_settings):
        """Test that filtering requires auth token."""
        with pytest.raises(ValueError, match="auth_token is required"):
            data_store.get_databases(
                use_hms=True,
                filter_by_namespace=True,
                auth_token=None,
                settings=mock_settings,
            )


# =============================================================================
# Test get_tables
# =============================================================================


class TestGetTables:
    """Tests for the get_tables function."""

    def test_get_tables_via_hms(self, mock_settings):
        """Test getting tables via Hive Metastore."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_tables",
            return_value=["table1", "table2"],
        ):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store.get_tables(
                    database="testdb",
                    use_hms=True,
                    return_json=False,
                    settings=mock_settings,
                )

        assert result == ["table1", "table2"]

    def test_get_tables_via_spark(self, mock_spark_session, mock_settings):
        """Test getting tables via Spark."""
        spark = mock_spark_session(tables={"mydb": ["spark_table1", "spark_table2"]})

        result = data_store.get_tables(
            database="mydb",
            spark=spark,
            use_hms=False,
            return_json=False,
            settings=mock_settings,
        )

        assert "spark_table1" in result
        assert "spark_table2" in result

    def test_get_tables_returns_json(self, mock_settings):
        """Test that get_tables can return JSON."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_tables",
            return_value=["t1"],
        ):
            with patch(
                "src.delta_lake.data_store.get_settings", return_value=mock_settings
            ):
                result = data_store.get_tables(
                    database="db",
                    use_hms=True,
                    return_json=True,
                    settings=mock_settings,
                )

        assert result == '["t1"]'


# =============================================================================
# Test database_exists
# =============================================================================


class TestDatabaseExists:
    """Tests for the database_exists function."""

    def test_database_exists_true(self, mock_settings):
        """Test that existing database returns True."""
        with patch(
            "src.delta_lake.data_store.get_databases",
            return_value=["testdb", "otherdb"],
        ):
            result = data_store.database_exists("testdb", settings=mock_settings)

        assert result is True

    def test_database_exists_false(self, mock_settings):
        """Test that non-existing database returns False."""
        with patch(
            "src.delta_lake.data_store.get_databases",
            return_value=["otherdb"],
        ):
            result = data_store.database_exists("testdb", settings=mock_settings)

        assert result is False


# =============================================================================
# Test table_exists
# =============================================================================


class TestTableExists:
    """Tests for the table_exists function."""

    def test_table_exists_true(self, mock_settings):
        """Test that existing table returns True."""
        with patch(
            "src.delta_lake.data_store.get_tables",
            return_value=["users", "orders"],
        ):
            result = data_store.table_exists("testdb", "users", settings=mock_settings)

        assert result is True

    def test_table_exists_false(self, mock_settings):
        """Test that non-existing table returns False."""
        with patch(
            "src.delta_lake.data_store.get_tables",
            return_value=["orders"],
        ):
            result = data_store.table_exists("testdb", "users", settings=mock_settings)

        assert result is False


# =============================================================================
# Test get_db_structure
# =============================================================================


class TestGetDbStructure:
    """Tests for the get_db_structure function."""

    def test_get_structure_without_schema(self, mock_settings):
        """Test getting database structure without schemas."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_databases",
            return_value=["db1", "db2"],
        ):
            with patch(
                "src.delta_lake.data_store.hive_metastore.get_tables",
                side_effect=lambda database, **kwargs: (
                    ["table1"] if database == "db1" else ["table2", "table3"]
                ),
            ):
                with patch(
                    "src.delta_lake.data_store.get_settings", return_value=mock_settings
                ):
                    result = data_store.get_db_structure(
                        use_hms=True,
                        with_schema=False,
                        return_json=False,
                        settings=mock_settings,
                    )

        assert result == {"db1": ["table1"], "db2": ["table2", "table3"]}

    def test_get_structure_returns_json(self, mock_settings):
        """Test that get_db_structure can return JSON."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_databases",
            return_value=["db1"],
        ):
            with patch(
                "src.delta_lake.data_store.hive_metastore.get_tables",
                return_value=["t1"],
            ):
                with patch(
                    "src.delta_lake.data_store.get_settings", return_value=mock_settings
                ):
                    result = data_store.get_db_structure(
                        use_hms=True,
                        with_schema=False,
                        return_json=True,
                        settings=mock_settings,
                    )

        assert result == '{"db1": ["t1"]}'


# =============================================================================
# Test HTTP Client Management
# =============================================================================


class TestHttpClientManagement:
    """Tests for HTTP client management."""

    def test_get_http_client_cached(self):
        """Test that HTTP client is cached."""
        # Clear any existing cache
        data_store._get_http_client.cache_clear()

        with patch("src.delta_lake.data_store.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client1 = data_store._get_http_client()
            client2 = data_store._get_http_client()

            # Should only create one client
            assert mock_client_class.call_count == 1
            assert client1 is client2

        data_store._get_http_client.cache_clear()


# =============================================================================
# Spark Session Requirement Tests
# =============================================================================


class TestSparkSessionRequirement:
    """Tests for Spark session requirements."""

    def test_execute_with_spark_requires_session(self):
        """Test that _execute_with_spark requires a SparkSession."""
        with pytest.raises(ValueError, match="SparkSession must be provided"):
            data_store._execute_with_spark(lambda s: s, spark=None)

    def test_get_db_structure_with_schema_requires_spark(self, mock_settings):
        """Test that with_schema=True requires Spark session."""
        with patch(
            "src.delta_lake.data_store.hive_metastore.get_databases",
            return_value=["db1"],
        ):
            with patch(
                "src.delta_lake.data_store.hive_metastore.get_tables",
                return_value=["t1"],
            ):
                with patch(
                    "src.delta_lake.data_store.get_settings", return_value=mock_settings
                ):
                    with pytest.raises(
                        ValueError, match="SparkSession must be provided"
                    ):
                        data_store.get_db_structure(
                            spark=None,
                            use_hms=True,
                            with_schema=True,
                            return_json=False,
                            settings=mock_settings,
                        )
