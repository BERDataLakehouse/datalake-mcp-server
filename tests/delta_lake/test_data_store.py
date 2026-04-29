"""
Tests for the delta_lake data store module (Iceberg catalogs).

Tests cover:
- _format_output() - JSON/raw output formatting
- _list_iceberg_catalogs() - Catalog discovery with spark_catalog exclusion
- get_databases() - Iceberg namespace listing
- get_tables() - Table listing in Iceberg namespaces
- get_table_schema() - Column listing from Iceberg tables
- get_db_structure() - Full database structure retrieval
- database_exists() / table_exists() - Validation helpers
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
        data = {"my.demo": ["table1", "table2"], "kbase.shared": ["table3"]}
        result = data_store._format_output(data, return_json=True)
        assert result == json.dumps(data)


# =============================================================================
# Test _list_iceberg_catalogs
# =============================================================================


def _make_set_rows(*catalog_names: str) -> list:
    """Create mock SET command rows for catalog configs.

    For each catalog name, generates the top-level ``spark.sql.catalog.<name>``
    key plus a few sub-property keys to simulate realistic SET output.
    """
    rows = []
    for name in catalog_names:
        rows.append(
            {
                "key": f"spark.sql.catalog.{name}",
                "value": "org.apache.iceberg.spark.SparkCatalog",
            }
        )
        rows.append({"key": f"spark.sql.catalog.{name}.type", "value": "rest"})
        rows.append(
            {
                "key": f"spark.sql.catalog.{name}.uri",
                "value": "http://polaris:8181/api/catalog",
            }
        )
    # Add some unrelated config entries
    rows.append({"key": "spark.app.name", "value": "test"})
    rows.append(
        {
            "key": "spark.sql.extensions",
            "value": "io.delta.sql.DeltaSparkSessionExtension",
        }
    )
    return rows


class TestListIcebergCatalogs:
    """Tests for the _list_iceberg_catalogs function."""

    def test_excludes_spark_catalog(self):
        """Test that spark_catalog is excluded from results."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = _make_set_rows(
            "spark_catalog", "my", "kbase"
        )

        result = data_store._list_iceberg_catalogs(mock_spark)

        assert "spark_catalog" not in result
        assert result == ["kbase", "my"]

    def test_returns_sorted(self):
        """Test that catalogs are returned sorted."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = _make_set_rows(
            "zebra", "alpha"
        )

        result = data_store._list_iceberg_catalogs(mock_spark)

        assert result == ["alpha", "zebra"]

    def test_empty_catalogs(self):
        """Test handling when no Iceberg catalogs exist."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = _make_set_rows(
            "spark_catalog"
        )

        result = data_store._list_iceberg_catalogs(mock_spark)

        assert result == []


# =============================================================================
# Test get_databases
# =============================================================================


class TestGetDatabases:
    """Tests for the get_databases function."""

    def test_requires_spark_session(self):
        """Test that get_databases raises ValueError without SparkSession."""
        with pytest.raises(ValueError, match="SparkSession must be provided"):
            data_store.get_databases(spark=None, return_json=False)

    def test_returns_catalog_namespace_format(self):
        """Test that databases are returned in catalog.namespace format."""
        mock_spark = MagicMock()

        with (
            patch.object(
                data_store, "_list_iceberg_catalogs", return_value=["kbase", "my"]
            ),
            patch.object(data_store, "_list_hive_databases", return_value=[]),
        ):

            def sql_side_effect(query):
                result = MagicMock()
                if "SHOW NAMESPACES IN kbase" in query:
                    result.collect.return_value = [
                        {"namespace": "shared_data"},
                        {"namespace": "research"},
                    ]
                elif "SHOW NAMESPACES IN my" in query:
                    result.collect.return_value = [
                        {"namespace": "demo"},
                        {"namespace": "analysis"},
                    ]
                return result

            mock_spark.sql.side_effect = sql_side_effect

            result = data_store.get_databases(spark=mock_spark, return_json=False)

        assert result == [
            "kbase.research",
            "kbase.shared_data",
            "my.analysis",
            "my.demo",
        ]

    def test_returns_json(self):
        """Test that get_databases can return JSON."""
        mock_spark = MagicMock()

        with (
            patch.object(data_store, "_list_iceberg_catalogs", return_value=["my"]),
            patch.object(data_store, "_list_hive_databases", return_value=[]),
        ):
            mock_spark.sql.return_value.collect.return_value = [{"namespace": "demo"}]

            result = data_store.get_databases(spark=mock_spark, return_json=True)

        assert result == '["my.demo"]'

    def test_handles_inaccessible_catalog(self):
        """Test that inaccessible catalogs are skipped."""
        mock_spark = MagicMock()

        with (
            patch.object(
                data_store, "_list_iceberg_catalogs", return_value=["my", "broken"]
            ),
            patch.object(data_store, "_list_hive_databases", return_value=[]),
        ):

            def sql_side_effect(query):
                if "broken" in query:
                    raise Exception("Catalog not accessible")
                result = MagicMock()
                result.collect.return_value = [{"namespace": "demo"}]
                return result

            mock_spark.sql.side_effect = sql_side_effect

            result = data_store.get_databases(spark=mock_spark, return_json=False)

        assert result == ["my.demo"]

    def test_empty_catalogs(self):
        """Test get_databases with no catalogs."""
        mock_spark = MagicMock()

        with (
            patch.object(data_store, "_list_iceberg_catalogs", return_value=[]),
            patch.object(data_store, "_list_hive_databases", return_value=[]),
        ):
            result = data_store.get_databases(spark=mock_spark, return_json=False)

        assert result == []

    def test_includes_hive_databases(self):
        """Hive (spark_catalog) databases are listed alongside Iceberg namespaces."""
        mock_spark = MagicMock()

        with (
            patch.object(data_store, "_list_iceberg_catalogs", return_value=["my"]),
            patch.object(
                data_store,
                "_list_hive_databases",
                return_value=["u_alice__demo", "default"],
            ),
        ):
            mock_spark.sql.return_value.collect.return_value = [{"namespace": "demo"}]

            result = data_store.get_databases(spark=mock_spark, return_json=False)

        assert result == ["default", "my.demo", "u_alice__demo"]


# =============================================================================
# Test _list_hive_databases
# =============================================================================


class TestListHiveDatabases:
    """Tests for _list_hive_databases."""

    def test_returns_flat_database_names(self):
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = [
            {"namespace": "u_alice__demo"},
            {"namespace": "default"},
        ]

        result = data_store._list_hive_databases(mock_spark)

        mock_spark.sql.assert_called_once_with("SHOW DATABASES IN spark_catalog")
        assert result == ["default", "u_alice__demo"]

    def test_returns_empty_on_failure(self):
        """A broken/disconnected Hive Metastore must not break Iceberg listing."""
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = Exception("HMS unreachable")

        assert data_store._list_hive_databases(mock_spark) == []


# =============================================================================
# Test get_tables
# =============================================================================


class TestGetTables:
    """Tests for the get_tables function."""

    def test_requires_spark_session(self):
        """Test that get_tables raises ValueError without SparkSession."""
        with pytest.raises(ValueError, match="SparkSession must be provided"):
            data_store.get_tables(database="my.demo", spark=None)

    def test_returns_sorted_table_names(self):
        """Test that tables are returned sorted."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = [
            {"tableName": "users"},
            {"tableName": "orders"},
        ]

        result = data_store.get_tables(
            database="my.demo", spark=mock_spark, return_json=False
        )

        assert result == ["orders", "users"]
        mock_spark.sql.assert_called_with("SHOW TABLES IN my.demo")

    def test_returns_json(self):
        """Test that get_tables can return JSON."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = [{"tableName": "t1"}]

        result = data_store.get_tables(
            database="my.demo", spark=mock_spark, return_json=True
        )

        assert result == '["t1"]'

    def test_handles_error(self):
        """Test get_tables returns empty list on error."""
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = Exception("Namespace not found")

        result = data_store.get_tables(
            database="my.nonexistent", spark=mock_spark, return_json=False
        )

        assert result == []


# =============================================================================
# Test get_table_schema
# =============================================================================


class TestGetTableSchema:
    """Tests for the get_table_schema function."""

    def test_requires_spark_session(self):
        """Test that get_table_schema raises ValueError without SparkSession."""
        with pytest.raises(ValueError, match="SparkSession must be provided"):
            data_store.get_table_schema(database="my.demo", table="users", spark=None)

    def test_returns_column_names(self):
        """Test that column names are extracted correctly."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = [
            {"col_name": "id", "data_type": "int", "comment": ""},
            {"col_name": "name", "data_type": "string", "comment": ""},
            {"col_name": "age", "data_type": "int", "comment": ""},
        ]

        result = data_store.get_table_schema(
            database="my.demo", table="users", spark=mock_spark, return_json=False
        )

        assert result == ["id", "name", "age"]
        mock_spark.sql.assert_called_with("DESCRIBE my.demo.users")

    def test_filters_metadata_rows(self):
        """Test that partition/metadata rows starting with # are filtered."""
        mock_spark = MagicMock()
        mock_spark.sql.return_value.collect.return_value = [
            {"col_name": "id", "data_type": "int", "comment": ""},
            {"col_name": "# Partitioning", "data_type": "", "comment": ""},
            {"col_name": "# col_name", "data_type": "data_type", "comment": ""},
        ]

        result = data_store.get_table_schema(
            database="my.demo", table="users", spark=mock_spark, return_json=False
        )

        assert result == ["id"]

    def test_handles_error(self):
        """Test get_table_schema returns empty list on error."""
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = Exception("Table not found")

        result = data_store.get_table_schema(
            database="my.demo", table="nonexistent", spark=mock_spark, return_json=False
        )

        assert result == []


# =============================================================================
# Test get_db_structure
# =============================================================================


class TestGetDbStructure:
    """Tests for the get_db_structure function."""

    def test_requires_spark_session(self):
        """Test that get_db_structure raises ValueError without SparkSession."""
        with pytest.raises(ValueError, match="SparkSession must be provided"):
            data_store.get_db_structure(spark=None, return_json=False)

    def test_structure_without_schema(self):
        """Test getting database structure without schemas."""
        mock_spark = MagicMock()

        with patch.object(
            data_store, "get_databases", return_value=["my.demo", "kbase.shared"]
        ):
            with patch.object(
                data_store,
                "get_tables",
                side_effect=lambda database, **kwargs: {
                    "my.demo": ["table1", "table2"],
                    "kbase.shared": ["dataset"],
                }[database],
            ):
                result = data_store.get_db_structure(
                    spark=mock_spark, with_schema=False, return_json=False
                )

        assert result == {
            "my.demo": ["table1", "table2"],
            "kbase.shared": ["dataset"],
        }

    def test_structure_with_schema(self):
        """Test getting database structure with schemas."""
        mock_spark = MagicMock()

        with patch.object(data_store, "get_databases", return_value=["my.demo"]):
            with patch.object(data_store, "get_tables", return_value=["table1"]):
                with patch.object(
                    data_store, "get_table_schema", return_value=["col1", "col2"]
                ):
                    result = data_store.get_db_structure(
                        spark=mock_spark, with_schema=True, return_json=False
                    )

        assert result == {"my.demo": {"table1": ["col1", "col2"]}}

    def test_returns_json(self):
        """Test that get_db_structure can return JSON."""
        mock_spark = MagicMock()

        with patch.object(data_store, "get_databases", return_value=["my.demo"]):
            with patch.object(data_store, "get_tables", return_value=["t1"]):
                result = data_store.get_db_structure(
                    spark=mock_spark, with_schema=False, return_json=True
                )

        assert result == '{"my.demo": ["t1"]}'


# =============================================================================
# Test database_exists
# =============================================================================


class TestDatabaseExists:
    """Tests for the database_exists function."""

    def test_database_exists_true(self):
        """Test that existing database returns True."""
        mock_spark = MagicMock()

        with patch.object(
            data_store, "get_databases", return_value=["my.demo", "kbase.shared"]
        ):
            result = data_store.database_exists("my.demo", spark=mock_spark)

        assert result is True

    def test_database_exists_false(self):
        """Test that non-existing database returns False."""
        mock_spark = MagicMock()

        with patch.object(data_store, "get_databases", return_value=["my.demo"]):
            result = data_store.database_exists("kbase.shared", spark=mock_spark)

        assert result is False


# =============================================================================
# Test table_exists
# =============================================================================


class TestTableExists:
    """Tests for the table_exists function."""

    def test_table_exists_true(self):
        """Test that existing table returns True."""
        mock_spark = MagicMock()

        with patch.object(data_store, "get_tables", return_value=["users", "orders"]):
            result = data_store.table_exists("my.demo", "users", spark=mock_spark)

        assert result is True

    def test_table_exists_false(self):
        """Test that non-existing table returns False."""
        mock_spark = MagicMock()

        with patch.object(data_store, "get_tables", return_value=["orders"]):
            result = data_store.table_exists("my.demo", "users", spark=mock_spark)

        assert result is False
