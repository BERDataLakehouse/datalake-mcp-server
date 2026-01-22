"""
Tests for standalone_operations.py module.

Tests cover:
- _create_spark_session helper function
- _cleanup_spark_session helper function
- _reconstruct_settings helper function
- All subprocess operation wrappers (count, sample, query, select, list_databases, etc.)
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import AnyUrl
from py4j.protocol import Py4JJavaError

from src.service.standalone_operations import (
    _create_spark_session,
    _cleanup_spark_session,
    _reconstruct_settings,
    count_table_subprocess,
    sample_table_subprocess,
    query_table_subprocess,
    select_table_subprocess,
    list_databases_subprocess,
    list_tables_subprocess,
    get_table_schema_subprocess,
    get_db_structure_subprocess,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings_dict():
    """Create a mock settings dictionary for testing."""
    return {
        "USER": "testuser",
        "MINIO_ACCESS_KEY": "access123",
        "MINIO_SECRET_KEY": "secret456",
        "MINIO_ENDPOINT_URL": "http://minio:9000",
        "MINIO_SECURE": False,
        "SPARK_HOME": "/opt/spark",
        "SPARK_MASTER_URL": "spark://master:7077",
        "BERDL_HIVE_METASTORE_URI": "thrift://hive:9083",
        "SPARK_WORKER_COUNT": 2,
        "SPARK_WORKER_CORES": 2,
        "SPARK_WORKER_MEMORY": "2g",
        "SPARK_MASTER_CORES": 2,
        "SPARK_MASTER_MEMORY": "2g",
        "GOVERNANCE_API_URL": "http://governance:8080",
        "BERDL_POD_IP": "10.0.0.1",
    }


@pytest.fixture
def mock_spark_session():
    """Create a mock SparkSession for testing."""
    spark = MagicMock()
    spark.sparkContext = MagicMock()
    spark.sparkContext._jvm = MagicMock()
    return spark


# =============================================================================
# Tests for _create_spark_session
# =============================================================================


class TestCreateSparkSession:
    """Tests for _create_spark_session helper function."""

    def test_creates_session_with_settings(self, mock_settings_dict):
        """Test that _create_spark_session creates a session with proper settings."""
        mock_spark = MagicMock()

        with patch(
            "src.service.standalone_operations.get_spark_session",
            return_value=mock_spark,
        ) as mock_get_session:
            result = _create_spark_session(mock_settings_dict, "test_app")

            assert result == mock_spark
            mock_get_session.assert_called_once()
            call_kwargs = mock_get_session.call_args[1]
            assert call_kwargs["app_name"] == "test_app"
            assert call_kwargs["use_spark_connect"] is False

    def test_converts_spark_master_url_to_anyurl(self, mock_settings_dict):
        """Test that SPARK_MASTER_URL string is converted to AnyUrl."""
        mock_spark = MagicMock()

        with patch(
            "src.service.standalone_operations.get_spark_session",
            return_value=mock_spark,
        ) as mock_get_session:
            _create_spark_session(mock_settings_dict, "test_app")

            call_kwargs = mock_get_session.call_args[1]
            settings = call_kwargs["settings"]
            assert isinstance(settings.SPARK_MASTER_URL, AnyUrl)

    def test_converts_spark_connect_url_to_anyurl(self, mock_settings_dict):
        """Test that SPARK_CONNECT_URL string is converted to AnyUrl."""
        mock_spark = MagicMock()
        mock_settings_dict["SPARK_CONNECT_URL"] = "sc://connect:15002"

        with patch(
            "src.service.standalone_operations.get_spark_session",
            return_value=mock_spark,
        ):
            _create_spark_session(mock_settings_dict, "test_app")
            # No exception means conversion worked


# =============================================================================
# Tests for _cleanup_spark_session
# =============================================================================


class TestCleanupSparkSession:
    """Tests for _cleanup_spark_session helper function."""

    def test_clears_hadoop_filesystem_cache(self, mock_spark_session):
        """Test that Hadoop FileSystem cache is cleared."""
        _cleanup_spark_session(mock_spark_session)

        mock_spark_session.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.closeAll.assert_called_once()

    def test_stops_spark_session(self, mock_spark_session):
        """Test that spark.stop() is called."""
        _cleanup_spark_session(mock_spark_session)

        mock_spark_session.stop.assert_called_once()

    def test_handles_no_jvm(self, mock_spark_session):
        """Test that cleanup handles missing JVM gracefully."""
        mock_spark_session.sparkContext._jvm = None

        # Should not raise exception
        _cleanup_spark_session(mock_spark_session)
        mock_spark_session.stop.assert_called_once()

    def test_handles_jvm_exception(self, mock_spark_session):
        """Test that cleanup handles JVM exceptions gracefully."""
        mock_spark_session.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.closeAll.side_effect = Exception(
            "JVM error"
        )

        # Should not raise exception
        _cleanup_spark_session(mock_spark_session)
        mock_spark_session.stop.assert_called_once()

    def test_handles_py4j_error_on_stop(self, mock_spark_session):
        """Test that cleanup handles Py4JJavaError on stop gracefully."""
        mock_spark_session.stop.side_effect = Py4JJavaError("test", MagicMock())

        # Should not raise exception
        _cleanup_spark_session(mock_spark_session)

    def test_handles_general_exception_on_stop(self, mock_spark_session):
        """Test that cleanup handles general exceptions on stop gracefully."""
        mock_spark_session.stop.side_effect = Exception("General error")

        # Should not raise exception
        _cleanup_spark_session(mock_spark_session)


# =============================================================================
# Tests for _reconstruct_settings
# =============================================================================


class TestReconstructSettings:
    """Tests for _reconstruct_settings helper function."""

    def test_reconstructs_settings_from_dict(self, mock_settings_dict):
        """Test that settings are properly reconstructed."""
        result = _reconstruct_settings(mock_settings_dict)

        assert result.USER == "testuser"
        assert result.MINIO_ACCESS_KEY == "access123"

    def test_converts_urls_to_anyurl(self, mock_settings_dict):
        """Test that URL strings are converted to AnyUrl."""
        result = _reconstruct_settings(mock_settings_dict)

        assert isinstance(result.SPARK_MASTER_URL, AnyUrl)

    def test_does_not_modify_original_dict(self, mock_settings_dict):
        """Test that original dict is not modified."""
        original_url = mock_settings_dict["SPARK_MASTER_URL"]
        _reconstruct_settings(mock_settings_dict)

        assert mock_settings_dict["SPARK_MASTER_URL"] == original_url


# =============================================================================
# Tests for count_table_subprocess
# =============================================================================


class TestCountTableSubprocess:
    """Tests for count_table_subprocess function."""

    def test_count_table_success(self, mock_settings_dict):
        """Test successful table count."""
        mock_spark = MagicMock()

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch(
                "src.service.standalone_operations._cleanup_spark_session"
            ) as mock_cleanup,
            patch(
                "src.service.standalone_operations.delta_service.count_delta_table",
                return_value=42,
            ) as mock_count,
        ):
            result = count_table_subprocess(
                mock_settings_dict, "test_db", "test_table", "testuser", "mcp_count"
            )

            assert result == 42
            mock_count.assert_called_once_with(
                spark=mock_spark,
                database="test_db",
                table="test_table",
                username="testuser",
            )
            mock_cleanup.assert_called_once_with(mock_spark)

    def test_count_table_cleanup_on_exception(self, mock_settings_dict):
        """Test that cleanup happens even on exception."""
        mock_spark = MagicMock()

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch(
                "src.service.standalone_operations._cleanup_spark_session"
            ) as mock_cleanup,
            patch(
                "src.service.standalone_operations.delta_service.count_delta_table",
                side_effect=Exception("Count error"),
            ),
        ):
            with pytest.raises(Exception, match="Count error"):
                count_table_subprocess(mock_settings_dict, "test_db", "test_table")

            mock_cleanup.assert_called_once_with(mock_spark)


# =============================================================================
# Tests for sample_table_subprocess
# =============================================================================


class TestSampleTableSubprocess:
    """Tests for sample_table_subprocess function."""

    def test_sample_table_success(self, mock_settings_dict):
        """Test successful table sampling."""
        mock_spark = MagicMock()
        sample_data = [{"id": 1, "name": "test"}]

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.delta_service.sample_delta_table",
                return_value=sample_data,
            ) as mock_sample,
        ):
            result = sample_table_subprocess(
                mock_settings_dict,
                "test_db",
                "test_table",
                limit=10,
                columns=["id", "name"],
                where_clause="id > 0",
                username="testuser",
            )

            assert result == sample_data
            mock_sample.assert_called_once_with(
                spark=mock_spark,
                database="test_db",
                table="test_table",
                limit=10,
                columns=["id", "name"],
                where_clause="id > 0",
                username="testuser",
            )


# =============================================================================
# Tests for query_table_subprocess
# =============================================================================


class TestQueryTableSubprocess:
    """Tests for query_table_subprocess function."""

    def test_query_table_success(self, mock_settings_dict):
        """Test successful table query."""
        mock_spark = MagicMock()
        mock_response = MagicMock()
        mock_response.result = [{"id": 1}]
        mock_response.pagination = MagicMock()
        mock_response.pagination.model_dump.return_value = {
            "total": 1,
            "limit": 100,
            "offset": 0,
            "has_more": False,
        }

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.delta_service.query_delta_table",
                return_value=mock_response,
            ) as mock_query,
        ):
            result = query_table_subprocess(
                mock_settings_dict,
                "SELECT * FROM test_table",
                limit=100,
                offset=0,
                username="testuser",
            )

            assert result["result"] == [{"id": 1}]
            assert result["pagination"]["total"] == 1
            mock_query.assert_called_once()


# =============================================================================
# Tests for select_table_subprocess
# =============================================================================


class TestSelectTableSubprocess:
    """Tests for select_table_subprocess function."""

    def test_select_table_success(self, mock_settings_dict):
        """Test successful structured select."""
        mock_spark = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_response.pagination = MagicMock()
        mock_response.pagination.model_dump.return_value = {
            "total": 1,
            "limit": 100,
            "offset": 0,
            "has_more": False,
        }

        request_dict = {
            "database": "test_db",
            "table": "test_table",
        }

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.delta_service.select_from_delta_table",
                return_value=mock_response,
            ) as mock_select,
        ):
            result = select_table_subprocess(
                mock_settings_dict,
                request_dict,
                username="testuser",
            )

            assert result["data"] == [{"id": 1}]
            assert result["pagination"]["total"] == 1
            mock_select.assert_called_once()


# =============================================================================
# Tests for list_databases_subprocess
# =============================================================================


class TestListDatabasesSubprocess:
    """Tests for list_databases_subprocess function."""

    def test_list_databases_success(self, mock_settings_dict):
        """Test successful database listing."""
        mock_spark = MagicMock()

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.data_store.get_databases",
                return_value=["db1", "db2"],
            ) as mock_get_dbs,
        ):
            result = list_databases_subprocess(
                mock_settings_dict,
                use_hms=True,
                filter_by_namespace=False,
                auth_token=None,
            )

            assert result == ["db1", "db2"]
            mock_get_dbs.assert_called_once()


# =============================================================================
# Tests for list_tables_subprocess
# =============================================================================


class TestListTablesSubprocess:
    """Tests for list_tables_subprocess function."""

    def test_list_tables_success(self, mock_settings_dict):
        """Test successful table listing."""
        mock_spark = MagicMock()

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.data_store.get_tables",
                return_value=["table1", "table2"],
            ) as mock_get_tables,
        ):
            result = list_tables_subprocess(
                mock_settings_dict,
                "test_db",
                use_hms=True,
            )

            assert result == ["table1", "table2"]
            mock_get_tables.assert_called_once()


# =============================================================================
# Tests for get_table_schema_subprocess
# =============================================================================


class TestGetTableSchemaSubprocess:
    """Tests for get_table_schema_subprocess function."""

    def test_get_schema_success(self, mock_settings_dict):
        """Test successful schema retrieval."""
        mock_spark = MagicMock()

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.data_store.get_table_schema",
                return_value=["id", "name", "created_at"],
            ) as mock_get_schema,
        ):
            result = get_table_schema_subprocess(
                mock_settings_dict,
                "test_db",
                "test_table",
            )

            assert result == ["id", "name", "created_at"]
            mock_get_schema.assert_called_once()


# =============================================================================
# Tests for get_db_structure_subprocess
# =============================================================================


class TestGetDbStructureSubprocess:
    """Tests for get_db_structure_subprocess function."""

    def test_get_structure_success(self, mock_settings_dict):
        """Test successful structure retrieval."""
        mock_spark = MagicMock()
        structure = {"db1": ["table1", "table2"], "db2": ["table3"]}

        with (
            patch(
                "src.service.standalone_operations._create_spark_session",
                return_value=mock_spark,
            ),
            patch("src.service.standalone_operations._cleanup_spark_session"),
            patch(
                "src.service.standalone_operations.data_store.get_db_structure",
                return_value=structure,
            ) as mock_get_struct,
        ):
            result = get_db_structure_subprocess(
                mock_settings_dict,
                with_schema=False,
                use_hms=True,
            )

            assert result == structure
            mock_get_struct.assert_called_once()
