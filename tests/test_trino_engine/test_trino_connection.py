"""
Tests for the Trino connection management module.

Tests cover:
- _sanitize_identifier: username → Trino identifier
- _build_catalog_properties: property dict construction
- _catalog_exists: catalog presence check
- _escape_sql_string: SQL escaping
- _validate_connector: connector allowlist
- _create_dynamic_catalog: dynamic catalog creation
- create_trino_connection: end-to-end connection setup
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.exceptions import TrinoConnectionError
from src.trino_engine.trino_connection import (
    ALLOWED_CONNECTORS,
    _build_iceberg_catalog_properties,
    _build_catalog_properties,
    _catalog_exists,
    _create_dynamic_catalog,
    _escape_sql_string,
    _get_personal_catalog_alias,
    _get_tenant_catalog_alias,
    _iter_tenant_catalogs,
    _sanitize_identifier,
    _validate_connector,
    create_trino_connection,
)


# =============================================================================
# _sanitize_identifier Tests
# =============================================================================


class TestSanitizeIdentifier:
    """Tests for the _sanitize_identifier function."""

    def test_lowercase_conversion(self):
        assert _sanitize_identifier("TestUser") == "testuser"

    def test_special_characters_replaced(self):
        assert _sanitize_identifier("user.name@test") == "user_name_test"

    def test_hyphens_replaced(self):
        assert _sanitize_identifier("user-name") == "user_name"

    def test_underscores_preserved(self):
        assert _sanitize_identifier("user_name") == "user_name"

    def test_numbers_preserved(self):
        assert _sanitize_identifier("user123") == "user123"

    def test_empty_string(self):
        assert _sanitize_identifier("") == ""


# =============================================================================
# _build_catalog_properties Tests
# =============================================================================


class TestBuildCatalogProperties:
    """Tests for the _build_catalog_properties function."""

    def test_basic_properties(self):
        props = _build_catalog_properties(
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="http://minio:9000",
            minio_secure=False,
            access_key="ak",
            secret_key="sk",
        )
        assert props["hive.metastore.uri"] == "thrift://hms:9083"
        assert props["s3.endpoint"] == "http://minio:9000"
        assert props["s3.aws-access-key"] == "ak"
        assert props["s3.aws-secret-key"] == "sk"
        assert props["s3.path-style-access"] == "true"
        assert props["s3.region"] == "us-east-1"
        assert props["fs.native-s3.enabled"] == "true"

    def test_http_prefix_added_for_bare_endpoint(self):
        props = _build_catalog_properties(
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=False,
            access_key="ak",
            secret_key="sk",
        )
        assert props["s3.endpoint"] == "http://minio:9000"

    def test_https_prefix_when_secure(self):
        props = _build_catalog_properties(
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=True,
            access_key="ak",
            secret_key="sk",
        )
        assert props["s3.endpoint"] == "https://minio:9000"

    def test_existing_http_prefix_preserved(self):
        props = _build_catalog_properties(
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="http://minio:9000",
            minio_secure=True,
            access_key="ak",
            secret_key="sk",
        )
        assert props["s3.endpoint"] == "http://minio:9000"


# =============================================================================
# Polaris Catalog Helper Tests
# =============================================================================


class TestPolarisCatalogHelpers:
    """Tests for Polaris Iceberg catalog helpers."""

    def test_personal_catalog_alias(self):
        assert _get_personal_catalog_alias("user_testuser") == "testuser"
        assert _get_personal_catalog_alias("Test.User") == "test_user"
        assert _get_personal_catalog_alias(None) is None

    def test_tenant_catalog_alias(self):
        assert _get_tenant_catalog_alias("tenant_globalusers") == "globalusers"
        assert _get_tenant_catalog_alias("tenant_research-team") == "research_team"

    def test_iter_tenant_catalogs(self):
        assert _iter_tenant_catalogs("tenant_a, tenant_b,,") == [
            "tenant_a",
            "tenant_b",
        ]
        assert _iter_tenant_catalogs(None) == []

    def test_build_iceberg_catalog_properties(self):
        props = _build_iceberg_catalog_properties(
            polaris_catalog_uri="http://polaris:8181/api/catalog",
            polaris_credential="cid:secret",
            warehouse="user_testuser",
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=False,
            access_key="ak",
            secret_key="sk",
        )

        assert props["iceberg.catalog.type"] == "rest"
        assert props["iceberg.rest-catalog.uri"] == "http://polaris:8181/api/catalog"
        assert props["iceberg.rest-catalog.warehouse"] == "user_testuser"
        assert props["iceberg.rest-catalog.oauth2.credential"] == "cid:secret"
        assert (
            props["iceberg.rest-catalog.oauth2.server-uri"]
            == "http://polaris:8181/api/catalog/v1/oauth/tokens"
        )
        assert props["s3.endpoint"] == "http://minio:9000"
        assert "hive.metastore.uri" not in props


# =============================================================================
# _catalog_exists Tests
# =============================================================================


class TestCatalogExists:
    """Tests for the _catalog_exists function."""

    def test_returns_true_when_found(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [("system",), ("u_testuser",)]
        assert _catalog_exists(cursor, "u_testuser") is True

    def test_returns_false_when_not_found(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [("system",)]
        assert _catalog_exists(cursor, "u_testuser") is False

    def test_empty_catalogs(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        assert _catalog_exists(cursor, "u_testuser") is False


# =============================================================================
# _escape_sql_string Tests
# =============================================================================


class TestEscapeSqlString:
    """Tests for the _escape_sql_string function."""

    def test_no_quotes(self):
        assert _escape_sql_string("hello") == "hello"

    def test_single_quote_doubled(self):
        assert _escape_sql_string("it's") == "it''s"

    def test_multiple_quotes(self):
        assert _escape_sql_string("a'b'c") == "a''b''c"


# =============================================================================
# _validate_connector Tests
# =============================================================================


class TestValidateConnector:
    """Tests for the _validate_connector function."""

    def test_valid_connectors(self):
        for connector in ALLOWED_CONNECTORS:
            _validate_connector(connector)  # should not raise

    def test_invalid_connector_raises(self):
        with pytest.raises(ValueError, match="not allowed"):
            _validate_connector("postgres")


# =============================================================================
# _create_dynamic_catalog Tests
# =============================================================================


class TestCreateDynamicCatalog:
    """Tests for the _create_dynamic_catalog function."""

    def test_skips_existing_catalog(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [("u_testuser",)]

        _create_dynamic_catalog(cursor, "u_testuser", "delta_lake", {"k": "v"})

        # Only SHOW CATALOGS was executed (no CREATE CATALOG)
        cursor.execute.assert_called_once_with("SHOW CATALOGS")

    def test_creates_catalog_when_missing(self):
        cursor = MagicMock()
        # First fetchall for SHOW CATALOGS (no match), second for CREATE CATALOG
        cursor.fetchall.side_effect = [[("system",)], []]

        _create_dynamic_catalog(
            cursor, "u_testuser", "delta_lake", {"hive.metastore.uri": "thrift://x"}
        )

        assert cursor.execute.call_count == 2
        create_call = cursor.execute.call_args_list[1]
        assert "CREATE CATALOG IF NOT EXISTS" in create_call.args[0]
        assert '"u_testuser"' in create_call.args[0]

    def test_rejects_invalid_connector(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = []  # catalog does not exist

        with pytest.raises(ValueError, match="not allowed"):
            _create_dynamic_catalog(cursor, "u_testuser", "mysql", {"k": "v"})


# =============================================================================
# create_trino_connection Tests
# =============================================================================


class TestCreateTrinoConnection:
    """Tests for the create_trino_connection function."""

    @patch("src.trino_engine.trino_connection.trino.dbapi.connect")
    def test_creates_connection_and_catalog(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("system",)],  # SHOW CATALOGS — catalog doesn't exist
            [],  # CREATE CATALOG result
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn._client_session = MagicMock()
        mock_connect.return_value = mock_conn

        conn = create_trino_connection(
            username="test_user",
            auth_token="tok",
            access_key="ak",
            secret_key="sk",
            trino_host="trino",
            trino_port=8080,
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=False,
        )

        assert conn is mock_conn
        mock_connect.assert_called_once_with(
            host="trino",
            port=8080,
            user="test_user",
            extra_credential=[("kbase_auth_token", "tok")],
        )
        assert mock_conn._client_session.catalog == "u_test_user"

    @patch("src.trino_engine.trino_connection.trino.dbapi.connect")
    def test_sanitizes_username(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("u_test_user",)]  # exists
        mock_conn.cursor.return_value = mock_cursor
        mock_conn._client_session = MagicMock()
        mock_connect.return_value = mock_conn

        create_trino_connection(
            username="Test.User",
            auth_token="tok",
            access_key="ak",
            secret_key="sk",
            trino_host="trino",
            trino_port=8080,
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=False,
        )

        assert mock_conn._client_session.catalog == "u_test_user"

    @patch("src.trino_engine.trino_connection.trino.dbapi.connect")
    def test_wraps_exception_in_trino_connection_error(self, mock_connect):
        mock_connect.side_effect = Exception("Connection refused")

        with pytest.raises(TrinoConnectionError, match="Connection refused"):
            create_trino_connection(
                username="test",
                auth_token="tok",
                access_key="ak",
                secret_key="sk",
                trino_host="trino",
                trino_port=8080,
                hive_metastore_uri="thrift://hms:9083",
                minio_endpoint_url="minio:9000",
                minio_secure=False,
            )

    @patch("src.trino_engine.trino_connection.trino.dbapi.connect")
    def test_creates_polaris_catalogs_when_configured(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("system",)],  # u_test_user missing
            [],  # CREATE u_test_user
            [("system",), ("u_test_user",)],  # test_user missing
            [],  # CREATE test_user
            [("system",), ("u_test_user",), ("test_user",)],  # research missing
            [],  # CREATE research
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn._client_session = MagicMock()
        mock_connect.return_value = mock_conn

        create_trino_connection(
            username="test_user",
            auth_token="tok",
            access_key="ak",
            secret_key="sk",
            trino_host="trino",
            trino_port=8080,
            hive_metastore_uri="thrift://hms:9083",
            minio_endpoint_url="minio:9000",
            minio_secure=False,
            polaris_catalog_uri="http://polaris:8181/api/catalog",
            polaris_credential="cid:secret",
            polaris_personal_catalog="user_test_user",
            polaris_tenant_catalogs="tenant_research",
        )

        create_sql = "\n".join(
            call.args[0]
            for call in mock_cursor.execute.call_args_list
            if call.args[0].startswith("CREATE CATALOG")
        )
        assert '"u_test_user" USING delta_lake' in create_sql
        assert '"test_user" USING iceberg' in create_sql
        assert '"research" USING iceberg' in create_sql
        assert "iceberg.rest-catalog.oauth2.credential" in create_sql
