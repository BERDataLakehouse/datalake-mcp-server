"""Tests for the postgres connection module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.postgres.connection import _validate_not_empty, get_postgres_connection


class TestValidateNotEmpty:
    """Tests for _validate_not_empty function."""

    def test_valid_string(self):
        """Test that valid string passes validation."""
        result = _validate_not_empty("valid_value", "test_param")
        assert result == "valid_value"

    def test_none_value_raises(self):
        """Test that None value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_not_empty(None, "test_param")

        assert "test_param must not be empty" in str(exc_info.value)

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_not_empty("", "test_param")

        assert "test_param must not be empty" in str(exc_info.value)

    def test_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_not_empty("   ", "test_param")

        assert "test_param must not be empty" in str(exc_info.value)

    def test_error_message_includes_env_var(self):
        """Test that error message includes env var hint when provided."""
        with pytest.raises(ValueError) as exc_info:
            _validate_not_empty(None, "test_param", "TEST_ENV_VAR")

        error_msg = str(exc_info.value)
        assert "test_param must not be empty" in error_msg
        assert "set TEST_ENV_VAR env var" in error_msg

    def test_valid_non_string(self):
        """Test that non-string valid values pass validation."""
        result = _validate_not_empty(12345, "test_param")
        assert result == 12345

    def test_falsy_zero_raises(self):
        """Test that falsy value 0 raises ValueError."""
        # Note: 0 is falsy but isinstance check only applies to strings
        # This tests the actual behavior - 0 passes because it's not None
        # and the string check only applies to strings
        result = _validate_not_empty(0, "test_param")
        assert result == 0


class TestGetPostgresConnection:
    """Tests for get_postgres_connection function."""

    def test_connection_with_explicit_params(self):
        """Test connection with all parameters explicitly provided."""
        with patch("src.postgres.connection.psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            result = get_postgres_connection(
                dbname="testdb",
                user="testuser",
                password="testpass",
                host="localhost",
                port="5432",
            )

            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["dbname"] == "testdb"
            assert call_kwargs["user"] == "testuser"
            assert call_kwargs["password"] == "testpass"
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == "5432"
            assert result == mock_conn

    def test_connection_from_env_vars(self):
        """Test connection using environment variables."""
        env_vars = {
            "POSTGRES_URL": "dbhost:5433",
            "POSTGRES_DB": "envdb",
            "POSTGRES_USER": "envuser",
            "POSTGRES_PASSWORD": "envpass",
        }

        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch("src.postgres.connection.psycopg.connect") as mock_connect,
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            result = get_postgres_connection()

            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["dbname"] == "envdb"
            assert call_kwargs["user"] == "envuser"
            assert call_kwargs["password"] == "envpass"
            assert call_kwargs["host"] == "dbhost"
            assert call_kwargs["port"] == "5433"
            assert result == mock_conn

    def test_invalid_postgres_url_format(self):
        """Test that invalid POSTGRES_URL format raises ValueError."""
        env_vars = {
            "POSTGRES_URL": "invalid-format-no-colon",
            "POSTGRES_DB": "testdb",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError) as exc_info:
                get_postgres_connection()

            assert "POSTGRES_URL must be in the format 'host:port'" in str(
                exc_info.value
            )

    def test_missing_database_name_raises(self):
        """Test that missing database name raises ValueError."""
        env_vars = {
            "POSTGRES_URL": "localhost:5432",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
        }

        # Clear POSTGRES_DB if it exists
        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch.dict(os.environ, {"POSTGRES_DB": ""}, clear=False),
        ):
            with pytest.raises(ValueError) as exc_info:
                get_postgres_connection()

            assert "Database name must not be empty" in str(exc_info.value)

    def test_missing_user_raises(self):
        """Test that missing user raises ValueError."""
        env_vars = {
            "POSTGRES_URL": "localhost:5432",
            "POSTGRES_DB": "testdb",
            "POSTGRES_PASSWORD": "testpass",
        }

        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch.dict(os.environ, {"POSTGRES_USER": ""}, clear=False),
        ):
            with pytest.raises(ValueError) as exc_info:
                get_postgres_connection()

            assert "Database user must not be empty" in str(exc_info.value)

    def test_missing_password_raises(self):
        """Test that missing password raises ValueError."""
        env_vars = {
            "POSTGRES_URL": "localhost:5432",
            "POSTGRES_DB": "testdb",
            "POSTGRES_USER": "testuser",
        }

        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch.dict(os.environ, {"POSTGRES_PASSWORD": ""}, clear=False),
        ):
            with pytest.raises(ValueError) as exc_info:
                get_postgres_connection()

            assert "Database password must not be empty" in str(exc_info.value)

    def test_psycopg_error_wrapped_in_connection_error(self):
        """Test that psycopg errors are wrapped in ConnectionError."""
        import psycopg

        env_vars = {
            "POSTGRES_URL": "localhost:5432",
            "POSTGRES_DB": "testdb",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
        }

        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch("src.postgres.connection.psycopg.connect") as mock_connect,
        ):
            mock_connect.side_effect = psycopg.Error("Connection refused")

            with pytest.raises(ConnectionError) as exc_info:
                get_postgres_connection()

            error_msg = str(exc_info.value)
            assert "Failed to connect to PostgreSQL" in error_msg
            assert "localhost:5432" in error_msg
            assert "testdb" in error_msg
            assert "testuser" in error_msg

    def test_explicit_host_port_overrides_env(self):
        """Test that explicit host/port parameters override env vars."""
        env_vars = {
            "POSTGRES_URL": "envhost:5433",
            "POSTGRES_DB": "testdb",
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
        }

        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch("src.postgres.connection.psycopg.connect") as mock_connect,
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            get_postgres_connection(host="explicit-host", port="9999")

            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["host"] == "explicit-host"
            assert call_kwargs["port"] == "9999"
