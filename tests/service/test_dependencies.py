"""
Tests for the FastAPI dependencies module.

Tests cover:
- read_user_minio_credentials() - file reading, error handling
- get_user_from_request() - user extraction from request state
- get_spark_session() - session creation, fallback logic, cleanup
- is_spark_connect_reachable() - TCP check mocking
- construct_user_spark_connect_url() - URL construction
- Concurrent session creation
"""

import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.service.dependencies import (
    read_user_minio_credentials,
    get_user_from_request,
    construct_user_spark_connect_url,
    is_spark_connect_reachable,
    get_spark_session,
    DEFAULT_SPARK_POOL,
    SPARK_CONNECT_PORT,
)


# =============================================================================
# Test read_user_minio_credentials
# =============================================================================


class TestReadUserMinioCredentials:
    """Tests for the read_user_minio_credentials function."""

    def test_successful_credential_read(self, tmp_path):
        """Test successfully reading credentials from file."""
        creds_data = {
            "username": "testuser",
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
        }

        # Create a temporary credentials file
        creds_file = tmp_path / ".berdl_minio_credentials"
        creds_file.write_text(json.dumps(creds_data))

        with patch.object(Path, "__new__", return_value=creds_file):
            # We need to mock the path construction
            with patch("src.service.dependencies.Path") as mock_path_class:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.__str__.return_value = str(creds_file)

                # Mock the open call
                with patch(
                    "builtins.open",
                    mock_open(read_data=json.dumps(creds_data)),
                ):
                    mock_path_class.return_value = mock_path
                    access_key, secret_key = read_user_minio_credentials("testuser")

        assert access_key == "test_access_key"
        assert secret_key == "test_secret_key"

    def test_file_not_found_raises_error(self):
        """Test that missing credentials file raises FileNotFoundError."""
        with patch("src.service.dependencies.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path

            with pytest.raises(FileNotFoundError, match="credentials file not found"):
                read_user_minio_credentials("nonexistent_user")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with patch("src.service.dependencies.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True

            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data="invalid json{")):
                with pytest.raises(ValueError, match="Failed to parse"):
                    read_user_minio_credentials("testuser")

    def test_missing_access_key_raises_error(self):
        """Test that missing access_key raises ValueError."""
        creds_data = {"username": "testuser", "secret_key": "secret"}

        with patch("src.service.dependencies.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=json.dumps(creds_data))):
                with pytest.raises(ValueError, match="Invalid credentials format"):
                    read_user_minio_credentials("testuser")

    def test_missing_secret_key_raises_error(self):
        """Test that missing secret_key raises ValueError."""
        creds_data = {"username": "testuser", "access_key": "access"}

        with patch("src.service.dependencies.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path

            with patch("builtins.open", mock_open(read_data=json.dumps(creds_data))):
                with pytest.raises(ValueError, match="Invalid credentials format"):
                    read_user_minio_credentials("testuser")

    def test_correct_path_constructed(self):
        """Test that the correct path is constructed."""
        with patch("src.service.dependencies.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path

            with pytest.raises(FileNotFoundError):
                read_user_minio_credentials("myuser")

            # Verify path was constructed correctly
            mock_path_class.assert_called_with("/home/myuser/.berdl_minio_credentials")


# =============================================================================
# Test get_user_from_request
# =============================================================================


class TestGetUserFromRequest:
    """Tests for the get_user_from_request function."""

    def test_returns_authenticated_user(self, mock_request, mock_kbase_user):
        """Test that authenticated user is returned."""
        request = mock_request(user="testuser")

        result = get_user_from_request(request)

        assert result == "testuser"

    def test_unauthenticated_user_raises_error(self, mock_request):
        """Test that unauthenticated request raises error."""
        request = mock_request(user=None)

        with pytest.raises(Exception, match="not authenticated"):
            get_user_from_request(request)

    def test_missing_request_state_raises_error(self):
        """Test that missing request state raises error."""
        request = MagicMock()
        # Simulate missing _request_state
        request.state._request_state = None

        with patch(
            "src.service.dependencies.app_state.get_request_user", return_value=None
        ):
            with pytest.raises(Exception, match="not authenticated"):
                get_user_from_request(request)


# =============================================================================
# Test construct_user_spark_connect_url
# =============================================================================


class TestConstructUserSparkConnectUrl:
    """Tests for the construct_user_spark_connect_url function."""

    def test_custom_template_without_username(self):
        """Test custom template without username placeholder."""
        with patch.dict(
            "os.environ",
            {"SPARK_CONNECT_URL_TEMPLATE": "sc://spark-notebook:15002"},
            clear=False,
        ):
            url = construct_user_spark_connect_url("anyuser")

        assert url == "sc://spark-notebook:15002"

    def test_custom_template_with_username(self):
        """Test custom template with username placeholder."""
        with patch.dict(
            "os.environ",
            {"SPARK_CONNECT_URL_TEMPLATE": "sc://spark-notebook-{username}:15002"},
            clear=False,
        ):
            url = construct_user_spark_connect_url("testuser")

        assert url == "sc://spark-notebook-testuser:15002"

    def test_kubernetes_url_default_dev(self):
        """Test Kubernetes URL construction with default dev environment."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove template if exists
            with patch.dict(
                "os.environ",
                {"SPARK_CONNECT_URL_TEMPLATE": "", "K8S_ENVIRONMENT": "dev"},
            ):
                # We need to handle the case where template is empty
                pass

        # Test with no template
        with patch.object(__import__("os"), "getenv", side_effect=lambda k, d=None: d):
            with patch.dict("os.environ", {"K8S_ENVIRONMENT": "dev"}, clear=False):
                with patch(
                    "os.getenv",
                    side_effect=lambda k, d=None: None
                    if k == "SPARK_CONNECT_URL_TEMPLATE"
                    else ("dev" if k == "K8S_ENVIRONMENT" else d),
                ):
                    url = construct_user_spark_connect_url("myuser")

        assert "jupyter-myuser" in url
        assert "15002" in url

    def test_kubernetes_url_prod_environment(self):
        """Test Kubernetes URL construction with prod environment."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("prod" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("produser")

        assert "jupyterhub-prod" in url
        assert "jupyter-produser" in url

    def test_spark_connect_port_constant(self):
        """Test that SPARK_CONNECT_PORT is correct."""
        assert SPARK_CONNECT_PORT == "15002"


# =============================================================================
# Test is_spark_connect_reachable
# =============================================================================


class TestIsSparkConnectReachable:
    """Tests for the is_spark_connect_reachable function."""

    def test_reachable_returns_true(self):
        """Test that reachable host returns True."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0  # Success
            mock_socket_class.return_value = mock_socket

            result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is True
        mock_socket.close.assert_called_once()

    def test_unreachable_returns_false(self):
        """Test that unreachable host returns False."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 111  # Connection refused
            mock_socket_class.return_value = mock_socket

            result = is_spark_connect_reachable("sc://unreachable:15002")

        assert result is False

    def test_socket_error_returns_false(self):
        """Test that socket error returns False."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket_class.side_effect = socket.error("Network error")

            result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is False

    def test_invalid_url_returns_false(self):
        """Test that invalid URL returns False."""
        result = is_spark_connect_reachable("invalid-url")
        assert result is False

    def test_custom_timeout(self):
        """Test that custom timeout is used."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            is_spark_connect_reachable("sc://localhost:15002", timeout=5.0)

            mock_socket.settimeout.assert_called_with(5.0)

    def test_default_port_used_when_missing(self):
        """Test that default port 15002 is used when not specified."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            is_spark_connect_reachable("sc://localhost")

            # The connect_ex should be called with default port
            call_args = mock_socket.connect_ex.call_args[0][0]
            assert call_args[1] == 15002


# =============================================================================
# Test get_spark_session Generator
# =============================================================================


class TestGetSparkSession:
    """Tests for the get_spark_session dependency generator."""

    def test_spark_connect_success(self, mock_request, mock_settings):
        """Test successful Spark Connect session creation."""
        mock_request_obj = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.construct_user_spark_connect_url",
                    return_value="sc://jupyter-testuser:15002",
                ):
                    with patch(
                        "src.service.dependencies.is_spark_connect_reachable",
                        return_value=True,
                    ):
                        with patch(
                            "src.service.dependencies._get_spark_session"
                        ) as mock_get_spark:
                            mock_spark = MagicMock()
                            mock_get_spark.return_value = mock_spark

                            gen = get_spark_session(mock_request_obj, mock_settings)
                            spark = next(gen)

                            assert spark is mock_spark

                            # Cleanup
                            try:
                                next(gen)
                            except StopIteration:
                                pass

                            mock_spark.stop.assert_called_once()

    def test_spark_connect_fallback_to_cluster(self, mock_request, mock_settings):
        """Test fallback to shared cluster when Spark Connect unavailable."""
        mock_request_obj = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.construct_user_spark_connect_url",
                    return_value="sc://jupyter-testuser:15002",
                ):
                    with patch(
                        "src.service.dependencies.is_spark_connect_reachable",
                        return_value=False,  # Not reachable - trigger fallback
                    ):
                        with patch(
                            "src.service.dependencies._get_spark_session"
                        ) as mock_get_spark:
                            mock_spark = MagicMock()
                            mock_get_spark.return_value = mock_spark

                            gen = get_spark_session(mock_request_obj, mock_settings)
                            _spark = next(gen)  # noqa: F841

                            # Verify fallback was used (use_spark_connect=False)
                            call_kwargs = mock_get_spark.call_args[1]
                            assert call_kwargs["use_spark_connect"] is False

                            # Cleanup
                            try:
                                next(gen)
                            except StopIteration:
                                pass

    def test_missing_credentials_raises_error(self, mock_request, mock_settings):
        """Test that missing credentials raises appropriate error."""
        mock_request_obj = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                side_effect=FileNotFoundError("Credentials not found"),
            ):
                gen = get_spark_session(mock_request_obj, mock_settings)

                with pytest.raises(Exception, match="MinIO credentials file not found"):
                    next(gen)

    def test_unauthenticated_user_raises_error(self, mock_request, mock_settings):
        """Test that unauthenticated user raises error."""
        mock_request_obj = mock_request(user=None)

        with patch(
            "src.service.dependencies.get_user_from_request",
            side_effect=Exception("User not authenticated"),
        ):
            gen = get_spark_session(mock_request_obj, mock_settings)

            with pytest.raises(Exception, match="not authenticated"):
                next(gen)

    def test_session_stopped_even_on_error(self, mock_request, mock_settings):
        """Test that session is stopped even if request handler raises error."""
        mock_request_obj = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.construct_user_spark_connect_url",
                    return_value="sc://jupyter-testuser:15002",
                ):
                    with patch(
                        "src.service.dependencies.is_spark_connect_reachable",
                        return_value=True,
                    ):
                        with patch(
                            "src.service.dependencies._get_spark_session"
                        ) as mock_get_spark:
                            mock_spark = MagicMock()
                            mock_get_spark.return_value = mock_spark

                            gen = get_spark_session(mock_request_obj, mock_settings)
                            _spark = next(gen)  # noqa: F841

                            # Simulate error during cleanup
                            try:
                                gen.throw(ValueError("Simulated error"))
                            except ValueError:
                                pass

                            # Session should still be stopped
                            mock_spark.stop.assert_called_once()

    def test_session_stop_error_handled_gracefully(self, mock_request, mock_settings):
        """Test that errors during session.stop() are handled."""
        mock_request_obj = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.is_spark_connect_reachable",
                    return_value=True,
                ):
                    with patch(
                        "src.service.dependencies._get_spark_session"
                    ) as mock_get_spark:
                        mock_spark = MagicMock()
                        mock_spark.stop.side_effect = Exception("Stop failed")
                        mock_get_spark.return_value = mock_spark

                        gen = get_spark_session(mock_request_obj, mock_settings)
                        next(gen)

                        # Should not raise despite stop() error
                        try:
                            next(gen)
                        except StopIteration:
                            pass


# =============================================================================
# Concurrent Session Tests
# =============================================================================


class TestConcurrentSparkSessions:
    """Tests for concurrent Spark session creation."""

    def test_concurrent_session_creation(
        self, mock_request, mock_settings, concurrent_executor
    ):
        """Test that multiple concurrent requests get isolated sessions."""
        sessions_created = {"count": 0}

        def create_session(user_id):
            mock_req = MagicMock()
            mock_req.state._request_state = MagicMock()

            with patch(
                "src.service.dependencies.get_user_from_request",
                return_value=f"user_{user_id}",
            ):
                with patch(
                    "src.service.dependencies.read_user_minio_credentials",
                    return_value=("access", "secret"),
                ):
                    with patch(
                        "src.service.dependencies.is_spark_connect_reachable",
                        return_value=True,
                    ):
                        with patch(
                            "src.service.dependencies._get_spark_session"
                        ) as mock_get_spark:
                            mock_spark = MagicMock()
                            mock_spark.user_id = user_id
                            mock_get_spark.return_value = mock_spark
                            sessions_created["count"] += 1

                            gen = get_spark_session(mock_req, mock_settings)
                            spark = next(gen)
                            spark_id = spark.user_id

                            try:
                                next(gen)
                            except StopIteration:
                                pass

                            return spark_id

        args_list = [(i,) for i in range(5)]
        results, exceptions = concurrent_executor(
            create_session, args_list, max_workers=5
        )

        assert len(exceptions) == 0
        # Each user should get their own session
        assert sorted(results) == [0, 1, 2, 3, 4]


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_spark_pool_value(self):
        """Test DEFAULT_SPARK_POOL constant."""
        assert DEFAULT_SPARK_POOL == "default"

    def test_spark_connect_port_value(self):
        """Test SPARK_CONNECT_PORT constant."""
        assert SPARK_CONNECT_PORT == "15002"
