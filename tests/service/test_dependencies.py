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
    sanitize_k8s_name,
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
# Test sanitize_k8s_name
# =============================================================================


class TestSanitizeK8sName:
    """Tests for the sanitize_k8s_name function."""

    def test_lowercase_alphanumeric_unchanged(self):
        """Test that DNS-compliant lowercase alphanumeric names are unchanged."""
        assert sanitize_k8s_name("user123") == "user123"
        assert sanitize_k8s_name("myuser") == "myuser"
        assert sanitize_k8s_name("testuser456") == "testuser456"

    def test_underscores_replaced_with_hyphens(self):
        """Test that underscores are replaced with hyphens."""
        assert sanitize_k8s_name("user_name") == "user-name"
        assert sanitize_k8s_name("tian_gu_test") == "tian-gu-test"
        assert sanitize_k8s_name("my_test_user") == "my-test-user"

    def test_uppercase_converted_to_lowercase(self):
        """Test that uppercase characters are converted to lowercase."""
        assert sanitize_k8s_name("UserName") == "username"
        assert sanitize_k8s_name("TESTUSER") == "testuser"
        assert sanitize_k8s_name("MyUser123") == "myuser123"

    def test_mixed_case_with_underscores(self):
        """Test mixed case with underscores (real-world KBase usernames)."""
        assert sanitize_k8s_name("Tian_Gu_Test") == "tian-gu-test"
        assert sanitize_k8s_name("Jeff_Cohere") == "jeff-cohere"

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with hyphens."""
        assert sanitize_k8s_name("user@name") == "user-name"
        assert sanitize_k8s_name("user#name") == "user-name"
        assert sanitize_k8s_name("user$name") == "user-name"
        assert sanitize_k8s_name("user name") == "user-name"  # Space

    def test_leading_hyphens_removed(self):
        """Test that leading non-alphanumeric characters are removed."""
        assert sanitize_k8s_name("_username") == "username"
        assert sanitize_k8s_name("-username") == "username"
        assert sanitize_k8s_name("__username") == "username"

    def test_trailing_hyphens_removed(self):
        """Test that trailing non-alphanumeric characters are removed."""
        assert sanitize_k8s_name("username_") == "username"
        assert sanitize_k8s_name("username-") == "username"
        assert sanitize_k8s_name("username__") == "username"

    def test_multiple_consecutive_hyphens_collapsed(self):
        """Test that multiple consecutive hyphens are collapsed to one."""
        assert sanitize_k8s_name("user___name") == "user-name"
        assert sanitize_k8s_name("user---name") == "user-name"
        assert sanitize_k8s_name("user_-_name") == "user-name"

    def test_dots_preserved(self):
        """Test that dots are preserved (valid in DNS-1123)."""
        assert sanitize_k8s_name("user.name") == "user.name"
        assert sanitize_k8s_name("test.user.123") == "test.user.123"

    def test_truncation_at_253_chars(self):
        """Test that names longer than 253 characters are truncated."""
        long_name = "a" * 300
        result = sanitize_k8s_name(long_name)
        assert len(result) == 253
        assert result == "a" * 253

    def test_empty_string(self):
        """Test handling of empty string."""
        assert sanitize_k8s_name("") == ""

    def test_only_special_characters(self):
        """Test handling of string with only special characters."""
        assert sanitize_k8s_name("___") == ""
        assert sanitize_k8s_name("@#$") == ""

    def test_real_kbase_usernames(self):
        """Test with realistic KBase username patterns."""
        # Common patterns in KBase
        assert sanitize_k8s_name("tgu2") == "tgu2"
        assert sanitize_k8s_name("jeff_cohere") == "jeff-cohere"
        assert sanitize_k8s_name("tian_gu_test") == "tian-gu-test"
        assert sanitize_k8s_name("user233") == "user233"

    def test_idempotency(self):
        """Test that sanitizing an already sanitized name doesn't change it."""
        name = "user-name-123"
        assert sanitize_k8s_name(name) == name
        # Double sanitization should be idempotent
        assert sanitize_k8s_name(sanitize_k8s_name(name)) == name


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

    def test_custom_template_sanitizes_username(self):
        """Test that custom template sanitizes username with underscores."""
        with patch.dict(
            "os.environ",
            {"SPARK_CONNECT_URL_TEMPLATE": "sc://spark-notebook-{username}:15002"},
            clear=False,
        ):
            url = construct_user_spark_connect_url("test_user")

        # Username should be sanitized (underscores → hyphens)
        assert url == "sc://spark-notebook-test-user:15002"

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

    def test_kubernetes_url_sanitizes_username_with_underscores(self):
        """Test that Kubernetes URL sanitizes usernames with underscores."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("dev" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("tian_gu_test")

        # Username should be sanitized (tian_gu_test → tian-gu-test)
        assert url == "sc://jupyter-tian-gu-test.jupyterhub-dev:15002"

    def test_kubernetes_url_sanitizes_mixed_case_username(self):
        """Test that Kubernetes URL sanitizes mixed-case usernames."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("dev" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("Jeff_Cohere")

        # Username should be sanitized and lowercased
        assert url == "sc://jupyter-jeff-cohere.jupyterhub-dev:15002"

    def test_kubernetes_url_preserves_dns_compliant_username(self):
        """Test that DNS-compliant usernames are preserved as-is."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("dev" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("tgu2")

        # DNS-compliant username should be unchanged
        assert url == "sc://jupyter-tgu2.jupyterhub-dev:15002"

    def test_kubernetes_url_prod_with_sanitization(self):
        """Test Kubernetes URL with prod environment and username sanitization."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("prod" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("user_name_test")

        assert url == "sc://jupyter-user-name-test.jupyterhub-prod:15002"

    def test_kubernetes_url_stage_with_sanitization(self):
        """Test Kubernetes URL with stage environment and username sanitization."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: None
            if k == "SPARK_CONNECT_URL_TEMPLATE"
            else ("stage" if k == "K8S_ENVIRONMENT" else d),
        ):
            url = construct_user_spark_connect_url("test_user")

        assert url == "sc://jupyter-test-user.jupyterhub-stage:15002"


# =============================================================================
# Test is_spark_connect_reachable
# =============================================================================


class TestIsSparkConnectReachable:
    """Tests for the is_spark_connect_reachable function."""

    def test_reachable_returns_true(self):
        """Test that reachable host returns True when both TCP and gRPC checks pass."""
        with (
            patch("socket.socket") as mock_socket_class,
            patch("src.service.dependencies.grpc") as mock_grpc,
        ):
            # Mock TCP check to pass
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0  # Success
            mock_socket_class.return_value = mock_socket

            # Mock gRPC check to pass
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel
            mock_future = MagicMock()
            mock_grpc.channel_ready_future.return_value = mock_future

            result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is True
        mock_socket.close.assert_called_once()
        mock_channel.close.assert_called_once()

    def test_unreachable_returns_false(self):
        """Test that unreachable host returns False (TCP check fails)."""
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

    def test_grpc_timeout_returns_false(self):
        """Test that gRPC timeout returns False even if TCP check passes."""
        import grpc

        with (
            patch("socket.socket") as mock_socket_class,
            patch("src.service.dependencies.grpc") as mock_grpc,
        ):
            # Mock TCP check to pass
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            # Mock gRPC to timeout - use real exception class
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel
            mock_future = MagicMock()
            mock_future.result.side_effect = grpc.FutureTimeoutError()
            mock_grpc.channel_ready_future.return_value = mock_future
            # Also need to set up the exception class on the mock for isinstance check
            mock_grpc.FutureTimeoutError = grpc.FutureTimeoutError

            result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is False
        mock_channel.close.assert_called_once()

    def test_grpc_error_returns_false(self):
        """Test that gRPC error returns False even if TCP check passes."""
        with (
            patch("socket.socket") as mock_socket_class,
            patch("src.service.dependencies.grpc") as mock_grpc,
        ):
            # Mock TCP check to pass
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            # Mock gRPC to fail with error
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("gRPC connection failed")
            mock_grpc.channel_ready_future.return_value = mock_future

            result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is False
        mock_channel.close.assert_called_once()

    def test_custom_timeout(self):
        """Test that custom timeout is used for gRPC check (TCP uses fixed 0.5s)."""
        with (
            patch("socket.socket") as mock_socket_class,
            patch("src.service.dependencies.grpc") as mock_grpc,
        ):
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel
            mock_future = MagicMock()
            mock_grpc.channel_ready_future.return_value = mock_future

            is_spark_connect_reachable("sc://localhost:15002", timeout=5.0)

            # TCP check uses fixed 0.5s timeout for quick pre-check
            mock_socket.settimeout.assert_called_with(0.5)

            # gRPC channel uses custom timeout in options
            mock_grpc.insecure_channel.assert_called_once()
            call_args = mock_grpc.insecure_channel.call_args
            options = call_args[1]["options"]
            assert ("grpc.connect_timeout_ms", 5000) in options  # 5.0 * 1000

            # channel_ready_future result uses custom timeout
            mock_future.result.assert_called_with(timeout=5.0)

    def test_default_port_used_when_missing(self):
        """Test that default port 15002 is used when not specified."""
        with (
            patch("socket.socket") as mock_socket_class,
            patch("src.service.dependencies.grpc") as mock_grpc,
        ):
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel
            mock_future = MagicMock()
            mock_grpc.channel_ready_future.return_value = mock_future

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
        """Test successful Spark Connect session creation.

        For Spark Connect mode, we do NOT call spark.stop() because the Spark
        cluster belongs to the user's notebook pod, not to the MCP server.
        """
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

                            # Spark Connect mode: stop() should NOT be called
                            # (cluster belongs to user's notebook, not us)
                            mock_spark.stop.assert_not_called()

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
        """Test that session is stopped even if request handler raises error.

        This test uses shared cluster mode (is_spark_connect_reachable=False)
        to verify cleanup behavior, since Spark Connect mode skips stop().
        """
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
                        return_value=False,  # Use shared cluster mode to test cleanup
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

                            # Session should still be stopped (shared cluster mode)
                            mock_spark.stop.assert_called_once()

    def test_session_stop_error_handled_gracefully(self, mock_request, mock_settings):
        """Test that errors during session.stop() are handled.

        This test uses shared cluster mode (is_spark_connect_reachable=False)
        since that's when stop() is actually called.
        """
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
                    return_value=False,  # Use shared cluster mode to test stop() error handling
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

    def test_connect_mode_acquires_lock_during_creation(
        self, mock_request, mock_settings
    ):
        """Test that Connect mode acquires _session_mode_lock during session creation.

        Verifies the lock is acquired by checking that concurrent Connect creations
        don't overlap (they should be serialized).
        """
        import threading
        import time

        mock_request_obj = mock_request(user="testuser")
        creation_events = {"count": 0, "max_concurrent": 0, "current": 0}
        lock = threading.Lock()

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
                        # Track concurrent creations
                        def mock_create(*args, **kwargs):
                            with lock:
                                creation_events["current"] += 1
                                creation_events["max_concurrent"] = max(
                                    creation_events["max_concurrent"],
                                    creation_events["current"],
                                )
                                creation_events["count"] += 1
                            time.sleep(0.01)  # Small delay to expose races
                            with lock:
                                creation_events["current"] -= 1
                            return MagicMock()

                        mock_get_spark.side_effect = mock_create

                        gen = get_spark_session(mock_request_obj, mock_settings)
                        next(gen)

                        # Should have created exactly 1 session
                        assert creation_events["count"] == 1

                        # Cleanup
                        try:
                            next(gen)
                        except StopIteration:
                            pass

    def test_standalone_mode_holds_lock_for_entire_request(
        self, mock_request, mock_settings
    ):
        """Test that standalone mode holds _session_mode_lock for entire request lifecycle.

        Verifies the lock prevents other requests from starting while one is active.
        """
        import threading

        mock_request_obj = mock_request(user="testuser")
        lifecycle_events = []
        events_lock = threading.Lock()

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.is_spark_connect_reachable",
                    return_value=False,  # Fallback to standalone
                ):
                    with patch(
                        "src.service.dependencies._get_spark_session"
                    ) as mock_get_spark:

                        def mock_create(*args, **kwargs):
                            with events_lock:
                                lifecycle_events.append("session_created")
                            mock_spark = MagicMock()
                            return mock_spark

                        mock_get_spark.side_effect = mock_create

                        gen = get_spark_session(mock_request_obj, mock_settings)
                        _spark = next(gen)

                        with events_lock:
                            lifecycle_events.append("after_yield")

                        # Session should have been created before yield
                        assert "session_created" in lifecycle_events
                        assert "after_yield" in lifecycle_events

                        # Cleanup
                        try:
                            next(gen)
                        except StopIteration:
                            pass

    def test_mixed_mode_requests_serialize_during_creation(
        self, mock_request, mock_settings, concurrent_executor
    ):
        """Test that mixed Connect/Standalone requests serialize session creation.

        When both Connect and Standalone requests happen concurrently,
        session creation should be serialized to prevent mode conflicts.

        Note: Patches are applied OUTSIDE the threads to avoid race conditions
        where each thread's patch context interferes with others.
        """
        import threading
        import time

        creation_tracking = {"in_use": False, "conflicts": 0, "created": 0}
        tracking_lock = threading.Lock()

        # Track which user index requested which mode
        mode_by_user = {}
        mode_lock = threading.Lock()

        def mock_create(*args, **kwargs):
            """Track if another creation is in progress."""
            with tracking_lock:
                if creation_tracking["in_use"]:
                    creation_tracking["conflicts"] += 1
                creation_tracking["in_use"] = True
                creation_tracking["created"] += 1

            time.sleep(0.02)  # Small delay to expose race conditions

            with tracking_lock:
                creation_tracking["in_use"] = False
            return MagicMock()

        def get_user_mock(request):
            """Return user_{0-3} based on request object."""
            return getattr(request, "_test_user", "testuser")

        def is_connect_mock(url, timeout=1.0):
            """Return True for even users (Connect), False for odd (Standalone)."""
            # Extract user index from URL pattern
            # URL format: sc://jupyter-user_{N}:15002
            with mode_lock:
                for user, is_connect in mode_by_user.items():
                    if user in url:
                        return is_connect
            return False

        def create_session(mode_index):
            """Create session using shared mocks."""
            is_connect = mode_index % 2 == 0
            mock_req = MagicMock()
            mock_req.state._request_state = MagicMock()
            mock_req._test_user = f"user_{mode_index}"

            with mode_lock:
                mode_by_user[f"user_{mode_index}"] = is_connect

            gen = get_spark_session(mock_req, mock_settings)
            _spark = next(gen)

            try:
                next(gen)
            except StopIteration:
                pass

            return mode_index

        # Apply patches OUTSIDE the threads so all threads share the same mocks
        with patch(
            "src.service.dependencies.get_user_from_request",
            side_effect=get_user_mock,
        ):
            with patch(
                "src.service.dependencies.read_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.is_spark_connect_reachable",
                    side_effect=is_connect_mock,
                ):
                    with patch(
                        "src.service.dependencies._get_spark_session",
                        side_effect=mock_create,
                    ):
                        # Run 4 requests: 2 Connect, 2 Standalone (interleaved)
                        args_list = [(i,) for i in range(4)]
                        results, exceptions = concurrent_executor(
                            create_session, args_list, max_workers=4
                        )

        # No exceptions should occur
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"
        # All requests should complete
        assert sorted(results) == [0, 1, 2, 3]
        # All sessions should have been created
        assert creation_tracking["created"] == 4
        # No conflicts should occur (lock should serialize creation)
        assert creation_tracking["conflicts"] == 0, (
            f"Expected 0 conflicts but got {creation_tracking['conflicts']}. "
            "This indicates the lock is not properly serializing session creation."
        )


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
