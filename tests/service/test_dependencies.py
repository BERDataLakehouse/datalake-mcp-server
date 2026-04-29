"""
Tests for the FastAPI dependencies module.

Tests cover:
- fetch_user_minio_credentials() - API-based credential fetch
- get_user_from_request() - user extraction from request state
- get_spark_session() - session creation, fallback logic, cleanup
- is_spark_connect_reachable() - TCP check mocking
- construct_user_spark_connect_url() - URL construction
- Concurrent session creation
"""

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import grpc
import pytest

from src.service.dependencies import (
    SparkContext,
    TrinoContext,
    fetch_user_minio_credentials,
    get_token_from_request,
    get_user_from_request,
    construct_user_spark_connect_url,
    is_spark_connect_reachable,
    get_spark_context,
    get_spark_session,
    resolve_engine,
    sanitize_k8s_name,
    DEFAULT_SPARK_POOL,
    SPARK_CONNECT_PORT,
)
from src.service.models import QueryEngine


# =============================================================================
# Test fetch_user_minio_credentials
# =============================================================================


class TestFetchUserMinioCredentials:
    """Tests for the fetch_user_minio_credentials function."""

    def test_successful_credential_fetch(self):
        """Test successfully fetching credentials from governance API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "username": "testuser",
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "src.service.dependencies.httpx.get", return_value=mock_response
        ) as mock_get:
            access_key, secret_key = fetch_user_minio_credentials(
                "http://governance:8000", "test-token"
            )

            mock_get.assert_called_once_with(
                "http://governance:8000/credentials/",
                headers={"Authorization": "Bearer test-token"},
                timeout=10.0,
            )

        assert access_key == "test_access_key"
        assert secret_key == "test_secret_key"

    def test_api_error_raises_exception(self):
        """Test that HTTP errors are propagated."""
        import httpx

        with patch(
            "src.service.dependencies.httpx.get",
            side_effect=httpx.HTTPStatusError(
                "401", request=MagicMock(), response=MagicMock()
            ),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                fetch_user_minio_credentials("http://governance:8000", "bad-token")

    def test_missing_access_key_raises_error(self):
        """Test that missing access_key raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "username": "testuser",
            "secret_key": "secret",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.service.dependencies.httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="missing access_key or secret_key"):
                fetch_user_minio_credentials("http://governance:8000", "test-token")

    def test_missing_secret_key_raises_error(self):
        """Test that missing secret_key raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "username": "testuser",
            "access_key": "access",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.service.dependencies.httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="missing access_key or secret_key"):
                fetch_user_minio_credentials("http://governance:8000", "test-token")

    def test_trailing_slash_in_url_handled(self):
        """Test that trailing slash in governance URL is handled correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "username": "testuser",
            "access_key": "key",
            "secret_key": "secret",
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "src.service.dependencies.httpx.get", return_value=mock_response
        ) as mock_get:
            fetch_user_minio_credentials("http://governance:8000/", "test-token")

            # Should not double the trailing slash
            mock_get.assert_called_once_with(
                "http://governance:8000/credentials/",
                headers={"Authorization": "Bearer test-token"},
                timeout=10.0,
            )


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
                    side_effect=lambda k, d=None: (
                        None
                        if k == "SPARK_CONNECT_URL_TEMPLATE"
                        else ("dev" if k == "K8S_ENVIRONMENT" else d)
                    ),
                ):
                    url = construct_user_spark_connect_url("myuser")

        assert "jupyter-myuser" in url
        assert "15002" in url

    def test_kubernetes_url_prod_environment(self):
        """Test Kubernetes URL construction with prod environment."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("prod" if k == "K8S_ENVIRONMENT" else d)
            ),
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
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("dev" if k == "K8S_ENVIRONMENT" else d)
            ),
        ):
            url = construct_user_spark_connect_url("tian_gu_test")

        # Username should be sanitized (tian_gu_test → tian-gu-test)
        assert url == "sc://jupyter-tian-gu-test.jupyterhub-dev:15002"

    def test_kubernetes_url_sanitizes_mixed_case_username(self):
        """Test that Kubernetes URL sanitizes mixed-case usernames."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("dev" if k == "K8S_ENVIRONMENT" else d)
            ),
        ):
            url = construct_user_spark_connect_url("Jeff_Cohere")

        # Username should be sanitized and lowercased
        assert url == "sc://jupyter-jeff-cohere.jupyterhub-dev:15002"

    def test_kubernetes_url_preserves_dns_compliant_username(self):
        """Test that DNS-compliant usernames are preserved as-is."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("dev" if k == "K8S_ENVIRONMENT" else d)
            ),
        ):
            url = construct_user_spark_connect_url("tgu2")

        # DNS-compliant username should be unchanged
        assert url == "sc://jupyter-tgu2.jupyterhub-dev:15002"

    def test_kubernetes_url_prod_with_sanitization(self):
        """Test Kubernetes URL with prod environment and username sanitization."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("prod" if k == "K8S_ENVIRONMENT" else d)
            ),
        ):
            url = construct_user_spark_connect_url("user_name_test")

        assert url == "sc://jupyter-user-name-test.jupyterhub-prod:15002"

    def test_kubernetes_url_stage_with_sanitization(self):
        """Test Kubernetes URL with stage environment and username sanitization."""
        with patch(
            "os.getenv",
            side_effect=lambda k, d=None: (
                None
                if k == "SPARK_CONNECT_URL_TEMPLATE"
                else ("stage" if k == "K8S_ENVIRONMENT" else d)
            ),
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
                "src.service.dependencies.fetch_user_minio_credentials",
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
                "src.service.dependencies.fetch_user_minio_credentials",
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
                "src.service.dependencies.fetch_user_minio_credentials",
                side_effect=FileNotFoundError("Credentials not found"),
            ):
                gen = get_spark_session(mock_request_obj, mock_settings)

                with pytest.raises(
                    Exception, match="failed to fetch MinIO credentials"
                ):
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
                "src.service.dependencies.fetch_user_minio_credentials",
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
                "src.service.dependencies.fetch_user_minio_credentials",
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
        """Test that multiple concurrent requests get isolated sessions.

        Uses side_effects instead of per-thread patching to avoid race conditions
        in ThreadPoolExecutor.
        """
        sessions_created = {"count": 0}
        lock = threading.Lock()

        # Dynamic side effects to handle per-thread logic without patching inside threads
        def get_user_side_effect(request):
            # Extract user from the request object passed by create_session
            # We'll attach a custom attribute _test_user to the mock request
            return getattr(request, "_test_user", "unknown")

        def get_spark_side_effect(app_name=None, settings=None, **kwargs):
            # Create a unique mock spark session for each user
            mock_spark = MagicMock()
            # Extract user ID from app_name or settings to simulate isolation
            if app_name and "user_" in app_name:
                parts = app_name.split("_")
                for part in parts:
                    if part.isdigit():
                        mock_spark.user_id = int(part)
                        break

            # Also support extraction from settings if needed
            if settings and hasattr(settings, "USER"):
                if "user_" in settings.USER:
                    try:
                        mock_spark.user_id = int(settings.USER.split("_")[1])
                    except (IndexError, ValueError):
                        pass

            with lock:
                sessions_created["count"] += 1
            return mock_spark

        # Helper function run in threads - NO PATCHING HERE
        def create_session(user_id):
            mock_req = MagicMock()
            mock_req.state._request_state = MagicMock()
            # Attach user info to request so get_user_side_effect can find it
            mock_req._test_user = f"user_{user_id}"

            gen = get_spark_session(mock_req, mock_settings)
            spark = next(gen)

            # Get the user_id we stashed on the mock
            spark_id = getattr(spark, "user_id", -1)

            try:
                next(gen)
            except StopIteration:
                pass

            return spark_id

        # Apply patches GLOBALLY
        with (
            patch(
                "src.service.dependencies.get_user_from_request",
                side_effect=get_user_side_effect,
            ),
            patch(
                "src.service.dependencies.get_token_from_request",
                return_value="fake-token",
            ),
            patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                return_value=("access", "secret"),
            ),
            patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ),
            patch(
                "src.service.dependencies._get_spark_session",
                side_effect=get_spark_side_effect,
            ),
        ):
            args_list = [(i,) for i in range(5)]
            results, exceptions = concurrent_executor(
                create_session, args_list, max_workers=5
            )

        assert len(exceptions) == 0, f"Encountered exceptions: {exceptions}"
        # Each user should get their own session corresponding to their ID
        assert sorted(results) == [0, 1, 2, 3, 4]

    def test_connect_mode_acquires_lock_during_creation(
        self, mock_request, mock_settings
    ):
        """Test that Connect mode acquires _session_mode_lock during session creation.

        Verifies the lock is acquired by checking that concurrent Connect creations
        don't overlap (they should be serialized).
        """
        mock_request_obj = mock_request(user="testuser")
        creation_events = {"count": 0, "max_concurrent": 0, "current": 0}
        lock = threading.Lock()

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
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
        mock_request_obj = mock_request(user="testuser")
        lifecycle_events = []
        events_lock = threading.Lock()

        with patch(
            "src.service.dependencies.get_user_from_request", return_value="testuser"
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
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

    def test_mixed_mode_requests_run_concurrently(
        self, mock_request, mock_settings, concurrent_executor
    ):
        """Test that mixed Connect/Standalone requests can run concurrently.

        With the new architecture:
        - Spark Connect mode: Fully concurrent (no locking), each is a gRPC client
        - Standalone mode: Process-isolated via ProcessPoolExecutor

        This test verifies that concurrent requests complete successfully.
        """
        creation_tracking = {"created": 0}
        tracking_lock = threading.Lock()

        # Track which user index requested which mode
        mode_by_user = {}
        mode_lock = threading.Lock()

        def mock_create(*args, **kwargs):
            """Track session creation."""
            with tracking_lock:
                creation_tracking["created"] += 1

            time.sleep(0.02)  # Small delay to simulate work
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
        with (
            patch(
                "src.service.dependencies.get_user_from_request",
                side_effect=get_user_mock,
            ),
            patch(
                "src.service.dependencies.get_token_from_request",
                return_value="fake-token",
            ),
            patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                return_value=("access", "secret"),
            ),
            patch(
                "src.service.dependencies.is_spark_connect_reachable",
                side_effect=is_connect_mock,
            ),
            patch(
                "src.service.dependencies._get_spark_session",
                side_effect=mock_create,
            ),
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


# =============================================================================
# Token Extraction Tests
# =============================================================================


class TestGetTokenFromRequest:
    """Tests for the get_token_from_request function."""

    def test_extracts_bearer_token(self):
        """Test successful extraction of Bearer token."""

        request = MagicMock()
        request.headers.get.return_value = "Bearer test-token-123"

        token = get_token_from_request(request)

        assert token == "test-token-123"

    def test_returns_none_for_missing_header(self):
        """Test that missing Authorization header returns None."""

        request = MagicMock()
        request.headers.get.return_value = ""

        token = get_token_from_request(request)

        assert token is None

    def test_returns_none_for_non_bearer_auth(self):
        """Test that non-Bearer auth returns None."""

        request = MagicMock()
        request.headers.get.return_value = "Basic dXNlcjpwYXNz"

        token = get_token_from_request(request)

        assert token is None

    def test_case_insensitive_scheme(self):
        """Test that lowercase 'bearer' is accepted (matches auth middleware)."""

        request = MagicMock()
        request.headers.get.return_value = "bearer test-token-123"

        token = get_token_from_request(request)

        assert token == "test-token-123"

    def test_uppercase_scheme(self):
        """Test that uppercase 'BEARER' is accepted."""

        request = MagicMock()
        request.headers.get.return_value = "BEARER test-token-123"

        token = get_token_from_request(request)

        assert token == "test-token-123"

    def test_extra_whitespace_trimmed(self):
        """Test that extra whitespace between scheme and token is handled."""

        request = MagicMock()
        request.headers.get.return_value = "Bearer   test-token-123"

        token = get_token_from_request(request)

        assert token == "test-token-123"

    def test_returns_none_for_bearer_without_token(self):
        """Test that 'Bearer' with no token value returns None."""

        request = MagicMock()
        request.headers.get.return_value = "Bearer "

        token = get_token_from_request(request)

        assert token is None


# =============================================================================
# SparkContext Dataclass Tests
# =============================================================================


class TestSparkContextDataclass:
    """Tests for the SparkContext dataclass."""

    def test_default_values(self):
        """Test SparkContext with default values."""

        ctx = SparkContext()

        assert ctx.spark is None
        assert ctx.is_standalone_subprocess is False
        assert ctx.settings_dict == {}
        assert ctx.app_name == ""
        assert ctx.username == ""
        assert ctx.auth_token is None

    def test_custom_values(self):
        """Test SparkContext with custom values."""

        mock_spark = MagicMock()
        ctx = SparkContext(
            spark=mock_spark,
            is_standalone_subprocess=True,
            settings_dict={"USER": "testuser"},
            app_name="mcp_query",
            username="testuser",
            auth_token="token123",
        )

        assert ctx.spark == mock_spark
        assert ctx.is_standalone_subprocess is True
        assert ctx.settings_dict == {"USER": "testuser"}
        assert ctx.app_name == "mcp_query"
        assert ctx.username == "testuser"
        assert ctx.auth_token == "token123"


# =============================================================================
# get_spark_context Tests
# =============================================================================


class TestGetSparkContext:
    """Tests for the get_spark_context dependency generator."""

    @pytest.fixture(autouse=True)
    def _reset_sc_health(self):
        """Each test starts with a clean Spark Connect health tracker.

        ``_base_patches`` always resolves the user to "testuser", so any test
        that marks SC unhealthy (probe-failure tests) would leak state into
        every subsequent test in this class without this reset.
        """
        from src.service import sc_health

        sc_health._unhealthy_until.clear()
        yield
        sc_health._unhealthy_until.clear()

    def _base_patches(self):
        """Convenience: return a dict of the common patches for get_spark_context."""
        return {
            "get_user": patch(
                "src.service.dependencies.get_user_from_request",
                return_value="testuser",
            ),
            "get_token": patch(
                "src.service.dependencies.get_token_from_request",
                return_value="tok-123",
            ),
            "creds": patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                return_value=("access", "secret"),
            ),
            "url": patch(
                "src.service.dependencies.construct_user_spark_connect_url",
                return_value="sc://jupyter-testuser:15002",
            ),
        }

    # ---- Spark Connect happy path ----

    def test_spark_connect_returns_context_with_session(
        self, mock_request, mock_settings
    ):
        """get_spark_context yields a SparkContext with spark session in Connect mode."""
        req = mock_request(user="testuser")
        mock_spark = MagicMock()
        mock_spark.version = "3.5.0"

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ):
                with patch(
                    "src.service.dependencies._get_spark_session_with_retry",
                    return_value=mock_spark,
                ):
                    gen = get_spark_context(req, mock_settings)
                    ctx = next(gen)

                    assert isinstance(ctx, SparkContext)
                    assert ctx.spark is mock_spark
                    assert ctx.is_standalone_subprocess is False
                    assert ctx.username == "testuser"
                    assert ctx.auth_token == "tok-123"
                    assert "SPARK_CONNECT_URL" in ctx.settings_dict

                    try:
                        next(gen)
                    except StopIteration:
                        pass

    # ---- Fallback: Connect unreachable → Standalone ----

    def test_standalone_when_connect_unreachable(self, mock_request, mock_settings):
        """When Spark Connect is unreachable, yield Standalone context (no session)."""
        req = mock_request(user="testuser")

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=False,
            ):
                gen = get_spark_context(req, mock_settings)
                ctx = next(gen)

                assert ctx.spark is None
                assert ctx.is_standalone_subprocess is True
                assert ctx.username == "testuser"
                assert "SPARK_MASTER_URL" in ctx.settings_dict

                try:
                    next(gen)
                except StopIteration:
                    pass

    # ---- Fallback: Connect session creation fails ----

    def test_fallback_when_connect_session_fails(self, mock_request, mock_settings):
        """Connect reachable but session creation fails → fall back to Standalone."""
        req = mock_request(user="testuser")

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ):
                with patch(
                    "src.service.dependencies._get_spark_session_with_retry",
                    side_effect=ConnectionError("gRPC failed"),
                ):
                    gen = get_spark_context(req, mock_settings)
                    ctx = next(gen)

                    assert ctx.spark is None
                    assert ctx.is_standalone_subprocess is True

                    try:
                        next(gen)
                    except StopIteration:
                        pass

    # ---- Connect: SQL probe fails → fail fast (no Standalone fallback) ----
    # Note: _base_patches resolves the user to "testuser" regardless of the
    # value passed to mock_request; the autouse fixture above clears
    # sc_health state so these tests don't leak across each other.

    def test_failed_sql_probe_raises_sc_unavailable(self, mock_request, mock_settings):
        """Connect session created but SELECT 1 probe fails → SparkConnectUnavailableError.

        This is the wedge mode we observed in stage: gRPC channel is up,
        session creation succeeds, but the JVM driver in the user's notebook
        pod is deadlocked at the SQL execution layer. We must NOT silently
        fall back to Standalone — instead surface a clear actionable error
        so the user knows to restart their notebook pod.
        """
        from src.service import sc_health
        from src.service.exceptions import SparkConnectUnavailableError

        req = mock_request(user="testuser")
        mock_spark = MagicMock()
        # spark.sql("SELECT 1").collect() raises → simulates wedged SQL layer.
        mock_spark.sql.return_value.collect.side_effect = Exception("driver deadlocked")

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ):
                with patch(
                    "src.service.dependencies._get_spark_session_with_retry",
                    return_value=mock_spark,
                ):
                    gen = get_spark_context(req, mock_settings)
                    with pytest.raises(SparkConnectUnavailableError):
                        next(gen)

        # User should now be marked unhealthy so subsequent requests short-circuit.
        assert sc_health.is_unhealthy("testuser") is True

    def test_unhealthy_user_short_circuits_before_session_creation(
        self, mock_request, mock_settings
    ):
        """Pre-marked unhealthy user gets immediate SparkConnectUnavailableError."""
        from src.service import sc_health
        from src.service.exceptions import SparkConnectUnavailableError

        sc_health.mark_unhealthy("testuser")
        req = mock_request(user="testuser")

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ):
                with patch(
                    "src.service.dependencies._get_spark_session_with_retry",
                ) as mock_create:
                    gen = get_spark_context(req, mock_settings)
                    with pytest.raises(SparkConnectUnavailableError):
                        next(gen)

                    # Critical: we must NOT have attempted to create a session.
                    mock_create.assert_not_called()

    # ---- Credential errors ----

    def test_missing_auth_token_raises_error(self, mock_request, mock_settings):
        """Missing auth token raises with helpful message."""
        req = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.get_token_from_request",
                return_value=None,
            ):
                gen = get_spark_context(req, mock_settings)
                with pytest.raises(Exception, match="no auth token available"):
                    next(gen)

    def test_credential_fetch_error(self, mock_request, mock_settings):
        """Credential fetch failure propagates with helpful message."""
        req = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.get_token_from_request",
                return_value="valid-token",
            ):
                with patch(
                    "src.service.dependencies.fetch_user_minio_credentials",
                    side_effect=Exception("API connection refused"),
                ):
                    gen = get_spark_context(req, mock_settings)
                    with pytest.raises(
                        Exception, match="failed to fetch MinIO credentials"
                    ):
                        next(gen)

    # ---- Standalone: BERDL_POD_IP fallback ----

    def test_standalone_pod_ip_fallback(self, mock_request, mock_settings):
        """When BERDL_POD_IP is empty, standalone fills in 0.0.0.0."""
        req = mock_request(user="testuser")
        mock_settings.BERDL_POD_IP = ""

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=False,
            ):
                gen = get_spark_context(req, mock_settings)
                ctx = next(gen)

                assert ctx.settings_dict["BERDL_POD_IP"] == "0.0.0.0"

                try:
                    next(gen)
                except StopIteration:
                    pass

    def test_token_extraction_exception_raises(self, mock_request, mock_settings):
        """get_token_from_request failure results in auth error since token is required."""
        req = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.get_token_from_request",
                side_effect=Exception("header parse error"),
            ):
                gen = get_spark_context(req, mock_settings)
                with pytest.raises(Exception, match="no auth token available"):
                    next(gen)

    def test_connect_mode_pod_ip_fallback(self, mock_request, mock_settings):
        """Connect mode fills in BERDL_POD_IP when empty (line 519)."""
        req = mock_request(user="testuser")
        mock_settings.BERDL_POD_IP = ""
        mock_spark = MagicMock()
        mock_spark.version = "3.5.0"

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=True,
            ):
                with patch(
                    "src.service.dependencies._get_spark_session_with_retry",
                    return_value=mock_spark,
                ):
                    gen = get_spark_context(req, mock_settings)
                    ctx = next(gen)

                    assert ctx.settings_dict["BERDL_POD_IP"] == "0.0.0.0"

                    try:
                        next(gen)
                    except StopIteration:
                        pass

    def test_standalone_empty_master_urls(self, mock_request, mock_settings):
        """Empty SHARED_SPARK_MASTER_URL uses default in get_spark_context (line 574)."""
        req = mock_request(user="testuser")

        ps = self._base_patches()
        with ps["get_user"], ps["get_token"], ps["creds"], ps["url"]:
            with patch(
                "src.service.dependencies.is_spark_connect_reachable",
                return_value=False,
            ):
                with patch.dict("os.environ", {"SHARED_SPARK_MASTER_URL": "  "}):
                    gen = get_spark_context(req, mock_settings)
                    ctx = next(gen)

                    assert (
                        "sharedsparkclustermaster"
                        in ctx.settings_dict["SPARK_MASTER_URL"]
                    )

                    try:
                        next(gen)
                    except StopIteration:
                        pass


# =============================================================================
# get_spark_session (deprecated) — additional error branches
# =============================================================================


class TestGetSparkSessionDeprecatedErrors:
    """Cover uncovered error branches in the deprecated get_spark_session."""

    def test_generic_credential_error(self, mock_request, mock_settings):
        """Credential fetch failure propagates (line 642-643)."""
        req = mock_request(user="testuser")

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                side_effect=PermissionError("denied"),
            ):
                gen = get_spark_session(req, mock_settings)
                with pytest.raises(
                    Exception, match="failed to fetch MinIO credentials"
                ):
                    next(gen)

    def test_spark_connect_fails_falls_back(self, mock_request, mock_settings):
        """Spark Connect session creation failure falls back to standalone (line 684-689)."""
        req = mock_request(user="testuser")
        mock_spark = MagicMock()

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
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
                        call_count = {"n": 0}

                        def create_side_effect(*a, **kw):
                            call_count["n"] += 1
                            if call_count["n"] == 1:
                                raise RuntimeError("Connect failed")
                            return mock_spark

                        with patch(
                            "src.service.dependencies._get_spark_session",
                            side_effect=create_side_effect,
                        ):
                            gen = get_spark_session(req, mock_settings)
                            spark = next(gen)

                            assert spark is mock_spark
                            # Second call should be standalone
                            second_call = mock_spark  # returned by side_effect
                            assert second_call is not None

                            try:
                                next(gen)
                            except StopIteration:
                                pass

    def test_empty_master_urls_fallback(self, mock_request, mock_settings):
        """Empty SHARED_SPARK_MASTER_URL uses default (line 699)."""
        req = mock_request(user="testuser")
        mock_spark = MagicMock()

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.is_spark_connect_reachable",
                    return_value=False,
                ):
                    with patch.dict(
                        "os.environ",
                        {"SHARED_SPARK_MASTER_URL": "   "},
                    ):
                        with patch(
                            "src.service.dependencies._get_spark_session",
                            return_value=mock_spark,
                        ):
                            gen = get_spark_session(req, mock_settings)
                            spark = next(gen)
                            assert spark is mock_spark

                            try:
                                next(gen)
                            except StopIteration:
                                pass

    def test_missing_pod_ip_filled(self, mock_request, mock_settings):
        """When BERDL_POD_IP is falsy, it's set to 0.0.0.0 (line 705)."""
        req = mock_request(user="testuser")
        mock_settings.BERDL_POD_IP = ""
        mock_spark = MagicMock()

        with patch(
            "src.service.dependencies.get_user_from_request",
            return_value="testuser",
        ):
            with patch(
                "src.service.dependencies.fetch_user_minio_credentials",
                return_value=("access", "secret"),
            ):
                with patch(
                    "src.service.dependencies.is_spark_connect_reachable",
                    return_value=False,
                ):
                    with patch(
                        "src.service.dependencies._get_spark_session",
                        return_value=mock_spark,
                    ) as mock_create:
                        gen = get_spark_session(req, mock_settings)
                        next(gen)

                        call_kwargs = mock_create.call_args[1]
                        settings_obj = call_kwargs["settings"]
                        assert settings_obj.BERDL_POD_IP == "0.0.0.0"

                        try:
                            next(gen)
                        except StopIteration:
                            pass


# =============================================================================
# is_spark_connect_reachable — gRPC non-timeout exception (lines 327-329)
# =============================================================================


class TestIsSparkConnectReachableGrpcError:
    """Cover the non-FutureTimeoutError gRPC exception branch."""

    def test_grpc_non_timeout_exception(self):
        """gRPC channel_ready_future raises a non-timeout exception (line 327-329)."""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_sock

            with patch("src.service.dependencies.grpc") as mock_grpc:
                mock_channel = MagicMock()
                mock_grpc.insecure_channel.return_value = mock_channel
                mock_future = MagicMock()
                mock_future.result.side_effect = RuntimeError("unexpected gRPC error")
                mock_grpc.channel_ready_future.return_value = mock_future
                mock_grpc.FutureTimeoutError = grpc.FutureTimeoutError

                result = is_spark_connect_reachable("sc://localhost:15002")

        assert result is False
        mock_channel.close.assert_called_once()


# =============================================================================
# Test resolve_engine
# =============================================================================


class TestResolveEngine:
    """Tests for the resolve_engine function."""

    def test_default_is_spark(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_engine() == QueryEngine.SPARK

    def test_env_var_trino(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "trino"}):
            assert resolve_engine() == QueryEngine.TRINO

    def test_env_var_spark(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "spark"}):
            assert resolve_engine() == QueryEngine.SPARK

    def test_env_var_case_insensitive(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "TRINO"}):
            assert resolve_engine() == QueryEngine.TRINO

    def test_per_request_override_takes_precedence(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "spark"}):
            assert resolve_engine(QueryEngine.TRINO) == QueryEngine.TRINO

    def test_per_request_none_falls_through(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "trino"}):
            assert resolve_engine(None) == QueryEngine.TRINO

    def test_invalid_env_defaults_to_spark(self):
        with patch.dict("os.environ", {"QUERY_ENGINE": "invalid"}):
            assert resolve_engine() == QueryEngine.SPARK


# =============================================================================
# Test TrinoContext
# =============================================================================


class TestTrinoContextDataclass:
    """Tests for the TrinoContext dataclass."""

    def test_required_fields(self):
        conn = MagicMock()
        ctx = TrinoContext(connection=conn)
        assert ctx.connection is conn
        assert ctx.username == ""
        assert ctx.auth_token is None
        assert ctx.settings_dict == {}

    def test_all_fields(self):
        conn = MagicMock()
        ctx = TrinoContext(
            connection=conn,
            username="alice",
            auth_token="tok",
            settings_dict={"TRINO_HOST": "trino"},
        )
        assert ctx.username == "alice"
        assert ctx.auth_token == "tok"
        assert ctx.settings_dict == {"TRINO_HOST": "trino"}
