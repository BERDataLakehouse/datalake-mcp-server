"""
Tests for the health routes module.

Tests cover:
- _timed_check helper: all return branches (True, False, string, exception)
- _check_redis: client None and successful ping
- _check_hive_metastore: successful database listing
- health_check endpoint: unhealthy, degraded, and healthy overall statuses
"""

from unittest.mock import MagicMock, patch

import pytest

from src.routes.health import (
    _check_hive_metastore,
    _check_redis,
    _timed_check,
    health_check,
)
from src.service.models import ComponentHealth


# =============================================================================
# _timed_check
# =============================================================================


class TestTimedCheck:
    """Cover all branches of the _timed_check helper (lines 39-70)."""

    def test_returns_healthy_when_true(self):
        """check_fn returning True produces a healthy ComponentHealth."""
        result = _timed_check("comp", lambda: True)

        assert isinstance(result, ComponentHealth)
        assert result.name == "comp"
        assert result.status == "healthy"
        assert result.message is None
        assert result.latency_ms >= 0

    def test_returns_unhealthy_when_false(self):
        """check_fn returning False produces an unhealthy ComponentHealth."""
        result = _timed_check("comp", lambda: False)

        assert result.status == "unhealthy"
        assert result.message == "Health check failed"

    def test_returns_degraded_when_string(self):
        """check_fn returning a string produces a degraded ComponentHealth."""
        result = _timed_check("comp", lambda: "warming up")

        assert result.status == "degraded"
        assert result.message == "warming up"

    def test_returns_unhealthy_on_exception(self):
        """check_fn raising an exception produces an unhealthy ComponentHealth."""

        def boom():
            raise ConnectionError("connection refused")

        result = _timed_check("comp", boom)

        assert result.status == "unhealthy"
        assert "connection refused" in result.message

    def test_exception_message_truncated_to_200_chars(self):
        """Long exception messages are truncated."""

        def boom():
            raise RuntimeError("x" * 500)

        result = _timed_check("comp", boom)
        assert len(result.message) <= 200


# =============================================================================
# _check_redis
# =============================================================================


class TestCheckRedis:
    """Cover _check_redis branches."""

    @patch("src.routes.health._get_redis_client", return_value=None)
    def test_redis_client_not_initialized(self, _mock_get):
        """Returns string when Redis client is None (line 78)."""
        result = _check_redis()
        assert result == "Redis client not initialized"

    @patch("src.routes.health._get_redis_client")
    def test_redis_ping_succeeds(self, mock_get):
        """Returns True when ping succeeds (line 82)."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_get.return_value = mock_client

        assert _check_redis() is True

    @patch("src.routes.health._get_redis_client")
    def test_redis_ping_returns_false(self, mock_get):
        """Returns False when ping returns non-True value."""
        mock_client = MagicMock()
        mock_client.ping.return_value = False
        mock_get.return_value = mock_client

        assert _check_redis() is False


# =============================================================================
# _check_hive_metastore
# =============================================================================


class TestCheckHiveMetastore:
    """Cover _check_hive_metastore branches (lines 94-95)."""

    @patch("src.routes.health.get_hive_metastore_client")
    @patch("src.routes.health.get_settings")
    def test_hive_metastore_healthy(self, mock_settings, mock_get_client):
        """Returns True when databases can be listed."""
        mock_client = MagicMock()
        mock_client.get_databases.return_value = ["default"]
        mock_get_client.return_value = mock_client

        result = _check_hive_metastore()

        assert result is True
        mock_client.open.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("src.routes.health.get_hive_metastore_client")
    @patch("src.routes.health.get_settings")
    def test_hive_metastore_empty_db_list_still_healthy(
        self, mock_settings, mock_get_client
    ):
        """An empty database list is still valid (len >= 0 is True)."""
        mock_client = MagicMock()
        mock_client.get_databases.return_value = []
        mock_get_client.return_value = mock_client

        assert _check_hive_metastore() is True

    @patch("src.routes.health.get_hive_metastore_client")
    @patch("src.routes.health.get_settings")
    def test_hive_metastore_close_called_on_error(self, mock_settings, mock_get_client):
        """close() is called even when open() fails."""
        mock_client = MagicMock()
        mock_client.open.side_effect = ConnectionError("thrift down")
        mock_get_client.return_value = mock_client

        with pytest.raises(ConnectionError):
            _check_hive_metastore()

        mock_client.close.assert_called_once()


# =============================================================================
# health_check endpoint
# =============================================================================


class TestHealthCheckEndpoint:
    """Cover health_check overall-status logic.

    The route is sync ``def`` (so FastAPI runs it in the threadpool and
    blocking HMS/Redis calls don't freeze the event loop), so these tests
    call it directly without awaiting.
    """

    @patch("src.routes.health._check_hive_metastore", return_value=True)
    @patch("src.routes.health._check_redis", return_value=True)
    def test_all_healthy(self, _mock_redis, _mock_hive):
        """All components healthy ⇒ overall healthy."""
        resp = health_check()

        assert resp.status == "healthy"
        assert resp.message == "All components healthy"
        assert len(resp.components) == 2

    @patch(
        "src.routes.health._check_hive_metastore",
        side_effect=ConnectionError("thrift down"),
    )
    @patch("src.routes.health._check_redis", return_value=True)
    def test_one_unhealthy(self, _mock_redis, _mock_hive):
        """One unhealthy component ⇒ overall unhealthy."""
        resp = health_check()

        assert resp.status == "unhealthy"
        assert "1 component(s) unhealthy" in resp.message

    @patch("src.routes.health._check_hive_metastore", return_value=True)
    @patch("src.routes.health._check_redis", return_value="warming up")
    def test_one_degraded(self, _mock_redis, _mock_hive):
        """One degraded component ⇒ overall degraded."""
        resp = health_check()

        assert resp.status == "degraded"
        assert "1 component(s) degraded" in resp.message

    @patch(
        "src.routes.health._check_hive_metastore",
        side_effect=RuntimeError("hive error"),
    )
    @patch(
        "src.routes.health._check_redis",
        side_effect=RuntimeError("redis error"),
    )
    def test_all_unhealthy(self, _mock_redis, _mock_hive):
        """All components unhealthy."""
        resp = health_check()

        assert resp.status == "unhealthy"
        assert "2 component(s) unhealthy" in resp.message

    @patch("src.routes.health._check_hive_metastore", return_value="slow")
    @patch("src.routes.health._check_redis", return_value="warming up")
    def test_multiple_degraded(self, _mock_redis, _mock_hive):
        """Multiple degraded components."""
        resp = health_check()

        assert resp.status == "degraded"
        assert "2 component(s) degraded" in resp.message

    @patch(
        "src.routes.health._check_hive_metastore",
        side_effect=RuntimeError("hive error"),
    )
    @patch("src.routes.health._check_redis", return_value="degraded")
    def test_unhealthy_takes_precedence_over_degraded(self, _mock_redis, _mock_hive):
        """Unhealthy takes precedence over degraded."""
        resp = health_check()

        assert resp.status == "unhealthy"
