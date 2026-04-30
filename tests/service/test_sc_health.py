"""Unit tests for src/service/sc_health.py."""

import time

import pytest

from src.service import sc_health


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with a clean unhealthy-tracker."""
    sc_health._unhealthy_until.clear()
    yield
    sc_health._unhealthy_until.clear()


def test_unmarked_user_is_healthy():
    assert sc_health.is_unhealthy("alice") is False


def test_mark_then_check_returns_true():
    sc_health.mark_unhealthy("alice")
    assert sc_health.is_unhealthy("alice") is True


def test_mark_one_user_does_not_affect_others():
    sc_health.mark_unhealthy("alice")
    assert sc_health.is_unhealthy("bob") is False


def test_clear_resets_unhealthy_state():
    sc_health.mark_unhealthy("alice")
    sc_health.clear("alice")
    assert sc_health.is_unhealthy("alice") is False


def test_ttl_expiry_drops_mark(monkeypatch):
    """After the TTL elapses, is_unhealthy should auto-drop the entry."""
    monkeypatch.setattr(sc_health, "SC_UNHEALTHY_TTL", 0.05)
    sc_health.mark_unhealthy("alice")
    assert sc_health.is_unhealthy("alice") is True
    time.sleep(0.1)
    assert sc_health.is_unhealthy("alice") is False
    # The entry should have been removed, not just hidden.
    assert "alice" not in sc_health._unhealthy_until


def test_clear_on_unmarked_user_is_noop():
    sc_health.clear("never_marked")  # must not raise
