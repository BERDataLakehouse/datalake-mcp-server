"""Tests for the postgres connection module."""

from src.postgres import connection


def test_connection_imports():
    """Test that postgres connection module can be imported."""
    assert connection is not None


def test_noop():
    """Simple placeholder test."""
    assert 1 == 1
