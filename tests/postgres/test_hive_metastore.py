"""Tests for the hive metastore module."""

from src.postgres import hive_metastore


def test_hive_metastore_imports():
    """Test that hive metastore module can be imported."""
    assert hive_metastore is not None


def test_noop():
    """Simple placeholder test."""
    assert 1 == 1
