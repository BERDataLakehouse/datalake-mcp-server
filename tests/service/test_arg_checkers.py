"""Tests for the argument checker module."""

from src.service import arg_checkers


def test_arg_checkers_imports():
    """Test that arg_checkers module can be imported."""
    assert arg_checkers is not None


def test_noop():
    """Simple placeholder test."""
    assert 1 == 1
