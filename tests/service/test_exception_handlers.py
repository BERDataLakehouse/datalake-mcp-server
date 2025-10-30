"""Tests for the exception handlers module."""

from src.service import exception_handlers


def test_exception_handlers_imports():
    """Test that exception_handlers module can be imported."""
    assert exception_handlers is not None


def test_noop():
    """Simple placeholder test."""
    assert 1 == 1
