"""
Tests for the Redis cache module.

Tests cover:
- get_cached_value() - cache hit, cache miss, connection errors
- set_cached_value() - successful write, TTL handling, error handling
- _build_cache_key() - namespace/key formatting
- Concurrent access - thread safety for cache operations
"""

import json
import threading
from unittest.mock import MagicMock, patch

from redis.exceptions import RedisError

from src.cache.redis_cache import (
    _build_cache_key,
    _get_redis_client,
    get_cached_value,
    set_cached_value,
    REDIS_NAMESPACE_PREFIX,
)


# =============================================================================
# Test _build_cache_key
# =============================================================================


class TestBuildCacheKey:
    """Tests for the _build_cache_key function."""

    def test_basic_key_construction(self):
        """Test basic key construction with namespace and cache_key."""
        key = _build_cache_key("namespace", "cache_key")
        assert key == f"{REDIS_NAMESPACE_PREFIX}:namespace:cache_key"

    def test_key_with_special_characters(self):
        """Test key construction with special characters."""
        key = _build_cache_key("db:table", "query:hash:123")
        assert key == f"{REDIS_NAMESPACE_PREFIX}:db:table:query:hash:123"

    def test_empty_namespace(self):
        """Test key construction with empty namespace."""
        key = _build_cache_key("", "key")
        assert key == f"{REDIS_NAMESPACE_PREFIX}::key"

    def test_empty_cache_key(self):
        """Test key construction with empty cache_key."""
        key = _build_cache_key("namespace", "")
        assert key == f"{REDIS_NAMESPACE_PREFIX}:namespace:"

    def test_namespace_prefix_value(self):
        """Test that the namespace prefix is as expected."""
        assert REDIS_NAMESPACE_PREFIX == "berdl-mcp"


# =============================================================================
# Test get_cached_value
# =============================================================================


class TestGetCachedValue:
    """Tests for the get_cached_value function."""

    def test_cache_hit_returns_data(self, mock_redis_client):
        """Test that cache hit returns parsed JSON data."""
        test_data = [{"id": 1, "name": "test"}]
        cached_json = json.dumps(test_data).encode()
        client = mock_redis_client(
            {f"{REDIS_NAMESPACE_PREFIX}:test_ns:test_key": cached_json}
        )

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("test_ns", "test_key")

        assert result == test_data

    def test_cache_miss_returns_none(self, mock_redis_client):
        """Test that cache miss returns None."""
        client = mock_redis_client({})  # Empty cache

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("test_ns", "nonexistent_key")

        assert result is None

    def test_redis_client_unavailable_returns_none(self):
        """Test that unavailable Redis client returns None."""
        with patch("src.cache.redis_cache._get_redis_client", return_value=None):
            result = get_cached_value("test_ns", "test_key")

        assert result is None

    def test_redis_error_returns_none(self, mock_redis_client):
        """Test that Redis errors are caught and return None."""
        client = mock_redis_client({})
        client.get.side_effect = RedisError("Connection failed")

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("test_ns", "test_key")

        assert result is None

    def test_json_decode_error_returns_none(self, mock_redis_client):
        """Test that invalid JSON returns None."""
        client = mock_redis_client(
            {f"{REDIS_NAMESPACE_PREFIX}:test_ns:test_key": b"invalid json{"}
        )

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("test_ns", "test_key")

        assert result is None

    def test_handles_bytes_response(self, mock_redis_client):
        """Test that bytes response is decoded correctly."""
        test_data = [{"count": 42}]
        client = mock_redis_client(
            {f"{REDIS_NAMESPACE_PREFIX}:count:hash123": json.dumps(test_data).encode()}
        )

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("count", "hash123")

        assert result == test_data

    def test_handles_string_response(self, mock_redis_client):
        """Test that string response is handled correctly."""
        # Some Redis clients might return strings instead of bytes
        test_data = [{"key": "value"}]
        client = MagicMock()
        client.get.return_value = json.dumps(test_data)  # String, not bytes

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            result = get_cached_value("test_ns", "test_key")

        assert result == test_data

    def test_correct_key_used_for_lookup(self, mock_redis_client):
        """Test that the correct formatted key is used."""
        client = mock_redis_client({})

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            get_cached_value("my_namespace", "my_key")

        expected_key = f"{REDIS_NAMESPACE_PREFIX}:my_namespace:my_key"
        client.get.assert_called_once_with(expected_key)


# =============================================================================
# Test set_cached_value
# =============================================================================


class TestSetCachedValue:
    """Tests for the set_cached_value function."""

    def test_successful_cache_write(self, mock_redis_client):
        """Test successful write to cache."""
        client = mock_redis_client({})

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("test_ns", "test_key", [{"data": "value"}], ttl=3600)

        expected_key = f"{REDIS_NAMESPACE_PREFIX}:test_ns:test_key"
        client.set.assert_called_once()
        call_args = client.set.call_args
        assert call_args[1]["name"] == expected_key
        assert call_args[1]["ex"] == 3600

    def test_data_serialized_as_json(self, mock_redis_client):
        """Test that data is serialized as JSON."""
        client = mock_redis_client({})
        test_data = [{"id": 1, "name": "test", "nested": {"key": "value"}}]

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("test_ns", "test_key", test_data, ttl=3600)

        call_args = client.set.call_args
        stored_value = call_args[1]["value"]
        assert json.loads(stored_value) == test_data

    def test_redis_client_unavailable_no_error(self):
        """Test that unavailable Redis client doesn't raise error."""
        with patch("src.cache.redis_cache._get_redis_client", return_value=None):
            # Should not raise
            set_cached_value("test_ns", "test_key", [{"data": "value"}], ttl=3600)

    def test_redis_error_caught_gracefully(self, mock_redis_client):
        """Test that Redis errors are caught gracefully."""
        client = mock_redis_client({})
        client.set.side_effect = RedisError("Connection failed")

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            # Should not raise
            set_cached_value("test_ns", "test_key", [{"data": "value"}], ttl=3600)

    def test_type_error_caught_gracefully(self, mock_redis_client):
        """Test that serialization errors are caught gracefully."""
        client = mock_redis_client({})

        # Create non-serializable data
        class NonSerializable:
            pass

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            # Should not raise
            set_cached_value(
                "test_ns", "test_key", [{"obj": NonSerializable()}], ttl=3600
            )

    def test_ttl_passed_correctly(self, mock_redis_client):
        """Test that TTL is passed to Redis correctly."""
        client = mock_redis_client({})

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("test_ns", "test_key", [{"data": "value"}], ttl=7200)

        call_args = client.set.call_args
        assert call_args[1]["ex"] == 7200


# =============================================================================
# Test _get_redis_client
# =============================================================================


class TestGetRedisClient:
    """Tests for the _get_redis_client function."""

    def test_client_is_cached(self):
        """Test that the Redis client is cached (lru_cache)."""
        # Clear the cache first
        _get_redis_client.cache_clear()

        with patch("src.cache.redis_cache.redis.ConnectionPool") as mock_pool:
            with patch("src.cache.redis_cache.redis.Redis") as mock_redis:
                mock_redis.return_value = MagicMock()

                # Call twice
                client1 = _get_redis_client()
                client2 = _get_redis_client()

                # Should only create one instance
                assert mock_pool.call_count == 1
                assert mock_redis.call_count == 1
                assert client1 is client2

        # Clear cache for other tests
        _get_redis_client.cache_clear()

    def test_returns_none_on_redis_error(self):
        """Test that RedisError returns None."""
        _get_redis_client.cache_clear()

        with patch(
            "src.cache.redis_cache.redis.ConnectionPool", side_effect=RedisError("err")
        ):
            client = _get_redis_client()
            assert client is None

        _get_redis_client.cache_clear()


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentCacheAccess:
    """Tests for concurrent cache access."""

    def test_concurrent_reads_thread_safe(self, mock_redis_client, concurrent_executor):
        """Test that concurrent reads are thread-safe."""
        test_data = [{"id": 1}]
        client = mock_redis_client(
            {f"{REDIS_NAMESPACE_PREFIX}:test:key": json.dumps(test_data).encode()}
        )

        def read_cache(_):
            with patch("src.cache.redis_cache._get_redis_client", return_value=client):
                return get_cached_value("test", "key")

        args_list = [(i,) for i in range(20)]
        results, exceptions = concurrent_executor(read_cache, args_list, max_workers=10)

        assert len(exceptions) == 0
        assert all(r == test_data for r in results)

    def test_concurrent_writes_thread_safe(
        self, mock_redis_client, concurrent_executor
    ):
        """Test that concurrent writes are thread-safe."""
        client = mock_redis_client({})
        write_count = {"value": 0}
        lock = threading.Lock()

        def counting_set(name, value, ex=None):
            with lock:
                write_count["value"] += 1
            client._stored_data[name] = (
                value.encode() if isinstance(value, str) else value
            )
            return True

        client.set.side_effect = counting_set

        def write_cache(i):
            with patch("src.cache.redis_cache._get_redis_client", return_value=client):
                set_cached_value("test", f"key_{i}", [{"id": i}], ttl=3600)
            return True

        args_list = [(i,) for i in range(20)]
        results, exceptions = concurrent_executor(
            write_cache, args_list, max_workers=10
        )

        assert len(exceptions) == 0
        assert write_count["value"] == 20

    def test_concurrent_read_write_mixed(self, mock_redis_client, concurrent_executor):
        """Test mixed concurrent reads and writes."""
        initial_data = {
            f"{REDIS_NAMESPACE_PREFIX}:test:existing": json.dumps(
                [{"existing": True}]
            ).encode()
        }
        client = mock_redis_client(initial_data)

        def mixed_operation(i):
            with patch("src.cache.redis_cache._get_redis_client", return_value=client):
                if i % 2 == 0:
                    # Read operation
                    return ("read", get_cached_value("test", "existing"))
                else:
                    # Write operation
                    set_cached_value("test", f"new_{i}", [{"id": i}], ttl=3600)
                    return ("write", True)

        args_list = [(i,) for i in range(20)]
        results, exceptions = concurrent_executor(
            mixed_operation, args_list, max_workers=10
        )

        assert len(exceptions) == 0
        reads = [r for r in results if r[0] == "read"]
        writes = [r for r in results if r[0] == "write"]
        assert len(reads) == 10
        assert len(writes) == 10


# =============================================================================
# Integration-like Tests (with mocks)
# =============================================================================


class TestCacheIntegration:
    """Integration-style tests for cache operations."""

    def test_write_then_read_returns_same_data(self, mock_redis_client):
        """Test that writing then reading returns the same data."""
        client = mock_redis_client({})

        test_data = [{"id": 1, "name": "test", "values": [1, 2, 3]}]

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("integration", "test_key", test_data, ttl=3600)
            result = get_cached_value("integration", "test_key")

        assert result == test_data

    def test_different_namespaces_isolated(self, mock_redis_client):
        """Test that different namespaces are isolated."""
        client = mock_redis_client({})

        data1 = [{"ns": "one"}]
        data2 = [{"ns": "two"}]

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("namespace_one", "key", data1, ttl=3600)
            set_cached_value("namespace_two", "key", data2, ttl=3600)

            result1 = get_cached_value("namespace_one", "key")
            result2 = get_cached_value("namespace_two", "key")

        assert result1 == data1
        assert result2 == data2

    def test_overwrite_existing_key(self, mock_redis_client):
        """Test that writing to existing key overwrites."""
        client = mock_redis_client({})

        original_data = [{"version": 1}]
        updated_data = [{"version": 2}]

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("overwrite", "key", original_data, ttl=3600)
            set_cached_value("overwrite", "key", updated_data, ttl=3600)
            result = get_cached_value("overwrite", "key")

        assert result == updated_data

    def test_complex_nested_data_structure(self, mock_redis_client):
        """Test caching complex nested data structures."""
        client = mock_redis_client({})

        complex_data = [
            {
                "id": 1,
                "nested": {"level1": {"level2": {"value": "deep"}}},
                "array": [1, 2, {"inner": True}],
                "null_value": None,
                "bool_value": False,
            }
        ]

        with patch("src.cache.redis_cache._get_redis_client", return_value=client):
            set_cached_value("complex", "data", complex_data, ttl=3600)
            result = get_cached_value("complex", "data")

        assert result == complex_data
