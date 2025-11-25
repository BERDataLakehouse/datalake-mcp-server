"""
Utility helpers for interacting with Redis as a distributed cache backend.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, cast

import redis
from redis.exceptions import RedisError

from src.settings import get_settings

logger = logging.getLogger(__name__)

CachePayload = List[Dict[str, Any]]
REDIS_NAMESPACE_PREFIX = "berdl-mcp"


def _build_cache_key(namespace: str, cache_key: str) -> str:
    return f"{REDIS_NAMESPACE_PREFIX}:{namespace}:{cache_key}"


@lru_cache(maxsize=1)
def _get_redis_client() -> Optional[redis.Redis]:
    """
    Lazily create a Redis client backed by a connection pool.
    """
    try:
        settings = get_settings()
        logger.info(
            "Initializing Redis client host=%s port=%s",
            settings.BERDL_REDIS_HOST,
            settings.BERDL_REDIS_PORT,
        )
        pool = redis.ConnectionPool(
            host=settings.BERDL_REDIS_HOST, port=settings.BERDL_REDIS_PORT
        )
        return redis.Redis(connection_pool=pool)
    except RedisError:
        logger.exception("Failed to establish Redis connection pool; caching disabled.")
        return None


def get_cached_value(namespace: str, cache_key: str) -> Optional[CachePayload]:
    """
    Retrieve a cached payload from Redis.
    """
    client = _get_redis_client()
    if client is None:
        logger.info(
            "Redis client unavailable; cache read skipped for namespace=%s key=%s",
            namespace,
            cache_key,
        )
        return None

    redis_key = _build_cache_key(namespace, cache_key)

    try:
        logger.info("Reading cache namespace=%s key=%s", namespace, cache_key)
        raw_value = client.get(redis_key)
        if raw_value is None:
            logger.info(
                "Cache miss for namespace=%s key=%s (redis key=%s)",
                namespace,
                cache_key,
                redis_key,
            )
            return None
        if isinstance(raw_value, bytes):
            decoded_value = raw_value.decode("utf-8")
        else:
            decoded_value = cast(str, raw_value)
        logger.info("Cache hit for namespace=%s key=%s", namespace, cache_key)
        return json.loads(decoded_value)
    except (RedisError, json.JSONDecodeError):
        logger.exception(
            "Redis connection error while fetching namespace=%s key=%s",
            namespace,
            cache_key,
        )
        return None


def set_cached_value(
    namespace: str,
    cache_key: str,
    data: CachePayload,
    ttl: int,
) -> None:
    """
    Store a payload in Redis with the provided TTL.
    """
    client = _get_redis_client()
    if client is None:
        logger.info(
            "Redis client unavailable; cache write skipped for namespace=%s key=%s",
            namespace,
            cache_key,
        )
        return

    redis_key = _build_cache_key(namespace, cache_key)

    try:
        payload = json.dumps(data)
        client.set(name=redis_key, value=payload, ex=ttl)
        logger.info(
            "Cached value namespace=%s key=%s ttl=%ss", namespace, cache_key, ttl
        )
    except (TypeError, RedisError):
        logger.exception(
            "Failed to cache value namespace=%s key=%s", namespace, cache_key
        )
