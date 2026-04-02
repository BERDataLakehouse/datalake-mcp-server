"""
Shared query execution logic for both sync and async query endpoints.

This module provides the SINGLE source of truth for dispatching SQL queries
to either Standalone (subprocess) or Spark Connect mode. Both the sync
query_table route and the AsyncQueryExecutor call this function.

Timeouts are controlled by two env vars (see service/timeouts.py and
service/spark_session_pool.py):
  - SPARK_CONNECT_QUERY_TIMEOUT: timeout around df.collect() in Connect mode
  - SPARK_STANDALONE_QUERY_TIMEOUT: timeout for the entire subprocess in Standalone mode
Both apply identically to sync and async code paths.
"""

from __future__ import annotations

import logging
from typing import Any

import trino

from src.delta_lake import delta_service
from src.delta_lake.delta_service import MAX_QUERY_ROWS
from src.service.dependencies import SparkContext
from src.service.models import PaginationInfo, TableQueryResponse
from src.service.spark_session_pool import run_in_spark_process
from src.service.standalone_operations import query_table_subprocess
from src.trino_engine import trino_service

logger = logging.getLogger(__name__)


def execute_query(
    ctx: SparkContext,
    query: str,
    limit: int,
    offset: int,
    username: str | None = None,
    max_rows: int = MAX_QUERY_ROWS,
    operation_name: str = "query_table",
) -> TableQueryResponse:
    """
    Execute a SQL query using the appropriate Spark mode.

    This is a blocking function. The sync route calls it directly;
    the async executor wraps it in asyncio.to_thread.

    Timeouts are handled by the mode-specific defaults:
      - Standalone: SPARK_STANDALONE_QUERY_TIMEOUT (pool-level, covers full subprocess)
      - Connect: SPARK_CONNECT_QUERY_TIMEOUT (around df.collect() in delta_service)

    Args:
        ctx: SparkContext with session and mode info.
        query: SQL query string.
        limit: Max rows to return.
        offset: Pagination offset.
        username: Username for cache isolation.
        max_rows: Maximum allowed row limit.
        operation_name: Name for logging/metrics.

    Returns:
        TableQueryResponse with result rows and pagination info.
    """
    if ctx.is_standalone_subprocess:
        result: dict[str, Any] = run_in_spark_process(
            query_table_subprocess,
            ctx.settings_dict,
            query=query,
            limit=limit,
            offset=offset,
            username=username,
            app_name=ctx.app_name,
            max_rows=max_rows,
            operation_name=operation_name,
        )
        return TableQueryResponse(
            result=result["result"],
            pagination=PaginationInfo(**result["pagination"]),
        )
    else:
        return delta_service.query_delta_table(
            spark=ctx.spark,
            query=query,
            limit=limit,
            offset=offset,
            username=username,
            max_rows=max_rows,
        )


def execute_query_trino(
    conn: trino.dbapi.Connection,
    query: str,
    limit: int,
    offset: int,
    username: str | None = None,
    max_rows: int = MAX_QUERY_ROWS,
) -> TableQueryResponse:
    """
    Execute a SQL query via Trino.

    Blocking function — the async executor wraps it in asyncio.to_thread.
    """
    return trino_service.query_via_trino(
        conn=conn,
        query=query,
        limit=limit,
        offset=offset,
        username=username,
        max_rows=max_rows,
    )
