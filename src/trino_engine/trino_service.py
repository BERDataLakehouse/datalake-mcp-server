"""
Trino query execution service layer.

Mirrors ``delta_lake/delta_service.py`` but uses a Trino DB-API cursor
instead of a SparkSession.  Reuses engine-agnostic helpers from
``delta_service`` (SQL validation, query builder, cache layer).
"""

import hashlib
import json
import logging
import re
from typing import Any

import sqlparse
from sqlparse import tokens as T
import trino

from src.cache.redis_cache import get_cached_value, set_cached_value
from src.delta_lake.delta_service import (
    MAX_QUERY_ROWS,
    MAX_SAMPLE_ROWS,
    CACHE_EXPIRY_SECONDS,
    _check_query_is_valid,
    _is_non_paginatable_query,
    _validate_identifier,
    build_select_query as _build_select_query_spark,
    _check_exists,
)
from src.service.exceptions import (
    SparkQueryError,
    TrinoOperationError,
    TrinoQueryError,
)
from src.service.models import (
    PaginationInfo,
    TableQueryResponse,
    TableSelectRequest,
    TableSelectResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------


def _validate_trino_identifier(name: str, identifier_type: str = "identifier") -> None:
    """Validate a SQL identifier, raising TrinoQueryError on failure."""
    try:
        _validate_identifier(name, identifier_type)
    except SparkQueryError as e:
        raise TrinoQueryError(str(e)) from e


# ---------------------------------------------------------------------------
# Trino SQL helpers
# ---------------------------------------------------------------------------


def _spark_sql_to_trino(sql: str) -> str:
    """Convert Spark SQL quoting/pagination to Trino dialect.

    - Backtick identifiers → double-quote identifiers (token-aware,
      leaves string literals untouched)
    - ``LIMIT n OFFSET m`` → ``OFFSET m LIMIT n``
    """
    parsed = sqlparse.parse(sql)[0]
    parts: list[str] = []
    for token in parsed.flatten():
        value = token.value
        if token.ttype in (T.Name, T.Name.Placeholder) and "`" in value:
            value = value.replace("`", '"')
        parts.append(value)
    sql = "".join(parts)
    sql = re.sub(
        r"\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)",
        r"OFFSET \2 LIMIT \1",
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def build_select_query_trino(
    request: TableSelectRequest, include_pagination: bool = True
) -> str:
    """Build a SELECT query using Trino SQL dialect."""
    return _spark_sql_to_trino(
        _build_select_query_spark(request, include_pagination=include_pagination)
    )


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


def _cursor_to_dicts(cursor: trino.dbapi.Cursor) -> list[dict[str, Any]]:
    """Convert Trino cursor results (tuples + description) to list of dicts."""
    rows = cursor.fetchall()
    if not rows:
        return []
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Cache helpers (same logic as delta_service, scoped per-user)
# ---------------------------------------------------------------------------


def _generate_cache_key(params: dict[str, Any], username: str | None = None) -> str:
    if username:
        params = {"_username": username, "_engine": "trino", **params}
    else:
        params = {"_engine": "trino", **params}
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def _get_from_cache(namespace: str, cache_key: str) -> list[dict[str, Any]] | None:
    return get_cached_value(namespace=namespace, cache_key=cache_key)


def _store_in_cache(
    namespace: str,
    cache_key: str,
    data: list[dict[str, Any]],
    ttl: int = CACHE_EXPIRY_SECONDS,
) -> None:
    set_cached_value(namespace=namespace, cache_key=cache_key, data=data, ttl=ttl)


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _execute_non_paginatable_query_trino(
    conn: trino.dbapi.Connection,
    query: str,
    username: str | None = None,
) -> TableQueryResponse:
    """Execute a metadata query (DESCRIBE / SHOW / EXPLAIN) via Trino."""
    base_query = query.rstrip()

    namespace = "metadata_query"
    params = {"query": base_query}
    cache_key = _generate_cache_key(params, username=username)

    cached = _get_from_cache(namespace, cache_key)
    if cached:
        logger.info(f"Cache hit for metadata query: {base_query[:50]}...")
        return TableQueryResponse(
            result=cached[0]["result"],
            pagination=PaginationInfo(**cached[0]["pagination"]),
        )

    logger.info(f"Executing Trino metadata query: {base_query[:100]}...")

    try:
        cursor = conn.cursor()
        cursor.execute(base_query)
        results = _cursor_to_dicts(cursor)
        logger.info(f"Metadata query returned {len(results)} rows.")

        pagination = PaginationInfo(
            limit=len(results),
            offset=0,
            total_count=len(results),
            has_more=False,
        )
        response = TableQueryResponse(result=results, pagination=pagination)

        _store_in_cache(
            namespace,
            cache_key,
            [{"result": results, "pagination": pagination.model_dump()}],
        )
        return response

    except Exception as e:
        logger.error(f"Error executing Trino metadata query: {e}")
        raise TrinoOperationError(f"Failed to execute metadata query: {e}") from e


def query_via_trino(
    conn: trino.dbapi.Connection,
    query: str,
    limit: int = 1000,
    offset: int = 0,
    username: str | None = None,
    max_rows: int = MAX_QUERY_ROWS,
) -> TableQueryResponse:
    """
    Execute a SQL query via Trino with pagination.

    Reuses ``_check_query_is_valid()`` from delta_service (engine-agnostic).
    """
    # Translate Spark SQL dialect (backticks, LIMIT/OFFSET order) to Trino
    query = _spark_sql_to_trino(query)

    _check_query_is_valid(query)

    if _is_non_paginatable_query(query):
        return _execute_non_paginatable_query_trino(conn, query, username=username)

    if limit > max_rows:
        raise TrinoQueryError(
            f"Limit ({limit}) exceeds maximum allowed ({max_rows}). "
            f"Please reduce your limit."
        )

    base_query = query.rstrip().rstrip(";")
    base_query = re.sub(
        r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*$",
        "",
        base_query,
        flags=re.IGNORECASE,
    )
    base_query = re.sub(
        r"\s+OFFSET\s+\d+(\s+LIMIT\s+\d+)?\s*$",
        "",
        base_query,
        flags=re.IGNORECASE,
    )

    if offset > 0 and not re.search(r"\bORDER\s+BY\b", base_query, flags=re.IGNORECASE):
        logger.warning(
            f"Pagination with offset={offset} but query has no ORDER BY. "
            "Results may be non-deterministic across pages."
        )

    namespace = "query"
    params = {"query": base_query, "limit": limit, "offset": offset}
    cache_key = _generate_cache_key(params, username=username)

    count_namespace = "query_count"
    count_cache_key = _generate_cache_key({"query": base_query}, username=username)

    cached = _get_from_cache(namespace, cache_key)
    if cached:
        logger.info(
            f"Cache hit for query: {base_query[:50]}{'...' if len(base_query) > 50 else ''}"
        )
        return TableQueryResponse(
            result=cached[0]["result"],
            pagination=PaginationInfo(**cached[0]["pagination"]),
        )

    logger.info(f"Executing Trino paginated query: {base_query[:100]}...")

    try:
        cursor = conn.cursor()
        paginated_query = f"{base_query} OFFSET {offset} LIMIT {limit}"
        cursor.execute(paginated_query)
        results = _cursor_to_dicts(cursor)
        logger.info(f"Trino query returned {len(results)} rows.")

        if len(results) < limit and len(results) > 0:
            # Incomplete page with results — we know the exact total without
            # an expensive COUNT scan (matches Spark implementation)
            total_count = offset + len(results)
            has_more = False
        elif len(results) == 0 and offset == 0:
            # Empty table / no matching rows
            total_count = 0
            has_more = False
        else:
            # Full page or empty page with offset > 0 — need accurate count
            cached_count = _get_from_cache(count_namespace, count_cache_key)
            if cached_count:
                total_count = cached_count[0]["count"]
            else:
                count_query = f"SELECT COUNT(*) AS cnt FROM ({base_query}) AS subquery"
                cursor.execute(count_query)
                total_count = cursor.fetchone()[0]
                _store_in_cache(
                    count_namespace, count_cache_key, [{"count": total_count}]
                )
            has_more = (offset + len(results)) < total_count

        pagination = PaginationInfo(
            limit=limit,
            offset=offset,
            total_count=total_count,
            has_more=has_more,
        )
        response = TableQueryResponse(result=results, pagination=pagination)

        _store_in_cache(
            namespace,
            cache_key,
            [{"result": results, "pagination": pagination.model_dump()}],
        )
        return response

    except (TrinoQueryError, TrinoOperationError):
        raise
    except Exception as e:
        logger.error(f"Error executing Trino query: {e}")
        raise TrinoOperationError(f"Failed to execute query: {e}") from e


def count_via_trino(
    conn: trino.dbapi.Connection,
    database: str,
    table: str,
    username: str | None = None,
) -> int:
    """Count rows in a table via Trino."""
    namespace = "count"
    params = {"database": database, "table": table}
    cache_key = _generate_cache_key(params, username=username)

    cached = _get_from_cache(namespace, cache_key)
    if cached:
        logger.info(f"Cache hit for count on {database}.{table}")
        return cached[0]["count"]

    _validate_trino_identifier(database, "database")
    _validate_trino_identifier(table, "table")
    _check_exists(database, table)
    full_table_name = f'"{database}"."{table}"'
    logger.info(f"Counting rows in {full_table_name} via Trino")

    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {full_table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"{full_table_name} has {count} rows.")

        _store_in_cache(namespace, cache_key, [{"count": count}])
        return count

    except Exception as e:
        logger.error(f"Error counting rows in {full_table_name}: {e}")
        raise TrinoOperationError(
            f"Failed to count rows in {full_table_name}: {e}"
        ) from e


def sample_via_trino(
    conn: trino.dbapi.Connection,
    database: str,
    table: str,
    limit: int = 10,
    columns: list[str] | None = None,
    where_clause: str | None = None,
    username: str | None = None,
) -> list[dict[str, Any]]:
    """Sample rows from a table via Trino."""
    namespace = "sample"
    params = {
        "database": database,
        "table": table,
        "limit": limit,
        "columns": sorted(columns) if columns else None,
        "where_clause": where_clause,
    }
    cache_key = _generate_cache_key(params, username=username)

    cached = _get_from_cache(namespace, cache_key)
    if cached:
        logger.info(f"Cache hit for sample on {database}.{table}")
        return cached

    if not 0 < limit <= MAX_SAMPLE_ROWS:
        raise ValueError(f"Limit must be between 1 and {MAX_SAMPLE_ROWS}, got {limit}")

    _validate_trino_identifier(database, "database")
    _validate_trino_identifier(table, "table")
    if columns:
        for col in columns:
            _validate_trino_identifier(col, "column")

    _check_exists(database, table)
    full_table_name = f'"{database}"."{table}"'
    logger.info(f"Sampling {limit} rows from {full_table_name} via Trino")

    try:
        col_expr = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql = f"SELECT {col_expr} FROM {full_table_name}"

        if where_clause:
            equivalent_query = f"SELECT * FROM {full_table_name} WHERE {where_clause}"
            _check_query_is_valid(equivalent_query)
            sql += f" WHERE {where_clause}"

        sql += f" LIMIT {limit}"

        cursor = conn.cursor()
        cursor.execute(sql)
        results = _cursor_to_dicts(cursor)
        logger.info(f"Sampled {len(results)} rows.")

        _store_in_cache(namespace, cache_key, results)
        return results

    except Exception as e:
        logger.error(f"Error sampling rows from {full_table_name}: {e}")
        raise TrinoOperationError(
            f"Failed to sample rows from {full_table_name}: {e}"
        ) from e


def select_via_trino(
    conn: trino.dbapi.Connection,
    request: TableSelectRequest,
    username: str | None = None,
) -> TableSelectResponse:
    """Execute a structured SELECT query via Trino."""
    namespace = "select"
    params = request.model_dump()
    # Remove engine field from cache key params (not relevant to caching)
    params.pop("engine", None)
    cache_key = _generate_cache_key(params, username=username)

    cached = _get_from_cache(namespace, cache_key)
    if cached:
        logger.info(f"Cache hit for select on {request.database}.{request.table}")
        return TableSelectResponse(
            data=cached[0]["data"],
            pagination=PaginationInfo(**cached[0]["pagination"]),
        )

    _check_exists(request.database, request.table)

    # Count query (without pagination)
    count_query = (
        f"SELECT COUNT(*) AS cnt FROM "
        f"({build_select_query_trino(request, include_pagination=False)})"
    )
    logger.info(f"Executing Trino count query: {count_query}")

    try:
        cursor = conn.cursor()
        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]
    except Exception as e:
        logger.error(f"Error executing Trino count query: {e}")
        raise TrinoOperationError(f"Failed to execute count query: {e}") from e

    # Main query with pagination
    main_query = build_select_query_trino(request, include_pagination=True)
    logger.info(f"Executing Trino select query: {main_query}")

    try:
        cursor = conn.cursor()
        cursor.execute(main_query)
        results = _cursor_to_dicts(cursor)
        logger.info(f"Trino select query returned {len(results)} rows.")

        has_more = (request.offset + len(results)) < total_count
        pagination = PaginationInfo(
            limit=request.limit,
            offset=request.offset,
            total_count=total_count,
            has_more=has_more,
        )

        response = TableSelectResponse(data=results, pagination=pagination)

        _store_in_cache(
            namespace,
            cache_key,
            [{"data": results, "pagination": pagination.model_dump()}],
        )
        return response

    except Exception as e:
        logger.error(f"Error executing Trino select query: {e}")
        raise TrinoOperationError(f"Failed to execute select query: {e}") from e
