"""
Service layer for interacting with Delta Lake tables via Spark.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

import sqlparse
from pyspark.sql import SparkSession

from src.cache.redis_cache import get_cached_value, set_cached_value
from src.delta_lake.data_store import database_exists, table_exists
from src.service.exceptions import (
    DeltaDatabaseNotFoundError,
    DeltaTableNotFoundError,
    SparkOperationError,
    SparkQueryError,
    SparkTimeoutError,
)
from src.service.timeouts import run_with_timeout, DEFAULT_SPARK_COLLECT_TIMEOUT
from src.service.models import (
    AggregationSpec,
    ColumnSpec,
    FilterCondition,
    JoinClause,
    PaginationInfo,
    TableQueryResponse,
    TableSelectRequest,
    TableSelectResponse,
)

logger = logging.getLogger(__name__)

# Row limits to prevent OOM and ensure service stability
MAX_SAMPLE_ROWS = 1000
MAX_QUERY_ROWS = 50000  # Maximum rows returned by arbitrary SQL queries
MAX_SELECT_ROWS = 10000  # Maximum rows for structured SELECT (enforced in model)
CACHE_EXPIRY_SECONDS = 3600  # Cache results for 1 hour by default

# Common SQL keywords that might indicate destructive operations
FORBIDDEN_KEYWORDS = {
    # NOTE: This might create false positives, legitemate queries might include these keywords
    # e.g. "SELECT * FROM orders ORDER BY created_at DESC"
    "drop",
    "truncate",
    "delete",
    "insert",
    "update",
    "create",
    "alter",
    "merge",
    "replace",
    "rename",
    "vacuum",
}

DISALLOW_SQL_META_CHARS = {
    "--",
    "/*",
    "*/",
    ";",
    "\\",
}

ALLOWED_STATEMENTS = {
    "select",
}

FORBIDDEN_POSTGRESQL_SCHEMAS = {
    # NOTE: This might create false positives, legitemate queries might include these schemas
    # e.g. "SELECT * FROM jpg_files"
    # NOTE: may also need to expand this if other databases are used
    "pg_",
    "pg_catalog",
    "information_schema",
}


def _extract_limit_from_query(query: str) -> Optional[int]:
    """
    Extract the LIMIT value from a SQL query if present.

    Args:
        query: SQL query string

    Returns:
        The LIMIT value as int, or None if no LIMIT clause found
    """
    # Simple regex to find LIMIT clause - handles most common cases
    # Pattern matches: LIMIT <number> with optional whitespace
    limit_pattern = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
    match = limit_pattern.search(query)
    if match:
        return int(match.group(1))
    return None


def _enforce_query_limit(query: str, max_rows: int = MAX_QUERY_ROWS) -> str:
    """
    Ensure a SQL query has a LIMIT clause that doesn't exceed max_rows.

    If the query has no LIMIT, adds one. If it has a LIMIT > max_rows,
    raises an error to inform the user.

    Args:
        query: SQL query string
        max_rows: Maximum allowed rows (default: MAX_QUERY_ROWS)

    Returns:
        Query with enforced LIMIT clause

    Raises:
        SparkQueryError: If query has LIMIT exceeding max_rows
    """
    existing_limit = _extract_limit_from_query(query)

    if existing_limit is not None:
        if existing_limit > max_rows:
            raise SparkQueryError(
                f"Query LIMIT ({existing_limit}) exceeds maximum allowed ({max_rows}). "
                f"Please reduce your LIMIT or use pagination."
            )
        # Query already has acceptable LIMIT
        return query

    # No LIMIT clause - add one
    # Strip trailing whitespace and add LIMIT
    query = query.rstrip()
    logger.info(f"Adding LIMIT {max_rows} to query without explicit limit")
    return f"{query} LIMIT {max_rows}"


def _check_query_is_valid(query: str) -> bool:
    """
    Check if a query is valid.

    Please note that this function is not a comprehensive SQL query validator.
    It only checks for basic syntax and structure.
    MCP server should be configured to use read-only user for both PostgreSQL and MinIO.
    """

    try:
        # NOTE: sqlparse does not validate SQL syntax; what happens with unexpected syntax is unknown
        statements = sqlparse.parse(query)
    except Exception as e:
        raise SparkQueryError(f"Query {query} is not a valid SQL query: {e}")

    if len(statements) != 1:
        raise SparkQueryError(f"Query {query} must contain exactly one statement")

    statement = statements[0]
    # NOTE: statement might have subqueries, we only check the main statement here!
    if statement.get_type().lower() not in ALLOWED_STATEMENTS:
        raise SparkQueryError(
            f"Query {query} must be one of the following: {', '.join(ALLOWED_STATEMENTS)}, got {statement.get_type()}"
        )

    if any(schema in query.lower() for schema in FORBIDDEN_POSTGRESQL_SCHEMAS):
        raise SparkQueryError(
            f"Query {query} contains forbidden PostgreSQL schema: {', '.join(FORBIDDEN_POSTGRESQL_SCHEMAS)}"
        )

    if any(char in query for char in DISALLOW_SQL_META_CHARS):
        raise SparkQueryError(
            f"Query {query} contains disallowed metacharacter: {', '.join(char for char in DISALLOW_SQL_META_CHARS if char in query)}"
        )

    if any(keyword in query.lower() for keyword in FORBIDDEN_KEYWORDS):
        raise SparkQueryError(
            f"Query {query} contains forbidden keyword: {', '.join(keyword for keyword in FORBIDDEN_KEYWORDS if keyword in query.lower())}"
        )

    return True


def _check_exists(database: str, table: str) -> bool:
    """
    Check if a table exists in a database.
    """
    if not database_exists(database):
        raise DeltaDatabaseNotFoundError(f"Database [{database}] not found")
    if not table_exists(database, table):
        raise DeltaTableNotFoundError(
            f"Table [{table}] not found in database [{database}]"
        )
    return True


def _generate_cache_key(params: Dict[str, Any]) -> str:
    """
    Generate a cache key from parameters.
    """
    # Convert parameters to a sorted JSON string to ensure consistency
    param_str = json.dumps(params, sort_keys=True)
    # Create a hash of the parameters to avoid very long keys
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return param_hash


def _get_from_cache(namespace: str, cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Try to get data from Redis cache.
    """
    return get_cached_value(namespace=namespace, cache_key=cache_key)


def _store_in_cache(
    namespace: str,
    cache_key: str,
    data: List[Dict[str, Any]],
    ttl: int = CACHE_EXPIRY_SECONDS,
) -> None:
    """
    Store data in Redis cache.
    """
    set_cached_value(namespace=namespace, cache_key=cache_key, data=data, ttl=ttl)


def count_delta_table(
    spark: SparkSession, database: str, table: str, use_cache: bool = True
) -> int:
    """
    Counts the number of rows in a specific Delta table.

    Args:
        spark: The SparkSession object.
        database: The database (namespace) containing the table.
        table: The name of the Delta table.
        use_cache: Whether to use the redis cache to store the result.

    Returns:
        The number of rows in the table.
    """

    namespace = "count"
    params = {"database": database, "table": table}
    cache_key = _generate_cache_key(params)

    if use_cache:
        cached_result = _get_from_cache(namespace, cache_key)
        if cached_result:
            logger.info(f"Cache hit for count on {database}.{table}")
            return cached_result[0]["count"]

    _check_exists(database, table)
    full_table_name = f"`{database}`.`{table}`"
    logger.info(f"Counting rows in {full_table_name}")
    try:
        # Use timeout wrapper for count operation
        count = run_with_timeout(
            lambda: spark.table(full_table_name).count(),
            timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
            operation_name=f"count_{database}.{table}",
        )
        logger.info(f"{full_table_name} has {count} rows.")

        if use_cache:
            _store_in_cache(namespace, cache_key, [{"count": count}])

        return count
    except SparkTimeoutError:
        raise  # Re-raise timeout errors as-is
    except Exception as e:
        logger.error(f"Error counting rows in {full_table_name}: {e}")
        raise SparkOperationError(
            f"Failed to count rows in {full_table_name}: {str(e)}"
        )


def sample_delta_table(
    spark: SparkSession,
    database: str,
    table: str,
    limit: int = 10,
    columns: List[str] | None = None,
    where_clause: str | None = None,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieves a sample of rows from a specific Delta table.

    Args:
        spark: The SparkSession object.
        database: The database (namespace) containing the table.
        table: The name of the Delta table.
        limit: The maximum number of rows to return.
        columns: The columns to return. If None, all columns will be returned.
        where_clause: A SQL WHERE clause to filter the rows. e.g. "id > 100"
        use_cache: Whether to use the redis cache to store the result.

    Returns:
        A list of dictionaries, where each dictionary represents a row.
    """
    namespace = "sample"
    params = {
        "database": database,
        "table": table,
        "limit": limit,
        "columns": sorted(columns) if columns else None,
        "where_clause": where_clause,
    }
    cache_key = _generate_cache_key(params)

    if use_cache:
        cached_result = _get_from_cache(namespace, cache_key)
        if cached_result:
            logger.info(f"Cache hit for sample on {database}.{table}")
            return cached_result

    if not 0 < limit <= MAX_SAMPLE_ROWS:
        raise ValueError(f"Limit must be between 1 and {MAX_SAMPLE_ROWS}, got {limit}")

    _check_exists(database, table)
    full_table_name = f"`{database}`.`{table}`"
    logger.info(f"Sampling {limit} rows from {full_table_name}")
    try:
        df = spark.table(full_table_name)
        if columns:
            df = df.select(columns)
        if where_clause:
            equivalent_query = f"SELECT * FROM {full_table_name} WHERE {where_clause}"
            _check_query_is_valid(equivalent_query)
            df = df.filter(where_clause)

        df = df.limit(limit)

        # Use timeout wrapper for collect operation
        results = run_with_timeout(
            lambda: [row.asDict() for row in df.collect()],
            timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
            operation_name=f"sample_{database}.{table}",
        )
        logger.info(f"Sampled {len(results)} rows.")

        if use_cache:
            _store_in_cache(namespace, cache_key, results)

        return results
    except SparkTimeoutError:
        raise  # Re-raise timeout errors as-is
    except Exception as e:
        logger.error(f"Error sampling rows from {full_table_name}: {e}")
        raise SparkOperationError(
            f"Failed to sample rows from {full_table_name}: {str(e)}"
        )


def query_delta_table(
    spark: SparkSession,
    query: str,
    limit: int = 1000,
    offset: int = 0,
    use_cache: bool = True,
) -> TableQueryResponse:
    """
    Executes a SQL query against Delta tables with pagination support.

    The query is validated for safety, then wrapped to support pagination.
    A count query is executed first to determine total rows, then the
    paginated results are returned.

    Args:
        spark: The SparkSession object.
        query: The SQL query string to execute.
        limit: Maximum number of rows to return (default: 1000, max: 50000).
        offset: Number of rows to skip for pagination (default: 0).
        use_cache: Whether to use the redis cache to store the result.

    Returns:
        TableQueryResponse with result data and pagination info.

    Raises:
        SparkQueryError: If query validation fails or LIMIT exceeds MAX_QUERY_ROWS
        SparkOperationError: If query execution fails
    """
    # Validate query structure first
    _check_query_is_valid(query)

    # Enforce max limit
    if limit > MAX_QUERY_ROWS:
        raise SparkQueryError(
            f"Limit ({limit}) exceeds maximum allowed ({MAX_QUERY_ROWS}). "
            f"Please reduce your limit."
        )

    # Strip any existing LIMIT/OFFSET from user query since we'll add our own
    # We use a subquery approach to handle pagination correctly
    base_query = query.rstrip().rstrip(";")

    # Remove trailing LIMIT clause if present (we'll add our own)
    existing_limit = _extract_limit_from_query(base_query)
    if existing_limit is not None:
        # Remove the LIMIT clause - simple regex replacement
        base_query = re.sub(r"\s+LIMIT\s+\d+\s*$", "", base_query, flags=re.IGNORECASE)

    # Warn if using offset without ORDER BY - results will be non-deterministic
    if offset > 0:
        has_order_by = bool(
            re.search(r"\bORDER\s+BY\b", base_query, flags=re.IGNORECASE)
        )
        if not has_order_by:
            logger.warning(
                f"Pagination with offset={offset} but query has no ORDER BY clause. "
                "Results may be non-deterministic across pages. "
                "Add ORDER BY to ensure consistent pagination."
            )

    namespace = "query"
    params = {"query": base_query, "limit": limit, "offset": offset}
    cache_key = _generate_cache_key(params)

    # Separate cache key for count (independent of limit/offset for reuse across pages)
    count_namespace = "query_count"
    count_cache_key = _generate_cache_key({"query": base_query})

    if use_cache:
        cached_result = _get_from_cache(namespace, cache_key)
        if cached_result:
            logger.info(
                f"Cache hit for query: {base_query[:50]}{'...' if len(base_query) > 50 else ''}"
            )
            # Reconstruct response from cached data
            return TableQueryResponse(
                result=cached_result[0]["result"],
                pagination=PaginationInfo(**cached_result[0]["pagination"]),
            )

    logger.info(f"Executing paginated query: {base_query[:100]}...")

    try:
        # Execute the paginated data query FIRST
        # This allows us to skip the COUNT query if we get fewer rows than limit
        paginated_query = f"{base_query} LIMIT {limit} OFFSET {offset}"
        logger.info(f"Executing data query with LIMIT {limit} OFFSET {offset}")

        df = spark.sql(paginated_query)
        results = run_with_timeout(
            lambda: [row.asDict() for row in df.collect()],
            timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
            operation_name="query_delta_table",
        )
        logger.info(f"Query returned {len(results)} rows.")

        # Optimization: If we got fewer rows than limit, we know the exact total
        # without executing an expensive COUNT query
        if len(results) < limit:
            total_count = offset + len(results)
            has_more = False
            logger.info(
                f"Skipped COUNT query: results ({len(results)}) < limit ({limit}), "
                f"total_count={total_count}"
            )
        else:
            # We filled the page, need to get actual count
            # First check if count is cached
            cached_count = None
            if use_cache:
                cached_count = _get_from_cache(count_namespace, count_cache_key)

            if cached_count:
                total_count = cached_count[0]["count"]
                logger.info(f"Cache hit for count: {total_count}")
            else:
                # Execute COUNT query
                count_query = f"SELECT COUNT(*) as cnt FROM ({base_query})"
                logger.info("Executing count query for pagination")

                count_result = run_with_timeout(
                    lambda: spark.sql(count_query).collect(),
                    timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
                    operation_name="query_count",
                )
                total_count = count_result[0]["cnt"]
                logger.info(f"Total count: {total_count}")

                # Cache the count separately for reuse across pages
                if use_cache:
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

        if use_cache:
            # Store serializable version in cache
            cache_data = [
                {
                    "result": results,
                    "pagination": pagination.model_dump(),
                }
            ]
            _store_in_cache(namespace, cache_key, cache_data)

        return response

    except SparkTimeoutError:
        raise  # Re-raise timeout errors as-is
    except SparkQueryError:
        raise  # Re-raise query validation errors as-is
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise SparkOperationError(f"Failed to execute query: {str(e)}")


# ---
# Query Builder Functions
# ---

# Valid identifier pattern: alphanumeric and underscores only
VALID_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, identifier_type: str = "identifier") -> None:
    """
    Validate that an identifier (table, column, database name) is safe.

    Args:
        name: The identifier to validate.
        identifier_type: Type of identifier for error messages.

    Raises:
        SparkQueryError: If the identifier is invalid.
    """
    if not name or not VALID_IDENTIFIER_PATTERN.match(name):
        raise SparkQueryError(
            f"Invalid {identifier_type}: '{name}'. "
            "Identifiers must start with a letter or underscore and contain "
            "only alphanumeric characters and underscores."
        )


def _escape_value(value: Any) -> str:
    """
    Escape a value for safe SQL use.

    Args:
        value: The value to escape.

    Returns:
        The escaped value as a SQL-safe string.
    """
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    else:
        # For other types, convert to string and escape
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"


def _build_column_expression(col: ColumnSpec) -> str:
    """
    Build a column expression from a ColumnSpec.

    Args:
        col: The column specification.

    Returns:
        SQL column expression string.
    """
    parts = []

    if col.table_alias:
        _validate_identifier(col.table_alias, "table alias")
        parts.append(f"`{col.table_alias}`.")

    _validate_identifier(col.column, "column")
    parts.append(f"`{col.column}`")

    if col.alias:
        _validate_identifier(col.alias, "column alias")
        parts.append(f" AS `{col.alias}`")

    return "".join(parts)


def _build_aggregation_expression(agg: AggregationSpec) -> str:
    """
    Build an aggregation expression from an AggregationSpec.

    Args:
        agg: The aggregation specification.

    Returns:
        SQL aggregation expression string.
    """
    if agg.column == "*":
        if agg.function != "COUNT":
            raise SparkQueryError(
                f"Aggregation function {agg.function} does not support '*'. "
                "Only COUNT(*) is valid."
            )
        expr = "COUNT(*)"
    else:
        _validate_identifier(agg.column, "aggregation column")
        expr = f"{agg.function}(`{agg.column}`)"

    if agg.alias:
        _validate_identifier(agg.alias, "aggregation alias")
        expr += f" AS `{agg.alias}`"

    return expr


def _build_filter_condition(condition: FilterCondition) -> str:
    """
    Build a single filter condition for WHERE or HAVING clause.

    Args:
        condition: The filter condition specification.

    Returns:
        SQL condition string.
    """
    _validate_identifier(condition.column, "filter column")
    col = f"`{condition.column}`"
    op = condition.operator

    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col} {op}"

    elif op in ("IN", "NOT IN"):
        if not condition.values:
            raise SparkQueryError(f"Operator {op} requires 'values' to be provided")
        escaped_values = [_escape_value(v) for v in condition.values]
        values_str = ", ".join(escaped_values)
        return f"{col} {op} ({values_str})"

    elif op == "BETWEEN":
        if not condition.values or len(condition.values) != 2:
            raise SparkQueryError(
                "Operator BETWEEN requires exactly 2 values in 'values'"
            )
        return (
            f"{col} BETWEEN {_escape_value(condition.values[0])} "
            f"AND {_escape_value(condition.values[1])}"
        )

    else:
        # Standard comparison operators: =, !=, <, >, <=, >=, LIKE, NOT LIKE
        if condition.value is None:
            raise SparkQueryError(f"Operator {op} requires 'value' to be provided")
        return f"{col} {op} {_escape_value(condition.value)}"


def _build_filter_clause(
    conditions: List[FilterCondition], clause_type: str = "WHERE"
) -> str:
    """
    Build a WHERE or HAVING clause from filter conditions.

    Args:
        conditions: List of filter conditions.
        clause_type: Either "WHERE" or "HAVING".

    Returns:
        SQL clause string (including the keyword) or empty string if no conditions.
    """
    if not conditions:
        return ""

    condition_strs = [_build_filter_condition(c) for c in conditions]
    return f" {clause_type} " + " AND ".join(condition_strs)


def _build_join_clause(join: JoinClause, main_table: str) -> str:
    """
    Build a JOIN clause.

    Args:
        join: The join specification.
        main_table: The name of the main table for the ON clause.

    Returns:
        SQL JOIN clause string.
    """
    _validate_identifier(join.database, "join database")
    _validate_identifier(join.table, "join table")
    _validate_identifier(join.on_left_column, "join left column")
    _validate_identifier(join.on_right_column, "join right column")

    join_table = f"`{join.database}`.`{join.table}`"
    join_type = join.join_type

    return (
        f" {join_type} JOIN {join_table} "
        f"ON `{main_table}`.`{join.on_left_column}` = "
        f"`{join.table}`.`{join.on_right_column}`"
    )


def build_select_query(
    request: TableSelectRequest, include_pagination: bool = True
) -> str:
    """
    Build a SQL SELECT query from a TableSelectRequest.

    Args:
        request: The structured select request.
        include_pagination: Whether to include LIMIT/OFFSET clauses.

    Returns:
        The constructed SQL query string.
    """
    _validate_identifier(request.database, "database")
    _validate_identifier(request.table, "table")

    # Build SELECT clause
    select_parts = []

    # Add DISTINCT keyword if requested
    distinct_keyword = "DISTINCT " if request.distinct else ""

    # Add columns
    if request.columns:
        select_parts.extend([_build_column_expression(c) for c in request.columns])

    # Add aggregations
    if request.aggregations:
        select_parts.extend(
            [_build_aggregation_expression(a) for a in request.aggregations]
        )

    # If no columns or aggregations, select all
    if not select_parts:
        select_clause = f"SELECT {distinct_keyword}*"
    else:
        select_clause = f"SELECT {distinct_keyword}" + ", ".join(select_parts)

    # Build FROM clause
    main_table = f"`{request.database}`.`{request.table}`"
    from_clause = f" FROM {main_table}"

    # Build JOIN clauses
    join_clauses = ""
    if request.joins:
        for join in request.joins:
            # Validate join table exists
            _check_exists(join.database, join.table)
            join_clauses += _build_join_clause(join, request.table)

    # Build WHERE clause
    where_clause = (
        _build_filter_clause(request.filters, "WHERE") if request.filters else ""
    )

    # Build GROUP BY clause
    group_by_clause = ""
    if request.group_by:
        for col in request.group_by:
            _validate_identifier(col, "group by column")
        group_by_cols = ", ".join([f"`{col}`" for col in request.group_by])
        group_by_clause = f" GROUP BY {group_by_cols}"

    # Build HAVING clause
    having_clause = (
        _build_filter_clause(request.having, "HAVING") if request.having else ""
    )

    # Build ORDER BY clause
    order_by_clause = ""
    if request.order_by:
        order_parts = []
        for order in request.order_by:
            _validate_identifier(order.column, "order by column")
            order_parts.append(f"`{order.column}` {order.direction}")
        order_by_clause = " ORDER BY " + ", ".join(order_parts)

    # Build LIMIT/OFFSET clause
    pagination_clause = ""
    if include_pagination:
        pagination_clause = f" LIMIT {request.limit} OFFSET {request.offset}"

    # Combine all parts
    query = (
        select_clause
        + from_clause
        + join_clauses
        + where_clause
        + group_by_clause
        + having_clause
        + order_by_clause
        + pagination_clause
    )

    return query


def select_from_delta_table(
    spark: SparkSession, request: TableSelectRequest, use_cache: bool = True
) -> TableSelectResponse:
    """
    Execute a structured SELECT query against Delta tables with pagination.

    Args:
        spark: The SparkSession object.
        request: The structured select request.
        use_cache: Whether to use the redis cache to store the result.

    Returns:
        TableSelectResponse with data and pagination info.
    """
    namespace = "select"

    # Generate cache key from request parameters
    params = request.model_dump()
    cache_key = _generate_cache_key(params)

    if use_cache:
        cached_result = _get_from_cache(namespace, cache_key)
        if cached_result:
            logger.info(f"Cache hit for select on {request.database}.{request.table}")
            # Reconstruct response from cached data
            return TableSelectResponse(
                data=cached_result[0]["data"],
                pagination=PaginationInfo(**cached_result[0]["pagination"]),
            )

    # Validate main table exists
    _check_exists(request.database, request.table)

    # Build and execute count query (without pagination) for total count
    count_query = f"SELECT COUNT(*) as cnt FROM ({build_select_query(request, include_pagination=False)})"
    logger.info(f"Executing count query: {count_query}")

    try:
        # Use timeout wrapper for count query
        count_result = run_with_timeout(
            lambda: spark.sql(count_query).collect(),
            timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
            operation_name=f"count_select_{request.database}.{request.table}",
        )
        total_count = count_result[0]["cnt"]
    except SparkTimeoutError:
        raise  # Re-raise timeout errors as-is
    except Exception as e:
        logger.error(f"Error executing count query: {e}")
        raise SparkOperationError(f"Failed to execute count query: {str(e)}")

    # Build and execute main query with pagination
    main_query = build_select_query(request, include_pagination=True)
    logger.info(f"Executing select query: {main_query}")

    try:
        df = spark.sql(main_query)
        # Use timeout wrapper for data query
        results = run_with_timeout(
            lambda: [row.asDict() for row in df.collect()],
            timeout_seconds=DEFAULT_SPARK_COLLECT_TIMEOUT,
            operation_name=f"select_{request.database}.{request.table}",
        )
        logger.info(f"Select query returned {len(results)} rows.")

        # Calculate pagination info
        has_more = (request.offset + len(results)) < total_count

        pagination = PaginationInfo(
            limit=request.limit,
            offset=request.offset,
            total_count=total_count,
            has_more=has_more,
        )

        response = TableSelectResponse(data=results, pagination=pagination)

        if use_cache:
            # Store serializable version in cache
            cache_data = [
                {
                    "data": results,
                    "pagination": pagination.model_dump(),
                }
            ]
            _store_in_cache(namespace, cache_key, cache_data)

        return response

    except SparkTimeoutError:
        raise  # Re-raise timeout errors as-is
    except Exception as e:
        logger.error(f"Error executing select query: {e}")
        raise SparkOperationError(f"Failed to execute select query: {str(e)}")
