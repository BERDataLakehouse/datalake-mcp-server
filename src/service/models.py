"""
Pydantic models for the Spark Manager API.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, Field

MAX_ASYNC_QUERY_ROWS = int(os.getenv("ASYNC_QUERY_MAX_ROWS", "5000"))


class QueryEngine(str, Enum):
    """Query engine to use for execution."""

    SPARK = "spark"
    TRINO = "trino"


MAX_CONCURRENT_ASYNC_JOBS_PER_USER = int(
    os.getenv("MAX_CONCURRENT_ASYNC_JOBS_PER_USER", "10")
)


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: Annotated[int | None, Field(description="Error code")] = None
    error_type: Annotated[str | None, Field(description="Error type")] = None
    message: Annotated[str | None, Field(description="Error message")] = None


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: Annotated[str, Field(description="Component name")]
    status: Annotated[
        Literal["healthy", "unhealthy", "degraded"],
        Field(description="Component health status"),
    ]
    message: Annotated[str | None, Field(description="Optional status message")] = None
    latency_ms: Annotated[
        float | None, Field(description="Response time in milliseconds")
    ] = None


class DeepHealthResponse(BaseModel):
    """Health check response with component-level details."""

    status: Annotated[
        Literal["healthy", "unhealthy", "degraded"],
        Field(description="Overall health status"),
    ]
    components: Annotated[
        List[ComponentHealth], Field(description="Health status of each component")
    ]
    message: Annotated[
        str | None, Field(description="Summary message about system health")
    ] = None


class DatabaseListRequest(BaseModel):
    """Request model for listing databases."""

    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True

    filter_by_namespace: Annotated[
        bool,
        Field(
            description="Whether to filter databases by user/tenant namespace prefixes"
        ),
    ] = True


class DatabaseListResponse(BaseModel):
    """Response model for listing databases."""

    databases: Annotated[List[str], Field(description="List of database names")]


class TableListRequest(BaseModel):
    """Request model for listing tables in a database."""

    database: Annotated[
        str, Field(description="Name of the database to list tables from")
    ]
    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True


class TableListResponse(BaseModel):
    """Response model for listing tables."""

    tables: Annotated[
        List[str], Field(description="List of table names in the specified database")
    ]


class TableSchemaRequest(BaseModel):
    """Request model for getting table schema."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to get schema for")]


class TableSchemaResponse(BaseModel):
    """Response model for table schema."""

    columns: Annotated[
        List[str], Field(description="List of column names in the table")
    ]


class DatabaseStructureRequest(BaseModel):
    """Request model for getting database structure."""

    with_schema: Annotated[
        bool, Field(description="Whether to include table schemas in the response")
    ] = False
    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True


class DatabaseStructureResponse(BaseModel):
    """Response model for database structure."""

    structure: Annotated[
        Dict[str, Any],
        Field(
            description="Database structure with tables and optionally their schemas"
        ),
    ]


# ---
# Models for Delta Table Data Operations
# ---


class TableQueryRequest(BaseModel):
    """Request model for querying a Delta table.

    Note: For deterministic pagination with offset > 0, include an ORDER BY
    clause in your query. Without ORDER BY, rows may appear in different
    order across pages, causing duplicates or missing records.
    """

    query: Annotated[str, Field(description="SQL query to execute against the table")]
    limit: Annotated[
        int,
        Field(
            description="Maximum number of rows to return",
            gt=0,
            le=1000,
        ),
    ] = 100
    offset: Annotated[
        int,
        Field(
            description=(
                "Number of rows to skip for pagination. "
                "Requires ORDER BY in query for deterministic results."
            ),
            ge=0,
        ),
    ] = 0
    engine: Annotated[
        QueryEngine | None,
        Field(
            description=(
                "Query engine to use. Overrides the global QUERY_ENGINE setting. "
                "If not specified, uses the QUERY_ENGINE env var (default: spark)."
            ),
        ),
    ] = None


class TableQueryResponse(BaseModel):
    """Response model for Delta table query results."""

    result: Annotated[
        List[Any],
        Field(description="List of rows returned by the query, each as a dictionary"),
    ]
    pagination: Annotated["PaginationInfo", Field(description="Pagination metadata")]


class TableCountRequest(BaseModel):
    """Request model for counting rows in a Delta table."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to count rows in")]


class TableCountResponse(BaseModel):
    """Response model for Delta table row count."""

    count: Annotated[int, Field(description="Total number of rows in the table")]


class TableSampleRequest(BaseModel):
    """Request model for sampling data from a Delta table."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to sample from")]
    limit: Annotated[
        int,
        Field(
            description="Maximum number of rows to return in the sample", gt=0, le=100
        ),
    ] = 10
    columns: Annotated[
        List[str] | None, Field(description="List of columns to return in the sample")
    ] = None
    where_clause: Annotated[
        str | None, Field(description="SQL WHERE clause to filter the rows")
    ] = None


class TableSampleResponse(BaseModel):
    """Response model for Delta table data sample."""

    sample: Annotated[
        List[Any],
        Field(description="List of sample rows, each as a dictionary"),
    ]


# ---
# Models for Query Builder (Structured SELECT)
# ---


class JoinClause(BaseModel):
    """Model for JOIN clause specification."""

    join_type: Annotated[
        Literal["INNER", "LEFT", "RIGHT", "FULL"],
        Field(description="Type of JOIN operation"),
    ]
    database: Annotated[str, Field(description="Database containing the table to join")]
    table: Annotated[str, Field(description="Table to join")]
    on_left_column: Annotated[
        str, Field(description="Column from the left/main table for the join condition")
    ]
    on_right_column: Annotated[
        str, Field(description="Column from the joined table for the join condition")
    ]


class ColumnSpec(BaseModel):
    """Model for column specification in SELECT clause."""

    column: Annotated[str, Field(description="Column name to select")]
    table_alias: Annotated[
        str | None,
        Field(description="Table alias for disambiguation in JOINs"),
    ] = None
    alias: Annotated[
        str | None, Field(description="Output alias for the column (AS clause)")
    ] = None


class AggregationSpec(BaseModel):
    """Model for aggregation function specification."""

    function: Annotated[
        Literal["COUNT", "SUM", "AVG", "MIN", "MAX"],
        Field(description="Aggregation function to apply"),
    ]
    column: Annotated[
        str,
        Field(description="Column to aggregate, or '*' for COUNT(*)"),
    ]
    alias: Annotated[
        str | None, Field(description="Output alias for the aggregation result")
    ] = None


class FilterCondition(BaseModel):
    """Model for WHERE/HAVING filter conditions."""

    column: Annotated[str, Field(description="Column name to filter on")]
    operator: Annotated[
        Literal[
            "=",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "IN",
            "NOT IN",
            "LIKE",
            "NOT LIKE",
            "IS NULL",
            "IS NOT NULL",
            "BETWEEN",
        ],
        Field(description="Comparison operator"),
    ]
    value: Annotated[
        Any | None,
        Field(description="Value for comparison (None for IS NULL/IS NOT NULL)"),
    ] = None
    values: Annotated[
        List[Any] | None,
        Field(description="List of values for IN, NOT IN, or BETWEEN operators"),
    ] = None


class OrderBySpec(BaseModel):
    """Model for ORDER BY clause specification."""

    column: Annotated[str, Field(description="Column name to sort by")]
    direction: Annotated[
        Literal["ASC", "DESC"], Field(description="Sort direction")
    ] = "ASC"


class PaginationInfo(BaseModel):
    """Pagination metadata for query results."""

    limit: Annotated[int, Field(description="Number of rows requested")]
    offset: Annotated[int, Field(description="Number of rows skipped")]
    total_count: Annotated[int, Field(description="Total number of matching rows")]
    has_more: Annotated[
        bool, Field(description="Whether more rows are available beyond this page")
    ]


class TableSelectRequest(BaseModel):
    """Request model for structured SELECT query builder."""

    database: Annotated[
        str, Field(description="Name of the primary database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the primary table to query")]
    joins: Annotated[
        List[JoinClause] | None,
        Field(description="Optional list of JOIN clauses"),
    ] = None
    columns: Annotated[
        List[ColumnSpec] | None,
        Field(description="Columns to select (None for SELECT *)"),
    ] = None
    distinct: Annotated[
        bool, Field(description="Whether to apply DISTINCT to results")
    ] = False
    aggregations: Annotated[
        List[AggregationSpec] | None,
        Field(description="Optional aggregation functions to apply"),
    ] = None
    filters: Annotated[
        List[FilterCondition] | None,
        Field(description="Optional WHERE clause conditions"),
    ] = None
    group_by: Annotated[
        List[str] | None,
        Field(description="Optional columns to GROUP BY"),
    ] = None
    having: Annotated[
        List[FilterCondition] | None,
        Field(description="Optional HAVING clause conditions (filters after GROUP BY)"),
    ] = None
    order_by: Annotated[
        List[OrderBySpec] | None,
        Field(description="Optional ORDER BY specifications"),
    ] = None
    limit: Annotated[
        int,
        Field(
            description="Maximum number of rows to return",
            gt=0,
            le=1000,
        ),
    ] = 100
    offset: Annotated[
        int,
        Field(description="Number of rows to skip for pagination", ge=0),
    ] = 0
    engine: Annotated[
        QueryEngine | None,
        Field(
            description=(
                "Query engine to use. Overrides the global QUERY_ENGINE setting. "
                "If not specified, uses the QUERY_ENGINE env var (default: spark)."
            ),
        ),
    ] = None


class TableSelectResponse(BaseModel):
    """Response model for structured SELECT query results."""

    data: Annotated[
        List[Dict[str, Any]],
        Field(description="Query result rows, each as a dictionary"),
    ]
    pagination: Annotated[PaginationInfo, Field(description="Pagination metadata")]


# ---
# Models for Async Query Execution
# ---


class JobStatus(str, Enum):
    """Status of an async query job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class JobRecord(BaseModel):
    """Internal model for tracking async query job state in the S3/MinIO-backed job store."""

    job_id: Annotated[str, Field(description="Unique job identifier (UUID4)")]
    user: Annotated[str, Field(description="KBase username who submitted the job")]
    query: Annotated[str, Field(description="SQL query submitted")]
    status: Annotated[JobStatus, Field(description="Current job status")]
    limit: Annotated[int, Field(description="Requested row limit")]
    offset: Annotated[int, Field(description="Requested pagination offset")]
    created_at: Annotated[datetime, Field(description="Job creation timestamp")]
    started_at: Annotated[
        datetime | None, Field(description="When execution started")
    ] = None
    completed_at: Annotated[
        datetime | None, Field(description="When execution completed")
    ] = None
    error_message: Annotated[
        str | None, Field(description="Error details if job failed")
    ] = None
    result_path: Annotated[
        str | None, Field(description="S3 path where results are stored")
    ] = None
    row_count: Annotated[
        int | None, Field(description="Number of rows in the result")
    ] = None
    total_count: Annotated[
        int | None, Field(description="Total matching rows (for pagination)")
    ] = None
    has_more: Annotated[
        bool | None, Field(description="Whether more rows are available")
    ] = None


class AsyncQuerySubmitRequest(BaseModel):
    """Request model for submitting an async query."""

    query: Annotated[
        str, Field(description="SQL query to execute against Delta tables")
    ]
    limit: Annotated[
        int,
        Field(
            description="Maximum number of rows to return",
            gt=0,
            le=MAX_ASYNC_QUERY_ROWS,
        ),
    ] = 1000
    offset: Annotated[
        int,
        Field(
            description=(
                "Number of rows to skip for pagination. "
                "Requires ORDER BY in query for deterministic results."
            ),
            ge=0,
        ),
    ] = 0
    engine: Annotated[
        QueryEngine | None,
        Field(
            description=(
                "Query engine to use. Overrides the global QUERY_ENGINE setting. "
                "If not specified, uses the QUERY_ENGINE env var (default: spark)."
            ),
        ),
    ] = None


class AsyncQuerySubmitResponse(BaseModel):
    """Response model for async query submission."""

    job_id: Annotated[str, Field(description="Unique identifier for the submitted job")]


class AsyncQueryStatusResponse(BaseModel):
    """Response model for async query job status."""

    job_id: Annotated[str, Field(description="Unique job identifier")]
    status: Annotated[JobStatus, Field(description="Current job status")]
    query: Annotated[str, Field(description="SQL query that was submitted")]
    limit: Annotated[int, Field(description="Requested row limit")]
    offset: Annotated[int, Field(description="Requested pagination offset")]
    created_at: Annotated[datetime, Field(description="Job creation timestamp")]
    started_at: Annotated[
        datetime | None, Field(description="When execution started")
    ] = None
    completed_at: Annotated[
        datetime | None, Field(description="When execution completed")
    ] = None
    error_message: Annotated[
        str | None, Field(description="Error details if job failed")
    ] = None
    row_count: Annotated[
        int | None, Field(description="Number of rows in the result")
    ] = None
