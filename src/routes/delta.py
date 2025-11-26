"""
API routes for Delta Lake operations.
"""

import logging
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, Request, status

from src.delta_lake import data_store, delta_service
from src.service.dependencies import auth, get_spark_session
from src.settings import get_settings
from src.service.models import (
    DatabaseListRequest,
    DatabaseListResponse,
    DatabaseStructureRequest,
    DatabaseStructureResponse,
    TableCountRequest,
    TableCountResponse,
    TableListRequest,
    TableListResponse,
    TableQueryRequest,
    TableQueryResponse,
    TableSampleRequest,
    TableSampleResponse,
    TableSchemaRequest,
    TableSchemaResponse,
    TableSelectRequest,
    TableSelectResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/delta", tags=["Delta Lake"])


def _extract_token_from_request(request: Request) -> Optional[str]:
    """Extract the Bearer token from the request Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    return None


@router.post(
    "/databases/list",
    response_model=DatabaseListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all databases in the Hive metastore",
    description="Lists all databases available in the Hive metastore, optionally using PostgreSQL for faster retrieval and filtered by user namespace.",
    operation_id="list_databases",
)
def list_databases(
    body: DatabaseListRequest,
    http_request: Request,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> DatabaseListResponse:
    """
    Endpoint to list all databases in the Hive metastore.
    Optionally filters by user/tenant namespace prefixes.
    """
    # Extract auth token for namespace filtering
    auth_token = None
    if body.filter_by_namespace:
        auth_token = _extract_token_from_request(http_request)
        if not auth_token:
            raise ValueError("Authorization token required for namespace filtering")

    databases = cast(
        list[str],
        data_store.get_databases(
            spark=spark,
            use_hms=body.use_hms,
            return_json=False,
            filter_by_namespace=body.filter_by_namespace,
            auth_token=auth_token,
        ),
    )

    return DatabaseListResponse(databases=databases)


@router.post(
    "/databases/tables/list",
    response_model=TableListResponse,
    status_code=status.HTTP_200_OK,
    summary="List tables in a database",
    description="Lists all tables in a specific database, optionally using PostgreSQL for faster retrieval.",
    operation_id="list_database_tables",
)
def list_database_tables(
    request: TableListRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableListResponse:
    """
    Endpoint to list tables in a specific database.
    """
    settings = get_settings()
    tables = cast(
        list[str],
        data_store.get_tables(
            database=request.database,
            spark=spark,
            use_hms=request.use_hms,
            return_json=False,
            settings=settings,
        ),
    )
    return TableListResponse(tables=tables)


@router.post(
    "/databases/tables/schema",
    response_model=TableSchemaResponse,
    status_code=status.HTTP_200_OK,
    summary="Get table schema",
    description="Gets the schema (column names) of a specific table in a database.",
    operation_id="get_table_schema",
)
def get_table_schema(
    request: TableSchemaRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableSchemaResponse:
    """
    Endpoint to get the schema of a specific table in a database.
    """
    columns = cast(
        list[str],
        data_store.get_table_schema(
            database=request.database,
            table=request.table,
            spark=spark,
            return_json=False,
        ),
    )
    return TableSchemaResponse(columns=columns)


@router.post(
    "/databases/structure",
    response_model=DatabaseStructureResponse,
    status_code=status.HTTP_200_OK,
    summary="Get database structure",
    description="Gets the complete structure of all databases, optionally including table schemas.",
    operation_id="get_database_structure",
)
def get_database_structure(
    request: DatabaseStructureRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> DatabaseStructureResponse:
    """
    Endpoint to get the complete structure of all databases.
    """
    settings = get_settings()
    structure = cast(
        dict[str, list[str] | dict[str, list[str]]],
        data_store.get_db_structure(
            spark=spark,
            with_schema=request.with_schema,
            use_hms=request.use_hms,
            return_json=False,
            settings=settings,
        ),
    )
    return DatabaseStructureResponse(structure=structure)


@router.post(
    "/tables/count",
    response_model=TableCountResponse,
    status_code=status.HTTP_200_OK,
    summary="Count rows in a Delta table",
    description="Gets the total row count for a specified Delta table.",
    operation_id="count_delta_table",
)
def count_table(
    request: TableCountRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableCountResponse:
    """
    Endpoint to count rows in a specific Delta table.
    """

    count = delta_service.count_delta_table(
        spark=spark, database=request.database, table=request.table
    )
    return TableCountResponse(count=count)


@router.post(
    "/tables/sample",
    response_model=TableSampleResponse,
    status_code=status.HTTP_200_OK,
    summary="Sample data from a Delta table",
    description="Retrieves a small sample of rows from a specified Delta table.",
    operation_id="sample_delta_table",
)
def sample_table(
    request: TableSampleRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableSampleResponse:
    """
    Endpoint to get a sample of data from a specific Delta table.
    """
    sample: List[Dict[str, Any]] = delta_service.sample_delta_table(
        spark=spark,
        database=request.database,
        table=request.table,
        limit=request.limit,
        columns=request.columns,
        where_clause=request.where_clause,
    )
    return TableSampleResponse(sample=sample)


@router.post(
    "/tables/query",
    response_model=TableQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query a Delta table",
    description="Executes a SQL query against a specified Delta table.",
    operation_id="query_delta_table",
)
def query_table(
    request: TableQueryRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableQueryResponse:
    """
    Endpoint to execute a query against a specific Delta table.
    """
    result: List[Dict[str, Any]] = delta_service.query_delta_table(
        spark=spark,
        query=request.query,
    )
    return TableQueryResponse(result=result)


@router.post(
    "/tables/select",
    response_model=TableSelectResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a structured SELECT query",
    description=(
        "Builds and executes a SELECT query from structured parameters. "
        "Supports column selection, aggregations (COUNT, SUM, AVG, MIN, MAX), "
        "JOINs, WHERE filters, GROUP BY, HAVING, ORDER BY, DISTINCT, and pagination. "
        "The backend builds the query safely, preventing SQL injection."
    ),
    operation_id="select_delta_table",
)
def select_table(
    request: TableSelectRequest,
    spark=Depends(get_spark_session),
    auth=Depends(auth),
) -> TableSelectResponse:
    """
    Endpoint to execute a structured SELECT query with pagination support.

    This endpoint allows users to build complex queries without writing raw SQL.
    The backend constructs the query from the provided parameters, ensuring
    security and proper escaping of all values.
    """
    return delta_service.select_from_delta_table(spark=spark, request=request)
