"""
API routes for Delta Lake operations.

Routes support full concurrency:
- Spark Connect mode: Direct SparkSession usage
- Standalone mode: Dispatched to ProcessPoolExecutor for isolated execution
"""

import logging
from typing import Annotated, Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, Request, status

from src.delta_lake import data_store, delta_service
from src.service.dependencies import SparkContext, auth, get_spark_context
from src.service.spark_session_pool import run_in_spark_process
from src.service.standalone_operations import (
    count_table_subprocess,
    get_db_structure_subprocess,
    get_table_schema_subprocess,
    list_databases_subprocess,
    list_tables_subprocess,
    query_table_subprocess,
    sample_table_subprocess,
    select_table_subprocess,
)
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
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

    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        databases = run_in_spark_process(
            list_databases_subprocess,
            ctx.settings_dict,
            use_hms=body.use_hms,
            filter_by_namespace=body.filter_by_namespace,
            auth_token=auth_token,
            app_name=ctx.app_name,
            operation_name="list_databases",
        )
    else:
        # Use Spark Connect session directly
        databases = cast(
            list[str],
            data_store.get_databases(
                spark=ctx.spark,
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableListResponse:
    """
    Endpoint to list tables in a specific database.
    """
    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        tables = run_in_spark_process(
            list_tables_subprocess,
            ctx.settings_dict,
            database=request.database,
            use_hms=request.use_hms,
            app_name=ctx.app_name,
            operation_name="list_tables",
        )
    else:
        # Use Spark Connect session directly
        settings = get_settings()
        tables = cast(
            list[str],
            data_store.get_tables(
                database=request.database,
                spark=ctx.spark,
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableSchemaResponse:
    """
    Endpoint to get the schema of a specific table in a database.
    """
    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        columns = run_in_spark_process(
            get_table_schema_subprocess,
            ctx.settings_dict,
            database=request.database,
            table=request.table,
            app_name=ctx.app_name,
            operation_name="get_table_schema",
        )
    else:
        # Use Spark Connect session directly
        columns = cast(
            list[str],
            data_store.get_table_schema(
                database=request.database,
                table=request.table,
                spark=ctx.spark,
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> DatabaseStructureResponse:
    """
    Endpoint to get the complete structure of all databases.
    """
    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        structure = run_in_spark_process(
            get_db_structure_subprocess,
            ctx.settings_dict,
            with_schema=request.with_schema,
            use_hms=request.use_hms,
            app_name=ctx.app_name,
            operation_name="get_db_structure",
        )
    else:
        # Use Spark Connect session directly
        settings = get_settings()
        structure = cast(
            dict[str, list[str] | dict[str, list[str]]],
            data_store.get_db_structure(
                spark=ctx.spark,
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableCountResponse:
    """
    Endpoint to count rows in a specific Delta table.
    """
    # Pass username to service layer for user-scoped cache isolation
    username = auth.user if auth else None

    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        count = run_in_spark_process(
            count_table_subprocess,
            ctx.settings_dict,
            database=request.database,
            table=request.table,
            username=username,
            app_name=ctx.app_name,
            operation_name="count_table",
        )
    else:
        # Use Spark Connect session directly
        count = delta_service.count_delta_table(
            spark=ctx.spark,
            database=request.database,
            table=request.table,
            username=username,
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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableSampleResponse:
    """
    Endpoint to get a sample of data from a specific Delta table.
    """
    # Pass username to service layer for user-scoped cache isolation
    username = auth.user if auth else None

    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        sample = run_in_spark_process(
            sample_table_subprocess,
            ctx.settings_dict,
            database=request.database,
            table=request.table,
            limit=request.limit,
            columns=request.columns,
            where_clause=request.where_clause,
            username=username,
            app_name=ctx.app_name,
            operation_name="sample_table",
        )
    else:
        # Use Spark Connect session directly
        sample: List[Dict[str, Any]] = delta_service.sample_delta_table(
            spark=ctx.spark,
            database=request.database,
            table=request.table,
            limit=request.limit,
            columns=request.columns,
            where_clause=request.where_clause,
            username=username,
        )
    return TableSampleResponse(sample=sample)


@router.post(
    "/tables/query",
    response_model=TableQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Delta tables with pagination",
    description=(
        "Executes a SQL query against Delta tables with pagination support. "
        "Returns query results along with pagination metadata including total count. "
        "Use limit and offset parameters to paginate through large result sets. "
        "IMPORTANT: Include an ORDER BY clause in your query for deterministic "
        "pagination results. Without ORDER BY, rows may appear in different order "
        "across pages, causing duplicates or missing records."
    ),
    operation_id="query_delta_table",
)
def query_table(
    request: TableQueryRequest,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableQueryResponse:
    """
    Endpoint to execute a query against Delta tables with pagination support.
    """
    # Pass username to service layer for user-scoped cache isolation
    username = auth.user if auth else None

    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        result = run_in_spark_process(
            query_table_subprocess,
            ctx.settings_dict,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            username=username,
            app_name=ctx.app_name,
            operation_name="query_table",
        )
        # Result is a dict, need to convert back to response model
        from src.service.models import PaginationInfo

        return TableQueryResponse(
            result=result["result"],
            pagination=PaginationInfo(**result["pagination"]),
        )
    else:
        # Use Spark Connect session directly
        return delta_service.query_delta_table(
            spark=ctx.spark,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            username=username,
        )


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
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableSelectResponse:
    """
    Endpoint to execute a structured SELECT query with pagination support.

    This endpoint allows users to build complex queries without writing raw SQL.
    The backend constructs the query from the provided parameters, ensuring
    security and proper escaping of all values.
    """
    # Pass username to service layer for user-scoped cache isolation
    username = auth.user if auth else None

    if ctx.is_standalone_subprocess:
        # Dispatch to process pool for Standalone mode
        # Convert request to dict for pickling
        request_dict = request.model_dump()
        result = run_in_spark_process(
            select_table_subprocess,
            ctx.settings_dict,
            request_dict=request_dict,
            username=username,
            app_name=ctx.app_name,
            operation_name="select_table",
        )
        # Result is a dict, need to convert back to response model
        from src.service.models import PaginationInfo

        return TableSelectResponse(
            data=result["data"],
            pagination=PaginationInfo(**result["pagination"]),
        )
    else:
        # Use Spark Connect session directly
        return delta_service.select_from_delta_table(
            spark=ctx.spark,
            request=request,
            username=username,
        )
