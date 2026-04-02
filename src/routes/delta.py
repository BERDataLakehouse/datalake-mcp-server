"""
API routes for Delta Lake operations.

Routes support full concurrency:
- Spark Connect mode: Direct SparkSession usage
- Standalone mode: Dispatched to ProcessPoolExecutor for isolated execution
- Trino mode: Per-request DB-API connection (lightweight HTTP, fully concurrent)

Engine selection:
- Global default: QUERY_ENGINE env var (spark | trino, default spark)
- Per-request override: ``engine`` field on query/select/async-submit requests
"""

import logging
from contextlib import contextmanager
from typing import Annotated, Any, Dict, Generator, List, cast

from fastapi import APIRouter, Depends, Request, Response, status

from src.delta_lake import data_store, delta_service
from src.service.exceptions import MissingTokenError
from src.service.dependencies import (
    SparkContext,
    TrinoContext,
    auth,
    get_spark_context,
    get_token_from_request,
    get_trino_context,
    resolve_engine,
)
from src.service.models import (
    DatabaseListRequest,
    DatabaseListResponse,
    DatabaseStructureRequest,
    DatabaseStructureResponse,
    PaginationInfo,
    QueryEngine,
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
from src.service.query_executor import execute_query, execute_query_trino
from src.service.spark_session_pool import run_in_spark_process
from src.service.standalone_operations import (
    count_table_subprocess,
    get_db_structure_subprocess,
    get_table_schema_subprocess,
    list_databases_subprocess,
    list_tables_subprocess,
    sample_table_subprocess,
    select_table_subprocess,
)
from src.settings import get_settings
from src.trino_engine import trino_data_store, trino_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/delta", tags=["Delta Lake"])


@contextmanager
def _make_trino_ctx(request: Request) -> Generator[TrinoContext, None, None]:
    """
    Create a TrinoContext from a FastAPI request, managing its lifecycle.

    Used when the route handler needs a Trino context (based on engine
    resolution) without eagerly creating it via Depends().
    """
    settings = get_settings()
    gen = get_trino_context(request, settings)
    ctx = next(gen)
    try:
        yield ctx
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


# ---------------------------------------------------------------------------
# Metadata endpoints (use global QUERY_ENGINE only - no per-request override)
# ---------------------------------------------------------------------------


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
    auth_token = None
    if body.filter_by_namespace:
        auth_token = get_token_from_request(http_request)
        if not auth_token:
            raise MissingTokenError(
                "Authorization token required for namespace filtering"
            )

    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            databases = trino_data_store.get_databases_trino(
                conn=trino_ctx.connection,
                use_hms=body.use_hms,
                filter_by_namespace=body.filter_by_namespace,
                auth_token=auth_token,
            )
    elif ctx.is_standalone_subprocess:
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
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableListResponse:
    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            tables = trino_data_store.get_tables_trino(
                conn=trino_ctx.connection,
                database=request.database,
                use_hms=request.use_hms,
            )
    elif ctx.is_standalone_subprocess:
        tables = run_in_spark_process(
            list_tables_subprocess,
            ctx.settings_dict,
            database=request.database,
            use_hms=request.use_hms,
            app_name=ctx.app_name,
            operation_name="list_tables",
        )
    else:
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
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableSchemaResponse:
    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            columns = trino_data_store.get_table_schema_trino(
                conn=trino_ctx.connection,
                database=request.database,
                table=request.table,
            )
    elif ctx.is_standalone_subprocess:
        columns = run_in_spark_process(
            get_table_schema_subprocess,
            ctx.settings_dict,
            database=request.database,
            table=request.table,
            app_name=ctx.app_name,
            operation_name="get_table_schema",
        )
    else:
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
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> DatabaseStructureResponse:
    auth_token = None
    if request.filter_by_namespace:
        auth_token = get_token_from_request(http_request)
        if not auth_token:
            raise MissingTokenError(
                "Authorization token required for namespace filtering"
            )

    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            structure = trino_data_store.get_db_structure_trino(
                conn=trino_ctx.connection,
                with_schema=request.with_schema,
                use_hms=request.use_hms,
                filter_by_namespace=request.filter_by_namespace,
                auth_token=auth_token,
            )
    elif ctx.is_standalone_subprocess:
        structure = run_in_spark_process(
            get_db_structure_subprocess,
            ctx.settings_dict,
            with_schema=request.with_schema,
            use_hms=request.use_hms,
            filter_by_namespace=request.filter_by_namespace,
            auth_token=auth_token,
            app_name=ctx.app_name,
            operation_name="get_db_structure",
        )
    else:
        settings = get_settings()
        structure = cast(
            dict[str, list[str] | dict[str, list[str]]],
            data_store.get_db_structure(
                spark=ctx.spark,
                with_schema=request.with_schema,
                use_hms=request.use_hms,
                return_json=False,
                filter_by_namespace=request.filter_by_namespace,
                auth_token=auth_token,
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
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableCountResponse:
    username = auth.user if auth else None
    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            count = trino_service.count_via_trino(
                conn=trino_ctx.connection,
                database=request.database,
                table=request.table,
                username=username,
            )
    elif ctx.is_standalone_subprocess:
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
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> TableSampleResponse:
    username = auth.user if auth else None
    engine = resolve_engine()

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            sample: List[Dict[str, Any]] = trino_service.sample_via_trino(
                conn=trino_ctx.connection,
                database=request.database,
                table=request.table,
                limit=request.limit,
                columns=request.columns,
                where_clause=request.where_clause,
                username=username,
            )
    elif ctx.is_standalone_subprocess:
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
        sample = delta_service.sample_delta_table(
            spark=ctx.spark,
            database=request.database,
            table=request.table,
            limit=request.limit,
            columns=request.columns,
            where_clause=request.where_clause,
            username=username,
        )
    return TableSampleResponse(sample=sample)


# ---------------------------------------------------------------------------
# Query endpoints (support per-request engine override + deprecated)
# ---------------------------------------------------------------------------


@router.post(
    "/tables/query",
    response_model=TableQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="[DEPRECATED] Query Delta tables with pagination",
    description=(
        "Executes a SQL query against Delta tables with pagination support. "
        "Returns query results along with pagination metadata including total count. "
        "Use limit and offset parameters to paginate through large result sets. "
        "IMPORTANT: Include an ORDER BY clause in your query for deterministic "
        "pagination results. Without ORDER BY, rows may appear in different order "
        "across pages, causing duplicates or missing records."
        "\n\nDEPRECATED: This endpoint is deprecated. Use endpoint /delta/tables/query/async/submit for asynchronous query execution."
    ),
    operation_id="query_delta_table",
)
def query_table(
    request: TableQueryRequest,
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    response: Response,
    auth=Depends(auth),
) -> TableQueryResponse:
    """
    Endpoint to execute a query against Delta tables with pagination support.

    DEPRECATED: Use async endpoint /delta/tables/query/async/submit instead.
    """
    response.headers["Warning"] = (
        '299 - "Deprecated API: Use endpoint /delta/tables/query/async/submit for asynchronous query execution"'
    )

    username = auth.user if auth else None
    engine = resolve_engine(request.engine)

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            return execute_query_trino(
                trino_ctx.connection,
                request.query,
                request.limit,
                request.offset,
                username,
            )

    return execute_query(ctx, request.query, request.limit, request.offset, username)


@router.post(
    "/tables/select",
    response_model=TableSelectResponse,
    status_code=status.HTTP_200_OK,
    summary="[DEPRECATED] Execute a structured SELECT query",
    description=(
        "Builds and executes a SELECT query from structured parameters. "
        "Supports column selection, aggregations (COUNT, SUM, AVG, MIN, MAX), "
        "JOINs, WHERE filters, GROUP BY, HAVING, ORDER BY, DISTINCT, and pagination. "
        "The backend builds the query safely, preventing SQL injection."
        "\n\nDEPRECATED: This endpoint is deprecated. Use endpoint /delta/tables/query/async/submit for asynchronous query execution."
    ),
    operation_id="select_delta_table",
)
def select_table(
    request: TableSelectRequest,
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    response: Response,
    auth=Depends(auth),
) -> TableSelectResponse:
    """
    Endpoint to execute a structured SELECT query with pagination support.

    DEPRECATED: Use async endpoint /delta/tables/query/async/submit instead.
    """
    response.headers["Warning"] = (
        '299 - "Deprecated API: Use endpoint /delta/tables/query/async/submit for asynchronous query execution"'
    )

    username = auth.user if auth else None
    engine = resolve_engine(request.engine)

    if engine == QueryEngine.TRINO:
        with _make_trino_ctx(http_request) as trino_ctx:
            return trino_service.select_via_trino(
                conn=trino_ctx.connection,
                request=request,
                username=username,
            )

    if ctx.is_standalone_subprocess:
        request_dict = request.model_dump()
        result = run_in_spark_process(
            select_table_subprocess,
            ctx.settings_dict,
            request_dict=request_dict,
            username=username,
            app_name=ctx.app_name,
            operation_name="select_table",
        )
        return TableSelectResponse(
            data=result["data"],
            pagination=PaginationInfo(**result["pagination"]),
        )
    else:
        return delta_service.select_from_delta_table(
            spark=ctx.spark,
            request=request,
            username=username,
        )
