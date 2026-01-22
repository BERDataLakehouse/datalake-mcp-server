"""
Subprocess-safe Spark operations for Standalone mode.

This module contains top-level functions that can be pickled and executed
in separate processes via ProcessPoolExecutor. Each function:

1. Creates its own SparkSession in the subprocess
2. Executes the Spark operation
3. Returns serializable results (dicts, lists, primitives)
4. Stops the session to clean up resources

These functions enable true concurrency for Standalone Spark mode by
running each operation in an isolated process with its own JVM.

Usage:
    from src.service.spark_session_pool import run_in_spark_process
    from src.service.standalone_operations import query_table_subprocess

    result = run_in_spark_process(
        query_table_subprocess,
        settings_dict,
        query,
        limit,
        offset,
        username,
        operation_name="query_table"
    )
"""

import logging
from typing import Any

from py4j.protocol import Py4JJavaError
from pydantic import AnyUrl

from src.delta_lake import data_store, delta_service
from src.delta_lake.setup_spark_session import get_spark_session
from src.service.models import TableSelectRequest
from src.settings import BERDLSettings

logger = logging.getLogger(__name__)


def _create_spark_session(settings_dict: dict, app_name: str):
    """
    Create a SparkSession in the subprocess.

    Args:
        settings_dict: Dictionary of BERDLSettings values (picklable)
        app_name: Application name for the Spark session

    Returns:
        SparkSession instance
    """
    # Make a copy to avoid modifying the original dict
    settings_copy = settings_dict.copy()

    # Reconstruct settings from dict, converting URL strings to AnyUrl
    if "SPARK_MASTER_URL" in settings_copy and settings_copy["SPARK_MASTER_URL"]:
        settings_copy["SPARK_MASTER_URL"] = AnyUrl(settings_copy["SPARK_MASTER_URL"])
    if "SPARK_CONNECT_URL" in settings_copy and settings_copy["SPARK_CONNECT_URL"]:
        settings_copy["SPARK_CONNECT_URL"] = AnyUrl(settings_copy["SPARK_CONNECT_URL"])

    settings = BERDLSettings(**settings_copy)

    spark = get_spark_session(
        app_name=app_name,
        settings=settings,
        use_spark_connect=False,  # Always Standalone in subprocess
    )
    return spark


def _cleanup_spark_session(spark) -> None:
    """
    Clean up a SparkSession and associated resources.

    Args:
        spark: SparkSession to clean up
    """
    try:
        # Clear Hadoop FileSystem cache to prevent credential leakage
        try:
            jvm = spark.sparkContext._jvm
            if jvm:
                jvm.org.apache.hadoop.fs.FileSystem.closeAll()
        except Exception:
            pass  # Ignore cleanup errors

        spark.stop()
    except Py4JJavaError:
        pass  # Session may already be stopped
    except Exception:
        pass  # Ignore cleanup errors


def _reconstruct_settings(settings_dict: dict) -> BERDLSettings:
    """
    Reconstruct BERDLSettings from a picklable dict.

    Args:
        settings_dict: Dictionary with settings values

    Returns:
        BERDLSettings instance
    """
    settings_copy = settings_dict.copy()
    if "SPARK_MASTER_URL" in settings_copy and settings_copy["SPARK_MASTER_URL"]:
        settings_copy["SPARK_MASTER_URL"] = AnyUrl(settings_copy["SPARK_MASTER_URL"])
    if "SPARK_CONNECT_URL" in settings_copy and settings_copy["SPARK_CONNECT_URL"]:
        settings_copy["SPARK_CONNECT_URL"] = AnyUrl(settings_copy["SPARK_CONNECT_URL"])
    return BERDLSettings(**settings_copy)


# =============================================================================
# Subprocess-safe operation functions
# =============================================================================


def count_table_subprocess(
    settings_dict: dict,
    database: str,
    table: str,
    username: str | None = None,
    app_name: str = "mcp_count",
) -> int:
    """
    Count rows in a Delta table (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        database: Database name
        table: Table name
        username: Username for cache isolation
        app_name: Spark application name

    Returns:
        Row count
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        return delta_service.count_delta_table(
            spark=spark,
            database=database,
            table=table,
            username=username,
        )
    finally:
        if spark:
            _cleanup_spark_session(spark)


def sample_table_subprocess(
    settings_dict: dict,
    database: str,
    table: str,
    limit: int = 10,
    columns: list[str] | None = None,
    where_clause: str | None = None,
    username: str | None = None,
    app_name: str = "mcp_sample",
) -> list[dict[str, Any]]:
    """
    Sample rows from a Delta table (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        database: Database name
        table: Table name
        limit: Maximum rows to return
        columns: Columns to select (None = all)
        where_clause: Optional WHERE filter
        username: Username for cache isolation
        app_name: Spark application name

    Returns:
        List of row dictionaries
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        return delta_service.sample_delta_table(
            spark=spark,
            database=database,
            table=table,
            limit=limit,
            columns=columns,
            where_clause=where_clause,
            username=username,
        )
    finally:
        if spark:
            _cleanup_spark_session(spark)


def query_table_subprocess(
    settings_dict: dict,
    query: str,
    limit: int = 1000,
    offset: int = 0,
    username: str | None = None,
    app_name: str = "mcp_query",
) -> dict[str, Any]:
    """
    Execute a SQL query against Delta tables (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        query: SQL query string
        limit: Maximum rows to return
        offset: Pagination offset
        username: Username for cache isolation
        app_name: Spark application name

    Returns:
        Dict with 'result' (list of rows) and 'pagination' (dict)
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        response = delta_service.query_delta_table(
            spark=spark,
            query=query,
            limit=limit,
            offset=offset,
            username=username,
        )
        # Convert Pydantic model to dict for serialization
        return {
            "result": response.result,
            "pagination": response.pagination.model_dump(),
        }
    finally:
        if spark:
            _cleanup_spark_session(spark)


def select_table_subprocess(
    settings_dict: dict,
    request_dict: dict,
    username: str | None = None,
    app_name: str = "mcp_select",
) -> dict[str, Any]:
    """
    Execute a structured SELECT query (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        request_dict: Serialized TableSelectRequest as dict
        username: Username for cache isolation
        app_name: Spark application name

    Returns:
        Dict with 'data' (list of rows) and 'pagination' (dict)
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        # Reconstruct request from dict
        request = TableSelectRequest(**request_dict)
        response = delta_service.select_from_delta_table(
            spark=spark,
            request=request,
            username=username,
        )
        # Convert Pydantic model to dict for serialization
        return {
            "data": response.data,
            "pagination": response.pagination.model_dump(),
        }
    finally:
        if spark:
            _cleanup_spark_session(spark)


def list_databases_subprocess(
    settings_dict: dict,
    use_hms: bool = True,
    filter_by_namespace: bool = False,
    auth_token: str | None = None,
    app_name: str = "mcp_list_dbs",
) -> list[str]:
    """
    List databases in the Hive metastore (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        use_hms: Whether to use Hive Metastore direct query
        filter_by_namespace: Filter by user namespace
        auth_token: Auth token for namespace filtering
        app_name: Spark application name

    Returns:
        List of database names
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        settings = _reconstruct_settings(settings_dict)

        result = data_store.get_databases(
            spark=spark,
            use_hms=use_hms,
            return_json=False,
            filter_by_namespace=filter_by_namespace,
            auth_token=auth_token,
            settings=settings,
        )
        return list(result)
    finally:
        if spark:
            _cleanup_spark_session(spark)


def list_tables_subprocess(
    settings_dict: dict,
    database: str,
    use_hms: bool = True,
    app_name: str = "mcp_list_tables",
) -> list[str]:
    """
    List tables in a database (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        database: Database name
        use_hms: Whether to use Hive Metastore direct query
        app_name: Spark application name

    Returns:
        List of table names
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        settings = _reconstruct_settings(settings_dict)

        result = data_store.get_tables(
            database=database,
            spark=spark,
            use_hms=use_hms,
            return_json=False,
            settings=settings,
        )
        return list(result)
    finally:
        if spark:
            _cleanup_spark_session(spark)


def get_table_schema_subprocess(
    settings_dict: dict,
    database: str,
    table: str,
    app_name: str = "mcp_schema",
) -> list[str]:
    """
    Get table schema (column names) (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        database: Database name
        table: Table name
        app_name: Spark application name

    Returns:
        List of column names
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        result = data_store.get_table_schema(
            database=database,
            table=table,
            spark=spark,
            return_json=False,
        )
        return list(result)
    finally:
        if spark:
            _cleanup_spark_session(spark)


def get_db_structure_subprocess(
    settings_dict: dict,
    with_schema: bool = False,
    use_hms: bool = True,
    app_name: str = "mcp_structure",
) -> dict[str, Any]:
    """
    Get database structure (subprocess version).

    Args:
        settings_dict: Picklable dict of BERDLSettings values
        with_schema: Whether to include table schemas
        use_hms: Whether to use Hive Metastore direct query
        app_name: Spark application name

    Returns:
        Dict mapping database names to table lists or schema dicts
    """
    spark = None
    try:
        spark = _create_spark_session(settings_dict, app_name)
        settings = _reconstruct_settings(settings_dict)

        result = data_store.get_db_structure(
            spark=spark,
            with_schema=with_schema,
            use_hms=use_hms,
            return_json=False,
            settings=settings,
        )
        return dict(result)
    finally:
        if spark:
            _cleanup_spark_session(spark)
