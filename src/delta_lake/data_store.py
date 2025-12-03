"""Module for interacting with Spark databases and tables.

This module provides functions to retrieve information about databases, tables,
and their schemas from a Spark cluster or directly from Hive metastore in PostgreSQL.

Uses berdl_notebook_utils for shared functionality with the notebook environment.
"""

import atexit
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import httpx
from pyspark.sql import SparkSession

# Use local MCP copies that don't rely on environment variables
from src.delta_lake import hive_metastore

# Use shared utilities from berdl_notebook_utils for consistency with notebooks
from berdl_notebook_utils.spark import data_store as notebook_data_store

from src.settings import BERDLSettings, get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP CLIENT MANAGEMENT
# =============================================================================
# Shared httpx client with connection pooling for governance API calls.
# Creating a new client per request is inefficient and can exhaust connections.


@lru_cache(maxsize=1)
def _get_http_client() -> httpx.Client:
    """
    Get a shared httpx Client with connection pooling.

    The client is cached and reused across all requests for efficiency.
    Uses reasonable defaults for timeouts and connection limits.

    Returns:
        Shared httpx.Client instance
    """
    logger.info("Initializing shared httpx client for governance API")
    client = httpx.Client(
        timeout=httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=30.0,  # Read timeout
            write=10.0,  # Write timeout
            pool=5.0,  # Pool timeout
        ),
        limits=httpx.Limits(
            max_connections=100,  # Total max connections in pool
            max_keepalive_connections=20,  # Max idle keepalive connections
            keepalive_expiry=30.0,  # Keepalive connection expiry in seconds
        ),
    )
    return client


def _close_http_client() -> None:
    """Close the shared HTTP client on shutdown."""
    if _get_http_client.cache_info().currsize > 0:
        try:
            client = _get_http_client()
            client.close()
            logger.info("Shared httpx client closed")
        except Exception as e:
            logger.warning(f"Error closing httpx client: {e}")


# Register cleanup on process exit
atexit.register(_close_http_client)

# Re-export get_table_schema from berdl_notebook_utils (works identically in both contexts)
get_table_schema = notebook_data_store.get_table_schema

# get_tables is customized below to support use_hms parameter with settings


def _execute_with_spark(
    func: Any, spark: Optional[SparkSession] = None, *args, **kwargs
) -> Any:
    """
    Execute a function with a SparkSession.

    In the MCP server context, spark must be provided via FastAPI dependency injection.
    """
    if spark is None:
        raise ValueError(
            "SparkSession must be provided. In MCP server context, use FastAPI dependency injection."
        )
    return func(spark, *args, **kwargs)


def _format_output(data: Any, return_json: bool = True) -> Union[str, Any]:
    """
    Format the output based on the return_json flag.
    """
    return json.dumps(data) if return_json else data


def _extract_databases_from_paths(paths: List[str]) -> List[str]:
    """
    Extract unique database names from S3 SQL warehouse paths.

    Only considers paths in SQL warehouses (not general warehouses or logs):
    - s3a://cdm-lake/users-sql-warehouse/...
    - s3a://cdm-lake/tenant-sql-warehouse/...

    S3 paths are in format: s3a://bucket/warehouse/user_or_tenant/database.db/...
    Example: s3a://cdm-lake/users-sql-warehouse/tgu1/u_tgu1__sharing_test.db/employee_records_1/

    Args:
        paths: List of S3 paths from accessible paths API

    Returns:
        List of unique database names (without .db suffix)
    """
    databases = set()
    for path in paths:
        # Only process paths from SQL warehouses
        if not any(
            warehouse in path
            for warehouse in ["/users-sql-warehouse/", "/tenant-sql-warehouse/"]
        ):
            continue

        # Remove s3a:// prefix and split by /
        parts = path.replace("s3a://", "").split("/")

        # Look for .db directory (database directory in Hive convention)
        for part in parts:
            if part.endswith(".db"):
                db_name = part[:-3]  # Remove .db suffix
                databases.add(db_name)
                break

    return sorted(list(databases))


def _get_user_namespace_prefixes(auth_token: str) -> List[str]:
    """
    Get all namespace prefixes for the authenticated user (user + all groups).

    Args:
        auth_token: KBase authentication token

    Returns:
        List of namespace prefixes (user prefix + all group/tenant prefixes)
    """
    settings = get_settings()
    governance_url = str(settings.GOVERNANCE_API_URL).rstrip("/")
    prefixes = []

    headers = {"Authorization": f"Bearer {auth_token}"}
    client = _get_http_client()

    try:
        # Get user's namespace prefix
        response = client.get(
            f"{governance_url}/workspaces/me/namespace-prefix", headers=headers
        )
        response.raise_for_status()
        data = response.json()
        user_prefix = data.get("user_namespace_prefix")
        if user_prefix:
            prefixes.append(user_prefix)
            logger.debug(f"User namespace prefix: {user_prefix}")

        # Get user's groups
        response = client.get(f"{governance_url}/workspaces/me/groups", headers=headers)
        response.raise_for_status()
        groups_data = response.json()
        groups = groups_data.get("groups", [])
        logger.debug(f"User groups: {groups}")

        # Get namespace prefix for each group
        for group_name in groups:
            try:
                response = client.get(
                    f"{governance_url}/workspaces/me/namespace-prefix",
                    params={"tenant": group_name},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                tenant_prefix = data.get("tenant_namespace_prefix")
                if tenant_prefix:
                    prefixes.append(tenant_prefix)
                    logger.debug(
                        f"Tenant '{group_name}' namespace prefix: {tenant_prefix}"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not get namespace prefix for group {group_name}: {e}"
                )
                # Continue with other groups

        return prefixes

    except Exception as e:
        logger.error(f"Error fetching namespace prefixes from governance API: {e}")
        raise Exception(f"Could not filter databases by namespace: {e}") from e


def _get_accessible_paths(auth_token: str) -> List[str]:
    """
    Get all S3 paths accessible to the user from the governance API.

    Args:
        auth_token: KBase authentication token

    Returns:
        List of accessible S3 paths
    """
    settings = get_settings()
    governance_url = str(settings.GOVERNANCE_API_URL).rstrip("/")
    headers = {"Authorization": f"Bearer {auth_token}"}
    client = _get_http_client()

    try:
        response = client.get(
            f"{governance_url}/workspaces/me/accessible-paths", headers=headers
        )
        response.raise_for_status()
        data = response.json()
        accessible_paths = data.get("accessible_paths", [])
        logger.debug(f"Retrieved {len(accessible_paths)} accessible paths")
        return accessible_paths

    except Exception as e:
        logger.error(f"Error fetching accessible paths from governance API: {e}")
        raise Exception(f"Could not get accessible paths: {e}") from e


def _get_tables_with_schemas(
    db: str, tables: List[str], spark: SparkSession
) -> Dict[str, Any]:
    """
    Get schemas for a list of tables in a database.
    """
    return {
        table: get_table_schema(
            database=db, table=table, spark=spark, return_json=False
        )
        for table in tables
    }


def get_databases(
    spark: Optional[SparkSession] = None,
    use_hms: bool = True,
    return_json: bool = True,
    filter_by_namespace: bool = False,
    auth_token: Optional[str] = None,
    settings: Optional[BERDLSettings] = None,
) -> Union[str, List[str]]:
    """
    Get the list of databases in the Hive metastore.

    Args:
        spark: Optional SparkSession to use (if use_hms is False)
        use_hms: Whether to use Hive Metastore client direct query (faster) or Spark
        return_json: Whether to return JSON string or raw data
        filter_by_namespace: Whether to filter databases by user/group ownership AND shared access.
                           When True, returns:
                           - User's owned databases (u_username_*)
                           - Group/tenant databases (groupname_*)
                           - Databases shared with the user (from accessible paths API)
                           When False, returns all databases in the metastore.
        auth_token: KBase auth token (required if filter_by_namespace is True)
        settings: BERDLSettings instance (required if use_hms is True)

    Returns:
        List of database names, either as JSON string or raw list

    Raises:
        ValueError: If filter_by_namespace is True but auth_token is not provided
        ValueError: If use_hms is True but settings is not provided
    """

    # Validate early: auth_token is required if filtering by namespace
    if filter_by_namespace and not auth_token:
        raise ValueError("auth_token is required when filter_by_namespace is True")

    def _get_dbs(session: SparkSession) -> List[str]:
        return [db.name for db in session.catalog.listDatabases()]

    if use_hms:
        if settings is None:
            settings = get_settings()
        databases = hive_metastore.get_databases(settings=settings)
    else:
        databases = _execute_with_spark(_get_dbs, spark)

    # Apply filtering: owned databases (fast) + shared databases (API call)
    if filter_by_namespace:
        try:
            # Step 1: Get owned/group databases using namespace prefixes (fast)
            prefixes = _get_user_namespace_prefixes(auth_token)

            if prefixes:
                # Filter databases by namespace prefixes (owned + group databases)
                owned_databases = [
                    db for db in databases if db.startswith(tuple(prefixes))
                ]
                logger.info(
                    f"Found {len(owned_databases)} owned/group databases matching {len(prefixes)} prefix(es)"
                )
            else:
                logger.warning("No namespace prefixes found")
                owned_databases = []

            # Step 2: Get shared databases from accessible paths API
            # These are databases shared with the user that don't match their namespace
            accessible_paths = _get_accessible_paths(auth_token)
            shared_databases = _extract_databases_from_paths(accessible_paths)
            logger.info(
                f"Found {len(shared_databases)} shared databases from accessible paths"
            )

            # Combine owned and shared, remove duplicates
            all_accessible = set(owned_databases) | set(shared_databases)

            # Filter to only databases that exist in metastore
            databases = sorted([db for db in databases if db in all_accessible])
            logger.info(
                f"Total accessible databases: {len(databases)} (owned: {len(owned_databases)}, shared: {len(shared_databases)})"
            )

        except Exception as e:
            logger.error(f"Error filtering databases by namespace: {e}")
            raise

    return _format_output(databases, return_json)


def get_tables(
    database: str,
    spark: Optional[SparkSession] = None,
    use_hms: bool = True,
    return_json: bool = True,
    settings: Optional[BERDLSettings] = None,
) -> Union[str, List[str]]:
    """
    Get the list of tables in a database.

    Args:
        database: Name of the database
        spark: Optional SparkSession to use (if use_hms is False)
        use_hms: Whether to use Hive Metastore client direct query (faster) or Spark
        return_json: Whether to return JSON string or raw data
        settings: BERDLSettings instance (required if use_hms is True)

    Returns:
        List of table names, either as JSON string or raw list
    """

    def _get_tbls(session: SparkSession) -> List[str]:
        return [t.name for t in session.catalog.listTables(database)]

    if use_hms:
        if settings is None:
            settings = get_settings()
        tables = hive_metastore.get_tables(database=database, settings=settings)
    else:
        tables = _execute_with_spark(_get_tbls, spark)

    return _format_output(tables, return_json)


def get_db_structure(
    spark: Optional[SparkSession] = None,
    with_schema: bool = False,
    use_hms: bool = True,
    return_json: bool = True,
    settings: Optional[BERDLSettings] = None,
) -> Union[str, Dict]:
    """Get the structure of all databases in the Hive metastore.

    Args:
        spark: Optional SparkSession to use for operations
        with_schema: Whether to include table schemas
        use_hms: Whether to use Hive Metastore client for metadata retrieval
        return_json: Whether to return the result as a JSON string
        settings: BERDLSettings instance (required if use_hms is True)

    Returns:
        Database structure as either JSON string or dictionary:
        {
            "database_name": ["table1", "table2"] or
            "database_name": {
                "table1": ["column1", "column2"],
                "table2": ["column1", "column2"]
            }
        }
    """

    def _get_structure(
        session: SparkSession,
    ) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
        db_structure = {}
        databases = get_databases(spark=session, return_json=False)

        for db in databases:
            tables = get_tables(database=db, spark=session, return_json=False)
            if with_schema and isinstance(tables, list):
                db_structure[db] = _get_tables_with_schemas(db, tables, session)
            else:
                db_structure[db] = tables

        return db_structure

    if use_hms:
        if settings is None:
            settings = get_settings()
        db_structure = {}
        databases = hive_metastore.get_databases(settings=settings)

        for db in databases:
            tables = hive_metastore.get_tables(database=db, settings=settings)
            if with_schema and isinstance(tables, list):
                if spark is None:
                    raise ValueError(
                        "SparkSession must be provided for schema retrieval. "
                        "In MCP server context, use FastAPI dependency injection."
                    )
                db_structure[db] = _get_tables_with_schemas(db, tables, spark)
            else:
                db_structure[db] = tables

    else:
        db_structure = _execute_with_spark(_get_structure, spark)

    return _format_output(db_structure, return_json)


def database_exists(
    database: str,
    spark: Optional[SparkSession] = None,
    use_hms: bool = True,
    settings: Optional[BERDLSettings] = None,
) -> bool:
    """
    Check if a database exists in the Hive metastore.
    """
    if settings is None:
        settings = get_settings()
    return database in get_databases(
        spark=spark, use_hms=use_hms, return_json=False, settings=settings
    )


def table_exists(
    database: str,
    table: str,
    spark: Optional[SparkSession] = None,
    use_hms: bool = True,
    settings: Optional[BERDLSettings] = None,
) -> bool:
    """
    Check if a table exists in a database.
    """
    if settings is None:
        settings = get_settings()
    return table in get_tables(
        database=database,
        spark=spark,
        use_hms=use_hms,
        return_json=False,
        settings=settings,
    )
