"""Module for interacting with Iceberg catalog databases and tables.

This module provides functions to retrieve information about databases (namespaces),
tables, and their schemas from Iceberg catalogs via Spark SQL.

Iceberg catalogs use 3-level naming: catalog.namespace.table
- Personal catalog: ``my`` (e.g., ``my.demo.employees``)
- Tenant catalogs: ``globalusers``, ``kbase``, etc. (e.g., ``kbase.shared.dataset``)

Functions return "database" identifiers in ``catalog.namespace`` format
(e.g., ``my.demo``, ``globalusers.shared_data``), which can be used directly
in Spark SQL queries: ``SELECT * FROM {database}.{table}``.

Uses berdl_notebook_utils for shared functionality with the notebook environment.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from pyspark.sql import SparkSession

from src.delta_lake.setup_spark_session import (
    _get_personal_catalog_aliases,
    _get_tenant_catalog_alias,
)

logger = logging.getLogger(__name__)

# Catalogs to exclude from Iceberg listing (non-Iceberg catalogs)
_EXCLUDED_CATALOGS = {"spark_catalog"}

# Pattern to extract catalog name from spark.sql.catalog.<name> keys
# Matches top-level catalog registration keys only (no sub-properties)
_CATALOG_KEY_PATTERN = re.compile(r"^spark\.sql\.catalog\.([a-zA-Z_][a-zA-Z0-9_]*)$")


def _format_output(data: Any, return_json: bool = True) -> Union[str, Any]:
    """Format the output based on the return_json flag."""
    return json.dumps(data) if return_json else data


def _list_iceberg_catalogs(spark: SparkSession) -> List[str]:
    """List all Iceberg catalogs (excluding spark_catalog).

    In Spark 4.0 with Spark Connect, ``SHOW CATALOGS`` only returns catalogs
    registered in the client session's CatalogManager. Catalogs configured
    server-side (via ``spark-defaults.conf``) are accessible for queries but
    invisible to ``SHOW CATALOGS``.

    This function discovers catalogs by inspecting the Spark SQL configuration
    via the ``SET`` command, which returns all server-side configs through the
    Spark Connect gRPC channel. It looks for top-level
    ``spark.sql.catalog.<name>`` keys to identify registered catalogs.
    """
    rows = spark.sql("SET").collect()
    catalog_names = set()
    for row in rows:
        match = _CATALOG_KEY_PATTERN.match(row["key"])
        if match:
            catalog_names.add(match.group(1))
    logger.info(
        f"Discovered {len(catalog_names)} catalog(s) from Spark config: "
        f"{sorted(catalog_names)}"
    )
    return sorted(c for c in catalog_names if c not in _EXCLUDED_CATALOGS)


def _settings_get(settings: Any, key: str) -> Any:
    """Read ``key`` from a BERDLSettings instance or a dict."""
    if isinstance(settings, dict):
        return settings.get(key)
    return getattr(settings, key, None)


def _aliases_for_user(settings: Any) -> tuple[set[str], set[str]]:
    """Return ``(personal_aliases, tenant_aliases)`` for the current user.

    Personal aliases include ``my`` plus the sanitized stripped form of
    ``POLARIS_PERSONAL_CATALOG`` (e.g. ``user_alice`` → ``alice``). Tenant
    aliases are derived from each entry of ``POLARIS_TENANT_CATALOGS`` with
    the ``tenant_`` prefix stripped (e.g. ``tenant_globalusers`` →
    ``globalusers``).
    """
    personal = set(
        _get_personal_catalog_aliases(
            _settings_get(settings, "POLARIS_PERSONAL_CATALOG")
        )
    )
    tenant: set[str] = set()
    raw_tenants = _settings_get(settings, "POLARIS_TENANT_CATALOGS")
    if raw_tenants:
        for cat in str(raw_tenants).split(","):
            cat = cat.strip()
            if cat:
                tenant.add(_get_tenant_catalog_alias(cat))
    return personal, tenant


def _filter_to_user_namespaces(
    databases: List[str],
    username: str,
    personal_aliases: set[str],
    tenant_aliases: set[str],
) -> List[str]:
    """Filter ``databases`` to those owned by the user or accessible via tenants.

    Iceberg ``catalog.namespace`` entries are kept when the catalog matches
    the user's personal aliases or one of their tenant aliases. Hive flat
    names (Delta) are kept when they start with ``u_{username}__`` (own
    personal) or ``{tenant}_`` for any tenant the user belongs to.
    """
    user_prefix = f"u_{username}__"
    allowed_catalogs = personal_aliases | tenant_aliases

    result: List[str] = []
    for db in databases:
        if "." in db:
            catalog = db.split(".", 1)[0]
            if catalog in allowed_catalogs:
                result.append(db)
        else:
            if db.startswith(user_prefix):
                result.append(db)
            elif any(db.startswith(f"{t}_") for t in tenant_aliases):
                result.append(db)
    return result


def _list_hive_databases(spark: SparkSession) -> List[str]:
    """List Hive databases registered under spark_catalog (Delta tables).

    Iceberg catalogs use 3-level naming and are listed via ``SHOW NAMESPACES``;
    Delta/Hive databases live in ``spark_catalog`` (which is the Spark default
    catalog and is bound to ``DeltaCatalog`` in this service). They show up as
    flat names (e.g., ``u_alice__demo``) and are queryable as
    ``SELECT * FROM {database}.{table}`` because Spark resolves unqualified
    references against ``spark_catalog``.
    """
    try:
        rows = spark.sql("SHOW DATABASES IN spark_catalog").collect()
        return sorted(row["namespace"] for row in rows)
    except Exception as e:
        logger.warning(f"Failed to list Hive databases: {e}")
        return []


def get_databases(
    spark: Optional[SparkSession] = None,
    return_json: bool = True,
    filter_by_namespace: bool = True,
    settings: Optional[Any] = None,
) -> Union[str, List[str]]:
    """
    List all accessible databases across Iceberg and Hive catalogs.

    Iceberg namespaces are returned in ``catalog.namespace`` format (e.g.,
    ``my.demo``, ``globalusers.shared_data``). Hive databases (Delta tables
    registered under ``spark_catalog``) are returned as flat names (e.g.,
    ``u_alice__demo``). All forms can be used directly in table references:
    ``SELECT * FROM {database}.{table}``.

    Args:
        spark: SparkSession to use (required in MCP server context)
        return_json: Whether to return JSON string or raw list
        filter_by_namespace: Defaults to True — restrict the returned list to
            databases owned by the current user or accessible via the user's
            tenant catalogs. Requires ``settings`` with ``USER``,
            ``POLARIS_PERSONAL_CATALOG``, and ``POLARIS_TENANT_CATALOGS``
            populated (BERDLSettings instance or dict). Pass False to bypass
            filtering (e.g. for admin tooling).
        settings: Per-user settings used for namespace filtering.

    Returns:
        Sorted list of database identifiers (Iceberg ``catalog.namespace`` and
        Hive flat names interleaved)

    Raises:
        ValueError: If spark is not provided, or if ``filter_by_namespace`` is
            True but ``settings`` lacks the required identity fields.
    """
    if spark is None:
        raise ValueError(
            "SparkSession must be provided. In MCP server context, use FastAPI dependency injection."
        )

    catalogs = _list_iceberg_catalogs(spark)
    logger.info(f"Found {len(catalogs)} Iceberg catalog(s): {catalogs}")

    databases = []
    for catalog in catalogs:
        try:
            namespaces = spark.sql(f"SHOW NAMESPACES IN {catalog}").collect()
            for row in namespaces:
                databases.append(f"{catalog}.{row['namespace']}")
        except Exception as e:
            logger.warning(f"Failed to list namespaces in catalog '{catalog}': {e}")

    databases.extend(_list_hive_databases(spark))

    if filter_by_namespace:
        if settings is None:
            raise ValueError("settings must be provided when filter_by_namespace=True")
        username = _settings_get(settings, "USER")
        if not username:
            raise ValueError("settings.USER must be set when filter_by_namespace=True")
        personal, tenant = _aliases_for_user(settings)
        databases = _filter_to_user_namespaces(databases, username, personal, tenant)

    return _format_output(sorted(databases), return_json)


def get_tables(
    database: str,
    spark: Optional[SparkSession] = None,
    return_json: bool = True,
) -> Union[str, List[str]]:
    """
    List all tables in a specific Iceberg namespace.

    Args:
        database: Namespace in ``catalog.namespace`` format (e.g., ``my.demo``)
        spark: SparkSession to use (required in MCP server context)
        return_json: Whether to return JSON string or raw data

    Returns:
        List of table names

    Raises:
        ValueError: If spark is not provided
    """
    if spark is None:
        raise ValueError(
            "SparkSession must be provided. In MCP server context, use FastAPI dependency injection."
        )

    try:
        rows = spark.sql(f"SHOW TABLES IN {database}").collect()
        tables = sorted(row["tableName"] for row in rows)
    except Exception:
        tables = []

    return _format_output(tables, return_json)


def get_table_schema(
    database: str,
    table: str,
    spark: Optional[SparkSession] = None,
    return_json: bool = True,
) -> Union[str, List[str]]:
    """
    Get the column names of a specific table.

    Args:
        database: Namespace in ``catalog.namespace`` format (e.g., ``my.demo``)
        table: Name of the table
        spark: SparkSession to use (required in MCP server context)
        return_json: Whether to return JSON string or raw data

    Returns:
        List of column names

    Raises:
        ValueError: If spark is not provided
    """
    if spark is None:
        raise ValueError(
            "SparkSession must be provided. In MCP server context, use FastAPI dependency injection."
        )

    try:
        rows = spark.sql(f"DESCRIBE {database}.{table}").collect()
        # DESCRIBE returns col_name, data_type, comment — filter out partition/metadata rows
        columns = [
            row["col_name"]
            for row in rows
            if row["col_name"] and not row["col_name"].startswith("#")
        ]
    except Exception:
        logger.error(f"Error retrieving schema for table {table} in {database}")
        columns = []

    return _format_output(columns, return_json)


def get_db_structure(
    spark: Optional[SparkSession] = None,
    with_schema: bool = False,
    return_json: bool = True,
    filter_by_namespace: bool = True,
    auth_token: Optional[str] = None,
    settings: Optional[Any] = None,
) -> Union[str, Dict]:
    """
    Get the structure of all accessible Iceberg namespaces.

    Args:
        spark: SparkSession to use (required in MCP server context)
        with_schema: Whether to include table column names
        return_json: Whether to return the result as a JSON string
        filter_by_namespace: Whether to filter databases by user/group ownership
                           and shared access (delegates to get_databases)
        auth_token: KBase auth token (required if filter_by_namespace is True)
        settings: BERDLSettings instance (unused in Iceberg mode, kept for API compat)

    Returns:
        Dictionary mapping ``catalog.namespace`` to table lists or schema dicts::

            {
                "my.demo": ["table1", "table2"],
                "globalusers.shared": ["dataset"]
            }

        Or with ``with_schema=True``::

            {
                "my.demo": {
                    "table1": ["col1", "col2"],
                    "table2": ["col1", "col2"]
                }
            }

    Raises:
        ValueError: If spark is not provided
    """
    if spark is None:
        raise ValueError(
            "SparkSession must be provided. In MCP server context, use FastAPI dependency injection."
        )

    databases = get_databases(
        spark=spark,
        return_json=False,
        filter_by_namespace=filter_by_namespace,
        settings=settings,
    )

    db_structure: Dict[str, Any] = {}
    for db in databases:
        tables = get_tables(database=db, spark=spark, return_json=False)
        if with_schema:
            db_structure[db] = {
                tbl: get_table_schema(
                    database=db, table=tbl, spark=spark, return_json=False
                )
                for tbl in tables
            }
        else:
            db_structure[db] = tables

    return _format_output(db_structure, return_json)


def database_exists(
    database: str,
    spark: Optional[SparkSession] = None,
) -> bool:
    """Check if a database physically exists (no namespace filtering).

    This is a presence check used by the count/sample/select endpoints to
    surface a clean 404 before issuing the actual SQL. Polaris/Hive/MinIO
    ACLs still gate the underlying read, so we deliberately bypass the
    user-namespace filter here to keep this purely about physical existence.
    """
    return database in get_databases(
        spark=spark, return_json=False, filter_by_namespace=False
    )


def table_exists(
    database: str,
    table: str,
    spark: Optional[SparkSession] = None,
) -> bool:
    """Check if a table exists in a database (catalog.namespace)."""
    return table in get_tables(database=database, spark=spark, return_json=False)
