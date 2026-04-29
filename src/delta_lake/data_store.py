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

    Returns:
        Sorted list of database identifiers (Iceberg ``catalog.namespace`` and
        Hive flat names interleaved)

    Raises:
        ValueError: If spark is not provided
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
    filter_by_namespace: bool = False,
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

    databases = get_databases(spark=spark, return_json=False)

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
    """Check if a database (catalog.namespace) exists."""
    return database in get_databases(spark=spark, return_json=False)


def table_exists(
    database: str,
    table: str,
    spark: Optional[SparkSession] = None,
) -> bool:
    """Check if a table exists in a database (catalog.namespace)."""
    return table in get_tables(database=database, spark=spark, return_json=False)
