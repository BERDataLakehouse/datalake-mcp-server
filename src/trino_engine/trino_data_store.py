"""
Trino-native metadata operations.

For ``use_hms=True``, delegates to the existing Hive Metastore client for
legacy Delta/Hive tables. By default, executes Trino-native metadata queries so
Polaris Iceberg catalogs are visible in the same ``catalog.namespace`` shape
used by the Spark data-store layer.
"""

import logging
from typing import Any

import trino

from src.delta_lake import hive_metastore
from src.service.exceptions import TrinoOperationError
from src.trino_engine.trino_service import _validate_trino_identifier
from src.settings import BERDLSettings, get_settings

logger = logging.getLogger(__name__)

_EXCLUDED_CATALOGS = {"system"}
_EXCLUDED_SCHEMAS = {"information_schema"}


def _quote_trino_identifier(identifier: str) -> str:
    """Quote a single Trino identifier component."""
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def _quote_qualified_identifier(
    identifier: str,
    identifier_type: str = "database",
) -> str:
    """Validate and quote a dotted Trino identifier."""
    _validate_trino_identifier(identifier, identifier_type)
    return ".".join(_quote_trino_identifier(part) for part in identifier.split("."))


def get_databases_trino(
    conn: trino.dbapi.Connection,
    use_hms: bool = False,
    filter_by_namespace: bool = False,
    auth_token: str | None = None,
    settings: BERDLSettings | None = None,
) -> list[str]:
    """List databases via Trino or HMS."""
    if settings is None:
        settings = get_settings()

    if use_hms:
        databases = hive_metastore.get_databases(settings=settings)
    else:
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW CATALOGS")
            catalogs = sorted(
                row[0] for row in cursor.fetchall() if row[0] not in _EXCLUDED_CATALOGS
            )
            databases = []
            for catalog in catalogs:
                try:
                    cursor.execute(
                        f"SHOW SCHEMAS FROM {_quote_trino_identifier(catalog)}"
                    )
                    for row in cursor.fetchall():
                        schema = row[0]
                        if schema not in _EXCLUDED_SCHEMAS:
                            databases.append(f"{catalog}.{schema}")
                except Exception as e:
                    logger.warning(
                        "Failed to list schemas in Trino catalog '%s': %s",
                        catalog,
                        e,
                    )
            databases = sorted(databases)
        except Exception as e:
            raise TrinoOperationError(f"Failed to list databases via Trino: {e}") from e

    return databases


def get_tables_trino(
    conn: trino.dbapi.Connection,
    database: str,
    use_hms: bool = False,
    settings: BERDLSettings | None = None,
) -> list[str]:
    """List tables in a database."""
    if settings is None:
        settings = get_settings()

    if use_hms:
        return hive_metastore.get_tables(database=database, settings=settings)

    quoted_database = _quote_qualified_identifier(database, "database")
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW TABLES FROM {quoted_database}")
        return sorted(row[0] for row in cursor.fetchall())
    except Exception as e:
        raise TrinoOperationError(
            f"Failed to list tables in {database} via Trino: {e}"
        ) from e


def get_table_schema_trino(
    conn: trino.dbapi.Connection,
    database: str,
    table: str,
) -> list[str]:
    """Get column names for a table via Trino SHOW COLUMNS."""
    quoted_database = _quote_qualified_identifier(database, "database")
    quoted_table = _quote_qualified_identifier(table, "table")
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW COLUMNS FROM {quoted_database}.{quoted_table}")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        raise TrinoOperationError(
            f"Failed to get schema for {database}.{table} via Trino: {e}"
        ) from e


def get_db_structure_trino(
    conn: trino.dbapi.Connection,
    with_schema: bool = False,
    use_hms: bool = False,
    filter_by_namespace: bool = False,
    auth_token: str | None = None,
    settings: BERDLSettings | None = None,
) -> dict[str, Any]:
    """Get the structure of databases, optionally filtered by user namespace."""
    if settings is None:
        settings = get_settings()

    databases = get_databases_trino(
        conn,
        use_hms=use_hms,
        filter_by_namespace=filter_by_namespace,
        auth_token=auth_token,
        settings=settings,
    )

    structure: dict[str, Any] = {}
    for db in databases:
        tables = get_tables_trino(conn, database=db, use_hms=use_hms, settings=settings)
        if with_schema:
            structure[db] = {
                t: get_table_schema_trino(conn, database=db, table=t) for t in tables
            }
        else:
            structure[db] = tables

    return structure
