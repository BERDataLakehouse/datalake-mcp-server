"""
Trino-native metadata operations.

For ``use_hms=True`` (the default), delegates to the existing Hive Metastore
client — both Spark and Trino share the same HMS, so the results are identical.
For ``use_hms=False``, executes SHOW SCHEMAS / SHOW TABLES / SHOW COLUMNS
via the Trino cursor.
"""

import logging
from typing import Any

import trino

from src.delta_lake import hive_metastore
from src.delta_lake.data_store import (
    _get_user_namespace_prefixes,
    _get_accessible_paths,
    _extract_databases_from_paths,
)
from src.service.exceptions import TrinoOperationError
from src.trino_engine.trino_service import _validate_trino_identifier
from src.settings import BERDLSettings, get_settings

logger = logging.getLogger(__name__)


def get_databases_trino(
    conn: trino.dbapi.Connection,
    use_hms: bool = True,
    filter_by_namespace: bool = False,
    auth_token: str | None = None,
    settings: BERDLSettings | None = None,
) -> list[str]:
    """List databases, optionally filtered by user namespace."""
    if filter_by_namespace and not auth_token:
        raise ValueError("auth_token is required when filter_by_namespace is True")

    if settings is None:
        settings = get_settings()

    if use_hms:
        databases = hive_metastore.get_databases(settings=settings)
    else:
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW SCHEMAS")
            databases = sorted(row[0] for row in cursor.fetchall())
        except Exception as e:
            raise TrinoOperationError(f"Failed to list databases via Trino: {e}") from e

    if filter_by_namespace:
        try:
            prefixes = _get_user_namespace_prefixes(auth_token)
            owned = (
                [db for db in databases if db.startswith(tuple(prefixes))]
                if prefixes
                else []
            )
            accessible_paths = _get_accessible_paths(auth_token)
            shared = _extract_databases_from_paths(accessible_paths)
            all_accessible = set(owned) | set(shared)
            databases = sorted(db for db in databases if db in all_accessible)
        except Exception:
            logger.error("Error filtering databases by namespace", exc_info=True)
            raise

    return databases


def get_tables_trino(
    conn: trino.dbapi.Connection,
    database: str,
    use_hms: bool = True,
    settings: BERDLSettings | None = None,
) -> list[str]:
    """List tables in a database."""
    if settings is None:
        settings = get_settings()

    if use_hms:
        return hive_metastore.get_tables(database=database, settings=settings)

    _validate_trino_identifier(database, "database")
    try:
        cursor = conn.cursor()
        cursor.execute(f'SHOW TABLES FROM "{database}"')
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
    _validate_trino_identifier(database, "database")
    _validate_trino_identifier(table, "table")
    try:
        cursor = conn.cursor()
        cursor.execute(f'SHOW COLUMNS FROM "{database}"."{table}"')
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        raise TrinoOperationError(
            f"Failed to get schema for {database}.{table} via Trino: {e}"
        ) from e


def get_db_structure_trino(
    conn: trino.dbapi.Connection,
    with_schema: bool = False,
    use_hms: bool = True,
    filter_by_namespace: bool = True,
    auth_token: str | None = None,
    settings: BERDLSettings | None = None,
) -> dict[str, Any]:
    """Get the structure of databases, filtered by user namespace by default."""
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
