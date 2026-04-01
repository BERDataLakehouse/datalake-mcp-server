"""
Trino connection management for the MCP server.

Adapted from berdl_notebook_utils.setup_trino_session for the multi-user
MCP server context.  The notebook version calls get_minio_credentials()
which reads from the notebook's environment; here credentials are passed
in explicitly from the per-request dependency injection layer.

MAINTENANCE NOTE: Helper functions (_sanitize_identifier, _build_catalog_properties,
_create_dynamic_catalog, _catalog_exists, _escape_sql_string, _validate_connector)
are copied from:
    spark_notebook/notebook_utils/berdl_notebook_utils/setup_trino_session.py

When updating, keep them in sync — especially _sanitize_identifier which must
match the Java access-control plugin's sanitizeIdentifier().
"""

import logging
import re

import trino

from src.service.exceptions import TrinoConnectionError

logger = logging.getLogger(__name__)

# Connectors allowed in CREATE CATALOG statements (SQL-injection allowlist).
ALLOWED_CONNECTORS = frozenset({"delta_lake", "hive"})


# ---------------------------------------------------------------------------
# Helpers (copied from setup_trino_session.py — keep in sync)
# ---------------------------------------------------------------------------


def _sanitize_identifier(value: str) -> str:
    """
    Sanitize a string into a valid Trino identifier component.

    IMPORTANT: This logic must match ``sanitizeIdentifier()`` in the Trino
    access-control plugin (BerdlSystemAccessControl.java).
    """
    return re.sub(r"[^a-z0-9_]", "_", value.lower())


def _build_catalog_properties(
    hive_metastore_uri: str,
    minio_endpoint_url: str,
    minio_secure: bool,
    access_key: str,
    secret_key: str,
) -> dict[str, str]:
    """Build the WITH properties for CREATE CATALOG."""
    endpoint_url = str(minio_endpoint_url)
    if not endpoint_url.startswith("http"):
        protocol = "https" if minio_secure else "http"
        endpoint_url = f"{protocol}://{endpoint_url}"

    return {
        "hive.metastore.uri": str(hive_metastore_uri),
        "fs.native-s3.enabled": "true",
        "s3.endpoint": endpoint_url,
        "s3.aws-access-key": access_key,
        "s3.aws-secret-key": secret_key,
        "s3.path-style-access": "true",
        "s3.region": "us-east-1",
    }


def _catalog_exists(cursor: trino.dbapi.Cursor, catalog_name: str) -> bool:
    """Check if a Trino catalog already exists."""
    cursor.execute("SHOW CATALOGS")
    catalogs = [row[0] for row in cursor.fetchall()]
    return catalog_name in catalogs


def _escape_sql_string(value: str) -> str:
    """Escape single quotes in a SQL string literal by doubling them."""
    return value.replace("'", "''")


def _validate_connector(connector: str) -> None:
    """Validate connector name against the allowlist."""
    if connector not in ALLOWED_CONNECTORS:
        raise ValueError(
            f"Connector {connector!r} is not allowed. "
            f"Must be one of: {', '.join(sorted(ALLOWED_CONNECTORS))}"
        )


def _create_dynamic_catalog(
    cursor: trino.dbapi.Cursor,
    catalog_name: str,
    connector: str,
    properties: dict[str, str],
) -> None:
    """Create a dynamic catalog if it doesn't already exist."""
    if _catalog_exists(cursor, catalog_name):
        logger.info(f"Catalog '{catalog_name}' already exists, skipping creation")
        return

    _validate_connector(connector)

    props_sql = ",\n        ".join(
        f"\"{k}\" = '{_escape_sql_string(v)}'" for k, v in properties.items()
    )

    sql = (
        f'CREATE CATALOG IF NOT EXISTS "{catalog_name}" USING {connector}\n'
        f"    WITH (\n        {props_sql}\n    )"
    )

    logger.info(
        f"Creating dynamic Trino catalog: {catalog_name} (connector={connector})"
    )
    cursor.execute(sql)
    cursor.fetchall()
    logger.info(f"Catalog '{catalog_name}' ready")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_trino_connection(
    username: str,
    auth_token: str,
    access_key: str,
    secret_key: str,
    trino_host: str,
    trino_port: int,
    hive_metastore_uri: str,
    minio_endpoint_url: str,
    minio_secure: bool,
    connector: str = "delta_lake",
) -> trino.dbapi.Connection:
    """
    Create a per-user Trino connection with a dynamic catalog.

    Each user gets their own catalog ``u_{sanitized_username}`` containing
    their MinIO credentials, giving isolated S3 access — the same security
    model as Spark sessions.

    Args:
        username: KBase username.
        auth_token: KBase auth token (passed as extra credential for
            the access-control plugin to resolve tenant groups).
        access_key: User's MinIO access key.
        secret_key: User's MinIO secret key.
        trino_host: Trino coordinator hostname.
        trino_port: Trino coordinator port.
        hive_metastore_uri: Hive Metastore Thrift URI.
        minio_endpoint_url: MinIO/S3 endpoint URL.
        minio_secure: Whether to use HTTPS for MinIO.
        connector: Trino connector to use (default ``delta_lake``).

    Returns:
        ``trino.dbapi.Connection`` with the default catalog set to the
        user's dynamic catalog.
    """
    safe_username = _sanitize_identifier(username)
    catalog_name = f"u_{safe_username}"

    logger.info(
        f"Setting up Trino connection for user={username}, catalog={catalog_name}"
    )

    try:
        conn = trino.dbapi.connect(
            host=trino_host,
            port=trino_port,
            user=username,
            extra_credential=[("kbase_auth_token", auth_token)],
        )

        properties = _build_catalog_properties(
            hive_metastore_uri=hive_metastore_uri,
            minio_endpoint_url=minio_endpoint_url,
            minio_secure=minio_secure,
            access_key=access_key,
            secret_key=secret_key,
        )

        cursor = conn.cursor()
        _create_dynamic_catalog(cursor, catalog_name, connector, properties)

        # Set default catalog so queries use schema.table format.
        conn._client_session.catalog = catalog_name

        logger.info(
            f"Trino session ready: host={trino_host}:{trino_port}, "
            f"user={username}, catalog={catalog_name}"
        )
        return conn

    except Exception as e:
        raise TrinoConnectionError(
            f"Failed to create Trino connection for user {username}: "
            f"{type(e).__name__}: {e}"
        ) from e
