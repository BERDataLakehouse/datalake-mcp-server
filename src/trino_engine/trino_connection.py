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
#
# delta_lake/hive are used for the user's per-request HMS-backed catalog.
# iceberg is used for Polaris REST catalogs.
ALLOWED_CONNECTORS = frozenset({"delta_lake", "hive", "iceberg"})


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


def _get_polaris_uri(polaris_catalog_uri: str | None) -> str | None:
    """Return the configured Polaris REST catalog URI without a trailing slash."""
    if not polaris_catalog_uri:
        return None
    return str(polaris_catalog_uri).rstrip("/")


def _get_polaris_oauth2_server_uri(polaris_catalog_uri: str | None) -> str | None:
    """Return the Polaris OAuth2 token endpoint for Iceberg REST clients."""
    polaris_uri = _get_polaris_uri(polaris_catalog_uri)
    if not polaris_uri:
        return None
    if polaris_uri.endswith("/v1"):
        return f"{polaris_uri}/oauth/tokens"
    return f"{polaris_uri}/v1/oauth/tokens"


def _get_personal_catalog_alias(personal_catalog: str | None) -> str | None:
    """Return the portable Trino alias for a personal Polaris catalog."""
    if not personal_catalog:
        return None

    alias = personal_catalog.strip()
    if alias.startswith("user_"):
        alias = alias[len("user_") :]
    alias = _sanitize_identifier(alias).strip("_")
    return alias or None


def _get_tenant_catalog_alias(tenant_catalog: str) -> str:
    """Return the portable Trino alias for a tenant Polaris catalog."""
    alias = tenant_catalog.strip()
    if alias.startswith("tenant_"):
        alias = alias[len("tenant_") :]
    return _sanitize_identifier(alias).strip("_")


def _iter_tenant_catalogs(tenant_catalogs: str | None) -> list[str]:
    """Parse configured Polaris tenant catalog names."""
    if not tenant_catalogs:
        return []
    return [catalog.strip() for catalog in tenant_catalogs.split(",") if catalog.strip()]


def _build_iceberg_catalog_properties(
    polaris_catalog_uri: str | None,
    polaris_credential: str | None,
    warehouse: str,
    hive_metastore_uri: str,
    minio_endpoint_url: str,
    minio_secure: bool,
    access_key: str,
    secret_key: str,
) -> dict[str, str]:
    """Build the WITH properties for a Trino Iceberg REST catalog backed by Polaris."""
    polaris_uri = _get_polaris_uri(polaris_catalog_uri)
    oauth2_server_uri = _get_polaris_oauth2_server_uri(polaris_catalog_uri)
    if not polaris_uri or not oauth2_server_uri:
        raise ValueError("POLARIS_CATALOG_URI is required to create an Iceberg catalog")

    s3_properties = _build_catalog_properties(
        hive_metastore_uri=hive_metastore_uri,
        minio_endpoint_url=minio_endpoint_url,
        minio_secure=minio_secure,
        access_key=access_key,
        secret_key=secret_key,
    )
    s3_properties.pop("hive.metastore.uri", None)

    return {
        "iceberg.catalog.type": "rest",
        "iceberg.rest-catalog.uri": polaris_uri,
        "iceberg.rest-catalog.warehouse": warehouse,
        "iceberg.rest-catalog.security": "OAUTH2",
        "iceberg.rest-catalog.oauth2.credential": polaris_credential or "",
        "iceberg.rest-catalog.oauth2.scope": "PRINCIPAL_ROLE:ALL",
        "iceberg.rest-catalog.oauth2.server-uri": oauth2_server_uri,
        "iceberg.rest-catalog.oauth2.token-refresh-enabled": "false",
        "iceberg.rest-catalog.oauth2.token-exchange-enabled": "false",
        "iceberg.rest-catalog.vended-credentials-enabled": "false",
        "iceberg.security": "read_only",
        **s3_properties,
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


def _create_optional_dynamic_catalog(
    cursor: trino.dbapi.Cursor,
    catalog_name: str,
    connector: str,
    properties: dict[str, str],
) -> None:
    """Create a catalog, logging access-control failures for optional tenant catalogs."""
    try:
        _create_dynamic_catalog(cursor, catalog_name, connector, properties)
    except Exception as e:
        logger.warning(
            "Could not create optional Trino catalog '%s': %s: %s",
            catalog_name,
            type(e).__name__,
            e,
        )


def _create_polaris_catalogs(
    cursor: trino.dbapi.Cursor,
    polaris_catalog_uri: str | None,
    polaris_credential: str | None,
    polaris_personal_catalog: str | None,
    polaris_tenant_catalogs: str | None,
    hive_metastore_uri: str,
    minio_endpoint_url: str,
    minio_secure: bool,
    access_key: str,
    secret_key: str,
) -> None:
    """Create Trino Iceberg catalogs for configured Polaris warehouses."""
    if not polaris_catalog_uri or not polaris_credential:
        return

    personal_alias = _get_personal_catalog_alias(polaris_personal_catalog)
    if polaris_personal_catalog and personal_alias:
        properties = _build_iceberg_catalog_properties(
            polaris_catalog_uri=polaris_catalog_uri,
            polaris_credential=polaris_credential,
            warehouse=polaris_personal_catalog,
            hive_metastore_uri=hive_metastore_uri,
            minio_endpoint_url=minio_endpoint_url,
            minio_secure=minio_secure,
            access_key=access_key,
            secret_key=secret_key,
        )
        _create_dynamic_catalog(cursor, personal_alias, "iceberg", properties)

    for tenant_catalog in _iter_tenant_catalogs(polaris_tenant_catalogs):
        tenant_alias = _get_tenant_catalog_alias(tenant_catalog)
        if not tenant_alias:
            continue
        properties = _build_iceberg_catalog_properties(
            polaris_catalog_uri=polaris_catalog_uri,
            polaris_credential=polaris_credential,
            warehouse=tenant_catalog,
            hive_metastore_uri=hive_metastore_uri,
            minio_endpoint_url=minio_endpoint_url,
            minio_secure=minio_secure,
            access_key=access_key,
            secret_key=secret_key,
        )
        _create_optional_dynamic_catalog(cursor, tenant_alias, "iceberg", properties)


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
    polaris_catalog_uri: str | None = None,
    polaris_credential: str | None = None,
    polaris_personal_catalog: str | None = None,
    polaris_tenant_catalogs: str | None = None,
) -> trino.dbapi.Connection:
    """
    Create a per-user Trino connection with a dynamic catalog.

    Each user gets their own Delta/Hive catalog ``u_{sanitized_username}`` containing
    their MinIO credentials, giving isolated S3 access — the same security
    model as Spark sessions. If Polaris settings are provided, Trino also
    creates portable Iceberg catalog aliases such as ``testuser`` and
    ``globalusers`` for the user's personal and tenant Polaris warehouses.

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
        polaris_catalog_uri: Polaris REST catalog endpoint, if configured.
        polaris_credential: Polaris OAuth2 client_id:client_secret credential.
        polaris_personal_catalog: Polaris personal warehouse/catalog name.
        polaris_tenant_catalogs: Comma-separated tenant warehouse/catalog names.

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
        _create_polaris_catalogs(
            cursor=cursor,
            polaris_catalog_uri=polaris_catalog_uri,
            polaris_credential=polaris_credential,
            polaris_personal_catalog=polaris_personal_catalog,
            polaris_tenant_catalogs=polaris_tenant_catalogs,
            hive_metastore_uri=hive_metastore_uri,
            minio_endpoint_url=minio_endpoint_url,
            minio_secure=minio_secure,
            access_key=access_key,
            secret_key=secret_key,
        )

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
