"""
Module for querying Hive metastore information using direct HMS client.

This is a simplified version of berdl_notebook_utils.hive_metastore
that works in a shared multi-user service context without environment variable dependencies.

MAINTENANCE NOTE: This file is copied from:
/spark_notebook/notebook_utils/berdl_notebook_utils/hive_metastore.py
/spark_notebook/notebook_utils/berdl_notebook_utils/clients.py (get_hive_metastore_client)

When updating, copy the relevant functions from those files.
"""

import os
from typing import List

from hmsclient import HMSClient
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport

from src.settings import BERDLSettings

# Bound the HMS Thrift socket so a slow / unreachable metastore can never
# pin a thread (or block the /health check) indefinitely. The default HMS
# call latency is sub-millisecond when healthy; 10s is a generous ceiling.
_HMS_SOCKET_TIMEOUT_MS = int(os.getenv("HMS_SOCKET_TIMEOUT_MS", "10000"))


def get_hive_metastore_client(settings: BERDLSettings) -> HMSClient:
    """
    Get Hive Metastore client configured from settings, with a bounded socket timeout.

    Args:
        settings: BERDLSettings instance with HMS configuration

    Returns:
        Configured HMSClient instance
    """
    # Parse the HMS URI to extract host and port
    # Format: thrift://host:port
    hms_uri = str(settings.BERDL_HIVE_METASTORE_URI)

    if not hms_uri.startswith("thrift://"):
        raise ValueError(
            f"Invalid HMS URI format: {hms_uri}. Expected thrift://host:port"
        )

    # Remove the thrift:// prefix and split host:port
    host_port = hms_uri.replace("thrift://", "")

    if ":" in host_port:
        host, port_str = host_port.split(":", 1)
        port = int(port_str)
    else:
        host = host_port
        port = 9083  # Default HMS port

    # Build the transport explicitly so we can set a socket-level timeout.
    # HMSClient(host=..., port=...) constructs the same TBufferedTransport +
    # TBinaryProtocol stack but with no timeout, which means open() and any
    # subsequent call can block forever on a slow / unresponsive metastore.
    sock = TSocket.TSocket(host, port)
    sock.setTimeout(_HMS_SOCKET_TIMEOUT_MS)
    transport = TTransport.TBufferedTransport(sock)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    return HMSClient(iprot=protocol)


def get_databases(settings: BERDLSettings) -> List[str]:
    """
    Get list of databases from Hive metastore.

    Args:
        settings: BERDLSettings instance with HMS configuration

    Returns:
        List of database names
    """
    client = get_hive_metastore_client(settings)
    try:
        client.open()
        databases = client.get_databases("*")
        return databases
    except Exception as e:
        raise e
    finally:
        client.close()


def get_tables(database: str, settings: BERDLSettings) -> List[str]:
    """
    Get list of tables in a database from Hive metastore.

    Args:
        database: Name of the database
        settings: BERDLSettings instance with HMS configuration

    Returns:
        List of table names
    """
    client = get_hive_metastore_client(settings)
    try:
        client.open()
        tables = client.get_tables(database, "*")
        return tables
    except Exception as e:
        raise e
    finally:
        client.close()
