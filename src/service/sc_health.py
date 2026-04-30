"""
Per-user Spark Connect health tracking.

When a user's Spark Connect server (their notebook pod) is reachable at the
gRPC level but its JVM driver is wedged or its session is otherwise unusable,
every subsequent request from that user can hang the full request-timeout
window (115 s). Across the cells of a notebook this looks like "the whole
MCP service is broken" even though every other user is fine.

This module records a per-user "broken-until" deadline.
``get_spark_context`` runs a cheap ``spark.version`` probe after session
creation; on probe failure it marks the user unhealthy for
``SC_UNHEALTHY_TTL`` seconds and raises an immediate error. Subsequent
requests from that user during the window short-circuit on this mark —
converting a 115 s timeout into a sub-second 503 with a clear message:
restart the notebook pod. The mark is also set by a session-create
timeout (see ``_SC_CREATE_TIMEOUT_SECONDS``).
"""

import os
import threading
import time

# How long after a failed probe to short-circuit subsequent requests for the
# same user with an immediate error. Long enough that a hung Spark Connect
# server doesn't keep eating probe budget; short enough that users don't have
# to wait once they've actually restarted their notebook.
SC_UNHEALTHY_TTL = float(os.getenv("SC_UNHEALTHY_TTL_SECONDS", "60"))

_lock = threading.Lock()
_unhealthy_until: dict[str, float] = {}


def mark_unhealthy(username: str) -> None:
    """Record that ``username`` 's Spark Connect failed its SQL probe."""
    with _lock:
        _unhealthy_until[username] = time.monotonic() + SC_UNHEALTHY_TTL


def is_unhealthy(username: str) -> bool:
    """True if the user's SC is currently within the unhealthy window."""
    with _lock:
        until = _unhealthy_until.get(username)
        if until is None:
            return False
        if time.monotonic() >= until:
            # TTL elapsed — drop the entry so the next request re-probes.
            del _unhealthy_until[username]
            return False
        return True


def clear(username: str) -> None:
    """Drop a user's unhealthy mark (e.g. after a successful probe)."""
    with _lock:
        _unhealthy_until.pop(username, None)
