"""
A client for the KBase Auth2 server.
"""

# Mostly copied from https://github.com/kbase/cdm-task-service/blob/main/cdmtaskservice/kb_auth.py

import logging
import time
from enum import IntEnum
from typing import NamedTuple, Self

import aiohttp
from cacheout.lru import LRUCache

from src.service.arg_checkers import not_falsy as _not_falsy
from src.service.exceptions import InvalidTokenError, MissingRoleError

logger = logging.getLogger(__name__)


class AdminPermission(IntEnum):
    """
    The different levels of admin permissions.
    """

    NONE = 1
    # leave some space for potential future levels
    FULL = 10


class KBaseUser(NamedTuple):
    user: str
    admin_perm: AdminPermission


async def _check_error(r: aiohttp.ClientResponse) -> None:
    """Check for errors in the response from the auth server."""
    if r.status != 200:
        try:
            j = await r.json()
        except Exception:
            err = "Non-JSON response from KBase auth server, status code: " + str(
                r.status
            )
            logger.info("%s, response:\n%s", err, await r.text())
            raise IOError(err)
        # assume that if we get json then at least this is the auth server and we can
        # rely on the error structure.
        if j["error"].get("appcode") == 10020:  # Invalid token
            raise InvalidTokenError("KBase auth server reported token is invalid.")
        # don't really see any other error codes we need to worry about - maybe disabled?
        # worry about it later.
        raise IOError("Error from KBase auth server: " + j["error"]["message"])


class KBaseAuth:
    """
    A client for contacting the KBase authentication server.

    Uses a shared aiohttp.ClientSession for connection pooling and efficiency.
    The session should be closed via close() when the application shuts down.
    """

    @classmethod
    async def create(
        cls,
        auth_url: str,
        required_roles: list[str] | None = None,
        full_admin_roles: list[str] | None = None,
        cache_max_size: int = 10000,
        cache_expiration: int = 300,
    ) -> Self:
        """
        Create the client.

        Args:
            auth_url: The root url of the authentication service.
            required_roles: The KBase Auth2 roles that the user must possess
                in order to be allowed to use the service.
            full_admin_roles: The KBase Auth2 roles that determine that user
                is an administrator.
            cache_max_size: The maximum size of the token cache.
            cache_expiration: The expiration time for the token cache in seconds.

        Returns:
            A configured KBaseAuth instance with a shared HTTP session.
        """
        if not _not_falsy(auth_url, "auth_url").endswith("/"):
            auth_url += "/"

        # Create the shared session with reasonable defaults
        # Connection pooling is handled automatically by aiohttp
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=100,  # Max connections in pool
            limit_per_host=20,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL in seconds
        )
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

        try:
            # Verify the auth service is reachable and valid
            async with session.get(
                auth_url, headers={"Accept": "application/json"}
            ) as r:
                await _check_error(r)
                j = await r.json()

            return cls(
                auth_url,
                required_roles,
                full_admin_roles,
                cache_max_size,
                cache_expiration,
                j.get("servicename"),
                session,
            )
        except Exception:
            # Clean up session if initialization fails
            await session.close()
            raise

    def __init__(
        self,
        auth_url: str,
        required_roles: list[str] | None,
        full_admin_roles: list[str] | None,
        cache_max_size: int,
        cache_expiration: int,
        service_name: str,
        session: aiohttp.ClientSession,
    ):
        self._url = auth_url
        self._me_url = self._url + "api/V2/me"
        self._req_roles = set(required_roles) if required_roles else None
        self._full_roles = set(full_admin_roles) if full_admin_roles else set()
        self._cache_timer = time.time
        self._cache = LRUCache(
            timer=self._cache_timer, maxsize=cache_max_size, ttl=cache_expiration
        )
        self._session = session

        if service_name != "Authentication Service":
            raise IOError(
                f"The service at {self._url} does not appear to be the KBase "
                + "Authentication Service"
            )

        logger.info("KBaseAuth initialized with shared HTTP session")

    async def close(self) -> None:
        """
        Close the shared HTTP session.

        Should be called during application shutdown to cleanly release resources.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("KBaseAuth HTTP session closed")

    async def _get(self, url: str, headers: dict) -> dict:
        """
        Make a GET request using the shared session.

        Args:
            url: The URL to request
            headers: Request headers

        Returns:
            JSON response as dictionary

        Raises:
            IOError: If the request fails or returns an error
            InvalidTokenError: If the token is invalid
        """
        async with self._session.get(url, headers=headers) as r:
            await _check_error(r)
            return await r.json()

    async def get_user(self, token: str) -> KBaseUser:
        """
        Get a username from a token as well as the user's administration status.

        Verifies the user has all the required roles set in the create() method.

        Args:
            token: The user's token.

        Returns:
            The authenticated user with their admin permission level.

        Raises:
            InvalidTokenError: If the token is invalid.
            MissingRoleError: If the user lacks required roles.
        """
        _not_falsy(token, "token")

        admin_cache = self._cache.get(token, default=False)
        if admin_cache:
            return KBaseUser(admin_cache[0], admin_cache[1])

        j = await self._get(self._me_url, {"Authorization": token})
        croles = set(j["customroles"])

        if self._req_roles and not self._req_roles <= croles:
            required_roles_str = ", ".join(sorted(self._req_roles))
            raise MissingRoleError(
                f"The user is missing required authentication roles to use the service. Required roles: {required_roles_str}"
            )

        v = (j["user"], self._get_admin_role(croles))
        self._cache.set(token, v)
        return KBaseUser(v[0], v[1])

    def _get_admin_role(self, roles: set[str]) -> AdminPermission:
        if roles & self._full_roles:
            return AdminPermission.FULL
        return AdminPermission.NONE
