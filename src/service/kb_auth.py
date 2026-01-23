"""
A client for the KBase Auth2 server.
"""

# Mostly copied from https://github.com/kbase/cdm-task-service/blob/main/cdmtaskservice/kb_auth.py

import logging
import os
import time
from enum import IntEnum
from typing import NamedTuple, Self

import aiohttp
from cacheout.lru import LRUCache

from src.service.arg_checkers import not_falsy as _not_falsy
from src.service.exceptions import InvalidTokenError, MissingMFAError, MissingRoleError

logger = logging.getLogger(__name__)


class AdminPermission(IntEnum):
    """
    The different levels of admin permissions.
    """

    NONE = 1
    # leave some space for potential future levels
    FULL = 10


class MFAStatus(IntEnum):
    """
    MFA status for a token.
    """

    NOT_USED = 0  # Token was not created with MFA
    USED = 1  # Token was created with MFA
    UNKNOWN = 2  # MFA status could not be determined


class KBaseUser(NamedTuple):
    user: str
    admin_perm: AdminPermission
    mfa_status: MFAStatus = MFAStatus.UNKNOWN


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
        require_mfa: bool = True,
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
            require_mfa: If True, only tokens created with MFA will be accepted
                (unless user is in the MFA_EXEMPT_USERS env var list).
            cache_max_size: The maximum size of the token cache.
            cache_expiration: The expiration time for the token cache in seconds.

        Environment Variables:
            MFA_EXEMPT_USERS: Comma-separated list of usernames exempt from MFA.
                Service accounts (e.g., 'kbaselakehouseserviceaccount') should be
                listed here as they cannot use MFA.

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
                require_mfa,
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
        require_mfa: bool,
        cache_max_size: int,
        cache_expiration: int,
        service_name: str,
        session: aiohttp.ClientSession,
    ):
        self._url = auth_url
        self._me_url = self._url + "api/V2/me"
        self._token_url = self._url + "api/V2/token"
        self._require_mfa = require_mfa
        # Read MFA-exempt users from environment variable (comma-separated list)
        mfa_exempt_env = os.getenv("MFA_EXEMPT_USERS", "")
        self._mfa_exempt_users = {
            u.strip() for u in mfa_exempt_env.split(",") if u.strip()
        }
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
        Get a username from a token as well as the user's administration and MFA status.

        Verifies the user has all the required roles set in the create() method.
        Optionally verifies the token was created with MFA.

        Args:
            token: The user's token.

        Returns:
            The authenticated user with their admin permission level and MFA status.

        Raises:
            InvalidTokenError: If the token is invalid.
            MissingRoleError: If the user lacks required roles.
            MissingMFAError: If require_mfa is True and the token was not created with MFA.
        """
        _not_falsy(token, "token")

        cached = self._cache.get(token, default=False)
        if cached:
            return KBaseUser(cached[0], cached[1], cached[2])

        # Get user info from /api/V2/me
        me_response = await self._get(self._me_url, {"Authorization": token})
        croles = set(me_response["customroles"])

        if self._req_roles and not self._req_roles <= croles:
            required_roles_str = ", ".join(sorted(self._req_roles))
            raise MissingRoleError(
                f"The user is missing required authentication roles to use the service. Required roles: {required_roles_str}"
            )

        # Get token info from /api/V2/token to check MFA status
        token_response = await self._get(self._token_url, {"Authorization": token})
        mfa_value = token_response.get("mfa", "")

        # Parse MFA status from response
        if mfa_value == "Used":
            mfa_status = MFAStatus.USED
        elif mfa_value in ("", "NotUsed", None):
            mfa_status = MFAStatus.NOT_USED
        else:
            mfa_status = MFAStatus.UNKNOWN

        # Check MFA requirement if enabled (skip for exempt users like service accounts)
        username = me_response["user"]
        is_mfa_exempt = username in self._mfa_exempt_users
        if self._require_mfa and mfa_status != MFAStatus.USED and not is_mfa_exempt:
            raise MissingMFAError(
                "This service requires multi-factor authentication (MFA). "
                "Please log in with MFA enabled to access this service."
            )

        admin_perm = self._get_admin_role(croles)
        v = (username, admin_perm, mfa_status)
        self._cache.set(token, v)
        return KBaseUser(v[0], v[1], v[2])

    def _get_admin_role(self, roles: set[str]) -> AdminPermission:
        if roles & self._full_roles:
            return AdminPermission.FULL
        return AdminPermission.NONE
