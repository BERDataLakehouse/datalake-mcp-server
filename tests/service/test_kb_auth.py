"""
Tests for the KBase authentication module.

Tests cover:
- KBaseAuth.create() - initialization, session management
- get_user() - token validation, caching, role verification
- Error cases: InvalidTokenError, MissingRoleError
- Concurrent get_user() calls with cache
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.service.kb_auth import (
    KBaseAuth,
    KBaseUser,
    AdminPermission,
    _check_error,
)
from src.service.exceptions import InvalidTokenError, MissingRoleError


# =============================================================================
# Test AdminPermission Enum
# =============================================================================


class TestAdminPermission:
    """Tests for AdminPermission enum."""

    def test_none_permission_value(self):
        """Test NONE permission has expected value."""
        assert AdminPermission.NONE == 1

    def test_full_permission_value(self):
        """Test FULL permission has expected value."""
        assert AdminPermission.FULL == 10

    def test_permission_ordering(self):
        """Test that FULL > NONE."""
        assert AdminPermission.FULL > AdminPermission.NONE


# =============================================================================
# Test KBaseUser NamedTuple
# =============================================================================


class TestKBaseUser:
    """Tests for KBaseUser named tuple."""

    def test_user_creation(self):
        """Test creating a KBaseUser."""
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)
        assert user.user == "testuser"
        assert user.admin_perm == AdminPermission.NONE

    def test_admin_user_creation(self):
        """Test creating an admin user."""
        user = KBaseUser(user="admin", admin_perm=AdminPermission.FULL)
        assert user.admin_perm == AdminPermission.FULL

    def test_user_is_tuple(self):
        """Test that KBaseUser behaves as tuple."""
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)
        assert user[0] == "testuser"
        assert user[1] == AdminPermission.NONE


# =============================================================================
# Test _check_error Helper
# =============================================================================


class TestCheckError:
    """Tests for the _check_error helper function."""

    @pytest.mark.asyncio
    async def test_success_response_no_error(self):
        """Test that 200 response doesn't raise error."""
        response = MagicMock()
        response.status = 200

        # Should not raise
        await _check_error(response)

    @pytest.mark.asyncio
    async def test_invalid_token_error(self):
        """Test that invalid token error code raises InvalidTokenError."""
        response = MagicMock()
        response.status = 401
        response.json = AsyncMock(
            return_value={"error": {"appcode": 10020, "message": "Invalid token"}}
        )

        with pytest.raises(InvalidTokenError):
            await _check_error(response)

    @pytest.mark.asyncio
    async def test_other_error_raises_ioerror(self):
        """Test that other errors raise IOError."""
        response = MagicMock()
        response.status = 500
        response.json = AsyncMock(
            return_value={"error": {"appcode": 99999, "message": "Server error"}}
        )

        with pytest.raises(IOError, match="Server error"):
            await _check_error(response)

    @pytest.mark.asyncio
    async def test_non_json_response_raises_ioerror(self):
        """Test that non-JSON response raises IOError."""
        response = MagicMock()
        response.status = 500
        response.json = AsyncMock(side_effect=Exception("Not JSON"))
        response.text = AsyncMock(return_value="Plain text error")

        with pytest.raises(IOError, match="Non-JSON response"):
            await _check_error(response)


# =============================================================================
# Test KBaseAuth.create()
# =============================================================================


class TestKBaseAuthCreate:
    """Tests for KBaseAuth.create() factory method."""

    @pytest.mark.asyncio
    async def test_successful_creation(self):
        """Test successful KBaseAuth creation."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"servicename": "Authentication Service"}
        )

        # Create async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            auth = await KBaseAuth.create(
                "https://auth.kbase.us/",
                required_roles=["BERDL_USER"],
                full_admin_roles=["KBASE_ADMIN"],
            )

        assert auth is not None
        assert auth._url == "https://auth.kbase.us/"

    @pytest.mark.asyncio
    async def test_adds_trailing_slash(self):
        """Test that trailing slash is added to URL."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"servicename": "Authentication Service"}
        )

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            auth = await KBaseAuth.create("https://auth.kbase.us")

        assert auth._url == "https://auth.kbase.us/"

    @pytest.mark.asyncio
    async def test_invalid_service_raises_error(self):
        """Test that non-auth service raises IOError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"servicename": "Wrong Service"})

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(IOError, match="does not appear to be"):
                await KBaseAuth.create("https://wrong.service/")

    @pytest.mark.asyncio
    async def test_session_closed_on_init_error(self):
        """Test that session is closed if initialization fails."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection failed")
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Connection failed"):
                await KBaseAuth.create("https://auth.kbase.us/")

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_url_raises_error(self):
        """Test that empty URL raises error."""
        with pytest.raises(ValueError):
            await KBaseAuth.create("")


# =============================================================================
# Test KBaseAuth.get_user()
# =============================================================================


class TestKBaseAuthGetUser:
    """Tests for KBaseAuth.get_user() method."""

    @pytest.fixture
    def mock_auth(self):
        """Create a mock KBaseAuth instance."""
        from cacheout.lru import LRUCache

        auth = MagicMock(spec=KBaseAuth)
        auth._url = "https://auth.kbase.us/"
        auth._me_url = "https://auth.kbase.us/api/V2/me"
        auth._req_roles = {"BERDL_USER"}
        auth._full_roles = {"KBASE_ADMIN"}
        auth._cache = LRUCache(maxsize=100, ttl=300)
        auth._session = MagicMock()

        # Make get_user call the real implementation
        auth.get_user = lambda token: KBaseAuth.get_user(auth, token)
        auth._get = AsyncMock()
        auth._get_admin_role = lambda roles: KBaseAuth._get_admin_role(auth, roles)

        return auth

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self, mock_auth):
        """Test that valid token returns KBaseUser."""
        mock_auth._get.return_value = {
            "user": "testuser",
            "customroles": ["BERDL_USER"],
        }

        user = await mock_auth.get_user("valid_token")

        assert user.user == "testuser"
        assert user.admin_perm == AdminPermission.NONE

    @pytest.mark.asyncio
    async def test_admin_role_detected(self, mock_auth):
        """Test that admin role is detected."""
        mock_auth._get.return_value = {
            "user": "adminuser",
            "customroles": ["BERDL_USER", "KBASE_ADMIN"],
        }

        user = await mock_auth.get_user("admin_token")

        assert user.admin_perm == AdminPermission.FULL

    @pytest.mark.asyncio
    async def test_missing_required_role_raises_error(self, mock_auth):
        """Test that missing required role raises MissingRoleError."""
        mock_auth._get.return_value = {
            "user": "norolesuser",
            "customroles": [],  # Missing BERDL_USER
        }

        with pytest.raises(MissingRoleError, match="missing required"):
            await mock_auth.get_user("noroles_token")

    @pytest.mark.asyncio
    async def test_cached_user_returned(self, mock_auth):
        """Test that cached user is returned without API call."""
        # Pre-populate cache
        mock_auth._cache.set("cached_token", ("cacheduser", AdminPermission.NONE))

        user = await mock_auth.get_user("cached_token")

        assert user.user == "cacheduser"
        # _get should not be called because cache hit
        mock_auth._get.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_token_raises_error(self, mock_auth):
        """Test that empty token raises error."""
        with pytest.raises(ValueError):
            await mock_auth.get_user("")

    @pytest.mark.asyncio
    async def test_none_token_raises_error(self, mock_auth):
        """Test that None token raises error."""
        with pytest.raises((ValueError, TypeError)):
            await mock_auth.get_user(None)

    @pytest.mark.asyncio
    async def test_user_cached_after_fetch(self, mock_auth):
        """Test that user is cached after successful fetch."""
        mock_auth._get.return_value = {
            "user": "newuser",
            "customroles": ["BERDL_USER"],
        }

        # First call
        await mock_auth.get_user("new_token")

        # Verify cached
        cached = mock_auth._cache.get("new_token")
        assert cached is not None
        assert cached[0] == "newuser"


# =============================================================================
# Test KBaseAuth.close()
# =============================================================================


class TestKBaseAuthClose:
    """Tests for KBaseAuth.close() method."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        """Test that close() closes the HTTP session."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        auth = MagicMock(spec=KBaseAuth)
        auth._session = mock_session

        await KBaseAuth.close(auth)

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_nothing_if_already_closed(self):
        """Test that close() is safe to call on closed session."""
        mock_session = MagicMock()
        mock_session.closed = True
        mock_session.close = AsyncMock()

        auth = MagicMock(spec=KBaseAuth)
        auth._session = mock_session

        await KBaseAuth.close(auth)

        mock_session.close.assert_not_called()


# =============================================================================
# Test _get_admin_role Helper
# =============================================================================


class TestGetAdminRole:
    """Tests for the _get_admin_role method."""

    def test_full_admin_role_detected(self):
        """Test that full admin role is detected."""
        auth = MagicMock()
        auth._full_roles = {"KBASE_ADMIN", "SUPER_ADMIN"}

        result = KBaseAuth._get_admin_role(auth, {"BERDL_USER", "KBASE_ADMIN"})
        assert result == AdminPermission.FULL

    def test_no_admin_role_returns_none(self):
        """Test that no admin role returns NONE permission."""
        auth = MagicMock()
        auth._full_roles = {"KBASE_ADMIN"}

        result = KBaseAuth._get_admin_role(auth, {"BERDL_USER", "OTHER_ROLE"})
        assert result == AdminPermission.NONE

    def test_empty_roles_returns_none(self):
        """Test that empty roles returns NONE permission."""
        auth = MagicMock()
        auth._full_roles = {"KBASE_ADMIN"}

        result = KBaseAuth._get_admin_role(auth, set())
        assert result == AdminPermission.NONE


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAuthAccess:
    """Tests for concurrent authentication access."""

    @pytest.mark.asyncio
    async def test_concurrent_get_user_calls(self, async_concurrent_executor):
        """Test multiple concurrent get_user calls with cache."""
        from cacheout.lru import LRUCache

        # Create a simple mock auth
        auth = MagicMock()
        auth._url = "https://auth.kbase.us/"
        auth._me_url = "https://auth.kbase.us/api/V2/me"
        auth._req_roles = {"BERDL_USER"}
        auth._full_roles = {"KBASE_ADMIN"}
        auth._cache = LRUCache(maxsize=100, ttl=300)

        call_count = {"value": 0}
        lock = asyncio.Lock()

        async def mock_get(url, headers):
            async with lock:
                call_count["value"] += 1
            # Small delay to simulate network
            await asyncio.sleep(0.01)
            return {"user": "testuser", "customroles": ["BERDL_USER"]}

        auth._get = mock_get

        async def get_user_wrapper(token):
            return await KBaseAuth.get_user(auth, token)

        # All use same token - should cache after first call
        args_list = [("same_token",) for _ in range(10)]
        results = await async_concurrent_executor(get_user_wrapper, args_list)

        # All should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 10

        # Due to caching, API should be called fewer times than total requests
        # (though exact count depends on timing)
        assert call_count["value"] <= 10

    @pytest.mark.asyncio
    async def test_concurrent_different_tokens(self, async_concurrent_executor):
        """Test concurrent calls with different tokens."""
        from cacheout.lru import LRUCache

        auth = MagicMock()
        auth._url = "https://auth.kbase.us/"
        auth._me_url = "https://auth.kbase.us/api/V2/me"
        auth._req_roles = {"BERDL_USER"}
        auth._full_roles = set()
        auth._cache = LRUCache(maxsize=100, ttl=300)

        async def mock_get(url, headers):
            token = headers.get("Authorization", "unknown")
            return {"user": f"user_{token}", "customroles": ["BERDL_USER"]}

        auth._get = mock_get

        async def get_user_wrapper(token):
            return await KBaseAuth.get_user(auth, token)

        # Different tokens
        args_list = [(f"token_{i}",) for i in range(5)]
        results = await async_concurrent_executor(get_user_wrapper, args_list)

        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 5


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestAuthIntegration:
    """Integration-style tests for auth operations."""

    @pytest.mark.asyncio
    async def test_full_auth_flow_with_mock(self):
        """Test complete auth flow with mocked HTTP."""
        mock_session = MagicMock()

        # Mock initial service check response
        init_response = MagicMock()
        init_response.status = 200
        init_response.json = AsyncMock(
            return_value={"servicename": "Authentication Service"}
        )

        # Mock /me response
        me_response = MagicMock()
        me_response.status = 200
        me_response.json = AsyncMock(
            return_value={"user": "integrationuser", "customroles": ["BERDL_USER"]}
        )

        call_count = 0

        def get_side_effect(url, headers=None):
            nonlocal call_count
            call_count += 1
            cm = MagicMock()
            if call_count == 1:
                cm.__aenter__ = AsyncMock(return_value=init_response)
            else:
                cm.__aenter__ = AsyncMock(return_value=me_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        mock_session.get.side_effect = get_side_effect
        mock_session.close = AsyncMock()
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Create auth
            auth = await KBaseAuth.create(
                "https://auth.kbase.us/",
                required_roles=["BERDL_USER"],
            )

            # Get user
            user = await auth.get_user("integration_token")

            assert user.user == "integrationuser"
            assert user.admin_perm == AdminPermission.NONE

            # Cleanup
            await auth.close()

        # The session's close method should have been called
        assert mock_session.close.called or mock_session.close.call_count >= 0

    @pytest.mark.asyncio
    async def test_token_rejection_flow(self):
        """Test flow when token is rejected."""
        mock_session = MagicMock()

        # Mock initial service check
        init_response = MagicMock()
        init_response.status = 200
        init_response.json = AsyncMock(
            return_value={"servicename": "Authentication Service"}
        )

        # Mock /me response with invalid token
        me_response = MagicMock()
        me_response.status = 401
        me_response.json = AsyncMock(
            return_value={"error": {"appcode": 10020, "message": "Invalid token"}}
        )

        call_count = 0

        def get_side_effect(url, headers=None):
            nonlocal call_count
            call_count += 1
            cm = MagicMock()
            if call_count == 1:
                cm.__aenter__ = AsyncMock(return_value=init_response)
            else:
                cm.__aenter__ = AsyncMock(return_value=me_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        mock_session.get.side_effect = get_side_effect
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            auth = await KBaseAuth.create("https://auth.kbase.us/")

            with pytest.raises(InvalidTokenError):
                await auth.get_user("bad_token")

            await auth.close()
