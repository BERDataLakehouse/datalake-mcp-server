"""
Tests for the application state management module.

Tests cover:
- build_app() / destroy_app_state() - lifecycle management
- get_app_state() / set_request_user() - state retrieval/setting
- Error case: uninitialized state
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from src.service.app_state import (
    AppState,
    RequestState,
    build_app,
    destroy_app_state,
    get_app_state,
    _get_app_state_from_app,
    set_request_user,
    get_request_user,
)
from src.service.kb_auth import AdminPermission, KBaseUser


# =============================================================================
# Test AppState NamedTuple
# =============================================================================


class TestAppState:
    """Tests for the AppState named tuple."""

    def test_app_state_creation(self):
        """Test creating an AppState instance."""
        mock_auth = MagicMock()
        state = AppState(auth=mock_auth)

        assert state.auth is mock_auth

    def test_app_state_is_tuple(self):
        """Test that AppState behaves as tuple."""
        mock_auth = MagicMock()
        state = AppState(auth=mock_auth)

        assert state[0] is mock_auth


# =============================================================================
# Test RequestState NamedTuple
# =============================================================================


class TestRequestState:
    """Tests for the RequestState named tuple."""

    def test_request_state_with_user(self):
        """Test creating RequestState with a user."""
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)
        state = RequestState(user=user)

        assert state.user is user
        assert state.user.user == "testuser"

    def test_request_state_without_user(self):
        """Test creating RequestState without a user."""
        state = RequestState(user=None)
        assert state.user is None

    def test_request_state_is_tuple(self):
        """Test that RequestState behaves as tuple."""
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)
        state = RequestState(user=user)

        assert state[0] is user


# =============================================================================
# Test build_app()
# =============================================================================


class TestBuildApp:
    """Tests for the build_app function."""

    @pytest.mark.asyncio
    async def test_build_app_initializes_state(self):
        """Test that build_app initializes application state."""
        app = FastAPI()
        mock_auth = MagicMock()

        with patch(
            "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_auth

            await build_app(app)

        # Verify state is set
        assert hasattr(app.state, "_auth")
        assert hasattr(app.state, "_spark_state")
        assert app.state._auth is mock_auth

    @pytest.mark.asyncio
    async def test_build_app_creates_app_state(self):
        """Test that build_app creates AppState instance."""
        app = FastAPI()
        mock_auth = MagicMock()

        with patch(
            "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_auth

            await build_app(app)

        state = app.state._spark_state
        assert isinstance(state, AppState)
        assert state.auth is mock_auth

    @pytest.mark.asyncio
    async def test_build_app_uses_env_variables(self):
        """Test that build_app reads from environment variables."""
        app = FastAPI()

        with patch.dict(
            "os.environ",
            {
                "KBASE_AUTH_URL": "https://custom.auth.url/",
                "KBASE_ADMIN_ROLES": "ADMIN1,ADMIN2",
                "KBASE_REQUIRED_ROLES": "ROLE1,ROLE2",
            },
        ):
            with patch(
                "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
            ) as mock_create:
                mock_create.return_value = MagicMock()

                await build_app(app)

                # Verify KBaseAuth.create was called with correct arguments
                call_kwargs = mock_create.call_args
                assert call_kwargs[0][0] == "https://custom.auth.url/"
                assert "ROLE1" in call_kwargs[1]["required_roles"]
                assert "ROLE2" in call_kwargs[1]["required_roles"]
                assert "ADMIN1" in call_kwargs[1]["full_admin_roles"]
                assert "ADMIN2" in call_kwargs[1]["full_admin_roles"]

    @pytest.mark.asyncio
    async def test_build_app_uses_defaults_when_env_missing(self):
        """Test that build_app uses defaults when env vars missing."""
        app = FastAPI()

        # Clear specific env vars
        with patch.dict(
            "os.environ",
            {},
            clear=False,
        ):
            with patch(
                "os.environ.get",
                side_effect=lambda k, d=None: d,  # Always return default
            ):
                with patch(
                    "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
                ) as mock_create:
                    mock_create.return_value = MagicMock()

                    await build_app(app)

                    # Verify defaults were used
                    call_args = mock_create.call_args
                    assert "ci.kbase.us" in call_args[0][0]


# =============================================================================
# Test destroy_app_state()
# =============================================================================


class TestDestroyAppState:
    """Tests for the destroy_app_state function."""

    @pytest.mark.asyncio
    async def test_destroy_closes_auth_session(self):
        """Test that destroy_app_state closes the auth session."""
        app = FastAPI()
        mock_auth = MagicMock()
        mock_auth.close = AsyncMock()

        app.state._auth = mock_auth
        app.state._spark_state = AppState(auth=mock_auth)

        await destroy_app_state(app)

        mock_auth.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_destroy_handles_missing_auth(self):
        """Test that destroy_app_state handles missing auth gracefully."""
        app = FastAPI()
        # No _auth attribute set

        # Should not raise
        await destroy_app_state(app)

    @pytest.mark.asyncio
    async def test_destroy_handles_none_auth(self):
        """Test that destroy_app_state handles None auth."""
        app = FastAPI()
        app.state._auth = None

        # Should not raise
        await destroy_app_state(app)

    @pytest.mark.asyncio
    async def test_destroy_waits_for_pending_tasks(self):
        """Test that destroy_app_state waits for pending tasks."""
        app = FastAPI()
        mock_auth = MagicMock()
        mock_auth.close = AsyncMock()
        app.state._auth = mock_auth

        # Measure time to verify sleep occurred
        start = asyncio.get_event_loop().time()
        await destroy_app_state(app)
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited at least 0.25 seconds
        assert elapsed >= 0.2


# =============================================================================
# Test get_app_state()
# =============================================================================


class TestGetAppState:
    """Tests for the get_app_state function."""

    def test_returns_app_state_from_request(self):
        """Test that get_app_state returns state from request."""
        app = FastAPI()
        mock_auth = MagicMock()
        expected_state = AppState(auth=mock_auth)
        app.state._spark_state = expected_state

        request = MagicMock()
        request.app = app

        result = get_app_state(request)

        assert result is expected_state

    def test_raises_error_when_not_initialized(self):
        """Test that get_app_state raises error when not initialized."""
        app = FastAPI()
        # No _spark_state set

        request = MagicMock()
        request.app = app

        with pytest.raises(ValueError, match="not been initialized"):
            get_app_state(request)


# =============================================================================
# Test _get_app_state_from_app()
# =============================================================================


class TestGetAppStateFromApp:
    """Tests for the _get_app_state_from_app function."""

    def test_returns_state_from_app(self):
        """Test that function returns state from app."""
        app = FastAPI()
        mock_auth = MagicMock()
        expected_state = AppState(auth=mock_auth)
        app.state._spark_state = expected_state

        result = _get_app_state_from_app(app)

        assert result is expected_state

    def test_raises_error_when_not_initialized(self):
        """Test that function raises error when not initialized."""
        app = FastAPI()

        with pytest.raises(ValueError, match="not been initialized"):
            _get_app_state_from_app(app)

    def test_raises_error_when_state_is_none(self):
        """Test that function raises error when state is None."""
        app = FastAPI()
        app.state._spark_state = None

        with pytest.raises(ValueError, match="not been initialized"):
            _get_app_state_from_app(app)


# =============================================================================
# Test set_request_user()
# =============================================================================


class TestSetRequestUser:
    """Tests for the set_request_user function."""

    def test_sets_user_on_request(self):
        """Test that set_request_user sets user on request."""
        request = MagicMock()
        request.state = MagicMock()
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)

        set_request_user(request, user)

        assert request.state._request_state.user is user

    def test_sets_none_user_on_request(self):
        """Test that set_request_user can set None user."""
        request = MagicMock()
        request.state = MagicMock()

        set_request_user(request, None)

        assert request.state._request_state.user is None

    def test_creates_request_state(self):
        """Test that set_request_user creates RequestState."""
        request = MagicMock()
        request.state = MagicMock()
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.FULL)

        set_request_user(request, user)

        assert isinstance(request.state._request_state, RequestState)


# =============================================================================
# Test get_request_user()
# =============================================================================


class TestGetRequestUser:
    """Tests for the get_request_user function."""

    def test_returns_user_from_request(self):
        """Test that get_request_user returns user from request."""
        request = MagicMock()
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)
        request.state._request_state = RequestState(user=user)

        result = get_request_user(request)

        assert result is user

    def test_returns_none_when_no_request_state(self):
        """Test that get_request_user returns None when no state."""
        request = MagicMock()
        request.state._request_state = None

        result = get_request_user(request)

        assert result is None

    def test_returns_none_when_missing_attribute(self):
        """Test that get_request_user returns None when attribute missing."""
        request = MagicMock()
        # Simulate missing _request_state
        del request.state._request_state
        request.state = MagicMock(spec=[])  # No _request_state

        result = get_request_user(request)

        assert result is None

    def test_returns_none_user_when_set(self):
        """Test that get_request_user returns None user when set."""
        request = MagicMock()
        request.state._request_state = RequestState(user=None)

        result = get_request_user(request)

        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestAppStateIntegration:
    """Integration tests for app state management."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full app state lifecycle: build, use, destroy."""
        app = FastAPI()
        mock_auth = MagicMock()
        mock_auth.close = AsyncMock()

        with patch(
            "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_auth

            # Build
            await build_app(app)

            # Use
            request = MagicMock()
            request.app = app

            state = get_app_state(request)
            assert state.auth is mock_auth

            # Set and get user
            user = KBaseUser(user="lifecycle_user", admin_perm=AdminPermission.NONE)
            set_request_user(request, user)
            retrieved_user = get_request_user(request)
            assert retrieved_user.user == "lifecycle_user"

            # Destroy
            await destroy_app_state(app)
            mock_auth.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_requests_isolated(self):
        """Test that multiple requests have isolated user state."""
        app = FastAPI()
        mock_auth = MagicMock()
        mock_auth.close = AsyncMock()

        with patch(
            "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_auth

            await build_app(app)

            # Create multiple requests
            request1 = MagicMock()
            request1.app = app
            request1.state = MagicMock()

            request2 = MagicMock()
            request2.app = app
            request2.state = MagicMock()

            # Set different users
            user1 = KBaseUser(user="user1", admin_perm=AdminPermission.NONE)
            user2 = KBaseUser(user="user2", admin_perm=AdminPermission.FULL)

            set_request_user(request1, user1)
            set_request_user(request2, user2)

            # Verify isolation
            assert get_request_user(request1).user == "user1"
            assert get_request_user(request2).user == "user2"
            assert get_request_user(request1).admin_perm == AdminPermission.NONE
            assert get_request_user(request2).admin_perm == AdminPermission.FULL

            await destroy_app_state(app)


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAppStateAccess:
    """Tests for concurrent access to app state."""

    @pytest.mark.asyncio
    async def test_concurrent_user_setting(self, async_concurrent_executor):
        """Test concurrent user setting on different requests."""
        app = FastAPI()
        mock_auth = MagicMock()
        mock_auth.close = AsyncMock()

        with patch(
            "src.service.app_state.KBaseAuth.create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_auth
            await build_app(app)

        requests = []
        for i in range(10):
            req = MagicMock()
            req.app = app
            req.state = MagicMock()
            requests.append(req)

        async def set_and_get_user(idx):
            user = KBaseUser(user=f"user_{idx}", admin_perm=AdminPermission.NONE)
            set_request_user(requests[idx], user)
            await asyncio.sleep(0.01)  # Small delay
            return get_request_user(requests[idx]).user

        args_list = [(i,) for i in range(10)]
        results = await async_concurrent_executor(set_and_get_user, args_list)

        # Verify all users were set and retrieved correctly
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 10
        assert sorted(successful) == [f"user_{i}" for i in range(10)]

        await destroy_app_state(app)

    def test_concurrent_app_state_reading(self, concurrent_executor):
        """Test concurrent reading of app state from multiple threads."""
        app = FastAPI()
        mock_auth = MagicMock()
        app.state._spark_state = AppState(auth=mock_auth)

        def read_state(_):
            request = MagicMock()
            request.app = app
            state = get_app_state(request)
            return state.auth is mock_auth

        args_list = [(i,) for i in range(20)]
        results, exceptions = concurrent_executor(read_state, args_list, max_workers=10)

        assert len(exceptions) == 0
        assert all(results)
