"""
Shared test fixtures for the datalake-mcp-server test suite.

Provides reusable mocks for external dependencies:
- Spark/PySpark: Mock SparkSession, DataFrames, and Spark operations
- Redis: Mock redis.Redis client and connection pool
- KBase Auth: Mock aiohttp.ClientSession for auth API calls
- HTTP: Mock httpx.Client for governance API calls
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import AnyHttpUrl, AnyUrl

from src.service.app_state import RequestState
from src.service.dependencies import auth, get_spark_session
from src.service.exceptions import InvalidTokenError
from src.service.kb_auth import AdminPermission, KBaseUser
from src.settings import BERDLSettings

# =============================================================================
# Settings Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> BERDLSettings:
    """
    Create mock BERDLSettings for testing.

    Returns:
        BERDLSettings with test-appropriate defaults.
    """
    return BERDLSettings(
        KBASE_AUTH_TOKEN="test_token_12345",
        USER="testuser",
        MINIO_ENDPOINT_URL="localhost:9002",
        MINIO_ACCESS_KEY="test_access_key",
        MINIO_SECRET_KEY="test_secret_key",
        MINIO_SECURE=False,
        BERDL_REDIS_HOST="localhost",
        BERDL_REDIS_PORT=6379,
        SPARK_HOME="/usr/local/spark",
        SPARK_MASTER_URL=AnyUrl("spark://localhost:7077"),
        SPARK_CONNECT_URL=AnyUrl("sc://localhost:15002"),
        BERDL_HIVE_METASTORE_URI=AnyUrl("thrift://localhost:9083"),
        SPARK_WORKER_COUNT=1,
        SPARK_WORKER_CORES=1,
        SPARK_WORKER_MEMORY="2GiB",
        SPARK_MASTER_CORES=1,
        SPARK_MASTER_MEMORY="1GiB",
        GOVERNANCE_API_URL=AnyHttpUrl("http://localhost:8000"),
        BERDL_POD_IP="127.0.0.1",
    )


@pytest.fixture
def mock_settings_patch(mock_settings):
    """
    Patch get_settings to return mock settings.

    Usage:
        def test_something(mock_settings_patch):
            # get_settings() now returns mock_settings
    """
    with patch("src.settings.get_settings", return_value=mock_settings):
        yield mock_settings


# =============================================================================
# Spark Session Fixtures
# =============================================================================


@pytest.fixture
def mock_spark_row():
    """Factory for creating mock Spark Row objects."""

    def _create_row(data: Dict[str, Any]) -> MagicMock:
        row = MagicMock()
        row.asDict.return_value = data
        for key, value in data.items():
            setattr(row, key, value)
        return row

    return _create_row


@pytest.fixture
def mock_spark_dataframe(mock_spark_row):
    """
    Create a configurable mock Spark DataFrame.

    Usage:
        def test_query(mock_spark_dataframe):
            df = mock_spark_dataframe([{"id": 1, "name": "test"}])
            df.collect()  # Returns list of mock rows
    """

    def _create_df(data: List[Dict[str, Any]] = None, count: int = None):
        if data is None:
            data = []

        df = MagicMock()

        # Mock collect() and toLocalIterator() to return rows
        rows = [mock_spark_row(row) for row in data]
        df.collect.return_value = rows
        df.toLocalIterator.return_value = iter(rows)

        # Mock count()
        df.count.return_value = count if count is not None else len(data)

        # Mock chained operations
        df.select.return_value = df
        df.filter.return_value = df
        df.where.return_value = df
        df.limit.return_value = df
        df.orderBy.return_value = df
        df.groupBy.return_value = df

        return df

    return _create_df


@pytest.fixture
def mock_spark_session(mock_spark_dataframe):
    """
    Create a mock SparkSession for testing.

    The mock supports:
    - spark.sql(query) -> mock DataFrame
    - spark.table(name) -> mock DataFrame
    - spark.catalog.listDatabases() -> list of mock Database objects
    - spark.catalog.listTables(db) -> list of mock Table objects
    - spark.stop() -> no-op

    Usage:
        def test_query(mock_spark_session):
            spark = mock_spark_session()
            spark.sql("SELECT * FROM test").collect()
    """

    def _create_session(
        sql_results: List[Dict[str, Any]] = None,
        table_results: List[Dict[str, Any]] = None,
        databases: List[str] = None,
        tables: Dict[str, List[str]] = None,
    ):
        if sql_results is None:
            sql_results = []
        if table_results is None:
            table_results = sql_results
        if databases is None:
            databases = ["default", "testdb"]
        if tables is None:
            tables = {"default": ["table1"], "testdb": ["users", "orders"]}

        spark = MagicMock()

        # Mock sql() method
        sql_df = mock_spark_dataframe(sql_results)
        spark.sql.return_value = sql_df

        # Mock table() method
        table_df = mock_spark_dataframe(table_results)
        spark.table.return_value = table_df

        # Mock catalog
        catalog = MagicMock()

        # Mock listDatabases
        db_objects = []
        for db_name in databases:
            db_obj = MagicMock()
            db_obj.name = db_name
            db_objects.append(db_obj)
        catalog.listDatabases.return_value = db_objects

        # Mock listTables
        def list_tables_side_effect(db_name):
            table_list = tables.get(db_name, [])
            table_objects = []
            for table_name in table_list:
                table_obj = MagicMock()
                table_obj.name = table_name
                table_objects.append(table_obj)
            return table_objects

        catalog.listTables.side_effect = list_tables_side_effect

        spark.catalog = catalog

        # Mock stop
        spark.stop.return_value = None

        return spark

    return _create_session


# =============================================================================
# Redis Fixtures
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """
    Create a mock Redis client for testing.

    The mock supports:
    - client.get(key) -> bytes or None
    - client.set(name, value, ex=ttl) -> True
    - client.delete(key) -> int

    Usage:
        def test_cache(mock_redis_client):
            client = mock_redis_client({"key": b'{"data": "value"}'})
    """

    def _create_client(cache_data: Dict[str, bytes] = None):
        if cache_data is None:
            cache_data = {}

        client = MagicMock()
        stored_data = dict(cache_data)

        def get_side_effect(key):
            return stored_data.get(key)

        def set_side_effect(name, value, ex=None):
            stored_data[name] = value.encode() if isinstance(value, str) else value
            return True

        def delete_side_effect(*keys):
            count = 0
            for key in keys:
                if key in stored_data:
                    del stored_data[key]
                    count += 1
            return count

        client.get.side_effect = get_side_effect
        client.set.side_effect = set_side_effect
        client.delete.side_effect = delete_side_effect

        # Store reference to internal data for assertions
        client._stored_data = stored_data

        return client

    return _create_client


@pytest.fixture
def mock_redis_client_patch(mock_redis_client):
    """
    Patch the Redis client factory to return a mock client.

    Usage:
        def test_cache(mock_redis_client_patch):
            client, patch = mock_redis_client_patch()
            # All redis operations now use the mock
    """

    def _create_patched(cache_data: Dict[str, bytes] = None):
        client = mock_redis_client(cache_data)
        patcher = patch("src.cache.redis_cache._get_redis_client", return_value=client)
        patcher.start()
        return client, patcher

    return _create_patched


# =============================================================================
# KBase Auth Fixtures
# =============================================================================


@pytest.fixture
def mock_kbase_user():
    """Create a mock KBaseUser for testing."""

    def _create_user(
        username: str = "testuser",
        admin_perm: AdminPermission = AdminPermission.NONE,
    ):
        return KBaseUser(user=username, admin_perm=admin_perm)

    return _create_user


@pytest.fixture
def mock_kbase_auth(mock_kbase_user):
    """
    Create a mock KBaseAuth client for testing.

    Usage:
        def test_auth(mock_kbase_auth):
            auth = mock_kbase_auth()
            user = await auth.get_user("valid_token")
    """

    def _create_auth(
        valid_tokens: Dict[str, str] = None,
        admin_tokens: List[str] = None,
    ):
        if valid_tokens is None:
            valid_tokens = {"valid_token": "testuser"}
        if admin_tokens is None:
            admin_tokens = []

        auth = MagicMock()

        async def get_user_side_effect(token):
            if token in valid_tokens:
                username = valid_tokens[token]
                admin_perm = (
                    AdminPermission.FULL
                    if token in admin_tokens
                    else AdminPermission.NONE
                )
                return mock_kbase_user(username, admin_perm)
            raise InvalidTokenError("KBase auth server reported token is invalid.")

        auth.get_user = AsyncMock(side_effect=get_user_side_effect)
        auth.close = AsyncMock()

        return auth

    return _create_auth


@pytest.fixture
def mock_aiohttp_session():
    """
    Create a mock aiohttp ClientSession for testing.

    Useful for testing KBaseAuth initialization and HTTP calls.
    """

    def _create_session(responses: Dict[str, Any] = None):
        if responses is None:
            responses = {}

        session = MagicMock()

        async def mock_get(url, headers=None):
            response = MagicMock()
            response.status = 200

            async def mock_json():
                if url in responses:
                    return responses[url]
                # Default auth service response
                return {"servicename": "Authentication Service"}

            async def mock_text():
                return str(responses.get(url, ""))

            response.json = mock_json
            response.text = mock_text
            return response

        # Create async context manager
        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(
            side_effect=lambda: mock_get(context_manager._url, context_manager._headers)
        )
        context_manager.__aexit__ = AsyncMock(return_value=None)

        def get_side_effect(url, headers=None):
            context_manager._url = url
            context_manager._headers = headers
            return context_manager

        session.get.side_effect = get_side_effect
        session.close = AsyncMock()
        session.closed = False

        return session

    return _create_session


# =============================================================================
# HTTP Client Fixtures (for governance API)
# =============================================================================


@pytest.fixture
def mock_httpx_client():
    """
    Create a mock httpx Client for testing governance API calls.

    Usage:
        def test_governance(mock_httpx_client):
            client = mock_httpx_client({
                "http://localhost/api": {"data": "value"}
            })
    """

    def _create_client(responses: Dict[str, Any] = None):
        if responses is None:
            responses = {}

        client = MagicMock()

        def get_side_effect(url, headers=None, params=None):
            response = MagicMock()
            response.status_code = 200

            # Build full URL with params for lookup
            lookup_url = url
            if params:
                param_str = "&".join(f"{k}={v}" for k, v in params.items())
                lookup_url = f"{url}?{param_str}"

            def json_func():
                # Try exact match first, then base URL
                if lookup_url in responses:
                    return responses[lookup_url]
                if url in responses:
                    return responses[url]
                return {}

            response.json = json_func
            response.raise_for_status = MagicMock()

            return response

        client.get.side_effect = get_side_effect
        client.close = MagicMock()

        return client

    return _create_client


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================


@pytest.fixture
def mock_app_dependencies(mock_spark_session, mock_kbase_user, mock_settings):
    """
    Create a fully mocked FastAPI app for testing routes.

    This fixture patches both auth and spark session dependencies.
    """
    from src.main import create_application

    app = create_application()

    # Create mock spark session
    spark = mock_spark_session()

    # Create mock user
    user = mock_kbase_user()

    # Override dependencies
    def mock_get_spark():
        yield spark

    def mock_auth():
        return user

    app.dependency_overrides[get_spark_session] = mock_get_spark
    app.dependency_overrides[auth] = mock_auth

    return app, spark, user


@pytest.fixture
def test_client(mock_app_dependencies):
    """
    Create a TestClient with mocked dependencies.

    Usage:
        def test_endpoint(test_client):
            app, spark, user = mock_app_dependencies
            client = TestClient(app)
            response = client.get("/health")
    """
    app, spark, user = mock_app_dependencies
    return TestClient(app)


@pytest.fixture
def client():
    """
    Basic test client without mocked dependencies.

    Note: This requires actual services to be available.
    Use test_client fixture for unit tests with mocks.
    """
    from src.main import create_application

    app = create_application()
    return TestClient(app)


# =============================================================================
# Request/Response Fixtures
# =============================================================================


@pytest.fixture
def mock_request(mock_kbase_user):
    """
    Create a mock FastAPI Request object for testing.

    Usage:
        def test_handler(mock_request):
            request = mock_request(user="testuser")
    """

    def _create_request(
        user: str = "testuser",
        headers: Dict[str, str] = None,
        app: FastAPI = None,
    ):
        if headers is None:
            headers = {"Authorization": "Bearer valid_token"}

        request = MagicMock()
        request.headers = headers
        request.app = app or MagicMock()

        # Set up request state with user
        request.state = MagicMock()

        request.state._request_state = RequestState(
            user=mock_kbase_user(user) if user else None
        )

        return request

    return _create_request


# =============================================================================
# Concurrency Testing Fixtures
# =============================================================================


@pytest.fixture
def concurrent_executor():
    """
    Fixture for testing concurrent operations.

    Usage:
        def test_concurrent(concurrent_executor):
            results = concurrent_executor(my_func, args_list, max_workers=10)
    """

    def _execute(func, args_list, max_workers=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            results = []
            exceptions = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    exceptions.append(e)
            return results, exceptions

    return _execute


@pytest.fixture
def async_concurrent_executor():
    """
    Fixture for testing concurrent async operations.

    Usage:
        async def test_concurrent(async_concurrent_executor):
            results = await async_concurrent_executor(my_async_func, args_list)
    """

    async def _execute(func, args_list, max_concurrent=10):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_func(*args):
            async with semaphore:
                return await func(*args)

        tasks = [limited_func(*args) for args in args_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    return _execute
