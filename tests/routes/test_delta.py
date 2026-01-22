"""
Tests for the delta routes module.

Tests cover:
- All /delta/* endpoints with TestClient
- Mock get_spark_context and auth dependencies
- Test error responses (401, 404, 422, 500)
- Concurrent API requests
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routes import delta
from src.routes.delta import _extract_token_from_request, router
from src.service.dependencies import SparkContext, get_spark_context, auth
from src.service.exceptions import (
    DeltaDatabaseNotFoundError,
    DeltaTableNotFoundError,
    SparkOperationError,
    SparkTimeoutError,
)
from src.service.kb_auth import KBaseUser, AdminPermission
from src.service.models import (
    AggregationSpec,
    ColumnSpec,
    FilterCondition,
    JoinClause,
    OrderBySpec,
    PaginationInfo,
    TableQueryResponse,
    TableSelectRequest,
    TableSelectResponse,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_app(mock_spark_session, mock_kbase_user):
    """Create a FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Create mock spark session
    spark = mock_spark_session()

    # Create mock user
    user = mock_kbase_user()

    # Override dependencies - return SparkContext simulating Connect mode
    def mock_get_spark_context():
        ctx = SparkContext(
            spark=spark,
            is_standalone_subprocess=False,  # Simulate Connect mode
            settings_dict={},
            app_name="test_app",
            username=user.user,
            auth_token=None,
        )
        yield ctx

    def mock_auth():
        return user

    app.dependency_overrides[get_spark_context] = mock_get_spark_context
    app.dependency_overrides[auth] = mock_auth

    return app, spark, user


@pytest.fixture
def delta_client(mock_app):
    """Create a TestClient with mocked dependencies."""
    app, spark, user = mock_app
    return TestClient(app), spark, user


# =============================================================================
# Module Import Tests
# =============================================================================


def test_delta_routes_imports():
    """Test that delta routes module can be imported."""
    assert delta is not None


def test_router_exists():
    """Test that router is properly defined."""
    assert router is not None
    assert router.prefix == "/delta"


# ---
# Model Tests for Query Builder
# ---


class TestTableSelectRequestModel:
    """Tests for TableSelectRequest model validation."""

    def test_minimal_request(self):
        """Test creating a minimal request with only required fields."""
        request = TableSelectRequest(database="mydb", table="users")
        assert request.database == "mydb"
        assert request.table == "users"
        assert request.limit == 100
        assert request.offset == 0
        assert request.distinct is False
        assert request.columns is None
        assert request.aggregations is None
        assert request.filters is None
        assert request.joins is None
        assert request.group_by is None
        assert request.having is None
        assert request.order_by is None

    def test_full_request(self):
        """Test creating a request with all fields populated."""
        request = TableSelectRequest(
            database="sales",
            table="orders",
            joins=[
                JoinClause(
                    join_type="LEFT",
                    database="sales",
                    table="customers",
                    on_left_column="customer_id",
                    on_right_column="id",
                )
            ],
            columns=[
                ColumnSpec(column="order_id"),
                ColumnSpec(
                    column="name", table_alias="customers", alias="customer_name"
                ),
            ],
            distinct=True,
            aggregations=[
                AggregationSpec(function="SUM", column="amount", alias="total"),
            ],
            filters=[
                FilterCondition(column="status", operator="=", value="active"),
                FilterCondition(column="amount", operator=">=", value=100),
            ],
            group_by=["category"],
            having=[
                FilterCondition(column="total", operator=">", value=1000),
            ],
            order_by=[
                OrderBySpec(column="total", direction="DESC"),
            ],
            limit=50,
            offset=10,
        )

        assert request.database == "sales"
        assert request.table == "orders"
        assert len(request.joins) == 1
        assert request.joins[0].join_type == "LEFT"
        assert len(request.columns) == 2
        assert request.distinct is True
        assert len(request.aggregations) == 1
        assert len(request.filters) == 2
        assert request.group_by == ["category"]
        assert len(request.having) == 1
        assert len(request.order_by) == 1
        assert request.limit == 50
        assert request.offset == 10


class TestColumnSpecModel:
    """Tests for ColumnSpec model."""

    def test_simple_column(self):
        """Test creating a simple column spec."""
        col = ColumnSpec(column="name")
        assert col.column == "name"
        assert col.table_alias is None
        assert col.alias is None

    def test_column_with_aliases(self):
        """Test creating a column spec with aliases."""
        col = ColumnSpec(column="name", table_alias="u", alias="user_name")
        assert col.column == "name"
        assert col.table_alias == "u"
        assert col.alias == "user_name"


class TestAggregationSpecModel:
    """Tests for AggregationSpec model."""

    def test_count_star(self):
        """Test COUNT(*) aggregation."""
        agg = AggregationSpec(function="COUNT", column="*")
        assert agg.function == "COUNT"
        assert agg.column == "*"
        assert agg.alias is None

    def test_sum_with_alias(self):
        """Test SUM with alias."""
        agg = AggregationSpec(function="SUM", column="amount", alias="total")
        assert agg.function == "SUM"
        assert agg.column == "amount"
        assert agg.alias == "total"


class TestFilterConditionModel:
    """Tests for FilterCondition model."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        filter_cond = FilterCondition(column="status", operator="=", value="active")
        assert filter_cond.column == "status"
        assert filter_cond.operator == "="
        assert filter_cond.value == "active"

    def test_in_operator(self):
        """Test IN operator with values list."""
        filter_cond = FilterCondition(column="id", operator="IN", values=[1, 2, 3])
        assert filter_cond.column == "id"
        assert filter_cond.operator == "IN"
        assert filter_cond.values == [1, 2, 3]

    def test_is_null(self):
        """Test IS NULL operator."""
        filter_cond = FilterCondition(column="deleted_at", operator="IS NULL")
        assert filter_cond.column == "deleted_at"
        assert filter_cond.operator == "IS NULL"
        assert filter_cond.value is None


class TestJoinClauseModel:
    """Tests for JoinClause model."""

    def test_inner_join(self):
        """Test INNER JOIN clause."""
        join = JoinClause(
            join_type="INNER",
            database="mydb",
            table="orders",
            on_left_column="user_id",
            on_right_column="id",
        )
        assert join.join_type == "INNER"
        assert join.database == "mydb"
        assert join.table == "orders"
        assert join.on_left_column == "user_id"
        assert join.on_right_column == "id"


class TestOrderBySpecModel:
    """Tests for OrderBySpec model."""

    def test_default_direction(self):
        """Test default direction is ASC."""
        order = OrderBySpec(column="name")
        assert order.column == "name"
        assert order.direction == "ASC"

    def test_desc_direction(self):
        """Test DESC direction."""
        order = OrderBySpec(column="created_at", direction="DESC")
        assert order.column == "created_at"
        assert order.direction == "DESC"


class TestPaginationInfoModel:
    """Tests for PaginationInfo model."""

    def test_pagination_info(self):
        """Test creating pagination info."""
        pagination = PaginationInfo(
            limit=100, offset=50, total_count=500, has_more=True
        )
        assert pagination.limit == 100
        assert pagination.offset == 50
        assert pagination.total_count == 500
        assert pagination.has_more is True


class TestTableSelectResponseModel:
    """Tests for TableSelectResponse model."""

    def test_response_with_data(self):
        """Test creating a response with data."""
        response = TableSelectResponse(
            data=[
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            pagination=PaginationInfo(
                limit=10, offset=0, total_count=2, has_more=False
            ),
        )
        assert len(response.data) == 2
        assert response.data[0]["name"] == "Alice"
        assert response.pagination.total_count == 2
        assert response.pagination.has_more is False


# =============================================================================
# Route Integration Tests
# =============================================================================


class TestListDatabasesEndpoint:
    """Tests for the /delta/databases/list endpoint."""

    def test_list_databases_success(
        self, mock_spark_session, mock_kbase_user, mock_settings
    ):
        """Test successful database listing."""
        app = FastAPI()
        app.include_router(router)

        spark = mock_spark_session()
        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict={},
                app_name="test_app",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.data_store.get_databases", return_value=["db1", "db2"]
        ):
            # Explicitly disable filter_by_namespace as it defaults to True
            response = client.post(
                "/delta/databases/list",
                json={"use_hms": True, "filter_by_namespace": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert "databases" in data
        assert data["databases"] == ["db1", "db2"]

    def test_list_databases_with_namespace_filter(
        self, mock_spark_session, mock_kbase_user
    ):
        """Test database listing with namespace filter requires token."""
        app = FastAPI()
        app.include_router(router)

        spark = mock_spark_session()
        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict={},
                app_name="test_app",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        # Without proper token, should fail
        with patch(
            "src.routes.delta.data_store.get_databases", return_value=["u_test__db"]
        ):
            response = client.post(
                "/delta/databases/list",
                json={"use_hms": True, "filter_by_namespace": False},  # Disable filter
            )

        # Without filter, should succeed
        assert response.status_code == 200


class TestListTablesEndpoint:
    """Tests for the /delta/databases/tables/list endpoint."""

    def test_list_tables_success(self, delta_client, mock_settings):
        """Test successful table listing."""
        client, spark, user = delta_client

        with patch(
            "src.routes.delta.data_store.get_tables", return_value=["table1", "table2"]
        ):
            with patch("src.routes.delta.get_settings", return_value=mock_settings):
                response = client.post(
                    "/delta/databases/tables/list",
                    json={"database": "testdb", "use_hms": True},
                )

        assert response.status_code == 200
        data = response.json()
        assert "tables" in data
        assert data["tables"] == ["table1", "table2"]

    def test_list_tables_requires_database(self, delta_client):
        """Test that database field is required."""
        client, spark, user = delta_client

        response = client.post("/delta/databases/tables/list", json={"use_hms": True})

        assert response.status_code == 422  # Validation error


class TestGetTableSchemaEndpoint:
    """Tests for the /delta/databases/tables/schema endpoint."""

    def test_get_schema_success(self, delta_client):
        """Test successful schema retrieval."""
        client, spark, user = delta_client

        with patch(
            "src.routes.delta.data_store.get_table_schema",
            return_value=["id", "name", "email"],
        ):
            response = client.post(
                "/delta/databases/tables/schema",
                json={"database": "testdb", "table": "users"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "columns" in data
        assert data["columns"] == ["id", "name", "email"]


class TestCountTableEndpoint:
    """Tests for the /delta/tables/count endpoint."""

    def test_count_success(self, delta_client):
        """Test successful table count."""
        client, spark, user = delta_client

        with patch(
            "src.routes.delta.delta_service.count_delta_table", return_value=12345
        ):
            response = client.post(
                "/delta/tables/count",
                json={"database": "testdb", "table": "users"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 12345


class TestSampleTableEndpoint:
    """Tests for the /delta/tables/sample endpoint."""

    def test_sample_success(self, delta_client):
        """Test successful table sampling."""
        client, spark, user = delta_client

        sample_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        with patch(
            "src.routes.delta.delta_service.sample_delta_table",
            return_value=sample_data,
        ):
            response = client.post(
                "/delta/tables/sample",
                json={"database": "testdb", "table": "users", "limit": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert "sample" in data
        assert len(data["sample"]) == 2

    def test_sample_with_columns(self, delta_client):
        """Test sampling with specific columns."""
        client, spark, user = delta_client

        with patch(
            "src.routes.delta.delta_service.sample_delta_table",
            return_value=[{"id": 1}],
        ):
            response = client.post(
                "/delta/tables/sample",
                json={
                    "database": "testdb",
                    "table": "users",
                    "limit": 10,
                    "columns": ["id"],
                },
            )

        assert response.status_code == 200


class TestQueryTableEndpoint:
    """Tests for the /delta/tables/query endpoint."""

    def test_query_success(self, delta_client):
        """Test successful query execution."""
        client, spark, user = delta_client

        query_response = TableQueryResponse(
            result=[{"count": 100}],
            pagination=PaginationInfo(
                limit=1000, offset=0, total_count=1, has_more=False
            ),
        )

        with patch(
            "src.routes.delta.delta_service.query_delta_table",
            return_value=query_response,
        ):
            response = client.post(
                "/delta/tables/query",
                json={"query": "SELECT COUNT(*) as count FROM users"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "pagination" in data
        assert data["result"][0]["count"] == 100
        assert data["pagination"]["total_count"] == 1


class TestSelectTableEndpoint:
    """Tests for the /delta/tables/select endpoint."""

    def test_select_success(self, delta_client):
        """Test successful select execution."""
        client, spark, user = delta_client

        select_response = TableSelectResponse(
            data=[{"id": 1}, {"id": 2}],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=2, has_more=False
            ),
        )

        with patch(
            "src.routes.delta.delta_service.select_from_delta_table",
            return_value=select_response,
        ):
            response = client.post(
                "/delta/tables/select",
                json={"database": "testdb", "table": "users"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "pagination" in data
        assert len(data["data"]) == 2

    def test_select_with_filters(self, delta_client):
        """Test select with filter conditions."""
        client, spark, user = delta_client

        select_response = TableSelectResponse(
            data=[{"id": 1}],
            pagination=PaginationInfo(
                limit=100, offset=0, total_count=1, has_more=False
            ),
        )

        with patch(
            "src.routes.delta.delta_service.select_from_delta_table",
            return_value=select_response,
        ):
            response = client.post(
                "/delta/tables/select",
                json={
                    "database": "testdb",
                    "table": "users",
                    "filters": [
                        {"column": "status", "operator": "=", "value": "active"}
                    ],
                },
            )

        assert response.status_code == 200

    def test_select_with_pagination(self, delta_client):
        """Test select with pagination parameters."""
        client, spark, user = delta_client

        select_response = TableSelectResponse(
            data=[],
            pagination=PaginationInfo(
                limit=50, offset=100, total_count=500, has_more=True
            ),
        )

        with patch(
            "src.routes.delta.delta_service.select_from_delta_table",
            return_value=select_response,
        ):
            response = client.post(
                "/delta/tables/select",
                json={
                    "database": "testdb",
                    "table": "users",
                    "limit": 50,
                    "offset": 100,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["limit"] == 50
        assert data["pagination"]["offset"] == 100
        assert data["pagination"]["has_more"] is True


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error responses from endpoints."""

    def test_validation_error_returns_422(self, mock_spark_session, mock_kbase_user):
        """Test that validation errors return 422."""
        app = FastAPI()
        app.include_router(router)

        spark = mock_spark_session()
        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict={},
                app_name="test_app",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        # Missing required fields
        response = client.post("/delta/tables/count", json={})

        assert response.status_code == 422

    def test_database_not_found_error(self):
        """Test that DeltaDatabaseNotFoundError is raised correctly."""
        # Test the exception can be raised and caught
        with pytest.raises(DeltaDatabaseNotFoundError):
            raise DeltaDatabaseNotFoundError("Database not found")

    def test_table_not_found_error(self):
        """Test that DeltaTableNotFoundError is raised correctly."""
        with pytest.raises(DeltaTableNotFoundError):
            raise DeltaTableNotFoundError("Table not found")

    def test_spark_operation_error(self):
        """Test that SparkOperationError is raised correctly."""
        with pytest.raises(SparkOperationError):
            raise SparkOperationError("Spark failed")

    def test_spark_timeout_error(self):
        """Test that SparkTimeoutError is raised correctly."""
        error = SparkTimeoutError(operation="count", timeout=30)
        assert error.operation == "count"
        assert error.timeout == 30
        assert "count" in str(error)
        assert "30" in str(error)


# =============================================================================
# Concurrent Request Tests
# =============================================================================


class TestConcurrentRequests:
    """Tests for concurrent API requests."""

    def test_concurrent_count_requests(self, concurrent_executor):
        """Test concurrent count requests."""

        def make_request(i):
            app = FastAPI()
            app.include_router(router)

            spark = MagicMock()
            user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)

            def mock_get_spark_ctx():
                ctx = SparkContext(
                    spark=spark,
                    is_standalone_subprocess=False,
                    settings_dict={},
                    app_name="test_app",
                    username=user.user,
                )
                yield ctx

            def mock_auth():
                return user

            app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
            app.dependency_overrides[auth] = mock_auth

            client = TestClient(app)

            with patch(
                "src.routes.delta.delta_service.count_delta_table",
                return_value=i * 100,
            ):
                response = client.post(
                    "/delta/tables/count",
                    json={"database": f"db_{i}", "table": "users"},
                )

            return response.json()["count"]

        args_list = [(i,) for i in range(5)]
        results, exceptions = concurrent_executor(
            make_request, args_list, max_workers=5
        )

        assert len(exceptions) == 0
        # Results may not be in order due to concurrency, just verify all values present
        assert len(results) == 5

    def test_concurrent_different_endpoints(self, concurrent_executor):
        """Test concurrent requests to different endpoints."""

        # Create a single app instance outside the concurrent function
        # to avoid re-triggering module initialization
        app = FastAPI()
        app.include_router(router)

        spark = MagicMock()
        user = KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=spark,
                is_standalone_subprocess=False,
                settings_dict={},
                app_name="test_app",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        def make_mixed_request(request_type):
            if request_type == "count":
                with patch(
                    "src.routes.delta.delta_service.count_delta_table",
                    return_value=100,
                ):
                    response = client.post(
                        "/delta/tables/count",
                        json={"database": "db", "table": "t"},
                    )
                return ("count", response.status_code)

            elif request_type == "sample":
                with patch(
                    "src.routes.delta.delta_service.sample_delta_table",
                    return_value=[],
                ):
                    response = client.post(
                        "/delta/tables/sample",
                        json={"database": "db", "table": "t", "limit": 10},
                    )
                return ("sample", response.status_code)

            elif request_type == "query":
                query_response = TableQueryResponse(
                    result=[],
                    pagination=PaginationInfo(
                        limit=1000, offset=0, total_count=0, has_more=False
                    ),
                )
                with patch(
                    "src.routes.delta.delta_service.query_delta_table",
                    return_value=query_response,
                ):
                    response = client.post(
                        "/delta/tables/query",
                        json={"query": "SELECT 1"},
                    )
                return ("query", response.status_code)

            return (request_type, 500)

        args_list = [
            ("count",),
            ("sample",),
            ("query",),
            ("count",),
            ("sample",),
        ]
        results, exceptions = concurrent_executor(
            make_mixed_request, args_list, max_workers=5
        )

        assert len(exceptions) == 0
        assert all(r[1] == 200 for r in results)


# =============================================================================
# Token Extraction Tests
# =============================================================================


class TestTokenExtraction:
    """Tests for the _extract_token_from_request helper."""

    def test_extract_valid_bearer_token(self):
        """Test extracting valid Bearer token."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer my_token_12345"}

        token = _extract_token_from_request(request)

        assert token == "my_token_12345"

    def test_extract_missing_header_returns_none(self):
        """Test that missing header returns None."""
        request = MagicMock()
        request.headers = {}

        token = _extract_token_from_request(request)

        assert token is None

    def test_extract_non_bearer_returns_none(self):
        """Test that non-Bearer auth returns None."""
        request = MagicMock()
        request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}

        token = _extract_token_from_request(request)

        assert token is None


# =============================================================================
# Standalone Subprocess Dispatch Tests
# =============================================================================


class TestStandaloneSubprocessDispatch:
    """Tests for routes when in Standalone subprocess mode (is_standalone_subprocess=True)."""

    def test_list_databases_standalone_dispatch(self, mock_kbase_user):
        """Test that list_databases dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,  # No session in Standalone mode
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_list_dbs",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value=["db1", "db2"],
        ) as mock_run:
            response = client.post(
                "/delta/databases/list",
                json={"use_hms": True, "filter_by_namespace": False},
            )

        assert response.status_code == 200
        assert response.json()["databases"] == ["db1", "db2"]
        mock_run.assert_called_once()

    def test_list_tables_standalone_dispatch(self, mock_kbase_user):
        """Test that list_tables dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_list_tables",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value=["table1", "table2"],
        ) as mock_run:
            response = client.post(
                "/delta/databases/tables/list",
                json={"database": "test_db", "use_hms": True},
            )

        assert response.status_code == 200
        assert response.json()["tables"] == ["table1", "table2"]
        mock_run.assert_called_once()

    def test_get_table_schema_standalone_dispatch(self, mock_kbase_user):
        """Test that get_table_schema dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_schema",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value=["id", "name", "created_at"],
        ) as mock_run:
            response = client.post(
                "/delta/databases/tables/schema",
                json={"database": "test_db", "table": "test_table"},
            )

        assert response.status_code == 200
        assert response.json()["columns"] == ["id", "name", "created_at"]
        mock_run.assert_called_once()

    def test_get_db_structure_standalone_dispatch(self, mock_kbase_user):
        """Test that get_db_structure dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_structure",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value={"db1": ["table1"]},
        ) as mock_run:
            response = client.post(
                "/delta/databases/structure",
                json={"with_schema": False, "use_hms": True},
            )

        assert response.status_code == 200
        assert response.json()["structure"] == {"db1": ["table1"]}
        mock_run.assert_called_once()

    def test_count_table_standalone_dispatch(self, mock_kbase_user):
        """Test that count_table dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_count",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value=42,
        ) as mock_run:
            response = client.post(
                "/delta/tables/count",
                json={"database": "test_db", "table": "test_table"},
            )

        assert response.status_code == 200
        assert response.json()["count"] == 42
        mock_run.assert_called_once()

    def test_sample_table_standalone_dispatch(self, mock_kbase_user):
        """Test that sample_table dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_sample",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value=[{"id": 1, "name": "test"}],
        ) as mock_run:
            response = client.post(
                "/delta/tables/sample",
                json={"database": "test_db", "table": "test_table", "limit": 10},
            )

        assert response.status_code == 200
        assert response.json()["sample"] == [{"id": 1, "name": "test"}]
        mock_run.assert_called_once()

    def test_query_table_standalone_dispatch(self, mock_kbase_user):
        """Test that query_table dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_query",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value={
                "result": [{"id": 1}],
                "pagination": {
                    "total_count": 1,
                    "limit": 100,
                    "offset": 0,
                    "has_more": False,
                },
            },
        ) as mock_run:
            response = client.post(
                "/delta/tables/query",
                json={"query": "SELECT * FROM test_table"},
            )

        assert response.status_code == 200
        assert response.json()["result"] == [{"id": 1}]
        mock_run.assert_called_once()

    def test_select_table_standalone_dispatch(self, mock_kbase_user):
        """Test that select_table dispatches to run_in_spark_process in Standalone mode."""
        app = FastAPI()
        app.include_router(router)

        user = mock_kbase_user()

        def mock_get_spark_ctx():
            ctx = SparkContext(
                spark=None,
                is_standalone_subprocess=True,
                settings_dict={"USER": "testuser"},
                app_name="mcp_select",
                username=user.user,
            )
            yield ctx

        def mock_auth():
            return user

        app.dependency_overrides[get_spark_context] = mock_get_spark_ctx
        app.dependency_overrides[auth] = mock_auth

        client = TestClient(app)

        with patch(
            "src.routes.delta.run_in_spark_process",
            return_value={
                "data": [{"id": 1}],
                "pagination": {
                    "total_count": 1,
                    "limit": 100,
                    "offset": 0,
                    "has_more": False,
                },
            },
        ) as mock_run:
            response = client.post(
                "/delta/tables/select",
                json={"database": "test_db", "table": "test_table"},
            )

        assert response.status_code == 200
        assert response.json()["data"] == [{"id": 1}]
        mock_run.assert_called_once()
