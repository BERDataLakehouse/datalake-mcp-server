"""Tests for the delta routes module."""

from src.routes import delta
from src.service.models import (
    AggregationSpec,
    ColumnSpec,
    FilterCondition,
    JoinClause,
    OrderBySpec,
    PaginationInfo,
    TableSelectRequest,
    TableSelectResponse,
)


def test_delta_routes_imports():
    """Test that delta routes module can be imported."""
    assert delta is not None


def test_noop():
    """Simple placeholder test."""
    assert 1 == 1


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
