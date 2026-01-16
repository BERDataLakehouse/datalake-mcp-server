"""Tests for the delta_service module.

Tests cover:
- Query validation (_check_query_is_valid)
- Query building (build_select_query)
- Spark operations with mocked SparkSession:
  - count_delta_table()
  - sample_delta_table()
  - query_delta_table()
  - select_from_delta_table()
- Caching behavior
- Timeout handling
- Concurrent query execution
"""

from unittest.mock import MagicMock, patch

import pytest

from src.delta_lake import delta_service
from src.service.exceptions import (
    DeltaDatabaseNotFoundError,
    DeltaTableNotFoundError,
    SparkOperationError,
    SparkQueryError,
    SparkTimeoutError,
)
from src.service.models import (
    AggregationSpec,
    ColumnSpec,
    FilterCondition,
    OrderBySpec,
    TableSelectRequest,
    TableSelectResponse,
)


def test_delta_service_imports():
    """Test that delta_service module can be imported."""
    assert delta_service is not None


# Lists of valid and invalid queries to test validation logic
VALID_QUERIES = [
    "SELECT * FROM my_table",
    "SELECT id, name FROM users",
    "SELECT COUNT(*) FROM transactions",
    "SELECT AVG(amount) FROM payments",
    "SELECT * FROM table WHERE id > 100",
    "SELECT DISTINCT category FROM products",
    "SELECT * FROM orders ORDER BY id DESC",
    "SELECT * FROM customers LIMIT 10",
    "SELECT * FROM (SELECT id FROM inner_table) AS subquery",
    "SELECT t1.id, t2.name FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id",
    "SELECT * FROM my_table WHERE date BETWEEN '2023-01-01' AND '2023-12-31'",
    "SELECT id FROM events WHERE type IN ('click', 'view', 'purchase')",
    "SELECT * FROM sales WHERE region = 'North' AND amount > 1000",
    "SELECT COALESCE(email, phone, 'no contact') FROM contacts",
    "SELECT * FROM employees WHERE department LIKE 'eng%'",
]

# Invalid queries grouped by expected error message
INVALID_QUERIES = {
    "must contain exactly one statement": [
        # Multiple statements
        "SELECT * FROM users; SELECT * FROM roles",
        "SELECT * FROM table1; DROP TABLE table2",
        "SELECT * FROM users; WAITFOR DELAY '0:0:10'--",
        "SELECT * FROM users WHERE id = ABS(1); DROP TABLE logs; --",
        "SELECT * FROM users; DROP TABLE logs",
        # Empty statement
        "",
        "   ",
        "\n\n",
    ],
    "must be one of the following: select": [
        # Non-SELECT statements
        "INSERT INTO users VALUES (1, 'john')",
        "UPDATE users SET active = true WHERE id = 1",
        "DELETE FROM logs WHERE created_at < '2023-01-01'",
        "DROP TABLE old_data",
        "CREATE TABLE new_table (id INT, name VARCHAR)",
        "TRUNCATE TABLE logs",
        "ALTER TABLE users ADD COLUMN age INT",
        "MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN UPDATE SET target.val = source.val",
        "VACUUM delta_table",
        "CREATE OR REPLACE FUNCTION f() RETURNS void AS $$ SELECT 1; $$ LANGUAGE SQL",
        "WITH t AS (SELECT 1) DELETE FROM users",
        # Invalid SQL queries
        "/**/SEL/**/ECT * FR/**/OM users",
        "S%45L%45CT%20*%20%46ROM%20users",
        "S\tE\tL\tE\tC\tT * F\tR\tO\tM users",
    ],
    "contains forbidden keyword": [
        # Forbidden keywords in various contexts
        # Most of them are not valid SQL queries
        "Select * from users where id in (DELETE from users where id < 100)",
        "SELECT * FROM users WHERE DROP = 1",
        "SELECT ${1+drop}table FROM users",
        "SELECT * FROM users WHERE id = 1 OR drop = true",
        "SELECT * FROM users WHERE id = 1 OR drop table users",
        "SELECT * FROM users WHERE id = 1 OR delete from users",
        "SELECT * FROM users WHERE id = 1 AND update users set name = 'hacked'",
        "SELECT * FROM users WHERE id = 1 AND create table hacked (id int)",
        "SELECT * FROM users WHERE command = 'drop table'",
        "SELECT * FROM users WHERE col = 'value' OR drop table logs",
        "SELECT * FROM users WHERE DROP = true",
    ],
    "contains disallowed metacharacter": [
        # Metacharacters
        "SELECT * FROM users WHERE name = 'user;' ",
        "/* Comment */ SELECT id FROM users /* Another comment */",
        "SELECT * FROM users WHERE id = 1 -- bypass filter",
    ],
    "contains forbidden PostgreSQL schema": [
        # PostgreSQL system schemas that should be blocked
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_class",
        "SELECT tablename FROM pg_tables",
        "SELECT table_name FROM information_schema.tables",
        "SELECT * FROM pg_settings WHERE name = 'data_directory'",
        "SELECT * FROM information_schema.schemata",
        "SELECT * FROM pg_shadow",
        "SELECT * FROM pg_authid",
        "SELECT * FROM pg_stat_activity",
        "SELECT datname FROM pg_database",
    ],
}


def test_valid_queries():
    """Test that all valid queries pass validation."""
    for query in VALID_QUERIES:
        try:
            assert delta_service._check_query_is_valid(query) is True
        except Exception as e:
            pytest.fail(f"Valid query '{query}' failed validation: {str(e)}")


def test_invalid_queries():
    """Test that all invalid queries fail validation with the correct error messages."""
    for error_pattern, queries in INVALID_QUERIES.items():
        for query in queries:
            with pytest.raises(SparkQueryError, match=error_pattern):
                delta_service._check_query_is_valid(query)


# ---
# Query Builder Tests
# ---


class TestValidateIdentifier:
    """Tests for _validate_identifier function."""

    def test_valid_identifiers(self):
        """Test that valid identifiers pass validation."""
        valid_names = [
            "table_name",
            "column1",
            "_private",
            "TableName",
            "UPPERCASE",
            "a1b2c3",
            "_",
            "a",
        ]
        for name in valid_names:
            # Should not raise
            delta_service._validate_identifier(name)

    def test_invalid_identifiers(self):
        """Test that invalid identifiers fail validation."""
        invalid_names = [
            "",  # Empty
            "123start",  # Starts with number
            "has-dash",  # Contains dash
            "has space",  # Contains space
            "has.dot",  # Contains dot
            "has;semicolon",  # Contains semicolon
            "has'quote",  # Contains quote
            'has"double',  # Contains double quote
            "has`backtick",  # Contains backtick
        ]
        for name in invalid_names:
            with pytest.raises(SparkQueryError, match="Invalid"):
                delta_service._validate_identifier(name)


class TestEscapeValue:
    """Tests for _escape_value function."""

    def test_escape_none(self):
        """Test escaping None values."""
        assert delta_service._escape_value(None) == "NULL"

    def test_escape_bool(self):
        """Test escaping boolean values."""
        assert delta_service._escape_value(True) == "TRUE"
        assert delta_service._escape_value(False) == "FALSE"

    def test_escape_numbers(self):
        """Test escaping numeric values."""
        assert delta_service._escape_value(42) == "42"
        assert delta_service._escape_value(3.14) == "3.14"
        assert delta_service._escape_value(-100) == "-100"
        assert delta_service._escape_value(0) == "0"

    def test_escape_strings(self):
        """Test escaping string values."""
        assert delta_service._escape_value("hello") == "'hello'"
        assert delta_service._escape_value("O'Brien") == "'O''Brien'"
        assert delta_service._escape_value("test''double") == "'test''''double'"
        assert delta_service._escape_value("") == "''"


class TestBuildColumnExpression:
    """Tests for _build_column_expression function."""

    def test_simple_column(self):
        """Test building a simple column expression."""
        col = ColumnSpec(column="name")
        assert delta_service._build_column_expression(col) == "`name`"

    def test_column_with_alias(self):
        """Test building a column expression with alias."""
        col = ColumnSpec(column="name", alias="user_name")
        assert delta_service._build_column_expression(col) == "`name` AS `user_name`"

    def test_column_with_table_alias(self):
        """Test building a column expression with table alias."""
        col = ColumnSpec(column="name", table_alias="u")
        assert delta_service._build_column_expression(col) == "`u`.`name`"

    def test_column_with_both_aliases(self):
        """Test building a column expression with both table and column alias."""
        col = ColumnSpec(column="name", table_alias="u", alias="user_name")
        assert (
            delta_service._build_column_expression(col) == "`u`.`name` AS `user_name`"
        )


class TestBuildAggregationExpression:
    """Tests for _build_aggregation_expression function."""

    def test_count_star(self):
        """Test COUNT(*) aggregation."""
        agg = AggregationSpec(function="COUNT", column="*")
        assert delta_service._build_aggregation_expression(agg) == "COUNT(*)"

    def test_count_star_with_alias(self):
        """Test COUNT(*) with alias."""
        agg = AggregationSpec(function="COUNT", column="*", alias="total")
        assert delta_service._build_aggregation_expression(agg) == "COUNT(*) AS `total`"

    def test_sum_column(self):
        """Test SUM aggregation."""
        agg = AggregationSpec(function="SUM", column="amount")
        assert delta_service._build_aggregation_expression(agg) == "SUM(`amount`)"

    def test_avg_with_alias(self):
        """Test AVG with alias."""
        agg = AggregationSpec(function="AVG", column="price", alias="avg_price")
        assert (
            delta_service._build_aggregation_expression(agg)
            == "AVG(`price`) AS `avg_price`"
        )

    def test_min_max(self):
        """Test MIN and MAX aggregations."""
        min_agg = AggregationSpec(function="MIN", column="created_at")
        max_agg = AggregationSpec(function="MAX", column="updated_at")
        assert (
            delta_service._build_aggregation_expression(min_agg) == "MIN(`created_at`)"
        )
        assert (
            delta_service._build_aggregation_expression(max_agg) == "MAX(`updated_at`)"
        )

    def test_invalid_star_with_non_count(self):
        """Test that * is only valid for COUNT."""
        agg = AggregationSpec(function="SUM", column="*")
        with pytest.raises(SparkQueryError, match="does not support"):
            delta_service._build_aggregation_expression(agg)


class TestBuildFilterCondition:
    """Tests for _build_filter_condition function."""

    def test_equality(self):
        """Test equality filter."""
        condition = FilterCondition(column="status", operator="=", value="active")
        assert delta_service._build_filter_condition(condition) == "`status` = 'active'"

    def test_not_equal(self):
        """Test not equal filter."""
        condition = FilterCondition(column="type", operator="!=", value="deleted")
        assert delta_service._build_filter_condition(condition) == "`type` != 'deleted'"

    def test_comparison_operators(self):
        """Test numeric comparison operators."""
        test_cases = [
            (FilterCondition(column="age", operator=">", value=18), "`age` > 18"),
            (FilterCondition(column="age", operator=">=", value=21), "`age` >= 21"),
            (FilterCondition(column="price", operator="<", value=100), "`price` < 100"),
            (FilterCondition(column="qty", operator="<=", value=10), "`qty` <= 10"),
        ]
        for condition, expected in test_cases:
            assert delta_service._build_filter_condition(condition) == expected

    def test_in_operator(self):
        """Test IN operator."""
        condition = FilterCondition(
            column="status", operator="IN", values=["active", "pending"]
        )
        assert (
            delta_service._build_filter_condition(condition)
            == "`status` IN ('active', 'pending')"
        )

    def test_not_in_operator(self):
        """Test NOT IN operator."""
        condition = FilterCondition(column="id", operator="NOT IN", values=[1, 2, 3])
        assert (
            delta_service._build_filter_condition(condition) == "`id` NOT IN (1, 2, 3)"
        )

    def test_like_operator(self):
        """Test LIKE operator."""
        condition = FilterCondition(column="name", operator="LIKE", value="%john%")
        assert (
            delta_service._build_filter_condition(condition) == "`name` LIKE '%john%'"
        )

    def test_not_like_operator(self):
        """Test NOT LIKE operator."""
        condition = FilterCondition(column="email", operator="NOT LIKE", value="%spam%")
        assert (
            delta_service._build_filter_condition(condition)
            == "`email` NOT LIKE '%spam%'"
        )

    def test_is_null(self):
        """Test IS NULL operator."""
        condition = FilterCondition(column="deleted_at", operator="IS NULL")
        assert (
            delta_service._build_filter_condition(condition) == "`deleted_at` IS NULL"
        )

    def test_is_not_null(self):
        """Test IS NOT NULL operator."""
        condition = FilterCondition(column="email", operator="IS NOT NULL")
        assert delta_service._build_filter_condition(condition) == "`email` IS NOT NULL"

    def test_between_operator(self):
        """Test BETWEEN operator."""
        condition = FilterCondition(
            column="created_at", operator="BETWEEN", values=["2023-01-01", "2023-12-31"]
        )
        assert (
            delta_service._build_filter_condition(condition)
            == "`created_at` BETWEEN '2023-01-01' AND '2023-12-31'"
        )

    def test_in_without_values_raises_error(self):
        """Test that IN operator without values raises error."""
        condition = FilterCondition(column="status", operator="IN")
        with pytest.raises(SparkQueryError, match="requires 'values'"):
            delta_service._build_filter_condition(condition)

    def test_between_with_wrong_values_count(self):
        """Test that BETWEEN with wrong number of values raises error."""
        condition = FilterCondition(column="age", operator="BETWEEN", values=[18])
        with pytest.raises(SparkQueryError, match="requires exactly 2 values"):
            delta_service._build_filter_condition(condition)

    def test_equality_without_value_raises_error(self):
        """Test that equality without value raises error."""
        condition = FilterCondition(column="status", operator="=")
        with pytest.raises(SparkQueryError, match="requires 'value'"):
            delta_service._build_filter_condition(condition)


class TestBuildSelectQuery:
    """Tests for build_select_query function."""

    def test_simple_select_all(self):
        """Test SELECT * query."""
        request = TableSelectRequest(database="mydb", table="users")
        query = delta_service.build_select_query(request)
        assert query == "SELECT * FROM `mydb`.`users` LIMIT 100 OFFSET 0"

    def test_select_specific_columns(self):
        """Test SELECT with specific columns."""
        request = TableSelectRequest(
            database="mydb",
            table="users",
            columns=[
                ColumnSpec(column="id"),
                ColumnSpec(column="name"),
            ],
        )
        query = delta_service.build_select_query(request)
        assert "SELECT `id`, `name` FROM" in query

    def test_select_with_distinct(self):
        """Test SELECT DISTINCT."""
        request = TableSelectRequest(
            database="mydb",
            table="users",
            columns=[ColumnSpec(column="status")],
            distinct=True,
        )
        query = delta_service.build_select_query(request)
        assert "SELECT DISTINCT `status`" in query

    def test_select_with_aggregations(self):
        """Test SELECT with aggregations."""
        request = TableSelectRequest(
            database="mydb",
            table="orders",
            aggregations=[
                AggregationSpec(function="COUNT", column="*", alias="total"),
                AggregationSpec(function="SUM", column="amount", alias="total_amount"),
            ],
        )
        query = delta_service.build_select_query(request)
        assert "COUNT(*) AS `total`" in query
        assert "SUM(`amount`) AS `total_amount`" in query

    def test_select_with_where(self):
        """Test SELECT with WHERE clause."""
        request = TableSelectRequest(
            database="mydb",
            table="users",
            filters=[
                FilterCondition(column="status", operator="=", value="active"),
                FilterCondition(column="age", operator=">=", value=18),
            ],
        )
        query = delta_service.build_select_query(request)
        assert "WHERE `status` = 'active' AND `age` >= 18" in query

    def test_select_with_group_by(self):
        """Test SELECT with GROUP BY."""
        request = TableSelectRequest(
            database="mydb",
            table="orders",
            columns=[ColumnSpec(column="category")],
            aggregations=[AggregationSpec(function="COUNT", column="*", alias="count")],
            group_by=["category"],
        )
        query = delta_service.build_select_query(request)
        assert "GROUP BY `category`" in query

    def test_select_with_having(self):
        """Test SELECT with HAVING clause."""
        request = TableSelectRequest(
            database="mydb",
            table="orders",
            columns=[ColumnSpec(column="category")],
            aggregations=[AggregationSpec(function="COUNT", column="*", alias="count")],
            group_by=["category"],
            having=[FilterCondition(column="count", operator=">", value=10)],
        )
        query = delta_service.build_select_query(request)
        assert "HAVING `count` > 10" in query

    def test_select_with_order_by(self):
        """Test SELECT with ORDER BY."""
        request = TableSelectRequest(
            database="mydb",
            table="users",
            order_by=[
                OrderBySpec(column="created_at", direction="DESC"),
                OrderBySpec(column="name", direction="ASC"),
            ],
        )
        query = delta_service.build_select_query(request)
        assert "ORDER BY `created_at` DESC, `name` ASC" in query

    def test_select_with_pagination(self):
        """Test SELECT with custom pagination."""
        request = TableSelectRequest(
            database="mydb", table="users", limit=50, offset=100
        )
        query = delta_service.build_select_query(request)
        assert "LIMIT 50 OFFSET 100" in query

    def test_select_without_pagination(self):
        """Test SELECT without pagination clause."""
        request = TableSelectRequest(database="mydb", table="users")
        query = delta_service.build_select_query(request, include_pagination=False)
        assert "LIMIT" not in query
        assert "OFFSET" not in query

    def test_complex_query(self):
        """Test a complex query with multiple features."""
        request = TableSelectRequest(
            database="sales",
            table="orders",
            columns=[
                ColumnSpec(column="customer_id"),
                ColumnSpec(column="category"),
            ],
            aggregations=[
                AggregationSpec(function="SUM", column="amount", alias="total"),
                AggregationSpec(function="COUNT", column="*", alias="order_count"),
            ],
            distinct=False,
            filters=[
                FilterCondition(column="status", operator="=", value="completed"),
                FilterCondition(column="created_at", operator=">=", value="2023-01-01"),
            ],
            group_by=["customer_id", "category"],
            having=[FilterCondition(column="total", operator=">", value=1000)],
            order_by=[OrderBySpec(column="total", direction="DESC")],
            limit=20,
            offset=0,
        )
        query = delta_service.build_select_query(request)

        # Verify all parts are present
        assert "SELECT `customer_id`, `category`" in query
        assert "SUM(`amount`) AS `total`" in query
        assert "COUNT(*) AS `order_count`" in query
        assert "FROM `sales`.`orders`" in query
        assert "WHERE `status` = 'completed'" in query
        assert "`created_at` >= '2023-01-01'" in query
        assert "GROUP BY `customer_id`, `category`" in query
        assert "HAVING `total` > 1000" in query
        assert "ORDER BY `total` DESC" in query
        assert "LIMIT 20 OFFSET 0" in query


# =============================================================================
# Tests for Limit Enforcement
# =============================================================================


class TestQueryLimitEnforcement:
    """Tests for query limit enforcement functions."""

    def test_extract_limit_from_query_with_limit(self):
        """Test extracting LIMIT from query that has one."""
        query = "SELECT * FROM users LIMIT 50"
        limit = delta_service._extract_limit_from_query(query)
        assert limit == 50

    def test_extract_limit_from_query_without_limit(self):
        """Test extracting LIMIT from query that doesn't have one."""
        query = "SELECT * FROM users"
        limit = delta_service._extract_limit_from_query(query)
        assert limit is None

    def test_extract_limit_case_insensitive(self):
        """Test that LIMIT extraction is case insensitive."""
        query = "SELECT * FROM users limit 100"
        limit = delta_service._extract_limit_from_query(query)
        assert limit == 100

    def test_enforce_query_limit_adds_limit(self):
        """Test that _enforce_query_limit adds LIMIT when missing."""
        query = "SELECT * FROM users"
        result = delta_service._enforce_query_limit(query, max_rows=1000)
        assert "LIMIT 1000" in result

    def test_enforce_query_limit_keeps_acceptable_limit(self):
        """Test that acceptable LIMIT is kept."""
        query = "SELECT * FROM users LIMIT 500"
        result = delta_service._enforce_query_limit(query, max_rows=1000)
        assert result == query

    def test_enforce_query_limit_rejects_excessive_limit(self):
        """Test that excessive LIMIT raises error."""
        query = "SELECT * FROM users LIMIT 100000"
        with pytest.raises(SparkQueryError, match="exceeds maximum"):
            delta_service._enforce_query_limit(query, max_rows=50000)


# =============================================================================
# Tests for count_delta_table with Mocked Spark
# =============================================================================


class TestCountDeltaTable:
    """Tests for count_delta_table function with mocked Spark."""

    def test_count_returns_correct_value(self, mock_spark_session):
        """Test that count_delta_table returns correct count."""
        spark = mock_spark_session()
        spark.table.return_value.count.return_value = 12345

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service.run_with_timeout",
                side_effect=lambda func, **kwargs: func(),
            ):
                with patch(
                    "src.delta_lake.delta_service._get_from_cache", return_value=None
                ):
                    with patch("src.delta_lake.delta_service._store_in_cache"):
                        count = delta_service.count_delta_table(
                            spark, "testdb", "testtable", use_cache=False
                        )

        assert count == 12345

    def test_count_uses_cache_hit(self, mock_spark_session):
        """Test that count_delta_table uses cached value."""
        spark = mock_spark_session()

        with patch(
            "src.delta_lake.delta_service._get_from_cache",
            return_value=[{"count": 9999}],
        ):
            count = delta_service.count_delta_table(
                spark, "testdb", "testtable", use_cache=True
            )

        assert count == 9999
        # Spark should not be called
        spark.table.assert_not_called()

    def test_count_database_not_found(self, mock_spark_session):
        """Test that missing database raises error."""
        spark = mock_spark_session()

        with patch("src.delta_lake.delta_service.database_exists", return_value=False):
            with pytest.raises(DeltaDatabaseNotFoundError):
                delta_service.count_delta_table(spark, "nonexistent", "table")

    def test_count_table_not_found(self, mock_spark_session):
        """Test that missing table raises error."""
        spark = mock_spark_session()

        with patch("src.delta_lake.delta_service.database_exists", return_value=True):
            with patch("src.delta_lake.delta_service.table_exists", return_value=False):
                with pytest.raises(DeltaTableNotFoundError):
                    delta_service.count_delta_table(spark, "testdb", "nonexistent")

    def test_count_timeout_raises_error(self, mock_spark_session):
        """Test that timeout raises SparkTimeoutError."""
        spark = mock_spark_session()

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout",
                    side_effect=SparkTimeoutError(operation="count", timeout=30),
                ):
                    with pytest.raises(SparkTimeoutError):
                        delta_service.count_delta_table(
                            spark, "testdb", "table", use_cache=False
                        )

    def test_count_spark_error_wrapped(self, mock_spark_session):
        """Test that Spark errors are wrapped in SparkOperationError."""
        spark = mock_spark_session()
        spark.table.return_value.count.side_effect = Exception("Spark failed")

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout",
                    side_effect=lambda func, **kwargs: func(),
                ):
                    with pytest.raises(SparkOperationError, match="Spark failed"):
                        delta_service.count_delta_table(
                            spark, "testdb", "table", use_cache=False
                        )

    def test_count_with_username_uses_user_scoped_cache(self, mock_spark_session):
        """Test that count with username uses user-scoped cache key."""
        spark = mock_spark_session()
        spark.table.return_value.count.return_value = 100

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ) as mock_cache_get:
                with patch(
                    "src.delta_lake.delta_service._store_in_cache"
                ) as mock_cache_set:
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout",
                        side_effect=lambda func, **kwargs: func(),
                    ):
                        delta_service.count_delta_table(
                            spark, "testdb", "testtable", username="alice"
                        )

            # Verify cache was checked/stored with user-scoped key
            assert mock_cache_get.called
            assert mock_cache_set.called

            # Get the cache key used - it should include the username
            cache_key_alice = mock_cache_get.call_args[0][1]

        # Now test with a different user
        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ) as mock_cache_get2:
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout",
                        side_effect=lambda func, **kwargs: func(),
                    ):
                        delta_service.count_delta_table(
                            spark, "testdb", "testtable", username="bob"
                        )

            cache_key_bob = mock_cache_get2.call_args[0][1]

        # Different users should have different cache keys
        assert cache_key_alice != cache_key_bob


# =============================================================================
# Tests for sample_delta_table with Mocked Spark
# =============================================================================


class TestSampleDeltaTable:
    """Tests for sample_delta_table function with mocked Spark."""

    def test_sample_returns_rows(self, mock_spark_session, mock_spark_row):
        """Test that sample_delta_table returns sample rows."""
        test_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        spark = mock_spark_session(table_results=test_data)

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout",
                        side_effect=lambda func, **kwargs: func(),
                    ):
                        result = delta_service.sample_delta_table(
                            spark, "testdb", "testtable", limit=10, use_cache=False
                        )

        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_sample_with_columns(self, mock_spark_session):
        """Test sampling with specific columns."""
        spark = mock_spark_session()
        spark.table.return_value.select.return_value = spark.table.return_value

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout",
                        side_effect=lambda func, **kwargs: func(),
                    ):
                        delta_service.sample_delta_table(
                            spark,
                            "testdb",
                            "testtable",
                            columns=["id", "name"],
                            use_cache=False,
                        )

        spark.table.return_value.select.assert_called_with(["id", "name"])

    def test_sample_with_where_clause(self, mock_spark_session):
        """Test sampling with WHERE clause."""
        spark = mock_spark_session()

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout",
                        side_effect=lambda func, **kwargs: func(),
                    ):
                        delta_service.sample_delta_table(
                            spark,
                            "testdb",
                            "testtable",
                            where_clause="id > 100",
                            use_cache=False,
                        )

        spark.table.return_value.filter.assert_called_with("id > 100")

    def test_sample_invalid_limit_raises_error(self, mock_spark_session):
        """Test that invalid limit raises ValueError."""
        spark = mock_spark_session()

        with pytest.raises(ValueError, match="Limit must be between"):
            delta_service.sample_delta_table(spark, "db", "table", limit=0)

        with pytest.raises(ValueError, match="Limit must be between"):
            delta_service.sample_delta_table(spark, "db", "table", limit=1000)

    def test_sample_uses_cache(self, mock_spark_session):
        """Test that sample uses cached results."""
        spark = mock_spark_session()
        cached_data = [{"id": 1, "cached": True}]

        with patch(
            "src.delta_lake.delta_service._get_from_cache", return_value=cached_data
        ):
            result = delta_service.sample_delta_table(
                spark, "testdb", "testtable", use_cache=True
            )

        assert result == cached_data
        spark.table.assert_not_called()


# =============================================================================
# Tests for query_delta_table with Mocked Spark
# =============================================================================


class TestQueryDeltaTable:
    """Tests for query_delta_table function with mocked Spark."""

    def test_query_returns_results_skips_count(self, mock_spark_session):
        """Test that query skips COUNT when results < limit."""
        test_data = [{"count": 100}]
        spark = mock_spark_session(sql_results=test_data)

        with patch("src.delta_lake.delta_service._get_from_cache", return_value=None):
            with patch("src.delta_lake.delta_service._store_in_cache"):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout"
                ) as mock_timeout:
                    # Only data query runs - 1 result < 1000 limit, so COUNT is skipped
                    mock_timeout.side_effect = [
                        test_data,  # Data query result (only call)
                    ]
                    result = delta_service.query_delta_table(
                        spark, "SELECT COUNT(*) as count FROM users", use_cache=False
                    )

        # COUNT was skipped, so total_count = offset (0) + len(results) (1)
        assert result.result[0]["count"] == 100
        assert result.pagination.total_count == 1
        assert result.pagination.limit == 1000  # default
        assert result.pagination.offset == 0  # default
        assert result.pagination.has_more is False
        # Verify only 1 call was made (no COUNT query)
        assert mock_timeout.call_count == 1

    def test_query_invalid_sql_rejected(self, mock_spark_session):
        """Test that invalid SQL is rejected."""
        spark = mock_spark_session()

        with pytest.raises(SparkQueryError):
            delta_service.query_delta_table(spark, "DROP TABLE users")

    def test_query_with_pagination_runs_count(self, mock_spark_session):
        """Test query runs COUNT when page is full."""
        spark = mock_spark_session()

        # Return exactly 10 results (== limit), so COUNT query will run
        data_results = [{"id": i} for i in range(10)]

        count_row = MagicMock()
        count_row.__getitem__ = lambda self, key: 500 if key == "cnt" else None

        with patch("src.delta_lake.delta_service._get_from_cache", return_value=None):
            with patch("src.delta_lake.delta_service._store_in_cache"):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout"
                ) as mock_timeout:
                    # Data query first, then count query (because results == limit)
                    mock_timeout.side_effect = [
                        data_results,  # Data query returns 10 rows
                        [count_row],  # Count query runs because page is full
                    ]
                    result = delta_service.query_delta_table(
                        spark,
                        "SELECT * FROM users",
                        limit=10,
                        offset=20,
                        use_cache=False,
                    )

        assert result.pagination.limit == 10
        assert result.pagination.offset == 20
        assert result.pagination.total_count == 500
        assert result.pagination.has_more is True
        # Verify 2 calls were made (data + count)
        assert mock_timeout.call_count == 2

    def test_query_uses_cache(self, mock_spark_session):
        """Test that query uses cached results."""
        spark = mock_spark_session()
        cached_data = [
            {
                "result": [{"id": 1}],
                "pagination": {
                    "limit": 1000,
                    "offset": 0,
                    "total_count": 1,
                    "has_more": False,
                },
            }
        ]

        with patch(
            "src.delta_lake.delta_service._get_from_cache", return_value=cached_data
        ):
            result = delta_service.query_delta_table(
                spark, "SELECT * FROM users", use_cache=True
            )

        assert result.result == [{"id": 1}]
        assert result.pagination.total_count == 1
        spark.sql.assert_not_called()

    def test_query_timeout_raises_error(self, mock_spark_session):
        """Test that query timeout raises SparkTimeoutError."""
        spark = mock_spark_session()

        with patch("src.delta_lake.delta_service._get_from_cache", return_value=None):
            with patch(
                "src.delta_lake.delta_service.run_with_timeout",
                side_effect=SparkTimeoutError(operation="query", timeout=30),
            ):
                with pytest.raises(SparkTimeoutError):
                    delta_service.query_delta_table(
                        spark, "SELECT * FROM users", use_cache=False
                    )

    def test_query_strips_existing_limit(self, mock_spark_session):
        """Test that existing LIMIT clause is stripped and replaced by pagination params."""
        spark = mock_spark_session()
        test_data = [{"id": 1}]

        with patch("src.delta_lake.delta_service._get_from_cache", return_value=None):
            with patch("src.delta_lake.delta_service._store_in_cache"):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout"
                ) as mock_timeout:
                    mock_timeout.side_effect = [test_data]
                    result = delta_service.query_delta_table(
                        spark,
                        "SELECT * FROM users LIMIT 100",  # User has LIMIT 100
                        limit=10,  # But pagination param is 10
                        use_cache=False,
                    )

        # Pagination param (10) should override user's LIMIT (100)
        assert result.pagination.limit == 10
        # Verify the SQL call used the pagination limit, not the user's limit
        sql_call = spark.sql.call_args[0][0]
        assert "LIMIT 10" in sql_call

    def test_query_strips_existing_limit_offset(self, mock_spark_session):
        """Test that existing LIMIT and OFFSET clauses are stripped."""
        spark = mock_spark_session()
        test_data = [{"id": 1}]

        with patch("src.delta_lake.delta_service._get_from_cache", return_value=None):
            with patch("src.delta_lake.delta_service._store_in_cache"):
                with patch(
                    "src.delta_lake.delta_service.run_with_timeout"
                ) as mock_timeout:
                    mock_timeout.side_effect = [test_data]
                    result = delta_service.query_delta_table(
                        spark,
                        "SELECT * FROM users LIMIT 100 OFFSET 50",  # User has both
                        limit=10,
                        offset=5,  # Pagination params should override
                        use_cache=False,
                    )

        # Pagination params should override user's LIMIT/OFFSET
        assert result.pagination.limit == 10
        assert result.pagination.offset == 5
        # Verify the SQL call used pagination params
        sql_call = spark.sql.call_args[0][0]
        assert "LIMIT 10 OFFSET 5" in sql_call
        assert "LIMIT 100" not in sql_call
        assert "OFFSET 50" not in sql_call


# =============================================================================
# Tests for select_from_delta_table with Mocked Spark
# =============================================================================


class TestSelectFromDeltaTable:
    """Tests for select_from_delta_table function with mocked Spark."""

    def test_select_returns_response(self, mock_spark_session):
        """Test that select_from_delta_table returns TableSelectResponse."""
        test_data = [{"id": 1}, {"id": 2}]
        spark = mock_spark_session(sql_results=test_data)

        # Mock the count query result
        count_row = MagicMock()
        count_row.__getitem__ = lambda self, key: 100 if key == "cnt" else None

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout"
                    ) as mock_timeout:
                        # First call is count, second is data
                        mock_timeout.side_effect = [
                            [count_row],  # Count query result
                            test_data,  # Data query result (already converted)
                        ]

                        request = TableSelectRequest(database="testdb", table="users")
                        result = delta_service.select_from_delta_table(
                            spark, request, use_cache=False
                        )

        assert isinstance(result, TableSelectResponse)
        assert result.pagination.total_count == 100

    def test_select_with_pagination(self, mock_spark_session):
        """Test select with pagination parameters."""
        spark = mock_spark_session()
        count_row = MagicMock()
        count_row.__getitem__ = lambda self, key: 500 if key == "cnt" else None

        with patch("src.delta_lake.delta_service._check_exists", return_value=True):
            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout"
                    ) as mock_timeout:
                        mock_timeout.side_effect = [
                            [count_row],
                            [],
                        ]

                        request = TableSelectRequest(
                            database="testdb", table="users", limit=50, offset=100
                        )
                        result = delta_service.select_from_delta_table(
                            spark, request, use_cache=False
                        )

        assert result.pagination.limit == 50
        assert result.pagination.offset == 100
        assert result.pagination.total_count == 500
        assert result.pagination.has_more is True

    def test_select_uses_cache(self, mock_spark_session):
        """Test that select uses cached results."""
        spark = mock_spark_session()
        cached_data = [
            {
                "data": [{"id": 1}],
                "pagination": {
                    "limit": 100,
                    "offset": 0,
                    "total_count": 1,
                    "has_more": False,
                },
            }
        ]

        with patch(
            "src.delta_lake.delta_service._get_from_cache", return_value=cached_data
        ):
            request = TableSelectRequest(database="testdb", table="users")
            result = delta_service.select_from_delta_table(
                spark, request, use_cache=True
            )

        assert isinstance(result, TableSelectResponse)
        assert result.data == [{"id": 1}]
        spark.sql.assert_not_called()


# =============================================================================
# Tests for Cache Key Generation
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_cache_key_deterministic(self):
        """Test that cache key generation is deterministic."""
        params = {"database": "testdb", "table": "users"}

        key1 = delta_service._generate_cache_key(params)
        key2 = delta_service._generate_cache_key(params)

        assert key1 == key2

    def test_generate_cache_key_different_for_different_params(self):
        """Test that different params produce different keys."""
        params1 = {"database": "db1", "table": "users"}
        params2 = {"database": "db2", "table": "users"}

        key1 = delta_service._generate_cache_key(params1)
        key2 = delta_service._generate_cache_key(params2)

        assert key1 != key2

    def test_generate_cache_key_order_independent(self):
        """Test that param order doesn't affect key."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}

        key1 = delta_service._generate_cache_key(params1)
        key2 = delta_service._generate_cache_key(params2)

        assert key1 == key2

    def test_generate_cache_key_with_username_isolation(self):
        """Test that different users get different cache keys for same query.

        This ensures cache isolation between users to prevent authorization bypass
        where one user's cached results could be returned to another user who may
        not have permission to access the underlying data.
        """
        params = {"database": "testdb", "table": "users"}

        key_user1 = delta_service._generate_cache_key(params, username="alice")
        key_user2 = delta_service._generate_cache_key(params, username="bob")

        assert key_user1 != key_user2

    def test_generate_cache_key_same_user_same_key(self):
        """Test that same user gets same cache key for same query."""
        params = {"database": "testdb", "table": "users"}

        key1 = delta_service._generate_cache_key(params, username="alice")
        key2 = delta_service._generate_cache_key(params, username="alice")

        assert key1 == key2

    def test_generate_cache_key_without_username_backward_compatible(self):
        """Test that cache key without username is backward compatible.

        When username is None, the key should be the same as before the
        user isolation feature was added.
        """
        params = {"database": "testdb", "table": "users"}

        key_no_user = delta_service._generate_cache_key(params, username=None)
        key_empty_user = delta_service._generate_cache_key(params)

        # Both should produce the same key (backward compatible)
        assert key_no_user == key_empty_user

    def test_generate_cache_key_with_username_differs_from_no_username(self):
        """Test that providing a username changes the cache key.

        Cache entries with a username should not collide with entries
        without a username (anonymous/legacy entries).
        """
        params = {"database": "testdb", "table": "users"}

        key_with_user = delta_service._generate_cache_key(params, username="alice")
        key_without_user = delta_service._generate_cache_key(params, username=None)

        assert key_with_user != key_without_user


# =============================================================================
# Concurrent Query Tests
# =============================================================================


class TestConcurrentQueries:
    """Tests for concurrent query execution."""

    def test_concurrent_count_queries(self, mock_spark_session, concurrent_executor):
        """Test concurrent count queries."""

        def count_query(i):
            spark = mock_spark_session()
            spark.table.return_value.count.return_value = i * 100

            with patch("src.delta_lake.delta_service._check_exists", return_value=True):
                with patch(
                    "src.delta_lake.delta_service._get_from_cache", return_value=None
                ):
                    with patch("src.delta_lake.delta_service._store_in_cache"):
                        with patch(
                            "src.delta_lake.delta_service.run_with_timeout",
                            side_effect=lambda func, **kwargs: func(),
                        ):
                            return delta_service.count_delta_table(
                                spark, f"db_{i}", "table", use_cache=False
                            )

        args_list = [(i,) for i in range(5)]
        results, exceptions = concurrent_executor(count_query, args_list)

        assert len(exceptions) == 0
        assert sorted(results) == [0, 100, 200, 300, 400]

    def test_concurrent_queries_with_timeout(
        self, mock_spark_session, concurrent_executor
    ):
        """Test that concurrent queries respect timeouts."""

        def query_with_timeout(i):
            spark = mock_spark_session(sql_results=[{"id": i}])

            with patch(
                "src.delta_lake.delta_service._get_from_cache", return_value=None
            ):
                with patch("src.delta_lake.delta_service._store_in_cache"):
                    with patch(
                        "src.delta_lake.delta_service.run_with_timeout"
                    ) as mock_timeout:
                        # Only data query runs since 1 result < 1000 limit
                        mock_timeout.side_effect = [
                            [{"id": i}],  # Data query only
                        ]
                        result = delta_service.query_delta_table(
                            spark,
                            f"SELECT {i} as id FROM dual",
                            use_cache=False,
                        )
                        return result.result[0]["id"]

        args_list = [(i,) for i in range(5)]
        results, exceptions = concurrent_executor(query_with_timeout, args_list)

        assert len(exceptions) == 0
        assert sorted(results) == [0, 1, 2, 3, 4]


# =============================================================================
# Constants Tests
# =============================================================================


class TestServiceConstants:
    """Tests for service constants."""

    def test_max_sample_rows(self):
        """Test MAX_SAMPLE_ROWS constant."""
        assert delta_service.MAX_SAMPLE_ROWS == 100

    def test_max_query_rows(self):
        """Test MAX_QUERY_ROWS constant."""
        assert delta_service.MAX_QUERY_ROWS == 1000

    def test_max_select_rows(self):
        """Test MAX_SELECT_ROWS constant."""
        assert delta_service.MAX_SELECT_ROWS == 1000

    def test_cache_expiry_seconds(self):
        """Test CACHE_EXPIRY_SECONDS constant."""
        assert delta_service.CACHE_EXPIRY_SECONDS == 3600

    def test_forbidden_keywords_set(self):
        """Test FORBIDDEN_KEYWORDS contains expected values."""
        assert "drop" in delta_service.FORBIDDEN_KEYWORDS
        assert "delete" in delta_service.FORBIDDEN_KEYWORDS
        assert "insert" in delta_service.FORBIDDEN_KEYWORDS
        assert "update" in delta_service.FORBIDDEN_KEYWORDS
