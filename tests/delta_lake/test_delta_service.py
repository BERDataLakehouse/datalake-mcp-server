"""Tests for the delta_service module."""

import pytest

from src.delta_lake import delta_service
from src.service.exceptions import SparkQueryError
from src.service.models import (
    AggregationSpec,
    ColumnSpec,
    FilterCondition,
    OrderBySpec,
    TableSelectRequest,
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
