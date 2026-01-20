"""Tests for the exception handlers module."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from src.service.exception_handlers import _format_error, universal_error_handler
from src.service.exceptions import (
    DeltaDatabaseNotFoundError,
    DeltaLakeError,
    DeltaSchemaError,
    DeltaTableNotFoundError,
    DeltaTableOperationError,
    InvalidAuthHeaderError,
    InvalidS3PathError,
    InvalidTokenError,
    MCPServerError,
    MissingRoleError,
    MissingTokenError,
    S3AccessError,
    SparkOperationError,
    SparkQueryError,
    SparkSessionError,
    SparkTimeoutError,
)


class TestFormatError:
    """Tests for _format_error function."""

    def test_format_error_with_all_fields(self):
        """Test formatting error with all fields provided."""
        response = _format_error(
            status_code=400,
            error_code=40001,
            error_type_str="invalid_request",
            message="Invalid request format",
        )

        assert response.status_code == 400
        content = response.body.decode()
        assert "40001" in content
        assert "invalid_request" in content
        assert "Invalid request format" in content

    def test_format_error_with_none_error_code(self):
        """Test formatting error with None error code."""
        response = _format_error(
            status_code=500,
            error_code=None,
            error_type_str="internal_error",
            message="Something went wrong",
        )

        assert response.status_code == 500
        content = response.body.decode()
        assert "internal_error" in content
        assert "Something went wrong" in content

    def test_format_error_message_fallback_to_error_type(self):
        """Test that message falls back to error_type when message is None."""
        response = _format_error(
            status_code=400,
            error_code=40001,
            error_type_str="validation_error",
            message=None,
        )

        assert response.status_code == 400
        content = response.body.decode()
        assert "validation_error" in content

    def test_format_error_message_fallback_to_unknown(self):
        """Test that message falls back to 'Unknown error' when both are None."""
        response = _format_error(
            status_code=500,
            error_code=None,
            error_type_str=None,
            message=None,
        )

        assert response.status_code == 500
        content = response.body.decode()
        assert "Unknown error" in content


class TestUniversalErrorHandler:
    """Tests for universal_error_handler function."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_handles_missing_token_error(self, mock_request):
        """Test handling MissingTokenError."""
        exc = MissingTokenError("Authorization header required")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 401
        content = response.body.decode()
        assert "Authorization header required" in content

    @pytest.mark.asyncio
    async def test_handles_invalid_token_error(self, mock_request):
        """Test handling InvalidTokenError."""
        exc = InvalidTokenError("Token is invalid or expired")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 401
        content = response.body.decode()
        assert "Token is invalid or expired" in content

    @pytest.mark.asyncio
    async def test_handles_invalid_auth_header_error(self, mock_request):
        """Test handling InvalidAuthHeaderError."""
        exc = InvalidAuthHeaderError("Invalid authorization header format")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 401
        content = response.body.decode()
        assert "Invalid authorization header format" in content

    @pytest.mark.asyncio
    async def test_handles_missing_role_error(self, mock_request):
        """Test handling MissingRoleError."""
        exc = MissingRoleError("User lacks required role")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 403
        content = response.body.decode()
        assert "User lacks required role" in content

    @pytest.mark.asyncio
    async def test_handles_delta_table_not_found_error(self, mock_request):
        """Test handling DeltaTableNotFoundError."""
        exc = DeltaTableNotFoundError("Table 'my_table' not found")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 404
        content = response.body.decode()
        assert "my_table" in content

    @pytest.mark.asyncio
    async def test_handles_delta_database_not_found_error(self, mock_request):
        """Test handling DeltaDatabaseNotFoundError."""
        exc = DeltaDatabaseNotFoundError("Database 'my_db' not found")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 404
        content = response.body.decode()
        assert "my_db" in content

    @pytest.mark.asyncio
    async def test_handles_invalid_s3_path_error(self, mock_request):
        """Test handling InvalidS3PathError."""
        exc = InvalidS3PathError("Invalid S3 path format")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Invalid S3 path" in content

    @pytest.mark.asyncio
    async def test_handles_delta_schema_error(self, mock_request):
        """Test handling DeltaSchemaError."""
        exc = DeltaSchemaError("Schema mismatch")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Schema mismatch" in content

    @pytest.mark.asyncio
    async def test_handles_s3_access_error(self, mock_request):
        """Test handling S3AccessError."""
        exc = S3AccessError("Access denied to S3 bucket")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Access denied" in content

    @pytest.mark.asyncio
    async def test_handles_delta_table_operation_error(self, mock_request):
        """Test handling DeltaTableOperationError."""
        exc = DeltaTableOperationError("Failed to write to table")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Failed to write" in content

    @pytest.mark.asyncio
    async def test_handles_spark_session_error(self, mock_request):
        """Test handling SparkSessionError."""
        exc = SparkSessionError("Failed to create Spark session")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 503
        content = response.body.decode()
        assert "Spark session" in content

    @pytest.mark.asyncio
    async def test_handles_spark_operation_error(self, mock_request):
        """Test handling SparkOperationError."""
        exc = SparkOperationError("Spark operation failed")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 503
        content = response.body.decode()
        assert "Spark operation" in content

    @pytest.mark.asyncio
    async def test_handles_spark_query_error(self, mock_request):
        """Test handling SparkQueryError."""
        exc = SparkQueryError("Invalid SQL syntax")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Invalid SQL" in content

    @pytest.mark.asyncio
    async def test_handles_spark_timeout_error(self, mock_request):
        """Test handling SparkTimeoutError."""
        exc = SparkTimeoutError(operation="query", timeout=30.0)

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 408
        content = response.body.decode()
        assert "timed out" in content

    @pytest.mark.asyncio
    async def test_handles_base_delta_lake_error(self, mock_request):
        """Test handling base DeltaLakeError."""
        exc = DeltaLakeError("Generic delta lake error")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "delta lake" in content.lower()

    @pytest.mark.asyncio
    async def test_handles_base_mcp_server_error(self, mock_request):
        """Test handling base MCPServerError (unmapped)."""
        exc = MCPServerError("Generic MCP error")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_handles_request_validation_error(self, mock_request):
        """Test handling RequestValidationError."""
        # Create a mock validation error
        exc = RequestValidationError(
            errors=[
                {
                    "loc": ("body", "name"),
                    "msg": "field required",
                    "type": "value_error.missing",
                }
            ]
        )

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 400
        content = response.body.decode()
        assert "Request validation failed" in content

    @pytest.mark.asyncio
    async def test_handles_http_exception(self, mock_request):
        """Test handling HTTPException."""
        exc = HTTPException(status_code=404, detail="Resource not found")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 404
        content = response.body.decode()
        assert "Resource not found" in content

    @pytest.mark.asyncio
    async def test_handles_http_exception_custom_status(self, mock_request):
        """Test handling HTTPException with custom status code."""
        exc = HTTPException(status_code=429, detail="Rate limit exceeded")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 429
        content = response.body.decode()
        assert "Rate limit exceeded" in content

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, mock_request):
        """Test handling generic Exception."""
        exc = Exception("Unexpected error")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 500
        content = response.body.decode()
        # Generic exceptions should not expose internal details
        assert "unexpected error occurred" in content.lower()

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self, mock_request):
        """Test handling RuntimeError (generic exception)."""
        exc = RuntimeError("Something broke")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_mcp_error_with_empty_message(self, mock_request):
        """Test MCPServerError subclass with empty message falls back to error type."""
        # Create an exception with empty string message
        exc = InvalidTokenError("")

        response = await universal_error_handler(mock_request, exc)

        assert response.status_code == 401
        # Should still have error type info even with empty message
        content = response.body.decode()
        assert "Invalid token" in content
