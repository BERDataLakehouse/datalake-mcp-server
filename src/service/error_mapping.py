"""
Map errors from exception type to custom error type and HTTP status.
"""

# mostly copied from https://github.com/kbase/cdm-task-service/blob/main/cdmtaskservice/error_mapping.py

from typing import NamedTuple

from fastapi import status

from src.service.errors import ErrorType
from src.service.exceptions import (
    AuthenticationError,
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

_H400 = status.HTTP_400_BAD_REQUEST
_H401 = status.HTTP_401_UNAUTHORIZED
_H403 = status.HTTP_403_FORBIDDEN
_H404 = status.HTTP_404_NOT_FOUND
_H408 = status.HTTP_408_REQUEST_TIMEOUT
_H503 = status.HTTP_503_SERVICE_UNAVAILABLE


class ErrorMapping(NamedTuple):
    """The application error type and HTTP status code for an exception."""

    err_type: ErrorType | None
    """ The type of application error. None if a 5XX error or Not Found."""
    http_code: int
    """ The HTTP code of the error. """


# Map only using ErrorTypes that actually exist in errors.py
_ERR_MAP = {
    # Authentication errors
    MissingTokenError: ErrorMapping(ErrorType.NO_TOKEN, _H401),
    InvalidAuthHeaderError: ErrorMapping(ErrorType.INVALID_AUTH_HEADER, _H401),
    InvalidTokenError: ErrorMapping(ErrorType.INVALID_TOKEN, _H401),
    MissingRoleError: ErrorMapping(ErrorType.MISSING_ROLE, _H403),
    AuthenticationError: ErrorMapping(ErrorType.AUTHENTICATION_FAILED, _H401),
    # Delta Lake specific errors
    InvalidS3PathError: ErrorMapping(ErrorType.INVALID_S3_PATH, _H400),
    DeltaTableNotFoundError: ErrorMapping(ErrorType.DELTA_TABLE_NOT_FOUND, _H404),
    DeltaDatabaseNotFoundError: ErrorMapping(ErrorType.DELTA_DATABASE_NOT_FOUND, _H404),
    DeltaSchemaError: ErrorMapping(ErrorType.DELTA_SCHEMA_ERROR, _H400),
    S3AccessError: ErrorMapping(ErrorType.S3_ACCESS_ERROR, _H400),
    DeltaTableOperationError: ErrorMapping(
        ErrorType.DELTA_TABLE_OPERATION_ERROR, _H400
    ),
    DeltaLakeError: ErrorMapping(ErrorType.DELTA_LAKE_ERROR, _H400),
    SparkSessionError: ErrorMapping(ErrorType.SPARK_SESSION_ERROR, _H503),
    SparkOperationError: ErrorMapping(ErrorType.SPARK_OPERATION_ERROR, _H503),
    SparkQueryError: ErrorMapping(ErrorType.SPARK_QUERY_ERROR, _H400),
    SparkTimeoutError: ErrorMapping(ErrorType.SPARK_TIMEOUT_ERROR, _H408),
    # Base error fallback
    MCPServerError: ErrorMapping(None, status.HTTP_500_INTERNAL_SERVER_ERROR),
}


def map_error(err: MCPServerError) -> ErrorMapping:
    """
    Map an error to an optional error type and a HTTP code.
    """
    # May need to add code to go up the error hierarchy if multiple errors have the same type
    mapping = _ERR_MAP.get(type(err))

    if not mapping:
        mapping = ErrorMapping(None, status.HTTP_500_INTERNAL_SERVER_ERROR)

    return mapping
