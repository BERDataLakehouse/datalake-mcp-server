"""
Custom exceptions for the Delta Lake MCP Server.
"""


class MCPServerError(Exception):
    """
    The super class of all MCP Server related errors.
    """


class SparkSessionError(MCPServerError):
    """
    An error thrown when there is an issue initializing or accessing the Spark session.
    """


class SparkOperationError(SparkSessionError):
    """
    An error thrown when a Spark operation fails.
    """


class AuthenticationError(MCPServerError):
    """
    Super class for authentication related errors.
    """


class MissingTokenError(AuthenticationError):
    """
    An error thrown when a token is required but absent.
    """


class InvalidAuthHeaderError(AuthenticationError):
    """
    An error thrown when an authorization header is invalid.
    """


class InvalidTokenError(AuthenticationError):
    """
    An error thrown when a user's token is invalid.
    """


class MissingRoleError(AuthenticationError):
    """
    An error thrown when a user is missing a required role.
    """


class DeltaLakeError(MCPServerError):
    """
    Base class for Delta Lake related errors.
    """


class InvalidS3PathError(DeltaLakeError):
    """
    An error thrown when an S3 path is invalid or does not follow required format.
    """


class DeltaTableNotFoundError(DeltaLakeError):
    """
    An error thrown when a Delta table is not found at the specified path.
    """


class DeltaDatabaseNotFoundError(DeltaLakeError):
    """
    An error thrown when a Delta database is not found.
    """


class DeltaSchemaError(DeltaLakeError):
    """
    An error thrown when there is an issue with a Delta table's schema.
    """


class S3AccessError(DeltaLakeError):
    """
    An error thrown when there is an issue accessing S3 storage.
    """


class DeltaTableOperationError(DeltaLakeError):
    """
    An error thrown when an operation on a Delta table fails.
    """
