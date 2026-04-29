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


class SparkConnectUnavailableError(SparkSessionError):
    """
    The user's Spark Connect server is reachable at the gRPC level but cannot
    execute SQL — typically because the JVM driver in their notebook pod is
    deadlocked. The fix is to restart the notebook pod; we surface this as a
    fast, clear 503 instead of letting every subsequent SQL call hang the
    full request-timeout window.
    """


class SparkOperationError(SparkSessionError):
    """
    An error thrown when a Spark operation fails.
    """


class SparkQueryError(SparkOperationError):
    """
    An error thrown when a Spark query fails.
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


class MissingMFAError(AuthenticationError):
    """
    An error thrown when a user's token was not created with MFA.
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


class SparkTimeoutError(SparkOperationError):
    """
    An error thrown when a Spark operation exceeds its timeout.

    This typically indicates a query that is too expensive or a system
    that is under heavy load.
    """

    def __init__(
        self,
        operation: str = "spark_operation",
        timeout: float = 0,
        message: str | None = None,
    ):
        self.operation = operation
        self.timeout = timeout
        if message:
            super().__init__(message)
        else:
            super().__init__(
                f"Spark operation '{operation}' timed out after {timeout} seconds. "
                f"Consider using pagination or reducing the query scope."
            )


class AsyncQueryError(MCPServerError):
    """
    Base class for async query related errors.
    """


class JobNotFoundError(AsyncQueryError):
    """
    An error thrown when a job_id does not exist.
    """


class JobNotReadyError(AsyncQueryError):
    """
    An error thrown when results are requested before a job has completed.
    """


class JobFailedError(AsyncQueryError):
    """
    An error thrown when results are requested for a job that failed.
    """


class JobAccessDeniedError(AsyncQueryError):
    """
    An error thrown when a user tries to access another user's job.
    """


class TooManyJobsError(AsyncQueryError):
    """
    An error thrown when a user has too many concurrent async jobs.
    """


# ----- Trino errors -----


class TrinoConnectionError(MCPServerError):
    """
    An error thrown when there is an issue connecting to Trino.
    """


class TrinoOperationError(MCPServerError):
    """
    An error thrown when a Trino operation fails.
    """


class TrinoQueryError(TrinoOperationError):
    """
    An error thrown when a Trino query fails.
    """
