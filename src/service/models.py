"""
Pydantic models for the Spark Manager API.
"""

from typing import Annotated, Any, Dict, List

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: Annotated[int | None, Field(description="Error code")] = None
    error_type: Annotated[str | None, Field(description="Error type")] = None
    message: Annotated[str | None, Field(description="Error message")] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: Annotated[str, Field(description="Health status")]


class DatabaseListRequest(BaseModel):
    """Request model for listing databases."""

    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True

    filter_by_namespace: Annotated[
        bool,
        Field(
            description="Whether to filter databases by user/tenant namespace prefixes"
        ),
    ] = True


class DatabaseListResponse(BaseModel):
    """Response model for listing databases."""

    databases: Annotated[List[str], Field(description="List of database names")]


class TableListRequest(BaseModel):
    """Request model for listing tables in a database."""

    database: Annotated[
        str, Field(description="Name of the database to list tables from")
    ]
    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True


class TableListResponse(BaseModel):
    """Response model for listing tables."""

    tables: Annotated[
        List[str], Field(description="List of table names in the specified database")
    ]


class TableSchemaRequest(BaseModel):
    """Request model for getting table schema."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to get schema for")]


class TableSchemaResponse(BaseModel):
    """Response model for table schema."""

    columns: Annotated[
        List[str], Field(description="List of column names in the table")
    ]


class DatabaseStructureRequest(BaseModel):
    """Request model for getting database structure."""

    with_schema: Annotated[
        bool, Field(description="Whether to include table schemas in the response")
    ] = False
    use_hms: Annotated[
        bool,
        Field(
            description="Whether to use Hive Metastore client for faster metadata retrieval"
        ),
    ] = True


class DatabaseStructureResponse(BaseModel):
    """Response model for database structure."""

    structure: Annotated[
        Dict[str, Any],
        Field(
            description="Database structure with tables and optionally their schemas"
        ),
    ]


# ---
# Models for Delta Table Data Operations
# ---


class TableQueryRequest(BaseModel):
    """Request model for querying a Delta table."""

    query: Annotated[str, Field(description="SQL query to execute against the table")]


class TableQueryResponse(BaseModel):
    """Response model for Delta table query results."""

    result: Annotated[
        List[Any],
        Field(description="List of rows returned by the query, each as a dictionary"),
    ]


class TableCountRequest(BaseModel):
    """Request model for counting rows in a Delta table."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to count rows in")]


class TableCountResponse(BaseModel):
    """Response model for Delta table row count."""

    count: Annotated[int, Field(description="Total number of rows in the table")]


class TableSampleRequest(BaseModel):
    """Request model for sampling data from a Delta table."""

    database: Annotated[
        str, Field(description="Name of the database containing the table")
    ]
    table: Annotated[str, Field(description="Name of the table to sample from")]
    limit: Annotated[
        int,
        Field(
            description="Maximum number of rows to return in the sample", gt=0, le=1000
        ),
    ] = 10
    columns: Annotated[
        List[str] | None, Field(description="List of columns to return in the sample")
    ] = None
    where_clause: Annotated[
        str | None, Field(description="SQL WHERE clause to filter the rows")
    ] = None


class TableSampleResponse(BaseModel):
    """Response model for Delta table data sample."""

    sample: Annotated[
        List[Any],
        Field(description="List of sample rows, each as a dictionary"),
    ]
