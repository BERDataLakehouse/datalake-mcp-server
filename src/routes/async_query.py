"""
API routes for async query execution.

Provides endpoints for submitting long-running Spark SQL queries,
polling job status, and retrieving results inline (same format as
the sync query endpoint). Results are stored temporarily in S3/MinIO
and returned to the client on retrieval, then deleted.

All routes are under /delta/tables/query/async/ to sit alongside
the existing sync /delta/tables/query endpoint.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, status

from src.delta_lake.delta_service import _check_query_is_valid
from src.async_query import job_store, s3_client
from src.async_query.executor import AsyncQueryExecutor
from src.service.dependencies import (
    SparkContext,
    auth,
    fetch_user_minio_credentials,
    get_spark_context,
    get_token_from_request,
    get_user_from_request,
)
from src.service.exceptions import (
    JobAccessDeniedError,
    JobNotFoundError,
    JobNotReadyError,
    MissingTokenError,
    TooManyJobsError,
)
from src.service.models import (
    MAX_CONCURRENT_ASYNC_JOBS_PER_USER,
    AsyncQueryStatusResponse,
    AsyncQuerySubmitRequest,
    AsyncQuerySubmitResponse,
    JobRecord,
    JobStatus,
    PaginationInfo,
    TableQueryResponse,
)
from src.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/delta/tables/query/async",
    tags=["Async Query"],
)


def _get_executor(request: Request) -> AsyncQueryExecutor:
    """Retrieve the AsyncQueryExecutor from app state."""
    return request.app.state.async_query_executor


def _require_auth_token(request: Request) -> str:
    """Extract and validate the auth token from the request.

    Raises MissingTokenError if the token cannot be parsed, ensuring
    a clean 401 instead of a 500 from downstream credential fetch.
    """
    token = get_token_from_request(request)
    if not token:
        raise MissingTokenError("Authorization token required for this operation")
    return token


def _validate_job_access(job: JobRecord, username: str) -> None:
    """Validate that the requesting user owns the job."""
    if job.user != username:
        raise JobAccessDeniedError(
            f"User '{username}' does not have access to job '{job.job_id}'"
        )


@router.post(
    "/submit",
    response_model=AsyncQuerySubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an async query",
    description=(
        "Submits a Spark SQL query for asynchronous execution. "
        "Returns a job_id immediately. Poll /{job_id}/status to track "
        "progress, then call /{job_id}/results to retrieve the data "
        "inline (same format as the sync query endpoint). "
        "Supports pagination with limit (max 5000) and offset parameters."
    ),
    operation_id="submit_async_query",
)
async def submit_async_query(
    request: AsyncQuerySubmitRequest,
    http_request: Request,
    ctx: Annotated[SparkContext, Depends(get_spark_context)],
    auth=Depends(auth),
) -> AsyncQuerySubmitResponse:
    """Submit an async query for background execution."""
    username = ctx.username

    # Validate query safety (same validation as sync endpoint)
    _check_query_is_valid(request.query)

    # Read user's MinIO credentials and create S3 client
    minio_access_key = ctx.settings_dict["MINIO_ACCESS_KEY"]
    minio_secret_key = ctx.settings_dict["MINIO_SECRET_KEY"]
    settings = get_settings()
    client = s3_client.create_s3_client(
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=settings.MINIO_SECURE,
    )

    # Enforce per-user concurrency limit
    active_count = await asyncio.to_thread(
        job_store.count_active_user_jobs, client, username
    )
    if active_count >= MAX_CONCURRENT_ASYNC_JOBS_PER_USER:
        raise TooManyJobsError(
            f"User '{username}' has {active_count} active job(s). "
            f"Maximum concurrent jobs per user is "
            f"{MAX_CONCURRENT_ASYNC_JOBS_PER_USER}. "
            f"Wait for existing jobs to complete before submitting new ones."
        )

    # Generate job ID and create initial record
    job_id = str(uuid4())

    job = JobRecord(
        job_id=job_id,
        user=username,
        query=request.query,
        status=JobStatus.PENDING,
        limit=request.limit,
        offset=request.offset,
        created_at=datetime.now(timezone.utc),
    )
    await asyncio.to_thread(job_store.create_job, client, job)

    # Submit to background executor
    executor = _get_executor(http_request)
    await executor.submit_query(
        job_id=job_id,
        user=username,
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        ctx=ctx,
        minio_endpoint=settings.MINIO_ENDPOINT_URL,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_secure=settings.MINIO_SECURE,
    )

    logger.info(f"Async query submitted: job_id={job_id} user={username}")
    return AsyncQuerySubmitResponse(job_id=job_id)


@router.get(
    "/{job_id}/status",
    response_model=AsyncQueryStatusResponse,
    summary="Get async query job status",
    description="Retrieves the current status and metadata of an async query job.",
    operation_id="get_async_query_status",
)
async def get_async_query_status(
    job_id: str,
    http_request: Request,
    auth=Depends(auth),
) -> AsyncQueryStatusResponse:
    """Get the status of an async query job."""
    username = get_user_from_request(http_request)
    auth_token = _require_auth_token(http_request)
    settings = get_settings()
    minio_access_key, minio_secret_key = fetch_user_minio_credentials(
        settings.GOVERNANCE_API_URL, auth_token
    )
    client = s3_client.create_s3_client(
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=settings.MINIO_SECURE,
    )

    job = await asyncio.to_thread(job_store.get_job, client, job_id, username)
    if job is None:
        raise JobNotFoundError(
            f"Job '{job_id}' not found. It may have already been consumed "
            f"(results are deleted after retrieval)."
        )

    _validate_job_access(job, username)

    # Expire stale PENDING/RUNNING jobs whose background task likely died
    job = await asyncio.to_thread(job_store.expire_stale_job, client, job)

    return AsyncQueryStatusResponse(
        job_id=job.job_id,
        status=job.status,
        query=job.query,
        limit=job.limit,
        offset=job.offset,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        row_count=job.row_count,
    )


@router.get(
    "/{job_id}/results",
    response_model=TableQueryResponse,
    summary="Get async query results",
    description=(
        "Retrieves the results of a completed async query. "
        "Returns the same format as the sync /delta/tables/query endpoint. "
        "Only available after the job has succeeded."
    ),
    operation_id="get_async_query_results",
)
async def get_async_query_results(
    job_id: str,
    http_request: Request,
    auth=Depends(auth),
) -> TableQueryResponse:
    """Get async query results (same format as sync query endpoint)."""
    username = get_user_from_request(http_request)
    auth_token = _require_auth_token(http_request)
    settings = get_settings()
    minio_access_key, minio_secret_key = fetch_user_minio_credentials(
        settings.GOVERNANCE_API_URL, auth_token
    )
    client = s3_client.create_s3_client(
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=settings.MINIO_SECURE,
    )

    job = await asyncio.to_thread(job_store.get_job, client, job_id, username)
    if job is None:
        raise JobNotFoundError(
            f"Job '{job_id}' not found. It may have already been consumed "
            f"(results are deleted after retrieval)."
        )

    _validate_job_access(job, username)

    # Expire stale PENDING/RUNNING jobs whose background task likely died
    job = await asyncio.to_thread(job_store.expire_stale_job, client, job)

    if job.status != JobStatus.SUCCEEDED:
        raise JobNotReadyError(
            f"Job '{job_id}' is not ready (status: {job.status.value}). "
            f"Results are only available for completed jobs."
        )

    # Download result and clean up from S3
    result_prefix = s3_client.build_result_path(username, job_id)
    bucket = s3_client.ASYNC_QUERY_RESULT_BUCKET

    def _download_result():
        return s3_client.download_result(client, bucket, result_prefix)

    result = await asyncio.to_thread(_download_result)

    # Clean up the job's result files from MinIO (including _metadata.json)
    def _cleanup():
        s3_client.delete_result_prefix(client, bucket, result_prefix)

    await asyncio.to_thread(_cleanup)

    # Build pagination info from job record
    pagination = PaginationInfo(
        limit=job.limit,
        offset=job.offset,
        total_count=job.total_count or len(result),
        has_more=job.has_more or False,
    )

    return TableQueryResponse(result=result, pagination=pagination)


@router.get(
    "/jobs",
    response_model=list[AsyncQueryStatusResponse],
    summary="List user's async query jobs",
    description="Lists all async query jobs for the authenticated user.",
    operation_id="list_async_query_jobs",
)
async def list_async_query_jobs(
    http_request: Request,
    auth=Depends(auth),
) -> list[AsyncQueryStatusResponse]:
    """List all async query jobs for the current user."""
    username = get_user_from_request(http_request)
    auth_token = _require_auth_token(http_request)
    settings = get_settings()
    minio_access_key, minio_secret_key = fetch_user_minio_credentials(
        settings.GOVERNANCE_API_URL, auth_token
    )
    client = s3_client.create_s3_client(
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=settings.MINIO_SECURE,
    )

    jobs = await asyncio.to_thread(job_store.list_user_jobs, client, username)

    # Expire any stale PENDING/RUNNING jobs inline so the list always
    # reflects accurate state (handles crashed background tasks)
    def _expire_stale_jobs(job_list: list[JobRecord]) -> list[JobRecord]:
        return [job_store.expire_stale_job(client, j) for j in job_list]

    jobs = await asyncio.to_thread(_expire_stale_jobs, jobs)

    return [
        AsyncQueryStatusResponse(
            job_id=job.job_id,
            status=job.status,
            query=job.query,
            limit=job.limit,
            offset=job.offset,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            row_count=job.row_count,
        )
        for job in jobs
    ]
