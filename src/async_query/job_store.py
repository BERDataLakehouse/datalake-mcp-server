"""
S3/MinIO-backed job metadata store for async query execution.

Stores JobRecord objects as _metadata.json files in MinIO alongside query results.
Each job's metadata lives at:
    users-general-warehouse/{user}/data/query_result/{job_id}/_metadata.json

All functions accept a boto3 S3 client parameter — callers are responsible for
creating the client with appropriate credentials (typically the user's own
MinIO credentials).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from botocore.exceptions import ClientError

from src.async_query import s3_client
from src.service.models import JobRecord, JobStatus
from src.service.spark_session_pool import SPARK_STANDALONE_QUERY_TIMEOUT
from src.service.timeouts import SPARK_CONNECT_QUERY_TIMEOUT

logger = logging.getLogger(__name__)

_BUCKET = s3_client.ASYNC_QUERY_RESULT_BUCKET


def _metadata_key(user: str, job_id: str) -> str:
    """S3 key for a job's metadata file."""
    return f"{s3_client.build_result_path(user, job_id)}_metadata.json"


def _serialize_job(job: JobRecord) -> str:
    """Serialize a JobRecord to JSON string for S3 storage."""
    data = job.model_dump()
    for field in ("created_at", "started_at", "completed_at"):
        if data[field] is not None:
            data[field] = data[field].isoformat()
    return json.dumps(data)


def _deserialize_job(raw: str | bytes) -> JobRecord:
    """Deserialize a JSON string from S3 into a JobRecord."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    for field in ("created_at", "started_at", "completed_at"):
        if data.get(field) is not None:
            data[field] = datetime.fromisoformat(data[field])
    return JobRecord(**data)


def create_job(client: Any, job: JobRecord) -> None:
    """
    Store a new job record as _metadata.json in S3.

    Args:
        client: boto3 S3 client.
        job: The job record to store.
    """
    key = _metadata_key(job.user, job.job_id)
    try:
        client.put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=_serialize_job(job),
            ContentType="application/json",
        )
        logger.info(f"Created job record: job_id={job.job_id} user={job.user}")
    except Exception:
        logger.exception(f"Failed to create job record: job_id={job.job_id}")
        raise


def get_job(client: Any, job_id: str, user: str) -> Optional[JobRecord]:
    """
    Retrieve a job record by ID and user.

    Args:
        client: boto3 S3 client.
        job_id: The unique job identifier.
        user: KBase username (needed to construct S3 path).

    Returns:
        JobRecord if found, None otherwise.
    """
    key = _metadata_key(user, job_id)
    try:
        response = client.get_object(Bucket=_BUCKET, Key=key)
        raw = response["Body"].read()
        return _deserialize_job(raw)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        logger.exception(f"Failed to retrieve job record: job_id={job_id}")
        return None
    except Exception:
        logger.exception(f"Failed to retrieve job record: job_id={job_id}")
        return None


def update_job_status(
    client: Any,
    job_id: str,
    status: JobStatus,
    user: str,
    **kwargs,
) -> None:
    """
    Update a job's status and optional fields.

    Reads the existing _metadata.json, applies updates, and writes back.

    Args:
        client: boto3 S3 client.
        job_id: The unique job identifier.
        status: The new job status.
        user: KBase username (needed to construct S3 path).
        **kwargs: Additional fields to update (started_at, completed_at,
                  error_message, row_count, total_count, has_more, result_path).
    """
    key = _metadata_key(user, job_id)

    try:
        response = client.get_object(Bucket=_BUCKET, Key=key)
        raw = response["Body"].read()
        job = _deserialize_job(raw)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning(f"Job not found for status update: job_id={job_id}")
            return
        logger.exception(f"Failed to read job for status update: job_id={job_id}")
        return
    except Exception:
        logger.exception(f"Failed to read job for status update: job_id={job_id}")
        return

    job.status = status
    for field, value in kwargs.items():
        if hasattr(job, field):
            setattr(job, field, value)

    try:
        client.put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=_serialize_job(job),
            ContentType="application/json",
        )
        logger.info(f"Updated job status: job_id={job_id} status={status.value}")
    except Exception:
        logger.exception(f"Failed to update job status: job_id={job_id}")


def list_user_jobs(client: Any, user: str) -> list[JobRecord]:
    """
    List all jobs for a specific user by scanning their query_result/ prefix.

    Args:
        client: boto3 S3 client.
        user: KBase username.

    Returns:
        List of JobRecord objects sorted by created_at descending.
    """
    prefix = s3_client.build_query_result_root_path(user)

    try:
        paginator = client.get_paginator("list_objects_v2")
        jobs = []
        for page in paginator.paginate(Bucket=_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("_metadata.json"):
                    continue
                try:
                    response = client.get_object(Bucket=_BUCKET, Key=key)
                    raw = response["Body"].read()
                    jobs.append(_deserialize_job(raw))
                except Exception:
                    logger.exception(f"Failed to read metadata: {key}")

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs
    except Exception:
        logger.exception(f"Failed to list jobs for user: {user}")
        return []


def count_active_user_jobs(client: Any, user: str) -> int:
    """
    Count the number of PENDING and RUNNING jobs for a user.

    Args:
        client: boto3 S3 client.
        user: KBase username.

    Returns:
        Number of active (PENDING or RUNNING) jobs.
    """
    jobs = list_user_jobs(client, user)
    return sum(1 for j in jobs if j.status in (JobStatus.PENDING, JobStatus.RUNNING))


# Buffer added on top of the query timeout before declaring a job stale.
# Accounts for session creation, S3 upload, and metadata write overhead.
_STALE_JOB_BUFFER_SECONDS = 120

_STALE_JOB_TIMEOUT = (
    max(SPARK_CONNECT_QUERY_TIMEOUT, SPARK_STANDALONE_QUERY_TIMEOUT)
    + _STALE_JOB_BUFFER_SECONDS
)


def _is_stale(job: JobRecord) -> bool:
    """Check if a PENDING/RUNNING job has exceeded the staleness threshold."""
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        return False
    reference_time = job.started_at or job.created_at
    elapsed = (datetime.now(timezone.utc) - reference_time).total_seconds()
    return elapsed > _STALE_JOB_TIMEOUT


def expire_stale_job(client: Any, job: JobRecord) -> JobRecord:
    """
    Check whether a PENDING or RUNNING job has exceeded the query timeout
    and, if so, mark it as FAILED.

    This catches jobs whose background task crashed, was OOM-killed, or
    was lost during a server restart — leaving the metadata stuck in an
    active state.

    The staleness threshold is the larger of SPARK_CONNECT_QUERY_TIMEOUT
    and SPARK_STANDALONE_QUERY_TIMEOUT plus a buffer for overhead.

    Args:
        client: boto3 S3 client.
        job: The job record to check.

    Returns:
        The (possibly updated) JobRecord.
    """
    if not _is_stale(job):
        return job

    reference_time = job.started_at or job.created_at
    elapsed = (datetime.now(timezone.utc) - reference_time).total_seconds()
    error_msg = (
        f"Job timed out after {elapsed:.0f}s (threshold: {_STALE_JOB_TIMEOUT:.0f}s). "
        f"The background task likely crashed or was terminated."
    )
    logger.warning(f"Expiring stale job: job_id={job.job_id} {error_msg}")

    update_job_status(
        client,
        job.job_id,
        JobStatus.FAILED,
        user=job.user,
        completed_at=datetime.now(timezone.utc),
        error_message=error_msg,
    )

    job.status = JobStatus.FAILED
    job.completed_at = datetime.now(timezone.utc)
    job.error_message = error_msg
    return job


def cleanup_stale_jobs(client: Any, user: str) -> int:
    """
    Find stale PENDING/RUNNING jobs and delete their entire S3 directory.

    Called at submission time to free up slots before counting active jobs.
    Jobs that have exceeded _STALE_JOB_TIMEOUT are considered abandoned
    (background task crashed, OOM-killed, or lost during server restart)
    and are deleted entirely — metadata, results, and .s3keep files.

    Args:
        client: boto3 S3 client.
        user: KBase username.

    Returns:
        Number of stale jobs cleaned up.
    """
    jobs = list_user_jobs(client, user)
    cleaned = 0

    for job in jobs:
        if not _is_stale(job):
            continue

        logger.warning(
            f"Cleaning up stale job: job_id={job.job_id} user={user} "
            f"status={job.status.value}"
        )

        result_prefix = s3_client.build_result_path(user, job.job_id)
        try:
            s3_client.delete_result_prefix(client, _BUCKET, result_prefix)
            cleaned += 1
        except Exception:
            logger.exception(
                f"Failed to clean up stale job directory: job_id={job.job_id}"
            )

    if cleaned:
        logger.info(f"Cleaned up {cleaned} stale job(s) for user {user}")

    return cleaned
