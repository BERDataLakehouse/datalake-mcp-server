"""
Tests for the S3/MinIO-backed job store module.

Tests cover:
- Job creation (S3 put_object)
- Job retrieval by ID and user
- Job status updates with field persistence
- Listing jobs per user (sorted)
- Counting active jobs
- Serialization/deserialization round-trips
- S3 unavailability handling (all error branches)
- Stale job expiration
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.async_query.job_store import (
    _STALE_JOB_TIMEOUT,
    _deserialize_job,
    _metadata_key,
    _serialize_job,
    create_job,
    count_active_user_jobs,
    expire_stale_job,
    get_job,
    list_user_jobs,
    update_job_status,
)
from src.service.models import JobRecord, JobStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_job():
    """Create a sample JobRecord for testing."""
    return JobRecord(
        job_id="test-job-123",
        user="testuser",
        query="SELECT * FROM db.table",
        status=JobStatus.PENDING,
        limit=1000,
        offset=0,
        created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        result_format="json",
    )


@pytest.fixture
def mock_s3():
    """Create a mock S3 client with in-memory object storage."""
    client = MagicMock()
    objects = {}  # key -> bytes

    def put_object(Bucket, Key, Body, **kwargs):
        body = Body.encode("utf-8") if isinstance(Body, str) else Body
        objects[Key] = body

    def get_object(Bucket, Key):
        if Key not in objects:
            error = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
            raise ClientError(error, "GetObject")
        body_stream = MagicMock()
        body_stream.read.return_value = objects[Key]
        return {"Body": body_stream}

    def list_objects_v2(Bucket, Prefix, **kwargs):
        matching = [k for k in objects if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in matching]}

    # Paginator that wraps list_objects_v2
    paginator = MagicMock()

    def paginate(Bucket, Prefix, **kwargs):
        matching = [k for k in objects if k.startswith(Prefix)]
        return [{"Contents": [{"Key": k} for k in matching]}]

    paginator.paginate = MagicMock(side_effect=paginate)
    client.get_paginator = MagicMock(return_value=paginator)

    client.put_object = MagicMock(side_effect=put_object)
    client.get_object = MagicMock(side_effect=get_object)
    client.list_objects_v2 = MagicMock(side_effect=list_objects_v2)
    client._objects = objects

    return client


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for job serialization/deserialization."""

    def test_serialize_deserialize_round_trip(self, sample_job):
        """Serialized and deserialized job should match original."""
        serialized = _serialize_job(sample_job)
        deserialized = _deserialize_job(serialized)

        assert deserialized.job_id == sample_job.job_id
        assert deserialized.user == sample_job.user
        assert deserialized.query == sample_job.query
        assert deserialized.status == sample_job.status
        assert deserialized.limit == sample_job.limit
        assert deserialized.offset == sample_job.offset
        assert deserialized.created_at == sample_job.created_at
        assert deserialized.result_format == sample_job.result_format

    def test_serialize_with_all_fields(self):
        """Serialization handles all optional fields."""
        job = JobRecord(
            job_id="full-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.SUCCEEDED,
            limit=500,
            offset=100,
            created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc),
            completed_at=datetime(2025, 1, 15, 10, 0, 5, tzinfo=timezone.utc),
            error_message=None,
            result_path="s3a://bucket/path/",
            row_count=42,
            total_count=100,
            has_more=True,
            result_format="parquet",
        )

        serialized = _serialize_job(job)
        deserialized = _deserialize_job(serialized)

        assert deserialized.started_at == job.started_at
        assert deserialized.completed_at == job.completed_at
        assert deserialized.result_path == "s3a://bucket/path/"
        assert deserialized.row_count == 42
        assert deserialized.total_count == 100
        assert deserialized.has_more is True
        assert deserialized.result_format == "parquet"

    def test_deserialize_from_bytes(self, sample_job):
        """Deserialization handles bytes input."""
        serialized = _serialize_job(sample_job)
        deserialized = _deserialize_job(serialized.encode("utf-8"))
        assert deserialized.job_id == sample_job.job_id


# =============================================================================
# Key Format Tests
# =============================================================================


class TestKeyFormat:
    """Tests for S3 metadata key formatting."""

    def test_metadata_key_format(self):
        key = _metadata_key("testuser", "abc-123")
        assert (
            key
            == "users-general-warehouse/testuser/data/query_result/abc-123/_metadata.json"
        )


# =============================================================================
# create_job Tests
# =============================================================================


class TestCreateJob:
    """Tests for create_job function."""

    def test_create_job_stores_record(self, mock_s3, sample_job):
        """create_job writes _metadata.json to S3."""
        create_job(mock_s3, sample_job)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Key"] == _metadata_key(sample_job.user, sample_job.job_id)
        assert call_kwargs["ContentType"] == "application/json"

        # Verify the stored data is valid JSON with correct fields
        stored = mock_s3._objects[_metadata_key(sample_job.user, sample_job.job_id)]
        data = json.loads(stored)
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "PENDING"

    def test_create_job_s3_error(self, sample_job):
        """create_job raises on S3 failure."""
        mock_client = MagicMock()
        mock_client.put_object.side_effect = Exception("S3 error")

        with pytest.raises(Exception, match="S3 error"):
            create_job(mock_client, sample_job)


# =============================================================================
# get_job Tests
# =============================================================================


class TestGetJob:
    """Tests for get_job function."""

    def test_get_existing_job(self, mock_s3, sample_job):
        """get_job returns the job when found in S3."""
        # Seed the store
        key = _metadata_key(sample_job.user, sample_job.job_id)
        mock_s3._objects[key] = _serialize_job(sample_job).encode()

        result = get_job(mock_s3, sample_job.job_id, sample_job.user)
        assert result is not None
        assert result.job_id == sample_job.job_id
        assert result.status == JobStatus.PENDING

    def test_get_nonexistent_job(self, mock_s3):
        """get_job returns None for missing job."""
        result = get_job(mock_s3, "nonexistent-id", "testuser")
        assert result is None

    def test_get_job_s3_error(self):
        """get_job returns None on unexpected S3 error."""
        mock_client = MagicMock()
        error = {"Error": {"Code": "InternalError", "Message": "oops"}}
        mock_client.get_object.side_effect = ClientError(error, "GetObject")

        result = get_job(mock_client, "any-id", "testuser")
        assert result is None


# =============================================================================
# update_job_status Tests
# =============================================================================


class TestUpdateJobStatus:
    """Tests for update_job_status function."""

    def test_update_status(self, mock_s3, sample_job):
        """update_job_status changes the status field."""
        key = _metadata_key(sample_job.user, sample_job.job_id)
        mock_s3._objects[key] = _serialize_job(sample_job).encode()

        update_job_status(
            mock_s3, sample_job.job_id, JobStatus.RUNNING, user=sample_job.user
        )

        stored = mock_s3._objects[key]
        data = json.loads(stored)
        assert data["status"] == "RUNNING"

    def test_update_with_kwargs(self, mock_s3, sample_job):
        """update_job_status applies additional field updates."""
        key = _metadata_key(sample_job.user, sample_job.job_id)
        mock_s3._objects[key] = _serialize_job(sample_job).encode()

        completed = datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
        update_job_status(
            mock_s3,
            sample_job.job_id,
            JobStatus.SUCCEEDED,
            user=sample_job.user,
            completed_at=completed,
            row_count=42,
            total_count=100,
            has_more=True,
            result_path="s3a://bucket/path/",
        )

        stored = mock_s3._objects[key]
        data = json.loads(stored)
        assert data["status"] == "SUCCEEDED"
        assert data["row_count"] == 42
        assert data["total_count"] == 100
        assert data["has_more"] is True
        assert data["result_path"] == "s3a://bucket/path/"

    def test_update_nonexistent_job(self, mock_s3):
        """update_job_status silently returns for missing job."""
        # Should not raise
        update_job_status(mock_s3, "missing-id", JobStatus.FAILED, user="testuser")


# =============================================================================
# list_user_jobs Tests
# =============================================================================


class TestListUserJobs:
    """Tests for list_user_jobs function."""

    def test_list_jobs_returns_sorted(self, mock_s3):
        """list_user_jobs returns jobs sorted by created_at descending."""
        job1 = JobRecord(
            job_id="job-1",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.PENDING,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
        job2 = JobRecord(
            job_id="job-2",
            user="testuser",
            query="SELECT 2",
            status=JobStatus.SUCCEEDED,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Seed S3 store
        mock_s3._objects[_metadata_key("testuser", "job-1")] = _serialize_job(
            job1
        ).encode()
        mock_s3._objects[_metadata_key("testuser", "job-2")] = _serialize_job(
            job2
        ).encode()

        jobs = list_user_jobs(mock_s3, "testuser")
        assert len(jobs) == 2
        assert jobs[0].job_id == "job-2"  # newer first
        assert jobs[1].job_id == "job-1"

    def test_list_jobs_empty(self, mock_s3):
        """list_user_jobs returns empty list when user has no jobs."""
        jobs = list_user_jobs(mock_s3, "nouser")
        assert jobs == []

    def test_list_jobs_skips_non_metadata(self, mock_s3):
        """list_user_jobs only reads _metadata.json files, ignoring result files."""
        job = JobRecord(
            job_id="job-1",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.SUCCEEDED,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        prefix = "users-general-warehouse/testuser/data/query_result/job-1/"
        mock_s3._objects[f"{prefix}_metadata.json"] = _serialize_job(job).encode()
        mock_s3._objects[f"{prefix}result.json"] = b'[{"id": 1}]'
        mock_s3._objects[f"{prefix}.s3keep"] = b""

        jobs = list_user_jobs(mock_s3, "testuser")
        assert len(jobs) == 1
        assert jobs[0].job_id == "job-1"


# =============================================================================
# count_active_user_jobs Tests
# =============================================================================


class TestCountActiveUserJobs:
    """Tests for count_active_user_jobs function."""

    def test_count_active_jobs(self, mock_s3):
        """count_active_user_jobs counts PENDING and RUNNING jobs."""
        for i, status in enumerate(
            [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED]
        ):
            job = JobRecord(
                job_id=f"job-{i}",
                user="testuser",
                query="SELECT 1",
                status=status,
                limit=100,
                offset=0,
                created_at=datetime(2025, 1, 15, i, 0, 0, tzinfo=timezone.utc),
            )
            mock_s3._objects[_metadata_key("testuser", f"job-{i}")] = _serialize_job(
                job
            ).encode()

        count = count_active_user_jobs(mock_s3, "testuser")
        assert count == 2  # PENDING + RUNNING

    def test_count_zero_when_empty(self, mock_s3):
        """count_active_user_jobs returns 0 when no jobs exist."""
        count = count_active_user_jobs(mock_s3, "testuser")
        assert count == 0


# =============================================================================
# Error Handling Tests — get_job
# =============================================================================


class TestGetJobErrors:
    """Tests for get_job error branches."""

    def test_get_job_non_nosuchkey_client_error(self):
        """get_job returns None on non-NoSuchKey ClientError (line 99-100)."""
        client = MagicMock()
        error = {"Error": {"Code": "InternalError", "Message": "S3 hiccup"}}
        client.get_object.side_effect = ClientError(error, "GetObject")

        result = get_job(client, "job-1", "testuser")
        assert result is None

    def test_get_job_generic_exception(self):
        """get_job returns None on generic (non-ClientError) exception (line 101-103)."""
        client = MagicMock()
        client.get_object.side_effect = ConnectionError("network down")

        result = get_job(client, "job-1", "testuser")
        assert result is None


# =============================================================================
# Error Handling Tests — update_job_status
# =============================================================================


class TestUpdateJobStatusErrors:
    """Tests for update_job_status error branches."""

    def test_update_non_nosuchkey_client_error_on_read(self):
        """update silently returns on non-NoSuchKey ClientError during read (line 136-137)."""
        client = MagicMock()
        error = {"Error": {"Code": "AccessDenied", "Message": "forbidden"}}
        client.get_object.side_effect = ClientError(error, "GetObject")

        update_job_status(client, "job-1", JobStatus.RUNNING, user="testuser")
        client.put_object.assert_not_called()

    def test_update_generic_exception_on_read(self):
        """update silently returns on generic exception during read (line 138-140)."""
        client = MagicMock()
        client.get_object.side_effect = RuntimeError("unexpected")

        update_job_status(client, "job-1", JobStatus.RUNNING, user="testuser")
        client.put_object.assert_not_called()

    def test_update_exception_on_write(self, mock_s3, sample_job):
        """update logs and swallows exception on put_object failure (line 155-156)."""
        key = _metadata_key(sample_job.user, sample_job.job_id)
        mock_s3._objects[key] = _serialize_job(sample_job).encode()

        mock_s3.put_object.side_effect = Exception("S3 write failed")

        # Should not raise — the error is logged and swallowed
        update_job_status(
            mock_s3, sample_job.job_id, JobStatus.RUNNING, user=sample_job.user
        )

    def test_update_ignores_unknown_kwargs(self, mock_s3, sample_job):
        """update_job_status skips kwargs that don't match JobRecord fields."""
        key = _metadata_key(sample_job.user, sample_job.job_id)
        mock_s3._objects[key] = _serialize_job(sample_job).encode()

        update_job_status(
            mock_s3,
            sample_job.job_id,
            JobStatus.RUNNING,
            user=sample_job.user,
            nonexistent_field="ignored",
        )

        stored = json.loads(mock_s3._objects[key])
        assert stored["status"] == "RUNNING"
        assert "nonexistent_field" not in stored


# =============================================================================
# Error Handling Tests — list_user_jobs
# =============================================================================


class TestListUserJobsErrors:
    """Tests for list_user_jobs error branches."""

    def test_list_skips_corrupt_metadata(self, mock_s3):
        """list_user_jobs skips individual files that fail to parse (line 184-185)."""
        good_job = JobRecord(
            job_id="good-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.SUCCEEDED,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        prefix = "users-general-warehouse/testuser/data/query_result/"
        mock_s3._objects[f"{prefix}good-job/_metadata.json"] = _serialize_job(
            good_job
        ).encode()
        mock_s3._objects[f"{prefix}bad-job/_metadata.json"] = b"not valid json"

        jobs = list_user_jobs(mock_s3, "testuser")
        assert len(jobs) == 1
        assert jobs[0].job_id == "good-job"

    def test_list_returns_empty_on_paginator_failure(self):
        """list_user_jobs returns [] when paginator itself throws (line 189-191)."""
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.side_effect = Exception("S3 unavailable")
        client.get_paginator.return_value = paginator

        jobs = list_user_jobs(client, "testuser")
        assert jobs == []

    def test_list_individual_get_object_failure(self):
        """list_user_jobs skips files where get_object fails (line 184-185)."""
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "prefix/job-1/_metadata.json"},
                    {"Key": "prefix/job-2/_metadata.json"},
                ]
            }
        ]
        client.get_paginator.return_value = paginator
        client.get_object.side_effect = Exception("intermittent failure")

        jobs = list_user_jobs(client, "testuser")
        assert jobs == []


# =============================================================================
# expire_stale_job Tests
# =============================================================================


class TestExpireStaleJob:
    """Tests for the stale job expiration logic."""

    def test_skips_succeeded_job(self, mock_s3):
        """expire_stale_job returns SUCCEEDED jobs unchanged."""
        job = JobRecord(
            job_id="done-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.SUCCEEDED,
            limit=100,
            offset=0,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

        result = expire_stale_job(mock_s3, job)
        assert result.status == JobStatus.SUCCEEDED
        mock_s3.get_object.assert_not_called()

    def test_skips_failed_job(self, mock_s3):
        """expire_stale_job returns FAILED jobs unchanged."""
        job = JobRecord(
            job_id="fail-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.FAILED,
            limit=100,
            offset=0,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

        result = expire_stale_job(mock_s3, job)
        assert result.status == JobStatus.FAILED

    def test_skips_recent_running_job(self, mock_s3):
        """expire_stale_job leaves a recently-started RUNNING job alone."""
        job = JobRecord(
            job_id="fresh-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.RUNNING,
            limit=100,
            offset=0,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            started_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )

        result = expire_stale_job(mock_s3, job)
        assert result.status == JobStatus.RUNNING

    def test_expires_stale_running_job(self, mock_s3):
        """expire_stale_job marks a long-running job as FAILED."""
        stale_time = datetime.now(timezone.utc) - timedelta(
            seconds=_STALE_JOB_TIMEOUT + 60
        )
        job = JobRecord(
            job_id="stale-running",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.RUNNING,
            limit=100,
            offset=0,
            created_at=stale_time - timedelta(seconds=5),
            started_at=stale_time,
        )

        key = _metadata_key(job.user, job.job_id)
        mock_s3._objects[key] = _serialize_job(job).encode()

        result = expire_stale_job(mock_s3, job)

        assert result.status == JobStatus.FAILED
        assert result.completed_at is not None
        assert "timed out" in result.error_message
        assert "crashed" in result.error_message

        stored = json.loads(mock_s3._objects[key])
        assert stored["status"] == "FAILED"

    def test_expires_stale_pending_job(self, mock_s3):
        """expire_stale_job marks a stale PENDING job as FAILED using created_at."""
        stale_time = datetime.now(timezone.utc) - timedelta(
            seconds=_STALE_JOB_TIMEOUT + 60
        )
        job = JobRecord(
            job_id="stale-pending",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.PENDING,
            limit=100,
            offset=0,
            created_at=stale_time,
        )

        key = _metadata_key(job.user, job.job_id)
        mock_s3._objects[key] = _serialize_job(job).encode()

        result = expire_stale_job(mock_s3, job)

        assert result.status == JobStatus.FAILED
        assert "timed out" in result.error_message

    def test_uses_started_at_over_created_at(self, mock_s3):
        """expire_stale_job uses started_at as the reference when available."""
        old_created = datetime.now(timezone.utc) - timedelta(
            seconds=_STALE_JOB_TIMEOUT + 300
        )
        recent_started = datetime.now(timezone.utc) - timedelta(seconds=30)

        job = JobRecord(
            job_id="started-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.RUNNING,
            limit=100,
            offset=0,
            created_at=old_created,
            started_at=recent_started,
        )

        result = expire_stale_job(mock_s3, job)
        assert result.status == JobStatus.RUNNING

    def test_job_just_under_threshold_not_expired(self, mock_s3):
        """Job just under the threshold should NOT be expired."""
        just_under = datetime.now(timezone.utc) - timedelta(
            seconds=_STALE_JOB_TIMEOUT - 30
        )
        job = JobRecord(
            job_id="boundary-job",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.RUNNING,
            limit=100,
            offset=0,
            created_at=just_under - timedelta(seconds=5),
            started_at=just_under,
        )

        result = expire_stale_job(mock_s3, job)
        assert result.status == JobStatus.RUNNING
