"""
Tests for the async query API routes.

Tests cover:
- POST /submit: submission, validation, 202 response
- GET /{job_id}/status: status retrieval, access control
- GET /{job_id}/results: inline result data, 409 not ready
- GET /jobs: list user's jobs
- User isolation (403 for wrong user)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routes.async_query import _validate_job_access, router
from src.async_query.executor import AsyncQueryExecutor
from src.service.dependencies import SparkContext, auth, get_spark_context
from src.service.exception_handlers import universal_error_handler
from src.service.exceptions import JobAccessDeniedError, SparkQueryError
from src.service.kb_auth import AdminPermission, KBaseUser
from src.service.models import (
    JobRecord,
    JobStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return KBaseUser(user="testuser", admin_perm=AdminPermission.NONE)


@pytest.fixture
def mock_executor():
    """Create a mock AsyncQueryExecutor."""
    exe = MagicMock(spec=AsyncQueryExecutor)
    # Make submit_query an async mock
    exe.submit_query = AsyncMock()
    return exe


@pytest.fixture
def mock_spark_ctx():
    """Create a mock SparkContext for get_spark_context dependency override."""
    return SparkContext(
        spark=None,
        is_standalone_subprocess=False,
        settings_dict={
            "USER": "testuser",
            "MINIO_ACCESS_KEY": "access_key",
            "MINIO_SECRET_KEY": "secret_key",
            "MINIO_ENDPOINT_URL": "localhost:9002",
            "MINIO_SECURE": False,
            "SPARK_HOME": "/usr/local/spark",
            "SPARK_MASTER_URL": None,
            "SPARK_CONNECT_URL": "sc://spark-connect:15002",
            "BERDL_HIVE_METASTORE_URI": "thrift://localhost:9083",
            "SPARK_WORKER_COUNT": 1,
            "SPARK_WORKER_CORES": 1,
            "SPARK_WORKER_MEMORY": "2GiB",
            "SPARK_MASTER_CORES": 1,
            "SPARK_MASTER_MEMORY": "1GiB",
            "GOVERNANCE_API_URL": "http://localhost:8000",
            "BERDL_POD_IP": "127.0.0.1",
        },
        app_name="datalake_mcp_server_testuser",
        username="testuser",
        auth_token="fake-token",
    )


@pytest.fixture
def sample_succeeded_job():
    """A completed job record for testing."""
    return JobRecord(
        job_id="job-succeed-1",
        user="testuser",
        query="SELECT * FROM db.table",
        status=JobStatus.SUCCEEDED,
        limit=1000,
        offset=0,
        created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        started_at=datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 15, 10, 0, 5, tzinfo=timezone.utc),
        row_count=42,
        total_count=100,
        has_more=True,
        result_path="s3a://cdm-lake/prefix/",
    )


@pytest.fixture
def sample_pending_job():
    """A pending job record for testing."""
    return JobRecord(
        job_id="job-pending-1",
        user="testuser",
        query="SELECT * FROM db.table",
        status=JobStatus.PENDING,
        limit=1000,
        offset=0,
        created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def app(mock_user, mock_executor, mock_spark_ctx):
    """Create a FastAPI app with mocked dependencies for testing."""
    test_app = FastAPI()
    test_app.add_exception_handler(Exception, universal_error_handler)
    test_app.include_router(router)

    # Store mock executor in app state
    test_app.state.async_query_executor = mock_executor

    # Override auth dependency
    def mock_auth():
        return mock_user

    # Override get_spark_context to return our mock context
    def mock_get_spark_context():
        yield mock_spark_ctx

    test_app.dependency_overrides[auth] = mock_auth
    test_app.dependency_overrides[get_spark_context] = mock_get_spark_context

    return test_app


@pytest.fixture
def client(app):
    """Create a TestClient that returns error responses instead of raising."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Helper to patch request user extraction
# =============================================================================


def _patch_get_user(username="testuser"):
    """Patch get_user_from_request to return given username."""
    return patch(
        "src.routes.async_query.get_user_from_request",
        return_value=username,
    )


# =============================================================================
# POST /submit Tests
# =============================================================================


class TestSubmitAsyncQuery:
    """Tests for the submit endpoint."""

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query._check_query_is_valid")
    @patch("src.routes.async_query.job_store")
    def test_submit_returns_202(
        self,
        mock_job_store,
        mock_check_query,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Successful submission returns 202 with job_id."""
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.count_active_user_jobs.return_value = 0

        response = client.post(
            "/delta/tables/query/async/submit",
            json={
                "query": "SELECT * FROM db.table",
                "limit": 500,
                "offset": 0,
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query._check_query_is_valid")
    @patch("src.routes.async_query.job_store")
    def test_submit_creates_job_record(
        self,
        mock_job_store,
        mock_check_query,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Submission creates a PENDING job record in the store."""
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.count_active_user_jobs.return_value = 0

        client.post(
            "/delta/tables/query/async/submit",
            json={"query": "SELECT 1"},
        )

        mock_job_store.create_job.assert_called_once()
        # create_job(client, job) — job is the second positional arg
        job_arg = mock_job_store.create_job.call_args.args[1]
        assert job_arg.status == JobStatus.PENDING
        assert job_arg.user == "testuser"

    @patch("src.routes.async_query._check_query_is_valid")
    def test_submit_invalid_query(self, mock_check_query, client):
        """Invalid query returns error."""
        mock_check_query.side_effect = SparkQueryError("DROP is not allowed")

        response = client.post(
            "/delta/tables/query/async/submit",
            json={"query": "DROP TABLE db.table"},
        )

        assert response.status_code >= 400

    def test_submit_limit_exceeds_max(self, client):
        """Limit exceeding MAX_ASYNC_QUERY_ROWS returns 422."""
        response = client.post(
            "/delta/tables/query/async/submit",
            json={"query": "SELECT 1", "limit": 999999},
        )

        assert response.status_code == 422


# =============================================================================
# GET /{job_id}/status Tests
# =============================================================================


class TestGetAsyncQueryStatus:
    """Tests for the status endpoint."""

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_status_returns_job(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
        sample_succeeded_job,
    ):
        """Status endpoint returns job metadata."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.get_job.return_value = sample_succeeded_job
        mock_job_store.expire_stale_job.side_effect = lambda client, job: job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/job-succeed-1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-succeed-1"
        assert data["status"] == "SUCCEEDED"
        assert data["row_count"] == 42

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_status_not_found(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Status for nonexistent job returns 404."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.get_job.return_value = None

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/nonexistent/status")

        assert response.status_code == 404

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_status_access_denied(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Status for another user's job returns 403."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        other_user_job = JobRecord(
            job_id="other-job",
            user="otheruser",
            query="SELECT 1",
            status=JobStatus.PENDING,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        mock_job_store.get_job.return_value = other_user_job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/other-job/status")

        assert response.status_code == 403


# =============================================================================
# GET /{job_id}/results Tests
# =============================================================================


class TestGetAsyncQueryResults:
    """Tests for the results endpoint."""

    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.job_store")
    def test_results_returns_inline_data(
        self,
        mock_job_store,
        mock_s3,
        mock_read_creds,
        mock_get_settings,
        client,
        sample_succeeded_job,
    ):
        """Results endpoint returns inline data (same format as sync endpoint)."""
        mock_job_store.get_job.return_value = sample_succeeded_job
        mock_job_store.expire_stale_job.side_effect = lambda client, job: job
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings

        mock_s3.create_s3_client.return_value = MagicMock()
        mock_s3.build_result_path.return_value = "prefix/"
        mock_s3.ASYNC_QUERY_RESULT_BUCKET = "cdm-lake"
        mock_s3.download_result.return_value = [{"id": 1}, {"id": 2}]

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/job-succeed-1/results")

        assert response.status_code == 200
        data = response.json()
        assert data["result"] == [{"id": 1}, {"id": 2}]
        assert data["pagination"]["total_count"] == 100
        assert data["pagination"]["has_more"] is True
        assert data["pagination"]["limit"] == 1000
        assert data["pagination"]["offset"] == 0

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_results_not_ready(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
        sample_pending_job,
    ):
        """Results for non-completed job returns 409."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.get_job.return_value = sample_pending_job
        mock_job_store.expire_stale_job.side_effect = lambda client, job: job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/job-pending-1/results")

        assert response.status_code == 409

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_results_not_found(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Results for nonexistent job returns 404."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.get_job.return_value = None

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/nonexistent/results")

        assert response.status_code == 404

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_results_access_denied(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """Results for another user's job returns 403."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        other_user_job = JobRecord(
            job_id="other-job",
            user="otheruser",
            query="SELECT 1",
            status=JobStatus.SUCCEEDED,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
            row_count=10,
        )
        mock_job_store.get_job.return_value = other_user_job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/other-job/results")

        assert response.status_code == 403


# =============================================================================
# GET /jobs Tests
# =============================================================================


class TestListAsyncQueryJobs:
    """Tests for the list jobs endpoint."""

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_list_jobs_returns_user_jobs(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """List endpoint returns all jobs for the user."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        jobs = [
            JobRecord(
                job_id=f"job-{i}",
                user="testuser",
                query=f"SELECT {i}",
                status=JobStatus.PENDING,
                limit=100,
                offset=0,
                created_at=datetime(2025, 1, 15, i, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(3)
        ]
        mock_job_store.list_user_jobs.return_value = jobs
        mock_job_store.expire_stale_job.side_effect = lambda client, job: job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    @patch("src.routes.async_query.s3_client")
    @patch("src.routes.async_query.get_settings")
    @patch("src.routes.async_query.fetch_user_minio_credentials")
    @patch("src.routes.async_query.job_store")
    def test_list_jobs_empty(
        self,
        mock_job_store,
        mock_read_creds,
        mock_get_settings,
        mock_s3,
        client,
    ):
        """List endpoint returns empty list when user has no jobs."""
        mock_read_creds.return_value = ("access_key", "secret_key")
        mock_settings = MagicMock()
        mock_settings.MINIO_ENDPOINT_URL = "localhost:9002"
        mock_settings.MINIO_SECURE = False
        mock_get_settings.return_value = mock_settings
        mock_s3.create_s3_client.return_value = MagicMock()
        mock_job_store.list_user_jobs.return_value = []
        mock_job_store.expire_stale_job.side_effect = lambda client, job: job

        with _patch_get_user("testuser"):
            response = client.get("/delta/tables/query/async/jobs")

        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# _validate_job_access Tests
# =============================================================================


class TestValidateJobAccess:
    """Tests for the job access validation helper."""

    def test_same_user_passes(self):
        """No exception when user matches job owner."""
        job = JobRecord(
            job_id="job-1",
            user="testuser",
            query="SELECT 1",
            status=JobStatus.PENDING,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        # Should not raise
        _validate_job_access(job, "testuser")

    def test_different_user_raises(self):
        """Raises JobAccessDeniedError for different user."""
        job = JobRecord(
            job_id="job-1",
            user="owner",
            query="SELECT 1",
            status=JobStatus.PENDING,
            limit=100,
            offset=0,
            created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        with pytest.raises(JobAccessDeniedError):
            _validate_job_access(job, "intruder")
