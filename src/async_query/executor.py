"""
Async query executor for background Spark query execution.

Manages background asyncio tasks that execute queries using the shared
execute_query function (same logic as the sync query_table endpoint),
then write results to S3.

Timeouts are controlled by the same settings as the sync path:
  - SPARK_CONNECT_QUERY_TIMEOUT: for Connect-mode queries
  - SPARK_STANDALONE_QUERY_TIMEOUT: for Standalone-mode queries
There is no separate async timeout — async and sync share the same
mode-based defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from src.async_query import job_store, s3_client
from src.service.dependencies import SparkContext
from src.service.models import MAX_ASYNC_QUERY_ROWS, JobStatus, QueryEngine
from src.service.query_executor import execute_query, execute_query_trino
from src.trino_engine.trino_connection import create_trino_connection

logger = logging.getLogger(__name__)


class AsyncQueryExecutor:
    """
    Manages background async query execution.

    Uses the shared execute_query function — the same code path as the
    sync query_table endpoint. Results are written to S3 via boto3 and
    job status is tracked in S3 _metadata.json files.
    """

    def __init__(self) -> None:
        self._active_tasks: dict[str, asyncio.Task] = {}
        logger.info("AsyncQueryExecutor initialized")

    async def submit_query(
        self,
        job_id: str,
        user: str,
        query: str,
        limit: int,
        offset: int,
        ctx: SparkContext,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_secure: bool,
        engine: QueryEngine = QueryEngine.SPARK,
        auth_token: str | None = None,
        trino_settings: dict | None = None,
    ) -> None:
        """
        Submit a query for background execution.

        Args:
            job_id: Unique job identifier.
            user: KBase username.
            query: SQL query string.
            limit: Max rows to return.
            offset: Pagination offset.
            ctx: SparkContext with session and mode info.
            minio_endpoint: MinIO endpoint URL.
            minio_access_key: MinIO access key for the user.
            minio_secret_key: MinIO secret key for the user.
            minio_secure: Whether to use HTTPS for MinIO.
            engine: Query engine to use (spark or trino).
            auth_token: KBase auth token (required for Trino connections).
            trino_settings: Trino-specific settings (TRINO_HOST, TRINO_PORT,
                BERDL_HIVE_METASTORE_URI). Required when engine is TRINO.
        """
        task = asyncio.create_task(
            self._execute_query(
                job_id,
                user,
                query,
                limit,
                offset,
                ctx,
                minio_endpoint,
                minio_access_key,
                minio_secret_key,
                minio_secure,
                engine,
                auth_token,
                trino_settings,
            )
        )
        self._active_tasks[job_id] = task
        logger.info(
            f"Submitted async query: job_id={job_id} user={user} engine={engine.value}"
        )

    async def _execute_query(
        self,
        job_id: str,
        user: str,
        query: str,
        limit: int,
        offset: int,
        ctx: SparkContext,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_secure: bool,
        engine: QueryEngine = QueryEngine.SPARK,
        auth_token: str | None = None,
        trino_settings: dict | None = None,
    ) -> None:
        """Background coroutine that orchestrates query execution."""
        bucket = s3_client.ASYNC_QUERY_RESULT_BUCKET
        # Create S3 client using user's MinIO credentials
        client = s3_client.create_s3_client(
            minio_endpoint, minio_access_key, minio_secret_key, minio_secure
        )
        try:
            # 1. Update status to RUNNING
            await asyncio.to_thread(
                job_store.update_job_status,
                client,
                job_id,
                JobStatus.RUNNING,
                user=user,
                started_at=datetime.now(timezone.utc),
            )

            # 2. Ensure directory structure exists
            result_prefix = s3_client.build_result_path(user, job_id)
            root_prefix = s3_client.build_query_result_root_path(user)
            await asyncio.to_thread(
                s3_client.create_s3keep, client, bucket, result_prefix
            )
            await asyncio.to_thread(
                s3_client.create_s3keep, client, bucket, root_prefix
            )

            # 3. Execute query using the selected engine
            if engine == QueryEngine.TRINO:
                response = await self._execute_trino_query(
                    user=user,
                    query=query,
                    limit=limit,
                    offset=offset,
                    auth_token=auth_token,
                    minio_access_key=minio_access_key,
                    minio_secret_key=minio_secret_key,
                    minio_endpoint=minio_endpoint,
                    minio_secure=minio_secure,
                    trino_settings=trino_settings,
                )
            else:
                response = await asyncio.to_thread(
                    execute_query,
                    ctx,
                    query,
                    limit,
                    offset,
                    user,
                    max_rows=MAX_ASYNC_QUERY_ROWS,
                )

            # 4. Write result to S3 via boto3
            result_json = json.dumps(response.result, default=str)
            await asyncio.to_thread(
                s3_client.upload_result, client, bucket, result_prefix, result_json
            )

            # 5. Update status to SUCCEEDED
            await asyncio.to_thread(
                job_store.update_job_status,
                client,
                job_id,
                JobStatus.SUCCEEDED,
                user=user,
                completed_at=datetime.now(timezone.utc),
                row_count=len(response.result),
                total_count=response.pagination.total_count,
                has_more=response.pagination.has_more,
                result_path=f"s3://{bucket}/{result_prefix}",
            )

            logger.info(
                f"Async query succeeded: job_id={job_id} "
                f"rows={len(response.result)} "
                f"total={response.pagination.total_count}"
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Async query failed: job_id={job_id} error={error_msg}")
            await asyncio.to_thread(
                job_store.update_job_status,
                client,
                job_id,
                JobStatus.FAILED,
                user=user,
                completed_at=datetime.now(timezone.utc),
                error_message=error_msg,
            )

        finally:
            self._active_tasks.pop(job_id, None)

    async def _execute_trino_query(
        self,
        user: str,
        query: str,
        limit: int,
        offset: int,
        auth_token: str | None,
        minio_access_key: str,
        minio_secret_key: str,
        minio_endpoint: str,
        minio_secure: bool,
        trino_settings: dict | None,
    ):
        """Create a fresh Trino connection and execute the query."""
        if not trino_settings:
            raise ValueError("trino_settings required for Trino engine")
        if not auth_token:
            raise ValueError("auth_token required for Trino engine")

        conn = await asyncio.to_thread(
            create_trino_connection,
            username=user,
            auth_token=auth_token,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            trino_host=trino_settings["TRINO_HOST"],
            trino_port=trino_settings["TRINO_PORT"],
            hive_metastore_uri=trino_settings["BERDL_HIVE_METASTORE_URI"],
            minio_endpoint_url=minio_endpoint,
            minio_secure=minio_secure,
        )
        try:
            return await asyncio.to_thread(
                execute_query_trino,
                conn,
                query,
                limit,
                offset,
                user,
                max_rows=MAX_ASYNC_QUERY_ROWS,
            )
        finally:
            conn.close()

    async def shutdown(self) -> None:
        """
        Graceful shutdown: cancel all active tasks and await their completion.

        Takes a snapshot of _active_tasks before iterating to avoid
        RuntimeError from concurrent dict mutation in _execute_query's
        finally block.
        """
        if not self._active_tasks:
            logger.info("No active async query tasks to cancel")
            return

        tasks_snapshot = list(self._active_tasks.items())
        logger.info(
            f"Shutting down AsyncQueryExecutor ({len(tasks_snapshot)} active tasks)"
        )
        for job_id, task in tasks_snapshot:
            logger.info(f"Cancelling async query task: job_id={job_id}")
            task.cancel()

        await asyncio.gather(
            *(task for _, task in tasks_snapshot), return_exceptions=True
        )
        self._active_tasks.clear()
        logger.info("AsyncQueryExecutor shutdown complete")
