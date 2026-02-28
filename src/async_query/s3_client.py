"""
S3/MinIO client utilities for async query result storage.

Provides functions for:
- Creating boto3 S3 clients configured for MinIO
- Generating presigned URLs for result files
- Writing job metadata alongside results
- Building user-scoped result paths
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

ASYNC_QUERY_RESULT_BUCKET = os.getenv("ASYNC_QUERY_RESULT_BUCKET", "cdm-lake")
ASYNC_QUERY_PRESIGNED_URL_EXPIRY = int(
    os.getenv("ASYNC_QUERY_PRESIGNED_URL_EXPIRY", "3600")
)

# Files to exclude when generating presigned URLs for results
_EXCLUDED_FILES = {"_SUCCESS", "_metadata.json", "_committed_", "_started_"}


def create_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    secure: bool = False,
) -> Any:
    """
    Create a boto3 S3 client configured for MinIO.

    Args:
        endpoint_url: MinIO endpoint (hostname:port or full URL).
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        secure: Whether to use HTTPS.

    Returns:
        A configured boto3 S3 client.
    """
    scheme = "https" if secure else "http"
    # Normalize endpoint URL
    if not endpoint_url.startswith(("http://", "https://")):
        full_url = f"{scheme}://{endpoint_url}"
    else:
        full_url = endpoint_url

    return boto3.client(
        "s3",
        endpoint_url=full_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # Required by boto3 but unused by MinIO
    )


def build_result_path(user: str, job_id: str) -> str:
    """
    Build the S3 key prefix for a job's results.

    Args:
        user: KBase username.
        job_id: Unique job identifier.

    Returns:
        S3 key prefix (without bucket name):
        users-general-warehouse/{user}/data/query_result/{job_id}/
    """
    return f"users-general-warehouse/{user}/data/query_result/{job_id}/"


def build_query_result_root_path(user: str) -> str:
    """
    Build the S3 key prefix for the user's query result root directory.

    Args:
        user: KBase username.

    Returns:
        S3 key prefix (without bucket name):
        users-general-warehouse/{user}/data/query_result/
    """
    return f"users-general-warehouse/{user}/data/query_result/"


def create_s3keep(
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> None:
    """
    Create an empty .s3keep file at the specified prefix.

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix where the .s3keep file should be created.
    """
    key = f"{prefix}.s3keep"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=b"",
            ContentType="application/octet-stream",
        )
        logger.info(f"Created .s3keep at s3://{bucket}/{key}")
    except Exception:
        logger.exception(f"Failed to create .s3keep at s3://{bucket}/{key}")


def generate_presigned_urls(
    s3_client: Any,
    bucket: str,
    prefix: str,
    expires_in: int | None = None,
) -> list[str]:
    """
    List objects under a prefix and generate presigned URLs for result files.

    Excludes internal Spark files like _SUCCESS and _metadata.json.

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix to list objects under.
        expires_in: Presigned URL expiry in seconds.

    Returns:
        List of presigned URLs for result data files.
    """
    if expires_in is None:
        expires_in = ASYNC_QUERY_PRESIGNED_URL_EXPIRY

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])

        urls = []
        for obj in contents:
            key = obj["Key"]
            filename = key.rsplit("/", 1)[-1] if "/" in key else key

            # Skip internal Spark/metadata files
            if any(filename.startswith(exc) for exc in _EXCLUDED_FILES):
                continue
            # Skip empty directory markers
            if obj.get("Size", 0) == 0:
                continue

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            urls.append(url)

        logger.info(
            f"Generated {len(urls)} presigned URLs for bucket={bucket} prefix={prefix}"
        )
        return urls
    except Exception:
        logger.exception(
            f"Failed to generate presigned URLs for bucket={bucket} prefix={prefix}"
        )
        raise


def upload_result(
    s3_client: Any,
    bucket: str,
    prefix: str,
    data: str,
    result_format: str = "json",
) -> None:
    """
    Upload query result data to S3 as a single file.

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix (result directory).
        data: Serialized result data (JSON string).
        result_format: Output format ("json" or "parquet").
    """
    key = f"{prefix}result.{result_format}"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data.encode("utf-8"),
        ContentType="application/json"
        if result_format == "json"
        else "application/octet-stream",
    )
    logger.info(f"Uploaded result to s3://{bucket}/{key}")


def download_result(
    s3_client: Any,
    bucket: str,
    prefix: str,
    result_format: str = "json",
) -> list[dict[str, Any]]:
    """
    Download query result data from S3 and return as a list of dicts.

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix (result directory).
        result_format: Output format ("json" or "parquet").

    Returns:
        List of row dictionaries (the query result).
    """
    key = f"{prefix}result.{result_format}"
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def delete_result_prefix(
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> None:
    """
    Delete all objects under a job's result prefix.

    Args:
        s3_client: A configured boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Key prefix (result directory) to delete.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            logger.info(f"No objects to delete at s3://{bucket}/{prefix}")
            return

        objects = [{"Key": obj["Key"]} for obj in contents]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects},
        )
        logger.info(f"Deleted {len(objects)} objects from s3://{bucket}/{prefix}")
    except Exception:
        logger.exception(f"Failed to delete objects at s3://{bucket}/{prefix}")
