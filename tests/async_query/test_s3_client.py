"""
Tests for the S3/MinIO client utility module.

Tests cover:
- S3 client creation with proper configuration
- Result path building
- Result upload and download
- Result prefix deletion
- Presigned URL generation (with file exclusion)
- .s3keep creation
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.async_query.s3_client import (
    build_query_result_root_path,
    build_result_path,
    create_s3_client,
    create_s3keep,
    delete_result_prefix,
    download_result,
    generate_presigned_urls,
    upload_result,
)


# =============================================================================
# build_result_path Tests
# =============================================================================


class TestBuildResultPath:
    """Tests for result path construction."""

    def test_basic_path(self):
        path = build_result_path("testuser", "job-123")
        assert path == "users-general-warehouse/testuser/data/query_result/job-123/"

    def test_path_with_special_username(self):
        path = build_result_path("tian_gu_test", "abc-def-ghi")
        assert (
            path
            == "users-general-warehouse/tian_gu_test/data/query_result/abc-def-ghi/"
        )

    def test_path_ends_with_slash(self):
        path = build_result_path("user", "job")
        assert path.endswith("/")


class TestBuildQueryResultRootPath:
    """Tests for query result root path construction."""

    def test_basic_path(self):
        path = build_query_result_root_path("testuser")
        assert path == "users-general-warehouse/testuser/data/query_result/"


class TestCreateS3Keep:
    """Tests for .s3keep creation."""

    def test_creates_s3keep_object(self):
        """Creates an empty .s3keep object."""
        mock_client = MagicMock()
        create_s3keep(mock_client, "bucket", "prefix/")

        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "bucket"
        assert call_kwargs["Key"] == "prefix/.s3keep"
        assert call_kwargs["Body"] == b""
        assert call_kwargs["ContentType"] == "application/octet-stream"

    def test_handles_errors_gracefully(self):
        """Does not raise if creation fails."""
        mock_client = MagicMock()
        mock_client.put_object.side_effect = Exception("S3 error")

        # Should not raise
        create_s3keep(mock_client, "bucket", "prefix/")


# =============================================================================
# create_s3_client Tests
# =============================================================================


class TestCreateS3Client:
    """Tests for S3 client creation."""

    @patch("src.async_query.s3_client.boto3")
    def test_create_client_http(self, mock_boto3):
        """Creates client with http:// for non-secure endpoint."""
        create_s3_client("localhost:9002", "access", "secret", secure=False)

        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "http://localhost:9002"
        assert call_kwargs.kwargs["aws_access_key_id"] == "access"
        assert call_kwargs.kwargs["aws_secret_access_key"] == "secret"

    @patch("src.async_query.s3_client.boto3")
    def test_create_client_https(self, mock_boto3):
        """Creates client with https:// for secure endpoint."""
        create_s3_client("minio.example.com", "key", "secret", secure=True)

        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "https://minio.example.com"

    @patch("src.async_query.s3_client.boto3")
    def test_create_client_preserves_existing_scheme(self, mock_boto3):
        """Doesn't double-prefix if endpoint already has scheme."""
        create_s3_client("http://localhost:9002", "key", "secret", secure=False)

        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "http://localhost:9002"


# =============================================================================
# generate_presigned_urls Tests
# =============================================================================


class TestGeneratePresignedUrls:
    """Tests for presigned URL generation."""

    def test_generates_urls_for_data_files(self):
        """Generates presigned URLs for actual data files."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/part-00000.json", "Size": 1024},
            ]
        }
        mock_client.generate_presigned_url.return_value = (
            "https://minio/presigned/part-00000.json"
        )

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/", expires_in=600)

        assert len(urls) == 1
        assert urls[0] == "https://minio/presigned/part-00000.json"
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "bucket", "Key": "prefix/part-00000.json"},
            ExpiresIn=600,
        )

    def test_excludes_spark_internal_files(self):
        """Excludes _SUCCESS, _metadata.json, _committed_, _started_ files."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/_SUCCESS", "Size": 0},
                {"Key": "prefix/_metadata.json", "Size": 512},
                {"Key": "prefix/_committed_123", "Size": 256},
                {"Key": "prefix/_started_123", "Size": 128},
                {"Key": "prefix/part-00000.json", "Size": 2048},
            ]
        }
        mock_client.generate_presigned_url.return_value = "https://url"

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/")

        # Only the data file should get a URL
        assert len(urls) == 1

    def test_excludes_empty_files(self):
        """Excludes zero-byte directory markers."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/", "Size": 0},
                {"Key": "prefix/part-00000.json", "Size": 1024},
            ]
        }
        mock_client.generate_presigned_url.return_value = "https://url"

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/")
        assert len(urls) == 1

    def test_empty_listing(self):
        """Returns empty list when no objects found."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {}

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/")
        assert urls == []

    def test_multiple_data_files(self):
        """Generates URLs for multiple data files."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/part-00000.json", "Size": 1024},
                {"Key": "prefix/part-00001.json", "Size": 2048},
                {"Key": "prefix/_SUCCESS", "Size": 0},
            ]
        }
        mock_client.generate_presigned_url.side_effect = [
            "https://url1",
            "https://url2",
        ]

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/")
        assert len(urls) == 2

    def test_raises_on_s3_error(self):
        """Propagates S3 exceptions."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = Exception("S3 error")

        with pytest.raises(Exception, match="S3 error"):
            generate_presigned_urls(mock_client, "bucket", "prefix/")

    def test_uses_default_expiry(self):
        """Uses ASYNC_QUERY_PRESIGNED_URL_EXPIRY when expires_in is None."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/part-00000.json", "Size": 1024},
            ]
        }
        mock_client.generate_presigned_url.return_value = "https://url"

        generate_presigned_urls(mock_client, "bucket", "prefix/")

        call_kwargs = mock_client.generate_presigned_url.call_args
        assert call_kwargs.kwargs["ExpiresIn"] == 3600  # default

    def test_excludes_s3keep_files(self):
        """Excludes .s3keep directory markers (zero-byte)."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/.s3keep", "Size": 0},
                {"Key": "prefix/part-00000.json", "Size": 1024},
            ]
        }
        mock_client.generate_presigned_url.return_value = "https://url"

        urls = generate_presigned_urls(mock_client, "bucket", "prefix/")
        assert len(urls) == 1


# =============================================================================
# upload_result Tests
# =============================================================================


class TestUploadResult:
    """Tests for uploading query results to S3."""

    def test_upload_json_result(self):
        """Uploads JSON result with correct key and content type."""
        mock_client = MagicMock()
        data = json.dumps([{"id": 1}, {"id": 2}])

        upload_result(mock_client, "cdm-lake", "prefix/", data, result_format="json")

        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "cdm-lake"
        assert call_kwargs["Key"] == "prefix/result.json"
        assert call_kwargs["Body"] == data.encode("utf-8")
        assert call_kwargs["ContentType"] == "application/json"

    def test_upload_parquet_result(self):
        """Uploads parquet result with octet-stream content type."""
        mock_client = MagicMock()

        upload_result(
            mock_client, "cdm-lake", "prefix/", "data", result_format="parquet"
        )

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "prefix/result.parquet"
        assert call_kwargs["ContentType"] == "application/octet-stream"

    def test_upload_default_format_is_json(self):
        """Default result_format is json."""
        mock_client = MagicMock()

        upload_result(mock_client, "cdm-lake", "prefix/", "[]")

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "prefix/result.json"

    def test_upload_propagates_s3_error(self):
        """Propagates S3 exceptions on upload failure."""
        mock_client = MagicMock()
        mock_client.put_object.side_effect = Exception("S3 write error")

        with pytest.raises(Exception, match="S3 write error"):
            upload_result(mock_client, "cdm-lake", "prefix/", "[]")


# =============================================================================
# download_result Tests
# =============================================================================


class TestDownloadResult:
    """Tests for downloading query results from S3."""

    def test_download_json_result(self):
        """Downloads and parses JSON result."""
        mock_client = MagicMock()
        result_data = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        body_stream = MagicMock()
        body_stream.read.return_value = json.dumps(result_data).encode("utf-8")
        mock_client.get_object.return_value = {"Body": body_stream}

        result = download_result(
            mock_client, "cdm-lake", "prefix/", result_format="json"
        )

        assert result == result_data
        mock_client.get_object.assert_called_once_with(
            Bucket="cdm-lake", Key="prefix/result.json"
        )

    def test_download_default_format_is_json(self):
        """Default result_format is json."""
        mock_client = MagicMock()
        body_stream = MagicMock()
        body_stream.read.return_value = b"[]"
        mock_client.get_object.return_value = {"Body": body_stream}

        download_result(mock_client, "cdm-lake", "prefix/")

        mock_client.get_object.assert_called_once_with(
            Bucket="cdm-lake", Key="prefix/result.json"
        )

    def test_download_parquet_key(self):
        """Uses .parquet extension for parquet format."""
        mock_client = MagicMock()
        body_stream = MagicMock()
        body_stream.read.return_value = b"[]"
        mock_client.get_object.return_value = {"Body": body_stream}

        download_result(mock_client, "cdm-lake", "prefix/", result_format="parquet")

        mock_client.get_object.assert_called_once_with(
            Bucket="cdm-lake", Key="prefix/result.parquet"
        )

    def test_download_empty_result(self):
        """Handles empty result list."""
        mock_client = MagicMock()
        body_stream = MagicMock()
        body_stream.read.return_value = b"[]"
        mock_client.get_object.return_value = {"Body": body_stream}

        result = download_result(mock_client, "cdm-lake", "prefix/")
        assert result == []

    def test_download_propagates_s3_error(self):
        """Propagates S3 exceptions on download failure."""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("S3 read error")

        with pytest.raises(Exception, match="S3 read error"):
            download_result(mock_client, "cdm-lake", "prefix/")


# =============================================================================
# delete_result_prefix Tests
# =============================================================================


class TestDeleteResultPrefix:
    """Tests for deleting all objects under a result prefix."""

    def test_deletes_all_objects(self):
        """Deletes all objects under the prefix."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/result.json"},
                {"Key": "prefix/_metadata.json"},
                {"Key": "prefix/.s3keep"},
            ]
        }

        delete_result_prefix(mock_client, "cdm-lake", "prefix/")

        mock_client.delete_objects.assert_called_once()
        call_kwargs = mock_client.delete_objects.call_args.kwargs
        assert call_kwargs["Bucket"] == "cdm-lake"
        deleted_keys = [obj["Key"] for obj in call_kwargs["Delete"]["Objects"]]
        assert "prefix/result.json" in deleted_keys
        assert "prefix/_metadata.json" in deleted_keys
        assert "prefix/.s3keep" in deleted_keys

    def test_no_objects_to_delete(self):
        """Does not call delete_objects when prefix is empty."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}

        delete_result_prefix(mock_client, "cdm-lake", "prefix/")

        mock_client.delete_objects.assert_not_called()

    def test_no_contents_key(self):
        """Handles response with no Contents key."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {}

        delete_result_prefix(mock_client, "cdm-lake", "prefix/")

        mock_client.delete_objects.assert_not_called()

    def test_handles_s3_error_gracefully(self):
        """Does not raise on S3 failure (logs instead)."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = Exception("S3 error")

        # Should not raise
        delete_result_prefix(mock_client, "cdm-lake", "prefix/")

    def test_single_object_delete(self):
        """Handles prefix with a single object."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "prefix/result.json"}]
        }

        delete_result_prefix(mock_client, "cdm-lake", "prefix/")

        call_kwargs = mock_client.delete_objects.call_args.kwargs
        assert len(call_kwargs["Delete"]["Objects"]) == 1
