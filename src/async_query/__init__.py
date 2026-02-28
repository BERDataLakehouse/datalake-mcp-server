"""
Async query execution subsystem.

Provides background query execution with S3/MinIO-backed job tracking
and result storage. Components:

- executor: Background task orchestration (AsyncQueryExecutor)
- job_store: Job metadata CRUD via S3 _metadata.json files
- s3_client: S3/MinIO operations (upload, download, presigned URLs)
"""
