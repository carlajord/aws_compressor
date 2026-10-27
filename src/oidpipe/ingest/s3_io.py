"""S3 upload/download helpers for the pipeline and eval results."""

from __future__ import annotations

import logging
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def get_client(region: str = "us-east-1"):
    """Return a boto3 S3 client for the given region."""
    return boto3.client("s3", region_name=region)


def upload_file(
    local_path: Path,
    bucket: str,
    s3_key: str,
    *,
    region: str = "us-east-1",
) -> str:
    """Upload a local file to S3. Returns the s3:// URI."""
    client = get_client(region)
    client.upload_file(str(local_path), bucket, s3_key)
    uri = f"s3://{bucket}/{s3_key}"
    logger.info("Uploaded %s → %s", local_path, uri)
    return uri


def list_keys(
    bucket: str,
    prefix: str,
    *,
    region: str = "us-east-1",
) -> list[str]:
    """List object keys under a prefix."""
    client = get_client(region)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_uri(bucket: str, key: str) -> str:
    """Build an s3:// URI."""
    return f"s3://{bucket}/{key}"
