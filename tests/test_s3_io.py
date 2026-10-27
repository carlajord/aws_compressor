"""Tests for s3_io upload/list helpers using moto."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from oidpipe.ingest.s3_io import list_keys, s3_uri, upload_file

BUCKET = "test-oid-bucket"
REGION = "us-east-1"


@pytest.fixture()
def s3():
    """Create a mocked S3 bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestUploadFile:
    def test_upload_and_verify(self, s3, tmp_path):
        local = tmp_path / "test.txt"
        local.write_text("hello")

        uri = upload_file(local, BUCKET, "data/test.txt", region=REGION)

        assert uri == f"s3://{BUCKET}/data/test.txt"
        body = s3.get_object(Bucket=BUCKET, Key="data/test.txt")["Body"].read()
        assert body == b"hello"

    def test_upload_binary(self, s3, tmp_path):
        local = tmp_path / "data.bin"
        local.write_bytes(b"\x00\x01\x02")

        upload_file(local, BUCKET, "bin/data.bin", region=REGION)

        body = s3.get_object(Bucket=BUCKET, Key="bin/data.bin")["Body"].read()
        assert body == b"\x00\x01\x02"


class TestListKeys:
    def test_list_empty_prefix(self, s3):
        keys = list_keys(BUCKET, "nonexistent/", region=REGION)
        assert keys == []

    def test_list_with_objects(self, s3):
        s3.put_object(Bucket=BUCKET, Key="raw/tags.parquet", Body=b"t")
        s3.put_object(Bucket=BUCKET, Key="raw/readings.parquet", Body=b"r")
        s3.put_object(Bucket=BUCKET, Key="other/file.txt", Body=b"o")

        keys = list_keys(BUCKET, "raw/", region=REGION)

        assert sorted(keys) == ["raw/readings.parquet", "raw/tags.parquet"]


class TestS3Uri:
    def test_format(self):
        assert s3_uri("mybucket", "path/file.db") == "s3://mybucket/path/file.db"
