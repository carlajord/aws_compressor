"""Tests for pipeline S3 mode helpers."""

from __future__ import annotations

from unittest.mock import patch

import duckdb
import pytest

from config.settings import get_settings
from oidpipe.pipeline import _resolve_raw_sources, _setup_s3_access


class TestResolveRawSources:
    def test_local_mode(self, tmp_path):
        settings = get_settings(env="local")
        with patch("oidpipe.pipeline.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.__truediv__ = lambda self, other: tmp_path / other
            sources = _resolve_raw_sources(settings)

        assert "tags" in sources
        assert "events" in sources
        assert "readings" in sources
        assert not sources["tags"].startswith("s3://")

    def test_aws_mode(self):
        settings = get_settings(env="aws", s3_bucket="my-bucket", raw_prefix="raw/")
        sources = _resolve_raw_sources(settings)

        assert sources["tags"] == "s3://my-bucket/raw/tags.parquet"
        assert sources["events"] == "s3://my-bucket/raw/events.parquet"
        assert sources["readings"] == "s3://my-bucket/raw/readings.parquet"

    def test_aws_mode_trailing_slash_stripped(self):
        settings = get_settings(env="aws", s3_bucket="bkt", raw_prefix="data/raw/")
        sources = _resolve_raw_sources(settings)

        assert sources["tags"] == "s3://bkt/data/raw/tags.parquet"


class TestSetupS3Access:
    def test_installs_httpfs(self):
        conn = duckdb.connect(":memory:")
        _setup_s3_access(conn, "us-east-1")
        result = conn.execute(
            "SELECT * FROM duckdb_extensions() WHERE extension_name = 'httpfs'"
        ).fetchone()
        assert result is not None
        conn.close()
