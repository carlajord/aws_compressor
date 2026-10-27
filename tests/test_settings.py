"""Tests for config/settings.py — defaults and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from config.settings import Environment, Settings, get_settings


class TestDefaults:
    def test_default_env_is_local(self):
        s = get_settings()
        assert s.env == Environment.LOCAL
        assert s.is_local is True

    def test_default_paths(self):
        s = get_settings()
        assert s.db_path == Path("data/oid.duckdb")
        assert s.traces_db_path == Path("data/traces.duckdb")

    def test_default_agent_params(self):
        s = get_settings()
        assert s.agent_max_steps == 5
        assert s.sql_row_limit == 500

    def test_default_anomaly_params(self):
        s = get_settings()
        assert s.zscore_threshold == 4.0
        assert s.iforest_contamination == 0.01


class TestEnvOverrides:
    def test_env_var_overrides_env(self):
        with patch.dict(os.environ, {"OID_ENV": "aws"}):
            s = Settings()
        assert s.env == Environment.AWS
        assert s.is_local is False

    def test_env_var_overrides_region(self):
        with patch.dict(os.environ, {"OID_AWS_REGION": "eu-west-1"}):
            s = Settings()
        assert s.aws_region == "eu-west-1"

    def test_env_var_overrides_db_path(self):
        with patch.dict(os.environ, {"OID_DB_PATH": "/tmp/test.duckdb"}):
            s = Settings()
        assert s.db_path == Path("/tmp/test.duckdb")

    def test_env_var_overrides_numeric(self):
        with patch.dict(
            os.environ,
            {
                "OID_AGENT_MAX_STEPS": "10",
                "OID_SQL_ROW_LIMIT": "100",
                "OID_ZSCORE_THRESHOLD": "3.5",
                "OID_IFOREST_CONTAMINATION": "0.05",
            },
        ):
            s = Settings()
        assert s.agent_max_steps == 10
        assert s.sql_row_limit == 100
        assert s.zscore_threshold == 3.5
        assert s.iforest_contamination == 0.05

    def test_env_var_overrides_s3_bucket(self):
        with patch.dict(os.environ, {"OID_S3_BUCKET": "my-test-bucket"}):
            s = Settings()
        assert s.s3_bucket == "my-test-bucket"


class TestKeywordOverrides:
    def test_get_settings_with_overrides(self):
        s = get_settings(env=Environment.AWS, aws_region="ap-southeast-1")
        assert s.env == Environment.AWS
        assert s.aws_region == "ap-southeast-1"
