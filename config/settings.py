"""Application settings loaded from environment variables (prefix OID_)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    AWS = "aws"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OID_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Environment.LOCAL
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    raw_prefix: str = "raw/"
    db_path: Path = Path("data/oid.duckdb")
    traces_db_path: Path = Path("data/traces.duckdb")
    llm_backend: str = "openai"  # "openai" or "bedrock"
    llm_model: str = "openai.gpt-oss-120b"
    agent_max_steps: int = Field(default=8, ge=1, le=20)
    sql_row_limit: int = Field(default=500, ge=1)
    zscore_threshold: float = Field(default=5.0, gt=0)
    iforest_contamination: float = Field(default=0.01, gt=0, lt=0.5)

    # Cognite SDK (extraction only, laptop)
    cognite_project: str = "publicdata"
    cognite_client_id: str = ""
    cognite_tenant_id: str = ""
    cognite_client_secret: str = ""
    cognite_cdf_cluster: str = "api"

    @property
    def is_local(self) -> bool:
        return self.env == Environment.LOCAL


def get_settings(**overrides: object) -> Settings:
    """Build a Settings instance, accepting keyword overrides for testing."""
    return Settings(**overrides)  # type: ignore[arg-type]
