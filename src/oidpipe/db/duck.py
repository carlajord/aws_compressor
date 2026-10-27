"""DuckDB connection, schema initialisation, and atomic file replacement."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

_SQL_DIR = Path(__file__).parent / "sql"


def connect(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB database file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only)
    logger.info("Connected to %s (read_only=%s)", db_path, read_only)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Run schema.sql to create tables with COMMENTs."""
    schema_sql = (_SQL_DIR / "schema.sql").read_text()
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    logger.info("Schema initialised")


def run_sql_file(conn: duckdb.DuckDBPyConnection, filename: str) -> None:
    """Execute an arbitrary .sql file from the sql/ directory."""
    sql = (_SQL_DIR / filename).read_text()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    logger.info("Executed %s", filename)


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """Atomically replace final_path with tmp_path via rename."""
    tmp_path.rename(final_path)
    logger.info("Replaced %s → %s", tmp_path, final_path)
