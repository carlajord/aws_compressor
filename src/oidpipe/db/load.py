"""Load raw Parquet files into DuckDB and run data quality checks."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


def load_tags(conn: duckdb.DuckDBPyConnection, parquet_path: Path | str) -> int:
    """Load tags.parquet into the tags table. Accepts local paths or S3 URIs."""
    conn.execute(
        "INSERT INTO tags SELECT * FROM read_parquet(?)",
        [str(parquet_path)],
    )
    count = conn.execute("SELECT count(*) FROM tags").fetchone()[0]
    logger.info("Loaded %d tags", count)
    return count


def load_events(conn: duckdb.DuckDBPyConnection, parquet_path: Path | str) -> int:
    """Load events.parquet into the events table. Accepts local paths or S3 URIs."""
    path_str = str(parquet_path)
    if not path_str.startswith("s3://"):
        import pyarrow.parquet as pq

        schema = pq.read_schema(parquet_path)
        if len(schema) == 0:
            logger.info("Loaded 0 events (empty file)")
            return 0

    conn.execute(
        "INSERT INTO events SELECT * FROM read_parquet(?)",
        [path_str],
    )
    count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    logger.info("Loaded %d events", count)
    return count


def load_readings(conn: duckdb.DuckDBPyConnection, parquet_path: Path | str) -> int:
    """Load readings.parquet into readings_10m, mapping external_id → logical name."""
    conn.execute(
        """
        INSERT INTO readings_10m (ts, tag, value)
        SELECT r.ts, t.logical_name, r.value
        FROM read_parquet(?) r
        JOIN tags t ON r.external_id = t.external_id
        """,
        [str(parquet_path)],
    )
    count = conn.execute("SELECT count(*) FROM readings_10m").fetchone()[0]
    logger.info("Loaded %d readings", count)
    return count


# ---------------------------------------------------------------------------
# Data quality checks (data contract §6)
# ---------------------------------------------------------------------------


class QualityCheckError(Exception):
    """Raised when a hard quality check fails."""


def check_timestamp_grid(conn: duckdb.DuckDBPyConnection) -> int:
    """Verify all timestamps are on the 10-minute grid. Returns violation count."""
    off_grid = conn.execute(
        """
        SELECT count(*) FROM readings_10m
        WHERE EXTRACT(MINUTE FROM ts) % 10 != 0
           OR EXTRACT(SECOND FROM ts) != 0
        """
    ).fetchone()[0]
    if off_grid > 0:
        raise QualityCheckError(f"{off_grid} timestamps are off the 10-minute grid")
    logger.info("Timestamp grid check passed")
    return off_grid


def check_duplicates(conn: duckdb.DuckDBPyConnection) -> int:
    """Verify no duplicate (ts, tag) pairs. Returns violation count."""
    dupes = conn.execute(
        """
        SELECT count(*) FROM (
            SELECT ts, tag, count(*) AS n
            FROM readings_10m
            GROUP BY ts, tag
            HAVING n > 1
        )
        """
    ).fetchone()[0]
    if dupes > 0:
        raise QualityCheckError(f"{dupes} duplicate (ts, tag) pairs found")
    logger.info("Duplicate check passed")
    return dupes


def check_tag_coverage(conn: duckdb.DuckDBPyConnection, min_pct: float = 50.0) -> dict[str, float]:
    """Check each tag has ≥ min_pct non-null coverage. Returns {tag: pct} for failures."""
    total_ts = conn.execute(
        "SELECT count(DISTINCT ts) FROM readings_10m"
    ).fetchone()[0]

    if total_ts == 0:
        raise QualityCheckError("No timestamps in readings_10m")

    coverage = conn.execute(
        """
        SELECT tag,
               count(*) AS n_rows,
               100.0 * count(*) / ? AS pct
        FROM readings_10m
        GROUP BY tag
        ORDER BY pct
        """,
        [total_ts],
    ).fetchall()

    failures: dict[str, float] = {}
    for tag, n_rows, pct in coverage:
        if pct < min_pct:
            failures[tag] = round(pct, 1)
            logger.warning("Tag '%s' has only %.1f%% coverage (%d rows)", tag, pct, n_rows)

    if failures:
        raise QualityCheckError(
            f"{len(failures)} tags below {min_pct}% coverage: {failures}"
        )
    logger.info("Tag coverage check passed (%d tags, all ≥ %.0f%%)", len(coverage), min_pct)
    return failures


def check_physical_sanity(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Run physical sanity checks. Returns {check_name: violation_count}. Logs warnings."""
    violations: dict[str, int] = {}

    checks = [
        (
            "pressure_range",
            """
            SELECT count(*) FROM readings_10m
            WHERE tag IN ('p_suction_barg', 'p_discharge_barg')
              AND (value < 0 OR value > 50)
            """,
        ),
        (
            "temperature_range",
            """
            SELECT count(*) FROM readings_10m
            WHERE tag LIKE 't\\_%' ESCAPE '\\'
              AND (value < -20 OR value > 200)
            """,
        ),
        (
            "valve_range",
            """
            SELECT count(*) FROM readings_10m
            WHERE tag IN ('asv_open_pct', 'stv_open_pct')
              AND (value < 0 OR value > 100)
            """,
        ),
    ]

    for name, sql in checks:
        count = conn.execute(sql).fetchone()[0]
        violations[name] = count
        if count > 0:
            logger.warning("Physical sanity '%s': %d violations", name, count)

    total = sum(violations.values())
    if total == 0:
        logger.info("Physical sanity checks passed")
    else:
        logger.warning("Physical sanity checks: %d total violations", total)
    return violations


def run_quality_checks(
    conn: duckdb.DuckDBPyConnection, min_coverage_pct: float = 50.0
) -> None:
    """Run all data quality checks. Raises QualityCheckError on hard failures."""
    check_timestamp_grid(conn)
    check_duplicates(conn)
    check_tag_coverage(conn, min_pct=min_coverage_pct)
    check_physical_sanity(conn)
    logger.info("All quality checks passed")
