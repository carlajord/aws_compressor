"""End-to-end pipeline: raw data → DuckDB → features → anomalies."""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from config.settings import Environment, get_settings
from oidpipe.anomaly.detect import run_anomaly_detection
from oidpipe.db.duck import atomic_replace, connect, init_schema, run_sql_file
from oidpipe.db.load import load_events, load_readings, load_tags, run_quality_checks
from oidpipe.features.build import build_features

logger = logging.getLogger(__name__)


def _table_counts(conn) -> dict[str, int]:
    """Return {table_name: row_count} for all user tables."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
        "ORDER BY table_name"
    ).fetchall()
    return {t[0]: conn.execute(f"SELECT count(*) FROM {t[0]}").fetchone()[0] for t in tables}


def _setup_s3_access(conn, region: str) -> None:
    """Install httpfs and configure S3 credential chain for DuckDB."""
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute(
        f"CREATE SECRET (TYPE s3, PROVIDER credential_chain, REGION '{region}')"
    )
    logger.info("Configured DuckDB httpfs with credential_chain (region=%s)", region)


def _resolve_raw_sources(settings) -> dict[str, str]:
    """Return {name: path_or_uri} for each raw Parquet source."""
    if settings.is_local:
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            logger.error("Raw data directory does not exist: %s", raw_dir)
            sys.exit(1)
        return {
            "tags": str(raw_dir / "tags.parquet"),
            "events": str(raw_dir / "events.parquet"),
            "readings": str(raw_dir / "readings.parquet"),
        }
    bucket = settings.s3_bucket
    prefix = settings.raw_prefix.rstrip("/")
    return {
        "tags": f"s3://{bucket}/{prefix}/tags.parquet",
        "events": f"s3://{bucket}/{prefix}/events.parquet",
        "readings": f"s3://{bucket}/{prefix}/readings.parquet",
    }


def _upload_db_to_s3(db_path: Path, settings) -> None:
    """Upload the built database to S3 with a timestamped key."""
    from oidpipe.ingest.s3_io import upload_file

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    s3_key = f"duckdb/oid_{ts}.duckdb"
    uri = upload_file(db_path, settings.s3_bucket, s3_key, region=settings.aws_region)
    print(f"Uploaded DB to {uri}")


def run_pipeline(env: Environment | None = None) -> Path:
    """Build the analytics DuckDB from raw Parquet files.

    Returns the final database path.
    """
    settings = get_settings() if env is None else get_settings(env=env)
    db_path = settings.db_path
    tmp_path = db_path.with_suffix(".duckdb.tmp")

    raw = _resolve_raw_sources(settings)
    logger.info("Pipeline starting (env=%s, db=%s)", settings.env.value, db_path)

    # Remove stale temp file from a previous aborted run
    if tmp_path.exists():
        tmp_path.unlink()

    # Build into a temp file, then atomically swap
    conn = connect(tmp_path)
    try:
        if not settings.is_local:
            _setup_s3_access(conn, settings.aws_region)

        init_schema(conn)

        n_tags = load_tags(conn, raw["tags"])
        n_events = load_events(conn, raw["events"])
        n_readings = load_readings(conn, raw["readings"])
        logger.info("Loaded: %d tags, %d events, %d readings", n_tags, n_events, n_readings)

        run_quality_checks(conn, min_coverage_pct=40.0)

        build_features(conn)

        run_anomaly_detection(
            conn,
            zscore_threshold=settings.zscore_threshold,
            iforest_contamination=settings.iforest_contamination,
        )

        run_sql_file(conn, "views.sql")

        counts = _table_counts(conn)
    finally:
        conn.close()

    atomic_replace(tmp_path, db_path)

    print("\n=== Pipeline complete ===")
    print(f"Database: {db_path}")
    for table, count in counts.items():
        print(f"  {table}: {count:,} rows")
    print()

    if not settings.is_local:
        _upload_db_to_s3(db_path, settings)

    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OID Compressor pipeline")
    parser.add_argument(
        "--local",
        action="store_const",
        const=Environment.LOCAL,
        dest="env",
        help="Force local mode (read from data/raw/)",
    )
    parser.add_argument(
        "--aws",
        action="store_const",
        const=Environment.AWS,
        dest="env",
        help="Force AWS mode (read from S3)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(env=args.env)


if __name__ == "__main__":
    main()
