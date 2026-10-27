"""Tests for db/load.py — loading Parquet into DuckDB and quality checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oidpipe.db.duck import connect, init_schema
from oidpipe.db.load import (
    QualityCheckError,
    check_duplicates,
    check_physical_sanity,
    check_tag_coverage,
    check_timestamp_grid,
    load_events,
    load_readings,
    load_tags,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic Parquet files
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path):
    conn = connect(tmp_path / "test.duckdb")
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Create synthetic Parquet files in a temp directory."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # tags.parquet
    tags = pd.DataFrame(
        {
            "external_id": ["EXT_P_SUCTION", "EXT_T_DISCHARGE"],
            "logical_name": ["p_suction_barg", "t_discharge_c"],
            "name": ["p_suction_barg", "t_discharge_c"],
            "description": ["Suction pressure", "Discharge temperature"],
            "unit": ["barg", "degC"],
            "asset_name": ["23-KA-9101", "23-KA-9101"],
        }
    )
    tags.to_parquet(raw / "tags.parquet", index=False)

    # readings.parquet — 6 timestamps (1 hour of 10-min data), 2 tags
    timestamps = pd.date_range("2024-03-01", periods=6, freq="10min")
    readings_rows = []
    for ts in timestamps:
        readings_rows.append(
            {"ts": ts, "external_id": "EXT_P_SUCTION", "value": 3.5}
        )
        readings_rows.append(
            {"ts": ts, "external_id": "EXT_T_DISCHARGE", "value": 95.0}
        )
    pd.DataFrame(readings_rows).to_parquet(raw / "readings.parquet", index=False)

    # events.parquet
    events = pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [pd.Timestamp("2024-03-01 00:00", tz="UTC")],
            "end_time": [pd.Timestamp("2024-03-01 01:00", tz="UTC")],
            "type": ["Maintenance"],
            "subtype": ["Planned"],
            "description": ["Test event"],
            "asset_names": ["1001"],
        }
    )
    events.to_parquet(raw / "events.parquet", index=False)

    return raw


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------


class TestLoadTags:
    def test_loads_correct_count(self, db, raw_dir: Path):
        count = load_tags(db, raw_dir / "tags.parquet")
        assert count == 2

    def test_logical_names_present(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        names = db.execute(
            "SELECT logical_name FROM tags ORDER BY logical_name"
        ).fetchall()
        assert [n[0] for n in names] == ["p_suction_barg", "t_discharge_c"]


class TestLoadEvents:
    def test_loads_correct_count(self, db, raw_dir: Path):
        count = load_events(db, raw_dir / "events.parquet")
        assert count == 1

    def test_event_data_preserved(self, db, raw_dir: Path):
        load_events(db, raw_dir / "events.parquet")
        row = db.execute("SELECT type, subtype FROM events").fetchone()
        assert row == ("Maintenance", "Planned")


class TestLoadReadings:
    def test_loads_correct_count(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        count = load_readings(db, raw_dir / "readings.parquet")
        assert count == 12  # 6 timestamps × 2 tags

    def test_maps_external_id_to_logical_name(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")

        tags = db.execute(
            "SELECT DISTINCT tag FROM readings_10m ORDER BY tag"
        ).fetchall()
        assert [t[0] for t in tags] == ["p_suction_barg", "t_discharge_c"]

    def test_no_external_ids_in_readings(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")

        ext = db.execute(
            "SELECT count(*) FROM readings_10m WHERE tag LIKE 'EXT_%'"
        ).fetchone()[0]
        assert ext == 0

    def test_unmapped_external_ids_dropped(self, db, raw_dir: Path, tmp_path: Path):
        """Readings with external_ids not in tags table are excluded by the JOIN."""
        load_tags(db, raw_dir / "tags.parquet")

        extra_readings = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2024-03-01")],
                "external_id": ["EXT_UNKNOWN"],
                "value": [99.0],
            }
        )
        extra_path = tmp_path / "extra.parquet"
        extra_readings.to_parquet(extra_path, index=False)

        load_readings(db, extra_path)
        count = db.execute("SELECT count(*) FROM readings_10m").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: quality checks
# ---------------------------------------------------------------------------


class TestCheckTimestampGrid:
    def test_passes_on_clean_data(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        assert check_timestamp_grid(db) == 0

    def test_fails_on_off_grid(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        db.execute(
            "INSERT INTO readings_10m VALUES ('2024-03-01 00:03:00', 'p_suction_barg', 1.0)"
        )
        with pytest.raises(QualityCheckError, match="off the 10-minute grid"):
            check_timestamp_grid(db)


class TestCheckDuplicates:
    def test_passes_on_clean_data(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        assert check_duplicates(db) == 0

    def test_pk_prevents_dupes(self, db, raw_dir: Path):
        """The PRIMARY KEY constraint blocks duplicate (ts, tag) inserts."""
        import duckdb as _duckdb

        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        with pytest.raises(_duckdb.ConstraintException):
            db.execute(
                "INSERT INTO readings_10m VALUES ('2024-03-01 00:00:00', 'p_suction_barg', 9.9)"
            )


class TestCheckTagCoverage:
    def test_passes_on_full_coverage(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        failures = check_tag_coverage(db)
        assert failures == {}

    def test_fails_on_sparse_tag(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        # Remove most readings for one tag (keep 1 of 6 = 16.7%)
        db.execute(
            "DELETE FROM readings_10m WHERE tag = 'p_suction_barg' AND ts > '2024-03-01 00:00:00'"
        )
        with pytest.raises(QualityCheckError, match="coverage"):
            check_tag_coverage(db)


class TestCheckPhysicalSanity:
    def test_passes_on_normal_data(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        violations = check_physical_sanity(db)
        assert violations["pressure_range"] == 0
        assert violations["temperature_range"] == 0

    def test_detects_pressure_out_of_range(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        db.execute(
            "INSERT INTO readings_10m VALUES ('2024-04-01', 'p_suction_barg', -5.0)"
        )
        violations = check_physical_sanity(db)
        assert violations["pressure_range"] == 1

    def test_detects_temperature_out_of_range(self, db, raw_dir: Path):
        load_tags(db, raw_dir / "tags.parquet")
        load_readings(db, raw_dir / "readings.parquet")
        db.execute(
            "INSERT INTO readings_10m VALUES ('2024-04-01', 't_discharge_c', 250.0)"
        )
        violations = check_physical_sanity(db)
        assert violations["temperature_range"] == 1
