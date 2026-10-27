"""Tests for db/duck.py — connect, init_schema, atomic_replace."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from oidpipe.db.duck import atomic_replace, connect, init_schema


class TestConnect:
    def test_creates_database_file(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        conn = connect(db_path)
        conn.close()
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        db_path = tmp_path / "nested" / "deep" / "test.duckdb"
        conn = connect(db_path)
        conn.close()
        assert db_path.exists()

    def test_read_only_on_existing_db(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        conn = connect(db_path)
        conn.close()

        ro_conn = connect(db_path, read_only=True)
        with pytest.raises(duckdb.InvalidInputException):
            ro_conn.execute("CREATE TABLE fail_test (x INT)")
        ro_conn.close()

    def test_returns_duckdb_connection(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        conn = connect(db_path)
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        conn.close()


class TestInitSchema:
    def test_creates_all_tables(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)

        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "tags" in table_names
        assert "readings_10m" in table_names
        assert "events" in table_names
        assert "anomalies" in table_names
        conn.close()

    def test_tags_has_expected_columns(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)

        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'tags' ORDER BY ordinal_position"
        ).fetchall()
        col_names = [c[0] for c in cols]

        assert col_names == [
            "external_id", "logical_name", "name", "description", "unit", "asset_name"
        ]
        conn.close()

    def test_readings_10m_has_expected_columns(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)

        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'readings_10m' ORDER BY ordinal_position"
        ).fetchall()
        col_names = [c[0] for c in cols]

        assert col_names == ["ts", "tag", "value"]
        conn.close()

    def test_idempotent(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)
        init_schema(conn)

        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        assert len(tables) == 4
        conn.close()

    def test_tables_have_comments(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)

        comment = conn.execute(
            "SELECT comment FROM duckdb_tables() WHERE table_name = 'readings_10m'"
        ).fetchone()
        assert comment is not None
        assert "10-minute" in comment[0]
        conn.close()

    def test_columns_have_comments(self, tmp_path: Path):
        conn = connect(tmp_path / "test.duckdb")
        init_schema(conn)

        comment = conn.execute(
            "SELECT comment FROM duckdb_columns() "
            "WHERE table_name = 'readings_10m' AND column_name = 'ts'"
        ).fetchone()
        assert comment is not None
        assert "UTC" in comment[0]
        conn.close()


class TestAtomicReplace:
    def test_renames_file(self, tmp_path: Path):
        tmp_file = tmp_path / "db.tmp"
        final_file = tmp_path / "db.duckdb"
        tmp_file.write_text("test data")

        atomic_replace(tmp_file, final_file)

        assert final_file.exists()
        assert not tmp_file.exists()
        assert final_file.read_text() == "test data"

    def test_overwrites_existing_final(self, tmp_path: Path):
        tmp_file = tmp_path / "db.tmp"
        final_file = tmp_path / "db.duckdb"
        final_file.write_text("old data")
        tmp_file.write_text("new data")

        atomic_replace(tmp_file, final_file)

        assert final_file.read_text() == "new data"
        assert not tmp_file.exists()
