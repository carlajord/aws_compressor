"""Tests for agent/tools.py — get_schema, run_sql, and SQL guards."""

from __future__ import annotations

import duckdb
import pytest

from oidpipe.agent.tools import dispatch_tool, get_schema, run_sql


@pytest.fixture()
def conn():
    """In-memory DuckDB with a tiny test table."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE sensors (
            ts TIMESTAMP NOT NULL,
            tag VARCHAR NOT NULL,
            value DOUBLE
        )
    """)
    c.execute("COMMENT ON TABLE sensors IS 'Test sensor readings'")
    c.execute("COMMENT ON COLUMN sensors.ts IS 'Measurement timestamp'")
    c.execute("""
        INSERT INTO sensors VALUES
        ('2024-01-01 00:00', 'temp', 25.0),
        ('2024-01-01 00:10', 'temp', 26.0),
        ('2024-01-01 00:20', 'temp', 27.0),
        ('2024-01-01 00:30', 'temp', 28.0),
        ('2024-01-01 00:40', 'temp', 29.0)
    """)
    c.execute("""
        CREATE VIEW v_avg AS
        SELECT tag, AVG(value) AS avg_val FROM sensors GROUP BY tag
    """)
    return c


# ── get_schema ─────────────────────────────────────────────────────────


class TestGetSchema:
    def test_list_all_tables(self, conn):
        result = get_schema(conn)
        names = {t["name"] for t in result["tables"]}
        assert "sensors" in names
        assert "v_avg" in names

    def test_table_details(self, conn):
        result = get_schema(conn, table="sensors")
        assert result["table"] == "sensors"
        col_names = [c["name"] for c in result["columns"]]
        assert col_names == ["ts", "tag", "value"]
        assert result["columns"][0]["type"] == "TIMESTAMP"
        assert len(result["sample_rows"]) == 3

    def test_table_comments(self, conn):
        result = get_schema(conn, table="sensors")
        ts_col = next(c for c in result["columns"] if c["name"] == "ts")
        assert ts_col.get("comment") == "Measurement timestamp"

    def test_view_schema(self, conn):
        result = get_schema(conn, table="v_avg")
        assert result["table"] == "v_avg"
        col_names = [c["name"] for c in result["columns"]]
        assert "tag" in col_names
        assert "avg_val" in col_names

    def test_nonexistent_table(self, conn):
        result = get_schema(conn, table="no_such_table")
        assert "error" in result

    def test_sql_injection_in_table_name(self, conn):
        result = get_schema(conn, table="sensors; DROP TABLE sensors")
        assert "error" in result or result.get("table") == "sensors DROP TABLE sensors"


# ── run_sql — valid queries ─────────────────────────────────────────────


class TestRunSqlValid:
    def test_simple_select(self, conn):
        result = run_sql(conn, "SELECT * FROM sensors")
        assert "error" not in result
        assert result["columns"] == ["ts", "tag", "value"]
        assert result["row_count"] == 5
        assert result["truncated"] is False

    def test_select_with_where(self, conn):
        result = run_sql(conn, "SELECT value FROM sensors WHERE value > 27")
        assert result["row_count"] == 2

    def test_with_cte(self, conn):
        sql = "WITH t AS (SELECT * FROM sensors) SELECT count(*) AS n FROM t"
        result = run_sql(conn, sql)
        assert result["rows"][0][0] == 5

    def test_select_from_view(self, conn):
        result = run_sql(conn, "SELECT * FROM v_avg")
        assert result["row_count"] == 1
        assert result["columns"] == ["tag", "avg_val"]

    def test_row_limit_truncation(self, conn):
        result = run_sql(conn, "SELECT * FROM sensors", row_limit=3)
        assert result["row_count"] == 3
        assert result["truncated"] is True

    def test_no_truncation_at_exact_limit(self, conn):
        result = run_sql(conn, "SELECT * FROM sensors", row_limit=5)
        assert result["row_count"] == 5
        assert result["truncated"] is False

    def test_trailing_semicolon(self, conn):
        result = run_sql(conn, "SELECT count(*) FROM sensors;")
        assert "error" not in result

    def test_comments_stripped(self, conn):
        sql = "-- get all\nSELECT count(*) AS n FROM sensors /* inline */"
        result = run_sql(conn, sql)
        assert result["rows"][0][0] == 5


# ── run_sql — rejected queries ──────────────────────────────────────────


class TestRunSqlRejected:
    @pytest.mark.parametrize(
        "sql,keyword",
        [
            ("INSERT INTO sensors VALUES ('2024-01-01', 'x', 1)", "INSERT"),
            ("UPDATE sensors SET value = 0", "UPDATE"),
            ("DELETE FROM sensors", "DELETE"),
            ("CREATE TABLE evil (id INT)", "CREATE"),
            ("DROP TABLE sensors", "DROP"),
            ("ATTACH ':memory:' AS other", "ATTACH"),
            ("COPY sensors TO '/tmp/out.csv'", "COPY"),
            ("INSTALL httpfs", "INSTALL"),
            ("LOAD httpfs", "LOAD"),
            ("PRAGMA database_list", "PRAGMA"),
            ("ALTER TABLE sensors ADD COLUMN x INT", "ALTER"),
        ],
    )
    def test_blocked_keyword(self, conn, sql, keyword):
        result = run_sql(conn, sql)
        assert "error" in result
        assert keyword in result["error"] or "Only SELECT/WITH" in result["error"]

    def test_multi_statement(self, conn):
        result = run_sql(conn, "SELECT 1; SELECT 2")
        assert "error" in result
        assert "Multi-statement" in result["error"]

    def test_empty_query(self, conn):
        result = run_sql(conn, "")
        assert "error" in result

    def test_whitespace_only(self, conn):
        result = run_sql(conn, "   \n  ")
        assert "error" in result

    def test_comment_hiding_drop(self, conn):
        result = run_sql(conn, "-- SELECT * FROM sensors\nDROP TABLE sensors")
        assert "error" in result

    def test_invalid_sql_returns_error(self, conn):
        result = run_sql(conn, "SELECT * FROM nonexistent_table_xyz")
        assert "error" in result

    def test_syntax_error_returns_error(self, conn):
        result = run_sql(conn, "SELECT FROM WHERE")
        assert "error" in result


# ── dispatch_tool ───────────────────────────────────────────────────────


class TestDispatchTool:
    def test_dispatch_get_schema(self, conn):
        result = dispatch_tool(conn, "get_schema", {})
        assert "tables" in result

    def test_dispatch_get_schema_with_table(self, conn):
        result = dispatch_tool(conn, "get_schema", {"table": "sensors"})
        assert result["table"] == "sensors"

    def test_dispatch_run_sql(self, conn):
        result = dispatch_tool(conn, "run_sql", {"sql": "SELECT 1 AS n"})
        assert result["rows"][0][0] == 1

    def test_dispatch_run_sql_with_limit(self, conn):
        result = dispatch_tool(
            conn, "run_sql", {"sql": "SELECT * FROM sensors"}, row_limit=2
        )
        assert result["truncated"] is True

    def test_dispatch_unknown_tool(self, conn):
        result = dispatch_tool(conn, "delete_everything", {})
        assert "error" in result
