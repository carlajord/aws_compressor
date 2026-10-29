"""Agent tools: get_schema and run_sql with safety guards."""

from __future__ import annotations

import json
import logging
import re

import duckdb

logger = logging.getLogger(__name__)

BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|ATTACH|DETACH|COPY|INSTALL|LOAD|PRAGMA|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

SQL_TIMEOUT_S = 10
SAMPLE_ROWS = 3

TOOL_DESCRIPTIONS: dict[str, dict] = {
    "get_schema": {
        "description": (
            "Return database schema information. "
            "If table is null, list all table and view names. "
            "If table is given, return column names, types, comments, and 3 sample rows."
        ),
        "parameters": {
            "table": {
                "type": "string",
                "description": "Table or view name. Omit to list all tables.",
                "required": False,
            },
        },
    },
    "run_sql": {
        "description": (
            "Execute a read-only SQL query against the DuckDB analytics database. "
            "Only SELECT/WITH statements are allowed. "
            "Returns columns, rows (up to the configured limit), total row_count, "
            "and whether results were truncated."
        ),
        "parameters": {
            "sql": {
                "type": "string",
                "description": "A SELECT or WITH statement to execute.",
                "required": True,
            },
        },
    },
}


def get_schema(conn: duckdb.DuckDBPyConnection, table: str | None = None) -> dict:
    """Return schema metadata for one table or a list of all tables/views."""
    if table is None:
        rows = conn.execute(
            "SELECT table_name, table_type "
            "FROM information_schema.tables "
            "WHERE table_schema = 'main' "
            "ORDER BY table_type, table_name"
        ).fetchall()

        col_rows = conn.execute(
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = 'main' "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        cols_by_table: dict[str, list[str]] = {}
        for tbl, col in col_rows:
            cols_by_table.setdefault(tbl, []).append(col)

        return {
            "tables": [
                {"name": r[0], "type": r[1], "columns": cols_by_table.get(r[0], [])}
                for r in rows
            ],
        }

    table_clean = table.strip().replace("'", "").replace(";", "")

    try:
        cols = conn.execute(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? "
            "ORDER BY ordinal_position",
            [table_clean],
        ).fetchall()
    except duckdb.Error as exc:
        return {"error": str(exc)}

    if not cols:
        return {"error": f"Table '{table_clean}' not found"}

    comments: dict[str, str] = {}
    try:
        comment_rows = conn.execute(
            "SELECT column_name, comment "
            "FROM duckdb_columns() "
            "WHERE schema_name = 'main' AND table_name = ? "
            "AND comment IS NOT NULL",
            [table_clean],
        ).fetchall()
        comments = {r[0]: r[1] for r in comment_rows}
    except duckdb.Error:
        pass

    columns = [
        {
            "name": c[0],
            "type": c[1],
            **({"comment": comments[c[0]]} if c[0] in comments else {}),
        }
        for c in cols
    ]

    try:
        sample = conn.execute(
            f'SELECT * FROM "{table_clean}" LIMIT {SAMPLE_ROWS}'
        ).fetchdf()
        sample_rows = json.loads(sample.to_json(orient="records", date_format="iso"))
    except duckdb.Error:
        sample_rows = []

    return {"table": table_clean, "columns": columns, "sample_rows": sample_rows}


def _validate_sql(sql: str) -> str | None:
    """Return an error message if the SQL is disallowed, else None."""
    stripped = COMMENT_RE.sub(" ", sql).strip()

    if not stripped:
        return "Empty query"

    if ";" in stripped:
        parts = [p.strip() for p in stripped.split(";") if p.strip()]
        if len(parts) > 1:
            return "Multi-statement queries are not allowed"

    first_word = stripped.split()[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return f"Only SELECT/WITH queries are allowed, got {first_word}"

    match = BLOCKED_KEYWORDS.search(stripped)
    if match:
        kw = match.group(0).upper()
        if first_word == "WITH" and kw == "SELECT":
            pass
        else:
            return f"Keyword '{kw}' is not allowed"

    return None


def run_sql(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    row_limit: int = 500,
) -> dict:
    """Execute a read-only query with guards. Returns a result dict."""
    error = _validate_sql(sql)
    if error:
        logger.warning("SQL rejected: %s — %s", error, sql[:200])
        return {"error": error}

    cleaned = COMMENT_RE.sub(" ", sql).strip().rstrip(";")
    wrapped = f"SELECT * FROM ({cleaned}) AS _q LIMIT {row_limit + 1}"

    try:
        result = conn.execute(wrapped)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        truncated = len(rows) > row_limit
        if truncated:
            rows = rows[:row_limit]

        rows_as_lists = [list(r) for r in rows]

        return {
            "columns": columns,
            "rows": rows_as_lists,
            "row_count": len(rows_as_lists),
            "truncated": truncated,
        }
    except duckdb.Error as exc:
        logger.warning("SQL error: %s — %s", exc, sql[:200])
        return {"error": str(exc)}


def dispatch_tool(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    arguments: dict,
    row_limit: int = 500,
) -> dict:
    """Route a tool call by name to the right handler."""
    if name == "get_schema":
        return get_schema(conn, table=arguments.get("table"))
    if name == "run_sql":
        sql = arguments.get("sql", "")
        return run_sql(conn, sql, row_limit=row_limit)
    return {"error": f"Unknown tool: {name}"}
