"""Shared DuckDB connection and chart helpers for the Streamlit app."""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_resource
def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a read-only DuckDB connection, cached across reruns."""
    db_path = Path("data/oid.duckdb")
    if not db_path.exists():
        st.error(f"Database not found at `{db_path}`. Run the pipeline first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)


def get_date_bounds(conn: duckdb.DuckDBPyConnection) -> tuple[datetime.date, datetime.date]:
    """Return (min_date, max_date) from features_10m."""
    row = conn.execute(
        "SELECT min(ts)::DATE, max(ts)::DATE FROM features_10m"
    ).fetchone()
    return row[0], row[1]


def date_range_slider(
    conn: duckdb.DuckDBPyConnection,
    sidebar: bool = True,
) -> tuple[datetime.date, datetime.date]:
    """Render a date-range slider and return (start, end)."""
    min_d, max_d = get_date_bounds(conn)
    target = st.sidebar if sidebar else st
    start, end = target.slider(
        "Date range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
        format="YYYY-MM-DD",
    )
    return start, end


def get_offline_periods(
    conn: duckdb.DuckDBPyConnection,
    start: datetime.date,
    end: datetime.date,
) -> list[tuple]:
    """Return list of (start_ts, end_ts) for contiguous offline blocks."""
    df = conn.execute("""
        WITH ordered AS (
            SELECT ts, is_running,
                   LAG(is_running) OVER (ORDER BY ts) AS prev_running
            FROM features_10m
            WHERE ts::DATE BETWEEN ? AND ?
        ),
        boundaries AS (
            SELECT ts,
                   CASE WHEN NOT is_running AND (prev_running OR prev_running IS NULL)
                        THEN 'start'
                        WHEN is_running AND NOT prev_running
                        THEN 'end'
                   END AS boundary
            FROM ordered
            WHERE (NOT is_running AND (prev_running OR prev_running IS NULL))
               OR (is_running AND NOT prev_running)
        )
        SELECT boundary, ts FROM boundaries ORDER BY ts
    """, [start, end]).fetchdf()

    periods = []
    current_start = None
    for _, row in df.iterrows():
        if row["boundary"] == "start":
            current_start = row["ts"]
        elif row["boundary"] == "end" and current_start is not None:
            periods.append((current_start, row["ts"]))
            current_start = None

    # If the last offline block extends to the end of the range
    if current_start is not None:
        last_ts = conn.execute(
            "SELECT max(ts) FROM features_10m WHERE ts::DATE BETWEEN ? AND ?",
            [start, end],
        ).fetchone()[0]
        periods.append((current_start, last_ts))

    return periods


def add_offline_shading(
    fig: go.Figure,
    periods: list[tuple],
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add semi-transparent gray rectangles for offline periods."""
    for start_ts, end_ts in periods:
        shape_kwargs = dict(
            type="rect",
            x0=start_ts, x1=end_ts,
            y0=0, y1=1, yref="paper",
            fillcolor="rgba(128, 128, 128, 0.15)",
            line_width=0,
            layer="below",
        )
        if row is not None and col is not None:
            shape_kwargs["xref"] = f"x{'' if row == 1 else row}"
            shape_kwargs["yref"] = f"y{'' if row == 1 else row} domain"
        fig.add_shape(**shape_kwargs)
