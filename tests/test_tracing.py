"""Tests for agent/tracing.py — Tracer, Span, and traces DB schema."""

from __future__ import annotations

import json

import duckdb
import pytest

from oidpipe.agent.tracing import Tracer, init_traces_db


@pytest.fixture()
def conn():
    """In-memory DuckDB with traces schema initialised."""
    c = duckdb.connect(":memory:")
    init_traces_db(c)
    return c


# ── init_traces_db ──────────────────────────────────────────────────────


def test_init_creates_tables(conn):
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
    }
    assert "agent_spans" in tables
    assert "agent_runs" in tables


def test_init_idempotent(conn):
    init_traces_db(conn)
    assert conn.execute("SELECT count(*) FROM agent_spans").fetchone()[0] == 0


# ── Span basics ─────────────────────────────────────────────────────────


class TestSpanLifecycle:
    def test_span_recorded(self, conn):
        tracer = Tracer(conn, trace_id="t1")
        with tracer.span("llm", "call-1"):
            pass

        rows = conn.execute("SELECT * FROM agent_spans WHERE trace_id='t1'").fetchall()
        assert len(rows) == 1

    def test_span_fields(self, conn):
        tracer = Tracer(conn, trace_id="t2")
        with tracer.span("llm", "chat", input={"prompt": "hello"}) as s:
            s.record(output={"text": "world"}, usage={"input_tokens": 10, "output_tokens": 5})

        row = conn.execute(
            "SELECT kind, name, status, input_json, output_json, "
            "input_tokens, output_tokens, latency_ms "
            "FROM agent_spans WHERE trace_id='t2'"
        ).fetchone()
        assert row[0] == "llm"
        assert row[1] == "chat"
        assert row[2] == "ok"
        assert json.loads(row[3]) == {"prompt": "hello"}
        assert json.loads(row[4]) == {"text": "world"}
        assert row[5] == 10
        assert row[6] == 5
        assert row[7] > 0  # latency_ms

    def test_span_latency_positive(self, conn):
        tracer = Tracer(conn, trace_id="t3")
        with tracer.span("tool", "get_schema"):
            pass

        latency = conn.execute(
            "SELECT latency_ms FROM agent_spans WHERE trace_id='t3'"
        ).fetchone()[0]
        assert latency >= 0

    def test_span_no_input_output(self, conn):
        tracer = Tracer(conn, trace_id="t4")
        with tracer.span("tool", "noop"):
            pass

        row = conn.execute(
            "SELECT input_json, output_json FROM agent_spans WHERE trace_id='t4'"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None


# ── Error handling ──────────────────────────────────────────────────────


class TestSpanErrors:
    def test_error_recorded(self, conn):
        tracer = Tracer(conn, trace_id="e1")
        with pytest.raises(ValueError, match="boom"):
            with tracer.span("llm", "bad-call"):
                raise ValueError("boom")

        row = conn.execute(
            "SELECT status, error FROM agent_spans WHERE trace_id='e1'"
        ).fetchone()
        assert row[0] == "error"
        assert "boom" in row[1]

    def test_error_span_still_written(self, conn):
        tracer = Tracer(conn, trace_id="e2")
        with pytest.raises(RuntimeError):
            with tracer.span("tool", "fail") as s:
                s.record(output={"partial": True})
                raise RuntimeError("oops")

        row = conn.execute(
            "SELECT status, output_json FROM agent_spans WHERE trace_id='e2'"
        ).fetchone()
        assert row[0] == "error"
        assert json.loads(row[1]) == {"partial": True}


# ── Nested spans / parent tracking ─────────────────────────────────────


class TestNestedSpans:
    def test_parent_id_set(self, conn):
        tracer = Tracer(conn, trace_id="n1")
        with tracer.span("agent", "ask") as outer:
            with tracer.span("llm", "call-1"):
                pass

        rows = conn.execute(
            "SELECT span_id, parent_id, kind FROM agent_spans "
            "WHERE trace_id='n1' ORDER BY started_at"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] is None  # outer has no parent
        assert rows[1][1] == rows[0][0]  # inner parent = outer span_id
        assert rows[1][2] == "llm"

    def test_three_level_nesting(self, conn):
        tracer = Tracer(conn, trace_id="n2")
        with tracer.span("agent", "top"):
            with tracer.span("llm", "mid"):
                with tracer.span("tool", "bottom"):
                    pass

        rows = conn.execute(
            "SELECT span_id, parent_id, kind FROM agent_spans "
            "WHERE trace_id='n2' ORDER BY started_at"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][1] is None
        assert rows[1][1] == rows[0][0]
        assert rows[2][1] == rows[1][0]

    def test_sibling_spans(self, conn):
        tracer = Tracer(conn, trace_id="n3")
        with tracer.span("agent", "ask") as parent:
            with tracer.span("llm", "call-1"):
                pass
            with tracer.span("tool", "run_sql"):
                pass

        rows = conn.execute(
            "SELECT span_id, parent_id, kind FROM agent_spans "
            "WHERE trace_id='n3' ORDER BY started_at"
        ).fetchall()
        assert len(rows) == 3
        # both children share the same parent
        assert rows[1][1] == rows[0][0]
        assert rows[2][1] == rows[0][0]


# ── record_run ──────────────────────────────────────────────────────────


class TestRecordRun:
    def test_run_recorded(self, conn):
        tracer = Tracer(conn, trace_id="r1")
        tracer.record_run(
            question="How many rows?",
            answer="42 rows",
            model_id="openai.gpt-oss-120b",
            prompt_version="v1",
            steps=3,
            total_tokens=500,
            latency_ms=1234.5,
            source="app",
        )

        row = conn.execute(
            "SELECT question, answer, model_id, prompt_version, steps, "
            "total_tokens, latency_ms, source "
            "FROM agent_runs WHERE trace_id='r1'"
        ).fetchone()
        assert row[0] == "How many rows?"
        assert row[1] == "42 rows"
        assert row[2] == "openai.gpt-oss-120b"
        assert row[3] == "v1"
        assert row[4] == 3
        assert row[5] == 500
        assert row[6] == 1234.5
        assert row[7] == "app"

    def test_run_eval_source(self, conn):
        tracer = Tracer(conn, trace_id="r2")
        tracer.record_run(
            question="What is the avg pressure ratio?",
            answer="3.5",
            model_id="openai.gpt-oss-120b",
            prompt_version="v2",
            steps=2,
            total_tokens=300,
            latency_ms=800.0,
            source="eval",
        )

        source = conn.execute(
            "SELECT source FROM agent_runs WHERE trace_id='r2'"
        ).fetchone()[0]
        assert source == "eval"

    def test_run_created_at_populated(self, conn):
        tracer = Tracer(conn, trace_id="r3")
        tracer.record_run(
            question="test",
            answer=None,
            model_id="m",
            prompt_version="v1",
            steps=1,
            total_tokens=100,
            latency_ms=50.0,
        )

        created = conn.execute(
            "SELECT created_at FROM agent_runs WHERE trace_id='r3'"
        ).fetchone()[0]
        assert created is not None


# ── Tracer auto-generates trace_id ──────────────────────────────────────


def test_auto_trace_id(conn):
    tracer = Tracer(conn)
    assert len(tracer.trace_id) == 16
    tracer2 = Tracer(conn)
    assert tracer.trace_id != tracer2.trace_id
