"""Agent tracing — records spans and run summaries to traces.duckdb."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import duckdb

logger = logging.getLogger(__name__)

_TRACES_SQL = Path(__file__).resolve().parent.parent / "db" / "sql" / "traces_schema.sql"


def init_traces_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the traces schema tables if they don't exist."""
    sql = _TRACES_SQL.read_text()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


class Span:
    """A single traced operation. Use via Tracer.span() context manager."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        kind: str,
        name: str,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.started_at = datetime.now(timezone.utc)
        self._start_ns = time.perf_counter_ns()
        self.latency_ms: float | None = None
        self.input_json: str | None = None
        self.output_json: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.status: str = "ok"
        self.error: str | None = None

    def record(
        self,
        output: Any = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Attach output data and token usage to this span."""
        if output is not None:
            self.output_json = json.dumps(output, default=str)
        if usage:
            self.input_tokens = usage.get("input_tokens")
            self.output_tokens = usage.get("output_tokens")

    def _finish(self, error: BaseException | None = None) -> None:
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self.latency_ms = elapsed_ns / 1_000_000
        if error is not None:
            self.status = "error"
            self.error = str(error)


class Tracer:
    """Records spans and run summaries for one agent invocation."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, trace_id: str | None = None) -> None:
        self.conn = conn
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._span_stack: list[str] = []

    @contextmanager
    def span(
        self,
        kind: str,
        name: str,
        input: Any = None,  # noqa: A002
    ) -> Generator[Span, None, None]:
        """Context manager that records a span on exit."""
        span_id = uuid.uuid4().hex[:12]
        parent_id = self._span_stack[-1] if self._span_stack else None

        s = Span(self.trace_id, span_id, parent_id, kind, name)
        if input is not None:
            s.input_json = json.dumps(input, default=str)

        self._span_stack.append(span_id)
        error: BaseException | None = None
        try:
            yield s
        except Exception as exc:
            error = exc
            raise
        finally:
            self._span_stack.pop()
            s._finish(error)
            self._write_span(s)

    def _write_span(self, s: Span) -> None:
        self.conn.execute(
            "INSERT INTO agent_spans "
            "(trace_id, span_id, parent_id, kind, name, started_at, latency_ms, "
            "input_json, output_json, input_tokens, output_tokens, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                s.trace_id,
                s.span_id,
                s.parent_id,
                s.kind,
                s.name,
                s.started_at,
                s.latency_ms,
                s.input_json,
                s.output_json,
                s.input_tokens,
                s.output_tokens,
                s.status,
                s.error,
            ],
        )

    def record_run(
        self,
        question: str,
        answer: str | None,
        model_id: str,
        prompt_version: str,
        steps: int,
        total_tokens: int,
        latency_ms: float,
        source: str = "app",
    ) -> None:
        """Write a summary row to agent_runs."""
        self.conn.execute(
            "INSERT INTO agent_runs "
            "(trace_id, question, answer, model_id, prompt_version, steps, "
            "total_tokens, latency_ms, created_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.trace_id,
                question,
                answer,
                model_id,
                prompt_version,
                steps,
                total_tokens,
                latency_ms,
                datetime.now(timezone.utc),
                source,
            ],
        )
