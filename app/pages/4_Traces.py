"""Traces page — agent run history and span details."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config.settings import get_settings
from oidpipe.agent.tracing import init_traces_db
from oidpipe.db.duck import connect

st.set_page_config(page_title="Traces — OID Compressor", layout="wide")
st.title("Agent Traces")

settings = get_settings()


@st.cache_resource
def get_traces_conn():
    if not settings.traces_db_path.exists():
        st.warning("No traces database found. Use the Chat page first.")
        st.stop()
    c = connect(settings.traces_db_path, read_only=False)
    init_traces_db(c)
    return c


tconn = get_traces_conn()

# ── Runs table ──────────────────────────────────────────────────────────

st.subheader("Agent runs")

runs = tconn.execute("""
    SELECT trace_id, question, answer, model_id, prompt_version,
           steps, total_tokens, ROUND(latency_ms / 1000, 1) AS latency_s,
           created_at, source
    FROM agent_runs
    ORDER BY created_at DESC
    LIMIT 100
""").fetchdf()

if len(runs) == 0:
    st.info("No agent runs recorded yet. Ask a question on the Chat page first.")
    st.stop()

source_filter = st.sidebar.multiselect(
    "Source", ["app", "eval"], default=["app", "eval"]
)
filtered = runs[runs["source"].isin(source_filter)]

st.caption(f"{len(filtered)} runs (showing up to 100 most recent)")

st.dataframe(
    filtered[["trace_id", "question", "steps", "total_tokens", "latency_s", "source", "created_at"]],
    hide_index=True,
    use_container_width=True,
)

# ── Summary metrics ─────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", len(filtered))
col2.metric("Avg steps", f"{filtered['steps'].mean():.1f}" if len(filtered) > 0 else "—")
col3.metric("Avg tokens", f"{filtered['total_tokens'].mean():,.0f}" if len(filtered) > 0 else "—")
col4.metric("Avg latency", f"{filtered['latency_s'].mean():.1f}s" if len(filtered) > 0 else "—")

st.divider()

# ── Span detail for selected run ────────────────────────────────────────

st.subheader("Span details")

trace_ids = filtered["trace_id"].tolist()
selected_trace = st.selectbox("Select a run (trace_id)", trace_ids, index=0)

if selected_trace:
    run_row = filtered[filtered["trace_id"] == selected_trace].iloc[0]
    st.markdown(f"**Question:** {run_row['question']}")
    st.markdown(f"**Answer:** {run_row['answer']}")

    spans = tconn.execute("""
        SELECT span_id, parent_id, kind, name,
               ROUND(latency_ms, 1) AS latency_ms,
               input_tokens, output_tokens, status, error,
               input_json, output_json
        FROM agent_spans
        WHERE trace_id = ?
        ORDER BY started_at
    """, [selected_trace]).fetchdf()

    if len(spans) == 0:
        st.info("No spans recorded for this run.")
    else:
        st.caption(f"{len(spans)} spans")

        st.dataframe(
            spans[["kind", "name", "latency_ms", "input_tokens", "output_tokens", "status"]],
            hide_index=True,
            use_container_width=True,
        )

        for idx, span in spans.iterrows():
            label = f"{span['kind']}: {span['name']}"
            if span["status"] == "error":
                label += " (ERROR)"
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Status:** {span['status']}")
                c2.write(f"**Latency:** {span['latency_ms']} ms")
                if pd.notna(span["input_tokens"]):
                    c3.write(f"**Tokens:** {int(span['input_tokens'])} in / {int(span['output_tokens'])} out")

                if pd.notna(span["error"]) and span["error"]:
                    st.error(span["error"])

                if pd.notna(span["input_json"]) and span["input_json"]:
                    try:
                        st.json(json.loads(span["input_json"]))
                    except json.JSONDecodeError:
                        st.code(span["input_json"])

                if pd.notna(span["output_json"]) and span["output_json"]:
                    try:
                        parsed = json.loads(span["output_json"])
                        if isinstance(parsed, dict) and "rows" in parsed and len(str(parsed)) > 2000:
                            st.write(f"*{len(parsed['rows'])} rows returned*")
                            st.json({k: v for k, v in parsed.items() if k != "rows"})
                        else:
                            st.json(parsed)
                    except json.JSONDecodeError:
                        st.code(span["output_json"])
