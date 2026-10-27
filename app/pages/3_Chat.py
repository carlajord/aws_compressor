"""Chat page — text-to-SQL agent interface."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path for config/ and src/ imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv

from config.settings import get_settings
from db import get_conn
from oidpipe.agent.sql_agent import ask
from oidpipe.agent.tracing import Tracer, init_traces_db
from oidpipe.db.duck import connect

load_dotenv()

st.set_page_config(page_title="Chat — OID Compressor", layout="wide")
st.title("Chat with your data")
st.caption("Ask questions about the compressor in plain English.")

conn = get_conn()
settings = get_settings()


@st.cache_resource
def get_traces_conn():
    c = connect(settings.traces_db_path, read_only=False)
    init_traces_db(c)
    return c


@st.cache_resource
def get_llm_client():
    if settings.llm_backend == "bedrock":
        return None
    from openai import OpenAI
    return OpenAI()


traces_conn = get_traces_conn()
client = get_llm_client()

# ── Chat history ────────────────────────────────────────────────────────

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg_idx, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("SQL & data"):
                for sql in msg["sql"]:
                    st.code(sql, language="sql")
                if msg.get("df") is not None:
                    st.dataframe(msg["df"], hide_index=True, use_container_width=True)
        if "chart" in msg:
            st.plotly_chart(msg["chart"], use_container_width=True, key=f"hist-chart-{msg_idx}")
        if "meta" in msg:
            st.caption(msg["meta"])

# ── User input ──────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about the compressor..."):
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tracer = Tracer(traces_conn)
            result = ask(
                question,
                conn,
                client,
                model=settings.llm_model,
                max_steps=settings.agent_max_steps,
                row_limit=settings.sql_row_limit,
                tracer=tracer,
                source="app",
                region=settings.aws_region,
            )

        st.markdown(result.answer)

        msg_data: dict = {
            "role": "assistant",
            "content": result.answer,
        }

        if result.sql_list:
            msg_data["sql"] = result.sql_list
            with st.expander("SQL & data"):
                for sql in result.sql_list:
                    st.code(sql, language="sql")
                if result.last_result_df is not None:
                    msg_data["df"] = result.last_result_df
                    st.dataframe(result.last_result_df, hide_index=True, use_container_width=True)

        if result.last_result_df is not None and len(result.last_result_df) > 1:
            df = result.last_result_df
            ts_cols = [c for c in df.columns if "ts" in c.lower() or "date" in c.lower() or "day" in c.lower()]
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if ts_cols and num_cols:
                fig = go.Figure()
                x_col = ts_cols[0]
                for col in num_cols[:4]:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=df[col], mode="lines", name=col,
                    ))
                fig.update_layout(height=300, margin=dict(t=30, b=30))
                chart_key = f"new-chart-{len(st.session_state.chat_messages)}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                msg_data["chart"] = fig

        meta = f"Steps: {result.steps} | Tokens: {result.input_tokens + result.output_tokens:,} | {result.latency_ms / 1000:.1f}s"
        if result.error:
            meta += f" | Error: {result.error}"
        st.caption(meta)
        msg_data["meta"] = meta

        st.session_state.chat_messages.append(msg_data)
