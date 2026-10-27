"""Cooler Performance page — duty, approach, gas ΔT with gap-aware rolling and anomaly overlay."""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from db import get_conn, date_range_slider, get_offline_periods, add_offline_shading

st.set_page_config(page_title="Cooler Performance — OID Compressor", layout="wide")
st.title("Cooler Performance — 23-HA-9103")

conn = get_conn()

# ── Sidebar ─────────────────────────────────────────────────────────────
start, end = date_range_slider(conn, sidebar=True)
show_rolling = st.sidebar.checkbox("Show 30-day rolling trend", value=False)
show_anomalies = st.sidebar.checkbox("Show anomaly flags", value=True)

offline = get_offline_periods(conn, start, end)

# ── Query features (running only) ──────────────────────────────────────
df = conn.execute("""
    SELECT ts, cooler_duty_kw, cooler_approach_c, cooler_gas_dt_c,
           t_gas_in_sc_c, t_gas_out_sc_c, t_cm_in_c, t_cm_out_c,
           q_cm_m3h, is_running
    FROM features_10m
    WHERE ts::DATE BETWEEN ? AND ?
    ORDER BY ts
""", [start, end]).fetchdf()

running = df[df["is_running"]].copy()
st.caption(f"{len(running):,} running-period rows out of {len(df):,} total")

if len(running) == 0:
    st.warning("No running data in the selected range.")
    st.stop()

# Gap-aware rolling: mask offline rows as NaN so rolling doesn't bridge gaps
ROLL_30D = 30 * 144


def gap_aware_rolling(full_df, col):
    """Compute rolling mean on full DataFrame, masking offline rows."""
    series = full_df[col].where(full_df["is_running"])
    return series.rolling(ROLL_30D, min_periods=144).mean()


# ── Query anomaly flags for cooler signals ──────────────────────────────
cooler_anomalies = {}
if show_anomalies:
    for sig in ["cooler_duty_kw", "cooler_approach_c", "cooler_gas_dt_c"]:
        anom = conn.execute("""
            SELECT a.ts, a.score
            FROM anomalies a
            WHERE a.method = 'zscore' AND a.is_anomaly AND a.signal = ?
              AND a.ts::DATE BETWEEN ? AND ?
        """, [sig, start, end]).fetchdf()
        if len(anom) > 0:
            cooler_anomalies[sig] = anom


def _make_cooler_chart(signal, title, ylabel, caption=None):
    """Build a chart for one cooler signal with rolling trend and anomaly flags."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=running["ts"], y=running[signal], mode="lines",
        name=signal, line=dict(color="#4c78a8", width=0.8), opacity=0.5,
    ))

    if show_rolling:
        roll = gap_aware_rolling(df, signal)
        roll_running = roll[df["is_running"]]
        fig.add_trace(go.Scatter(
            x=running["ts"], y=roll_running.values, mode="lines",
            name="30-day rolling mean", line=dict(color="#e45756", width=2),
        ))

    if show_anomalies and signal in cooler_anomalies:
        anom = cooler_anomalies[signal]
        merged = anom.merge(running[["ts", signal]], on="ts", how="inner")
        if len(merged) > 0:
            fig.add_trace(go.Scatter(
                x=merged["ts"], y=merged[signal], mode="markers",
                name=f"z-score flag ({len(merged)})",
                marker=dict(color="#e45756", size=6, symbol="circle"),
            ))

    add_offline_shading(fig, offline)
    fig.update_layout(height=350, margin=dict(t=30, b=30),
                      yaxis_title=ylabel, legend=dict(orientation="h", y=1.1))
    return fig


# ── Cooler duty ─────────────────────────────────────────────────────────
st.subheader("Cooler duty (kW)")
st.plotly_chart(
    _make_cooler_chart("cooler_duty_kw", "Cooler duty", "kW"),
    use_container_width=True,
)

# ── Approach temperature ────────────────────────────────────────────────
st.subheader("Approach temperature (°C)")
st.caption("Gas outlet minus cooling medium inlet — rising trend may indicate fouling")
st.plotly_chart(
    _make_cooler_chart("cooler_approach_c", "Approach", "°C"),
    use_container_width=True,
)

# ── Gas ΔT across cooler ───────────────────────────────────────────────
st.subheader("Gas ΔT across suction cooler (°C)")
st.caption("Gas inlet minus gas outlet — dropping trend may indicate reduced cooling")
st.plotly_chart(
    _make_cooler_chart("cooler_gas_dt_c", "Gas ΔT", "°C"),
    use_container_width=True,
)

# ── Temperature breakdown ──────────────────────────────────────────────
with st.expander("Raw temperatures & flow"):
    fig4 = go.Figure()
    for col, label, color in [
        ("t_gas_in_sc_c", "Gas in", "#4c78a8"),
        ("t_gas_out_sc_c", "Gas out", "#72b7b2"),
        ("t_cm_in_c", "CM in", "#f58518"),
        ("t_cm_out_c", "CM out", "#e45756"),
    ]:
        fig4.add_trace(go.Scatter(
            x=running["ts"], y=running[col], mode="lines",
            name=label, line=dict(color=color, width=1),
        ))
    add_offline_shading(fig4, offline)
    fig4.update_layout(height=350, margin=dict(t=30, b=30),
                       yaxis_title="°C", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=running["ts"], y=running["q_cm_m3h"], mode="lines",
        name="CM flow", line=dict(color="#b279a2", width=1),
    ))
    add_offline_shading(fig5, offline)
    fig5.update_layout(height=250, margin=dict(t=30, b=30),
                       yaxis_title="m³/h")
    st.plotly_chart(fig5, use_container_width=True)
