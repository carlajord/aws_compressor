"""Streamlit app entry point — Home page."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import get_conn, date_range_slider, get_offline_periods, add_offline_shading

st.set_page_config(page_title="Home — OID Compressor", layout="wide")
st.title("OID Compressor Copilot")

conn = get_conn()

# ── Date range slider ───────────────────────────────────────────────────
start_date, end_date = date_range_slider(conn, sidebar=False)

# ── Query daily KPIs for the selected range ─────────────────────────────
kpis = conn.execute("""
    SELECT * FROM v_daily_kpis
    WHERE day BETWEEN ? AND ?
    ORDER BY day
""", [start_date, end_date]).fetchdf()

offline = get_offline_periods(conn, start_date, end_date)

# ── KPI cards (computed from filtered range) ────────────────────────────
total_days = len(kpis)
total_running_h = kpis["running_hours"].sum()
running_kpis = kpis[kpis["running_hours"] > 0]
avg_pr = running_kpis["avg_pressure_ratio"].mean() if len(running_kpis) > 0 else 0
avg_duty = running_kpis["avg_cooler_duty_kw"].mean() if len(running_kpis) > 0 else 0
total_anomalies = int(kpis["anomaly_count"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Days", f"{total_days}")
c2.metric("Running hours", f"{total_running_h:,.0f}")
c3.metric("Avg pressure ratio", f"{avg_pr:.2f}")
c4.metric("Avg cooler duty", f"{avg_duty:,.0f} kW")
c5.metric("Total anomalies", f"{total_anomalies:,}")

st.divider()

# ── Daily anomaly count vs running hours ────────────────────────────────
st.subheader("Daily anomaly count & running hours")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(
    go.Bar(x=kpis["day"], y=kpis["anomaly_count"], name="Anomalies",
           marker_color="#e45756", opacity=0.7),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(x=kpis["day"], y=kpis["running_hours"], name="Running hours",
               line=dict(color="#4c78a8", width=1.5)),
    secondary_y=True,
)
add_offline_shading(fig1, offline)
fig1.update_layout(height=350, margin=dict(t=30, b=30),
                   legend=dict(orientation="h", y=1.1))
fig1.update_yaxes(title_text="Anomaly count", secondary_y=False)
fig1.update_yaxes(title_text="Running hours", secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# ── Daily avg pressure ratio & cooler duty (running only) ───────────────
st.subheader("Daily pressure ratio & cooler duty")
st.caption("Running periods only — offline days excluded")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Scatter(x=kpis["day"], y=kpis["avg_pressure_ratio"],
               name="Avg pressure ratio", line=dict(color="#4c78a8", width=1.5)),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(x=kpis["day"], y=kpis["avg_cooler_duty_kw"],
               name="Avg cooler duty (kW)", line=dict(color="#72b7b2", width=1.5)),
    secondary_y=True,
)
add_offline_shading(fig2, offline)
fig2.update_layout(height=350, margin=dict(t=30, b=30),
                   legend=dict(orientation="h", y=1.1))
fig2.update_yaxes(title_text="Pressure ratio", secondary_y=False)
fig2.update_yaxes(title_text="Cooler duty (kW)", secondary_y=True)
st.plotly_chart(fig2, use_container_width=True)

# ── Daily max vibration (running only) ──────────────────────────────────
st.subheader("Daily max vibration")
st.caption("Running periods only")

fig3 = go.Figure()
for col, label in [
    ("max_vib_mtr_de_1", "DE 1"),
    ("max_vib_mtr_de_2", "DE 2"),
    ("max_vib_mtr_nde_1", "NDE 1"),
    ("max_vib_mtr_nde_2", "NDE 2"),
]:
    fig3.add_trace(go.Scatter(x=kpis["day"], y=kpis[col],
                              name=label, mode="lines", line=dict(width=1.2)))

add_offline_shading(fig3, offline)
fig3.update_layout(height=350, margin=dict(t=30, b=30),
                   yaxis_title="Vibration (max daily)",
                   legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig3, use_container_width=True)

# ── Data summary table ──────────────────────────────────────────────────
with st.expander("Data summary"):
    counts = conn.execute("""
        SELECT 'tags' AS table_name, count(*) AS rows FROM tags
        UNION ALL SELECT 'readings_10m', count(*) FROM readings_10m
        UNION ALL SELECT 'features_10m', count(*) FROM features_10m
        UNION ALL SELECT 'anomalies', count(*) FROM anomalies
        UNION ALL SELECT 'events', count(*) FROM events
        ORDER BY table_name
    """).fetchdf()
    st.dataframe(counts, hide_index=True, use_container_width=True)
