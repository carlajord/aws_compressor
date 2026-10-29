"""Anomalies page — signal selector, anomaly markers, windows timeline, nearby events."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from db import add_offline_shading, date_range_slider, get_conn, get_offline_periods
from plotly.subplots import make_subplots

st.set_page_config(page_title="Anomalies — OID Compressor", layout="wide")
st.title("Anomalies")

conn = get_conn()

# ── Sidebar filters ─────────────────────────────────────────────────────
COOLER_SIGNALS = {"cooler_approach_c", "cooler_duty_kw", "cooler_gas_dt_c"}

signals = [
    s for s in conn.execute(
        "SELECT DISTINCT signal FROM anomalies ORDER BY signal"
    ).fetchdf()["signal"].tolist()
    if s not in COOLER_SIGNALS
]

selected_signal = st.sidebar.selectbox("Signal", signals, index=0)

methods = conn.execute(
    "SELECT DISTINCT method FROM anomalies ORDER BY method"
).fetchdf()["method"].tolist()
selected_methods = st.sidebar.multiselect("Methods", methods, default=methods)

start, end = date_range_slider(conn, sidebar=True)

only_flagged = st.sidebar.checkbox("Show only flagged anomalies", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "**Z-score** = |value − 24h rolling mean| / 24h rolling std. "
    "Flagged when score > configured threshold.\n\n"
    "**IForest score** = −decision_function (higher = more anomalous). "
    "Flagged by contamination parameter."
)

offline = get_offline_periods(conn, start, end)

# ── Query anomalies ─────────────────────────────────────────────────────
method_list = ", ".join(f"'{m}'" for m in selected_methods)
flag_filter = "AND is_anomaly" if only_flagged else ""

if not selected_methods:
    st.warning("Select at least one method.")
    st.stop()

anomalies = conn.execute(f"""
    SELECT ts, signal, method, score, is_anomaly
    FROM anomalies
    WHERE signal = ?
      AND method IN ({method_list})
      AND ts::DATE BETWEEN ? AND ?
      {flag_filter}
    ORDER BY ts
""", [selected_signal, start, end]).fetchdf()

n_flagged = int(anomalies["is_anomaly"].sum()) if len(anomalies) > 0 else 0
st.caption(f"{n_flagged:,} flagged anomalies / {len(anomalies):,} scored rows for **{selected_signal}**")

# ── Signal time-series with anomaly markers ─────────────────────────────
if selected_signal != "multivariate":
    raw = conn.execute("""
        SELECT ts, value FROM readings_10m
        WHERE tag = ? AND ts::DATE BETWEEN ? AND ?
        ORDER BY ts
    """, [selected_signal, start, end]).fetchdf()

    fig = go.Figure()
    if len(raw) > 0:
        fig.add_trace(go.Scatter(
            x=raw["ts"], y=raw["value"], mode="lines",
            name=selected_signal, line=dict(color="#4c78a8", width=1),
        ))

    for method in selected_methods:
        subset = anomalies[(anomalies["method"] == method) & anomalies["is_anomaly"]]
        if len(subset) == 0:
            continue
        merged = subset.merge(raw, on="ts", how="inner")
        color = "#e45756" if method == "zscore" else "#f58518"
        fig.add_trace(go.Scatter(
            x=merged["ts"], y=merged["value"], mode="markers",
            name=f"{method} flag",
            marker=dict(color=color, size=6, symbol="circle"),
        ))

    add_offline_shading(fig, offline)
    fig.update_layout(
        height=400, margin=dict(t=30, b=30),
        xaxis_title="Date", yaxis_title=selected_signal,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = go.Figure()
    normal = anomalies[~anomalies["is_anomaly"]]
    flagged = anomalies[anomalies["is_anomaly"]]
    if len(normal) > 0:
        fig.add_trace(go.Scatter(
            x=normal["ts"], y=normal["score"], mode="markers",
            marker=dict(size=3, color="#4c78a8", opacity=0.3), name="Normal",
        ))
    if len(flagged) > 0:
        fig.add_trace(go.Scatter(
            x=flagged["ts"], y=flagged["score"], mode="markers",
            marker=dict(size=5, color="#e45756"), name="Anomaly",
        ))
    add_offline_shading(fig, offline)
    fig.update_layout(
        height=400, margin=dict(t=30, b=30),
        xaxis_title="Date", yaxis_title="IForest score (−decision_function)",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Score distribution ──────────────────────────────────────────────────
with st.expander("Score distribution"):
    st.caption(
        "**Z-score**: |z| values — how many standard deviations from the 24h rolling mean. "
        "**IForest**: −decision_function — distance from the learned normal manifold."
    )
    flagged_only = anomalies[anomalies["is_anomaly"]]
    if len(flagged_only) > 0:
        fig_hist = go.Figure()
        for method in selected_methods:
            m_data = flagged_only[flagged_only["method"] == method]
            if len(m_data) > 0:
                fig_hist.add_trace(go.Histogram(
                    x=m_data["score"], name=method, opacity=0.7,
                ))
        fig_hist.update_layout(
            barmode="overlay", height=300,
            xaxis_title="Anomaly score", yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No flagged anomalies in the selected range.")

# ── Anomaly windows — timeline visualization ────────────────────────────
st.subheader("Anomaly windows")
st.caption("Consecutive flagged rows merged when gap ≤ 30 min")

windows = conn.execute(f"""
    SELECT window_id, method, signal, start_ts, end_ts, max_score, n_points
    FROM v_anomaly_windows
    WHERE signal = ?
      AND method IN ({method_list})
    ORDER BY start_ts
""", [selected_signal]).fetchdf()

if len(windows) > 0:
    fig_timeline = go.Figure()
    colors = {"zscore": "#e45756", "iforest": "#f58518"}

    for method in selected_methods:
        mw = windows[windows["method"] == method]
        if len(mw) == 0:
            continue
        fig_timeline.add_trace(go.Scatter(
            x=mw["start_ts"],
            y=mw["max_score"],
            mode="markers",
            marker=dict(
                size=mw["n_points"].clip(upper=30) + 4,
                color=colors.get(method, "#999"),
                opacity=0.7,
            ),
            name=f"{method} ({len(mw)} windows)",
            customdata=mw[["window_id", "n_points", "start_ts", "end_ts"]].values,
            hovertemplate=(
                "Window %{customdata[0]}<br>"
                "Score: %{y:.2f}<br>"
                "Points: %{customdata[1]}<br>"
                "%{customdata[2]} → %{customdata[3]}<extra></extra>"
            ),
        ))

    add_offline_shading(fig_timeline, offline)
    fig_timeline.update_layout(
        height=350, margin=dict(t=30, b=30),
        xaxis_title="Date", yaxis_title="Max score in window",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    with st.expander("Window details table"):
        st.dataframe(windows, hide_index=True, use_container_width=True)
else:
    st.info("No anomaly windows for this signal/method combination.")

# ── Nearby events ───────────────────────────────────────────────────────
st.subheader("Anomalies near events")

n_events = conn.execute("SELECT count(*) FROM events").fetchone()[0]
if n_events == 0:
    st.info("No events in the database. Run the extraction with events to populate this table.")
else:
    near = conn.execute(f"""
        SELECT ne.window_id, ne.event_id, ne.event_type, ne.days_between,
               w.method, w.signal, w.start_ts, w.max_score
        FROM v_anomalies_near_events ne
        JOIN v_anomaly_windows w ON ne.window_id = w.window_id
        WHERE w.signal = ?
          AND w.method IN ({method_list})
        ORDER BY ABS(ne.days_between)
        LIMIT 50
    """, [selected_signal]).fetchdf()

    if len(near) > 0:
        st.caption(f"{len(near)} anomaly–event pairs within ±7 days")
        st.dataframe(near, hide_index=True, use_container_width=True)
    else:
        st.info("No anomaly windows for this signal overlap with events (±7 days).")

# ── Shutdown drill-down ────────────────────────────────────────────────
st.subheader("Shutdown drill-down")
st.caption(
    "Raw signals ±3 days around each significant offline period (≥1 h), "
    "with anomaly markers overlaid. Reveals whether detectors flagged "
    "anything *before* the machine went offline."
)

shutdowns = conn.execute("""
    WITH numbered AS (
        SELECT ts,
               ts - INTERVAL 10 MINUTE * row_number() OVER (ORDER BY ts) AS grp
        FROM features_10m
        WHERE NOT is_running
    )
    SELECT min(ts) AS start_ts, max(ts) AS end_ts, count(*) AS n_points,
           EXTRACT(EPOCH FROM max(ts) - min(ts)) / 3600 AS duration_hrs
    FROM numbered
    GROUP BY grp
    HAVING count(*) >= 6
    ORDER BY start_ts
""").fetchdf()

if len(shutdowns) == 0:
    st.info("No offline periods ≥ 1 h found in the data.")
else:
    options = [
        f"Shutdown {i + 1}: {row['start_ts'].strftime('%Y-%m-%d %H:%M')} "
        f"({row['duration_hrs']:.0f}h offline)"
        for i, row in shutdowns.iterrows()
    ]
    chosen_idx = st.selectbox(
        "Select a shutdown period", range(len(options)), format_func=lambda i: options[i]
    )

    sd = shutdowns.iloc[chosen_idx]
    context_days = 3
    window_start = sd["start_ts"] - pd.Timedelta(days=context_days)
    window_end = sd["end_ts"] + pd.Timedelta(days=context_days)

    feat = conn.execute("""
        SELECT ts, vib_mtr_de_1, vib_mtr_de_2, vib_mtr_nde_1, vib_mtr_nde_2,
               t_bearing_comp_de, t_bearing_comp_nde, t_bearing_mtr_de, t_bearing_mtr_nde,
               pressure_ratio, is_running
        FROM features_10m
        WHERE ts BETWEEN ? AND ?
        ORDER BY ts
    """, [window_start, window_end]).fetchdf()

    anoms = conn.execute("""
        SELECT ts, signal, method, score
        FROM anomalies
        WHERE is_anomaly AND ts BETWEEN ? AND ?
    """, [window_start, window_end]).fetchdf()

    fig_drill = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=["Vibration", "Bearing temps", "Pressure ratio + flags"],
    )

    for sig in ["vib_mtr_de_1", "vib_mtr_de_2", "vib_mtr_nde_1", "vib_mtr_nde_2"]:
        if sig in feat.columns:
            fig_drill.add_trace(
                go.Scatter(x=feat["ts"], y=feat[sig], name=sig, line=dict(width=1)),
                row=1, col=1,
            )

    for sig in ["t_bearing_comp_de", "t_bearing_comp_nde", "t_bearing_mtr_de", "t_bearing_mtr_nde"]:
        if sig in feat.columns:
            fig_drill.add_trace(
                go.Scatter(x=feat["ts"], y=feat[sig], name=sig, line=dict(width=1)),
                row=2, col=1,
            )

    fig_drill.add_trace(
        go.Scatter(
            x=feat["ts"], y=feat["pressure_ratio"],
            name="pressure_ratio", line=dict(width=1, color="#4c78a8"),
        ),
        row=3, col=1,
    )

    pr_min = feat["pressure_ratio"].min() if len(feat) > 0 else 0
    for method, color, symbol in [("zscore", "#e45756", "triangle-up"),
                                   ("iforest", "#f58518", "diamond")]:
        subset = anoms[anoms["method"] == method]
        if len(subset) > 0:
            fig_drill.add_trace(
                go.Scatter(
                    x=subset["ts"], y=[pr_min] * len(subset), mode="markers",
                    marker=dict(symbol=symbol, size=8, color=color),
                    name=f"{method} flag",
                ),
                row=3, col=1,
            )

    fig_drill.add_vrect(
        x0=sd["start_ts"], x1=sd["end_ts"],
        fillcolor="grey", opacity=0.25, line_width=0,
    )

    n_zs = len(anoms[anoms["method"] == "zscore"])
    n_if = len(anoms[anoms["method"] == "iforest"])
    st.caption(f"{n_zs} zscore / {n_if} iforest flags in ±{context_days}d window")

    fig_drill.update_layout(
        height=700, margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_drill, use_container_width=True)
