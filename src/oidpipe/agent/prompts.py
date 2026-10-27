"""System prompt builder for the text-to-SQL agent."""

from __future__ import annotations

import json

import duckdb

from oidpipe.agent.tools import TOOL_DESCRIPTIONS, get_schema

PROMPT_VERSION = "v1"

GLOSSARY = """\
## Domain glossary
- 23-KA-9101: 1st-stage compressor (electric, fixed speed, centrifugal), ~3 → ~12 barg
- 23-HA-9103: 1st-stage suction cooler; 23-HA-9114/9115: discharge coolers
- 23-VG-9101: 1st-stage suction scrubber
- 23-FV-92543: anti-surge valve (ASV); 23-PV-92538: suction throttle valve (STV)
- barg = gauge pressure; bara = barg + 1.013
- pressure_ratio = (p_discharge_barg + 1.013) / (p_suction_barg + 1.013)
- cooler_duty_kw = heat removed by suction cooler in kW
- cooler_approach_c = gas outlet temp minus cooling medium inlet temp (°C)
- cooler_gas_dt_c = gas temperature drop across suction cooler (°C)
- is_running = TRUE when compressor is operating (motor current above threshold)
- Cooling medium: 70% water / 30% TEG, ρ ≈ 942 kg/m³, cp ≈ 4400 J/(kg·K)
- Vibration tags (vib_mtr_*): motor journal bearing proximity probes, DE = drive end, NDE = non-drive end
- Anomaly methods: 'zscore' (rolling z-score), 'iforest' (Isolation Forest)
"""

FEW_SHOT = """\
## Examples

Q: What is the average pressure ratio when the compressor is running?
Thought: I need to query features_10m for running periods.
<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT AVG(pressure_ratio) AS avg_pr FROM features_10m WHERE is_running"}}</tool_call>

Tool result: {"columns": ["avg_pr"], "rows": [[3.87]], "row_count": 1, "truncated": false}

The average pressure ratio when the compressor is running is **3.87**.

---

Q: How many anomaly windows were detected by z-score for vibration signals?
Thought: I should use the v_anomaly_windows view filtered by method and signal.
<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT signal, count(*) AS n_windows FROM v_anomaly_windows WHERE method = 'zscore' AND signal LIKE 'vib_%' GROUP BY signal ORDER BY n_windows DESC"}}</tool_call>

Tool result: {"columns": ["signal", "n_windows"], "rows": [["vib_mtr_de_1", 5], ["vib_mtr_nde_2", 3]], "row_count": 2, "truncated": false}

Z-score detected **5 anomaly windows** for vib_mtr_de_1 and **3** for vib_mtr_nde_2.

---

Q: What tables are available?
Thought: I'll use get_schema with no argument to list tables.
<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>

Tool result: {"tables": [{"name": "tags", "type": "BASE TABLE"}, {"name": "readings_10m", "type": "BASE TABLE"}, ...]}

The database has the following tables and views: tags, readings_10m, events, features_10m, anomalies, v_daily_kpis, v_anomaly_windows, v_anomalies_near_events.
"""


def _format_tool_docs() -> str:
    """Format tool descriptions for the system prompt."""
    lines = ["## Available tools", ""]
    lines.append(
        "To call a tool, output exactly one tag on its own line:\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n'
        "\nWait for the tool result before continuing. "
        "Do not guess the result of a tool call.\n"
    )
    for name, spec in TOOL_DESCRIPTIONS.items():
        params = spec["parameters"]
        param_strs = [
            f"  - {p}: ({info['type']}) {info['description']}"
            for p, info in params.items()
        ]
        lines.append(f"### {name}")
        lines.append(spec["description"])
        if param_strs:
            lines.append("Parameters:")
            lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)


def _format_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """Build a compact schema section from the live database."""
    listing = get_schema(conn)
    lines = ["## Database schema", ""]

    for tbl in listing.get("tables", []):
        name = tbl["name"]
        detail = get_schema(conn, table=name)
        cols = detail.get("columns", [])
        col_parts = []
        for c in cols:
            s = f"  {c['name']} ({c['type']})"
            if "comment" in c:
                s += f" — {c['comment']}"
            col_parts.append(s)
        lines.append(f"**{name}** ({tbl['type']}):")
        lines.extend(col_parts)
        lines.append("")

    return "\n".join(lines)


def build_system_prompt(conn: duckdb.DuckDBPyConnection) -> str:
    """Assemble the full system prompt with schema, glossary, tools, and examples."""
    parts = [
        (
            "You are a text-to-SQL assistant for the OID Compressor Copilot analytics database. "
            "This database contains sensor data, features, and anomaly detection results for the "
            "Valhall 1st-stage gas compressor (23-KA-9101).\n"
        ),
        "IMPORTANT: Always write your response as visible output text. "
        "Do not use internal reasoning or thinking — output everything directly.\n",
        "## Rules",
        "1. Always use tools to look up data. Never invent numbers or guess query results.",
        "2. State units in your answer (barg, °C, kW, %, etc.).",
        "3. Prefer views (v_daily_kpis, v_anomaly_windows, v_anomalies_near_events) when they fit.",
        "4. Filter by is_running = TRUE when computing operational statistics.",
        "5. Keep SQL simple and readable. Use CTEs for complex queries.",
        "6. If a query fails, read the error and fix the SQL before retrying.",
        "7. When answering, cite the data. Show key numbers from the query result.",
        "8. If you are unsure which table or column to use, call get_schema first.",
        "9. Use standard SQL operators: >= not ≥, <= not ≤, != not ≠. The database is DuckDB.",
        "",
        _format_tool_docs(),
        _format_schema(conn),
        GLOSSARY,
        FEW_SHOT,
        "\nREMINDER: Output your response directly as visible text. "
        "To call a tool, write the <tool_call> tag. Do not think silently.",
    ]
    return "\n".join(parts)
