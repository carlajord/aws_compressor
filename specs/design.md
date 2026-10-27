# Design — OID Compressor Copilot

## 1. Architecture

```
[Laptop]                                   [AWS, single region]
Cognite OID ──cognite-sdk──> data/raw/*.parquet ──(user: aws s3 cp)──> S3 raw/
                                                                         │
                                         EC2 t3.medium, Amazon Linux 2023, IAM role
                                         ├─ oidpipe.pipeline: S3 → DuckDB → features → anomalies
                                         ├─ Streamlit (port 8501, owner IP only)
                                         └─ SQL agent ──OpenAI API──> Bedrock (via proxy)
                                                                         │
                                         S3 duckdb/ (DB backup), evals/ (eval results)
```

Dev mode (`OID_ENV=local`): everything runs on the laptop, reading `data/raw/` instead of S3,
using the owner's local AWS profile only for Bedrock.

## 2. Configuration (`config/settings.py`)
pydantic-settings, env prefix `OID_`, optional `.env` file.

| Setting | Example | Notes |
|---|---|---|
| `env` | `local` / `aws` | switches raw-data source |
| `aws_region` | `us-east-1` | |
| `s3_bucket` | `oid-compressor-<name>` | |
| `raw_prefix` | `raw/` | |
| `db_path` | `data/oid.duckdb` | analytics DB |
| `traces_db_path` | `data/traces.duckdb` | agent traces |
| `llm_model` | `openai.gpt-oss-120b` | model ID for the OpenAI-compatible endpoint |
| `agent_max_steps` | `5` | |
| `sql_row_limit` | `500` | |
| `zscore_threshold` | `4.0` | z-score flag threshold; also defines iforest normal period |
| `iforest_contamination` | `0.01` | |

## 3. Modules

```
src/oidpipe/
├── ingest/
│   ├── cdf_extract.py   # CLI: extract tags/time series/events from CDF → data/raw/ (laptop, user-run)
│   └── s3_io.py         # list/download/upload helpers; used by pipeline on EC2
├── db/
│   ├── duck.py          # connect(read_only), init_schema(), atomic_replace()
│   └── sql/             # schema.sql, views.sql
├── features/
│   └── build.py         # pure functions on DataFrames + orchestration into DuckDB
├── anomaly/
│   ├── zscore.py
│   ├── iforest.py
│   └── detect.py        # runs both, writes anomalies table
├── agent/
│   ├── prompts.py       # system prompt template (schema + glossary + rules)
│   ├── tools.py         # get_schema, run_sql, TOOL_CONFIG
│   ├── sql_agent.py     # prompt-based tool-use loop → AgentResult
│   └── tracing.py       # Tracer, span context manager → traces.duckdb
└── pipeline.py          # CLI: python -m oidpipe.pipeline [--local]
app/
├── streamlit_app.py     # entry / Overview
└── pages/ 1_Anomalies.py, 2_Cooler_Performance.py, 3_Chat.py, 4_Traces_and_Evals.py
evals/
├── golden.yaml
└── run_evals.py
infra/
├── iam-ec2-policy.json  # least-privilege policy for the EC2 role
├── streamlit.service    # systemd unit
└── bootstrap.sh         # EC2 setup script (user runs)
```

## 4. Pipeline flow (`oidpipe.pipeline`)
1. Resolve raw source (S3 via DuckDB `httpfs` + `CREATE SECRET (TYPE s3, PROVIDER credential_chain)`, or local dir).
2. Create `data/oid.duckdb.tmp`, run `schema.sql`.
3. Load `tags`, `events`, `readings_10m` from Parquet.
4. Build `features_10m` (pivot long → wide on logical tag names, compute derived features).
5. Run anomaly detection → `anomalies`.
6. Create views (`v_daily_kpis`, `v_anomaly_windows`, `v_anomalies_near_events`).
7. Close, atomically rename `.tmp` → `oid.duckdb`.
8. (aws env) upload DB copy to `s3://bucket/duckdb/oid_<timestamp>.duckdb`.

## 5. Feature engineering
Logical names come from `data_contract.md`; all formulas are pure functions in `features/build.py`.

| Feature | Formula / method |
|---|---|
| `pressure_ratio` | `(p_discharge_barg + 1.013) / (p_suction_barg + 1.013)` |
| `cooler_duty_kw` | `m_dot * cp * (t_cm_out - t_cm_in) / 1000`, `m_dot = q_cm_m3h * 942 / 3600`, cp = 4400 (only if cooling-medium flow tag exists; else omitted and documented) |
| `cooler_approach_c` | `t_gas_out_suction_cooler - t_cm_in` |
| `cooler_gas_dt_c` | `t_gas_in - t_gas_out` across suction cooler |
| `is_running` | motor current (or speed/power proxy) > configurable threshold |
| `asv_open_pct` | direct, plus `asv_recycling` flag if > 5 % |
| `*_roll1h_mean/std`, `*_roll24h_mean/std` | rolling windows on vibration & bearing temps, running periods only |
| `*_roc` | first difference per 10 min |

## 6. Anomaly detection
- **Z-score** (`zscore.py`): for each monitored signal, z = (x − roll24h_mean) / roll24h_std on running
  periods; flag |z| > threshold. Score = |z|.
- **Isolation Forest** (`iforest.py`): scikit-learn, features = vibration + bearing temps + pressure ratio
  + cooler approach; StandardScaler fit on normal period; score = −decision_function; flag by contamination.
- Anomaly windows: consecutive flagged rows merged if gap ≤ 30 min.

## 7. Agent design

### LLM backend
- OpenAI-compatible API (wraps Bedrock models). Client: `openai.OpenAI()`, reads
  `OPENAI_API_KEY` and `OPENAI_BASE_URL` from env.
- Model set via `OID_LLM_MODEL` (default `openai.gpt-oss-120b`).
- **Prompt-based tool calling**: the endpoint does not support structured `tool_calls`, so the
  system prompt instructs the model to emit `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
  tags. The agent loop parses these from the text response.

### Tools
- `get_schema(table: str | None)` → table/column names, types, column comments, 3 sample rows.
- `run_sql(sql: str)` → `{columns, rows (≤50 shown), row_count, truncated}` or `{error}`.
  Guards: strip comments; must start with SELECT/WITH; reject `;`-separated multi-statements and
  keywords INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/COPY/INSTALL/LOAD/PRAGMA; wrap as
  `SELECT * FROM (<sql>) LIMIT <sql_row_limit>`; read-only connection; 10 s timeout.

### Loop (`sql_agent.ask(question, history) -> AgentResult`)
- System prompt: role, rules (always use tools, never invent numbers, state units, prefer views),
  compact schema, glossary, 3 few-shot Q→SQL examples, tool descriptions and `<tool_call>` format.
- Up to `agent_max_steps` chat completions; tool results appended as user messages.
- `AgentResult`: `answer, sql_list, last_result_df, steps, input_tokens, output_tokens, latency_ms, trace_id, error`.
- Errors from the API → one retry with backoff, then a graceful message.

### Tracing (`tracing.py`)
- `Tracer(trace_id)`; `with tracer.span(kind, name, input) as s: ... s.record(output=..., usage=...)`.
- Table `agent_spans(trace_id, span_id, parent_id, kind, name, started_at, latency_ms, input_json,
  output_json, input_tokens, output_tokens, status, error)`; table `agent_runs(trace_id, question,
  answer, model_id, prompt_version, steps, total_tokens, latency_ms, created_at, source)`.
- `source` = `app` or `eval`.

### Evaluation (`evals/run_evals.py`)
- For each golden item: run agent, take the last executed SQL's result; run reference SQL.
- **Execution match**: compare as sets of rows after rounding floats to 3 decimals and ignoring
  column names/order; for single-value answers compare with relative tolerance 1 %.
- Metrics: validity rate, execution accuracy (overall and by difficulty), mean steps, p50/p95 latency,
  tokens, estimated cost.
- Output: `evals/results/<timestamp>.json` (+ user uploads to S3).
- `prompt_version` constant in `prompts.py` is recorded with every run for comparison.

## 8. Streamlit app
- Shared cached read-only DuckDB connection (`st.cache_resource`).
- Plotly for charts.
- Chat page: `st.chat_message`, expander "SQL & data" per answer, auto line chart if result has a
  timestamp column + numeric column.
- Traces & Evals page: runs table → select → span tree; latest eval metrics and history chart.

## 9. AWS resources (created by the user, documented in tasks.md)
- S3 bucket (block public access on), prefixes `raw/`, `duckdb/`, `evals/`.
- IAM role `oid-ec2-role`: S3 Get/Put/List on the bucket; `bedrock:InvokeModel` (Converse uses it);
  managed policy `AmazonSSMManagedInstanceCore`.
- EC2 t3.medium, Amazon Linux 2023, 20 GB gp3, security group inbound TCP 8501 from owner IP only.
- Bedrock: chosen Claude model enabled in the console.
