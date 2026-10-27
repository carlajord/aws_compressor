# Requirements — OID Compressor Copilot

## 1. Goal
Demonstrate an end-to-end ML pipeline on AWS for an industrial use case:
monitor the Valhall 1st-stage gas compressor (23-KA-9101) and its suction cooler,
detect anomalous behaviour, and let users query the data in natural language through
a text-to-SQL agent on Amazon Bedrock. Prototype scope: as simple as possible while using
S3, EC2, IAM and Bedrock.

## 2. Out of scope (v1)
- Real-time / streaming ingestion (batch only)
- Multi-user auth on the app (access restricted by security group to the owner's IP)
- Containers, Terraform/CDK, CI/CD, load balancers, HTTPS
- Model training infrastructure (SageMaker); models are trained on the EC2 box

## 3. Users
- **Owner / data scientist**: builds and runs the pipeline, reviews anomalies, evaluates the agent.
- **Demo viewer** (e.g. interviewer/colleague): looks at dashboards, asks the bot questions.

## 4. User stories and acceptance criteria

### US-1 Data ingestion
As the owner, I want 3–6 months of compressor data extracted from Cognite OID and stored in S3,
so the pipeline has a durable, reproducible source.
- AC-1.1 Extraction script pulls the tags listed in `data_contract.md` at 10-minute average
  granularity for a configurable date range.
- AC-1.2 Output is Parquet in long format (`ts, tag, value`) plus `tags.parquet` and `events.parquet`.
- AC-1.3 Files are uploaded to `s3://<bucket>/raw/` under a dated prefix.
- AC-1.4 Re-running for the same range is idempotent (overwrites same keys).

### US-2 Storage and features
As the owner, I want the raw data loaded into DuckDB with engineered features,
so analytics and the agent query one clean, documented schema.
- AC-2.1 Pipeline reads Parquet from S3 (or local `data/raw/` in dev mode) into DuckDB.
- AC-2.2 Builds `readings_10m`, `features_10m`, `events`, `tags` tables per `data_contract.md`.
- AC-2.3 Features include at least: pressure ratio, cooler heat duty, cooler approach temperature,
  `is_running` flag, 1h/24h rolling mean and std for vibration/bearing temperatures.
- AC-2.4 Pipeline writes to a new DB file and atomically replaces the old one.
- AC-2.5 A copy of the DuckDB file is uploaded to `s3://<bucket>/duckdb/`.

### US-3 Anomaly detection
As the owner, I want simple, explainable anomaly detection,
so I can see when the compressor behaved unusually.
- AC-3.1 Method A: rolling z-score per signal, computed only on `is_running` periods; anomaly if |z| > threshold (configurable, default 4).
- AC-3.2 Method B: Isolation Forest on a fixed feature set, trained on a configurable "normal" period.
- AC-3.3 Results stored in `anomalies` table with method, score, and flag.
- AC-3.4 A summary query relates anomaly windows to maintenance events within ±7 days.

### US-4 Dashboard app
As a viewer, I want a web app with dashboards,
so I can explore compressor behaviour without writing code.
- AC-4.1 Streamlit app with pages: Overview, Anomalies, Cooler Performance, Chat, Traces & Evals.
- AC-4.2 Overview: KPIs (running hours, avg pressure ratio, anomaly count) and time-series plots with date filter.
- AC-4.3 Anomalies: plot signals with anomaly markers, plus table of events nearby.
- AC-4.4 App opens DuckDB read-only.
- AC-4.5 App runs on EC2, port 8501, reachable only from the owner's IP.

### US-5 Text-to-SQL agent
As a viewer, I want to ask questions in plain English and get answers backed by SQL,
so I can explore the data conversationally.
- AC-5.1 Agent uses Bedrock Converse API with tool use; tools: `get_schema`, `run_sql`.
- AC-5.2 `run_sql` only accepts SELECT/WITH, runs on a read-only connection, enforces a row limit.
- AC-5.3 Agent loop is capped at 5 tool steps.
- AC-5.4 The chat UI shows the answer, the SQL executed, and a result table (and a chart when the result is a time series).
- AC-5.5 The system prompt includes schema, units, and the domain glossary.
- AC-5.6 On SQL error, the agent receives the error message and may retry within the step cap.

### US-6 Tracing
As the owner, I want every agent run traced,
so I can debug and measure behaviour.
- AC-6.1 Each run has a `trace_id`; each LLM call and tool call is a span with input, output, latency, tokens, status.
- AC-6.2 Spans stored in a separate DuckDB file `traces.duckdb` (keeps the analytics DB read-only).
- AC-6.3 Traces page lists runs and shows the span tree for a selected run.

### US-7 Evaluation
As the owner, I want an offline evaluation of the agent,
so I can compare prompt/model versions objectively.
- AC-7.1 `evals/golden.yaml` with ≥ 20 questions, each with reference SQL and difficulty tag.
- AC-7.2 `evals/run_evals.py` reports: SQL validity rate, execution accuracy (result-set match), mean tool steps, p50/p95 latency, total tokens.
- AC-7.3 Results saved as timestamped JSON locally and to `s3://<bucket>/evals/`.
- AC-7.4 Evals page shows the latest run and a comparison across runs.

## 5. Non-functional requirements
- NFR-1 Cost: < $20/month; EC2 stopped when idle.
- NFR-2 Security: no AWS keys on EC2 (IAM instance role); S3 public access blocked; port 8501 open only to owner IP.
- NFR-3 Reproducibility: `uv.lock` committed; one command builds the DB from raw data.
- NFR-4 Testability: core logic covered by unit tests with mocked AWS.
