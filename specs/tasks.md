# Tasks — OID Compressor Copilot

Legend: `[ ]` todo, `[x]` done. **▶ USER** = the user runs it; Claude Code prints the commands,
explains them, and waits (see CLAUDE.md). Claude Code stops for review after each task.

## Phase 0 — Project setup (local)
- [x] **T0.1** Scaffold repo: `pyproject.toml` (uv, Python 3.11), `src/oidpipe/` package skeleton per
  design §3, `app/`, `evals/`, `infra/`, `tests/`, `.gitignore` (data/, .env, *.duckdb), `.env.example`,
  `Makefile` with targets `test`, `lint`, `pipeline-local`, `app`.
  Deps: duckdb, pandas, pyarrow, scikit-learn, plotly, streamlit, boto3, pydantic-settings, pyyaml,
  cognite-sdk; dev: pytest, ruff, mypy, moto.
  *Done when:* `uv run pytest` passes with one smoke test.
- [x] **T0.2** `config/settings.py` per design §2 + test that env vars override defaults.
- [x] **T0.3** ▶ USER: AWS account hygiene — MFA on root, budget alert ($20), choose region,
  create IAM user + access key, `aws configure`, verify with `aws sts get-caller-identity`.
  Claude Code: print step-by-step instructions (console clicks + commands) and wait.

## Phase 1 — Data extraction (laptop)
- [x] **T1.1** `ingest/cdf_extract.py` CLI skeleton: auth via Cognite OIDC interactive login
  (credentials from env), `--list-tags` mode to dump candidate time series under the 1st-stage
  compressor assets to `data/raw/tag_candidates.csv`.
- [x] **T1.2** ▶ USER: run `--list-tags`, then together fill in the CDF external IDs and units in
  `data_contract.md` §2 and generate `config/tags.yaml`. Record dropped features.
- [x] **T1.3** Extraction mode: `retrieve_dataframe(..., aggregates=["average"], granularity="10m")`
  in monthly chunks → `readings.parquet` (long), `tags.parquet`, `events.parquet`. Unit tests with a
  mocked CogniteClient.
- [x] **T1.4** ▶ USER: run the extraction for the chosen date range; report file sizes and row counts.

## Phase 2 — DuckDB + features (local)
- [x] **T2.1** `db/duck.py` + `db/sql/schema.sql`: connect (read_only flag), init schema with
  COMMENTs, atomic replace. Tests.
- [x] **T2.2** Load raw Parquet (local dir) into `tags`, `events`, `readings_10m` (map external_id →
  logical name). Data quality checks per data contract §6.
- [x] **T2.3** `features/build.py`: pure functions for each feature in design §5 + pivot to
  `features_10m`. Unit tests with synthetic data (e.g. known cooler duty).
- [x] **T2.4** `pipeline.py --local` end-to-end on local data; `make pipeline-local`.
  Claude Code may run this (local files only). Print table row counts.

## Phase 3 — Anomaly detection (local)
- [x] **T3.1** `anomaly/zscore.py` + tests (injected spike is detected; non-running periods ignored).
- [x] **T3.2** `anomaly/iforest.py` + tests (fit on normal window, flags injected multivariate shift).
- [x] **T3.3** `anomaly/detect.py` writes `anomalies`; create views from data contract §4; wire into pipeline.
- [x] **T3.4** Short notebook `notebooks/01_eda_anomalies.ipynb` comparing both methods against events
  (Claude Code writes it; user runs cells).

## Phase 4 — Streamlit dashboards (local)
- [x] **T4.1** Overview page (KPIs from `v_daily_kpis`, time-series with date filter).
- [x] **T4.2** Anomalies page (signal selector, anomaly markers, nearby events table).
- [x] **T4.3** Cooler Performance page (duty, approach, gas ΔT trends; 30-day rolling trend line).
- [x] **T4.4** ▶ USER: `make app` and review in browser at localhost:8501.

## Phase 5 — Text-to-SQL agent (local)
- [x] **T5.1** ▶ USER: set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OID_LLM_MODEL` in `.env`;
  run `uv run python scripts/bedrock_smoke.py` — verifies text generation and prompt-based
  tool calling via the OpenAI-compatible API.
- [x] **T5.2** `agent/tools.py`: `get_schema`, `run_sql` with all guards from design §7. Tests for
  every rejected statement type and for the row limit.
- [x] **T5.3** `agent/tracing.py` + `traces.duckdb` schema. Tests.
- [x] **T5.4** `agent/prompts.py` (schema injection, glossary, rules, few-shot, `PROMPT_VERSION`)
  and `agent/sql_agent.py` loop with prompt-based tool use (parse `<tool_call>` tags from text).
  Tests with a mocked OpenAI client (scripted responses, step cap, SQL error retry).
- [x] **T5.5** Chat page + Traces page in Streamlit.
- [ ] **T5.6** ▶ USER: try 10 questions in the app; note failures to feed the golden set.

## Phase 6 — Evaluation
- [ ] **T6.1** `evals/golden.yaml`: ≥ 20 questions (easy/medium/hard) with reference SQL; Claude Code
  drafts, user reviews domain correctness. A test validates every reference SQL runs.
- [ ] **T6.2** `evals/run_evals.py` per design §7 (metrics, result-set comparison, JSON output).
  Unit tests for the comparison logic.
- [ ] **T6.3** ▶ USER: run evals; review results; iterate on prompt (bump `PROMPT_VERSION`) and re-run.
- [ ] **T6.4** Evals section on Traces & Evals page (latest metrics + history by prompt version).

## Phase 7 — Deploy to AWS
- [x] **T7.1** Add S3 mode to pipeline (`httpfs` + credential_chain) and `s3_io.py` upload of DB/eval
  results. Tests with `moto`.
- [x] **T7.2** ▶ USER: create S3 bucket (block public access), upload `data/raw/`.
  Claude Code: print `aws s3 mb` / `aws s3 cp` commands with explanation.
- [x] **T7.3** Write `infra/iam-ec2-policy.json` (least privilege, bucket name placeholder).
  ▶ USER: create role `oid-ec2-role` with that policy + `AmazonSSMManagedInstanceCore`
  (console steps or CLI, explained).
- [x] **T7.4** Write `infra/bootstrap.sh` and `infra/streamlit.service`.
  ▶ USER: launch EC2 (t3.medium, AL2023, 20 GB, role attached, SG port 8501 from My IP),
  connect via Session Manager, run bootstrap, run pipeline in aws mode, enable the service.
- [x] **T7.5** ▶ USER: open `http://<ec2-public-ip>:8501`, run a chat question and the eval suite
  from EC2; confirm traces and eval JSON appear (and eval upload to S3).

## Phase 8 — Wrap-up
- [x] **T8.1** README: architecture diagram, how to run locally/on AWS, screenshots, eval results,
  cost notes, what I'd do next (CDK, containers, auth, scheduling).
- [ ] **T8.2** ▶ USER: stop EC2 when idle; at project end terminate EC2, empty/delete bucket,
  delete IAM user access key, verify billing.
