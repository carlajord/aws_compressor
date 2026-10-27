# OID Compressor Copilot

End-to-end ML pipeline on AWS for the Valhall 1st-stage gas compressor (23-KA-9101),
built with data from [Cognite Open Industrial Data](https://openindustrialdata.com/).

Sensor data flows from Cognite OID through feature engineering and anomaly detection
into a Streamlit app with interactive dashboards and a text-to-SQL agent.

## Architecture

```
Laptop                                        AWS (single region)
──────                                        ──────────────────
Cognite OID                                   S3
  │                                             raw/  ← Parquet files
  └─ cdf_extract.py ─→ data/raw/*.parquet ─→    duckdb/ ← DB snapshots
                                                │
                                              EC2 (t3.medium, Amazon Linux 2023)
                                                ├─ pipeline.py: S3 → DuckDB
                                                │    ├─ readings_10m, features_10m
                                                │    ├─ z-score + Isolation Forest → anomalies
                                                │    └─ views (daily KPIs, anomaly windows)
                                                ├─ Streamlit app (port 8501, owner IP only)
                                                │    ├─ Home (KPIs, time-series)
                                                │    ├─ Anomalies (markers, nearby events)
                                                │    ├─ Cooler Performance (duty, approach)
                                                │    ├─ Chat (text-to-SQL agent)
                                                │    └─ Traces (agent run history)
                                                └─ SQL agent → OpenAI-compatible API → LLM
```

**Dev mode** (`OID_ENV=local`): everything runs on the laptop reading `data/raw/`.

## Project structure

```
src/oidpipe/
  ingest/
    cdf_extract.py       Cognite SDK extraction (laptop, interactive login)
    s3_io.py             S3 upload/list helpers
  db/
    duck.py              DuckDB connect, schema init, atomic replace
    load.py              Parquet → DuckDB loaders
    sql/                 schema.sql, views.sql, traces_schema.sql
  features/
    build.py             Pure functions: pressure ratio, cooler duty, rolling stats
  anomaly/
    zscore.py            Rolling z-score on running periods
    iforest.py           Isolation Forest (scikit-learn)
    detect.py            Orchestrator → anomalies table
  agent/
    prompts.py           System prompt (schema, glossary, rules, few-shot)
    tools.py             get_schema, run_sql (SELECT-only, row limit)
    sql_agent.py         Prompt-based tool-use loop → AgentResult
    tracing.py           Span-level tracing → traces.duckdb
  pipeline.py            CLI: local or S3 mode

app/
  streamlit_app.py       Home page (KPIs, daily charts)
  db.py                  Shared DuckDB connection, chart helpers
  pages/                 Anomalies, Cooler Performance, Chat, Traces

config/
  settings.py            pydantic-settings (env prefix OID_)
  tags.yaml              Tag mapping (CDF external ID → logical name)

infra/
  iam-ec2-policy.json    Least-privilege IAM policy
  bootstrap.sh           EC2 setup script
  streamlit.service      systemd unit file

tests/                   209 tests (pytest, moto for S3, mocked LLM)
```

## Quick start — local

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your settings

# 3. Run the pipeline (requires data/raw/*.parquet)
uv run python -m oidpipe.pipeline --local

# 4. Start the app
uv run streamlit run app/streamlit_app.py --server.port 8501

# 5. Run tests
uv run pytest
```

## Deploy to AWS

Prerequisites: AWS CLI configured, S3 bucket created with raw Parquet uploaded.

```bash
# 1. Create IAM role with infra/iam-ec2-policy.json + AmazonSSMManagedInstanceCore
# 2. Launch EC2 (t3.medium, AL2023, 20 GB, role attached, SG port 8501 from your IP)
# 3. Connect via Session Manager and run:
sudo su - ec2-user
curl -O https://raw.githubusercontent.com/carlajord/aws_compressor/main/infra/bootstrap.sh
bash bootstrap.sh <YOUR_BUCKET> <REGION>

# 4. Start Streamlit
cd ~/aws_compressor
uv run streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# 5. (Optional) Install as a systemd service
sudo cp infra/streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now streamlit
```

To enable the Chat page, add your LLM API credentials to `.env` on the instance.

### Restarting the app

After stopping the EC2 instance, restart it with:

```bash
# 1. Start the instance
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region <REGION>

# 2. Wait ~30s, then get the new public IP
aws ec2 describe-instances --instance-ids <INSTANCE_ID> \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text --region <REGION>

# 3. Connect via Session Manager
aws ssm start-session --target <INSTANCE_ID> --region <REGION>

# 4. Update LLM credentials if they expired
sudo su - ec2-user
nano ~/aws_compressor/.env
# Update OPENAI_API_KEY, OPENAI_BASE_URL — save with Ctrl+O, exit with Ctrl+X

# 5. Start Streamlit
cd ~/aws_compressor
export PATH="$HOME/.local/bin:$PATH"
uv run streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# 6. Open http://<NEW_PUBLIC_IP>:8501
```

The public IP changes on each restart. If your home IP changed, update the security
group rule for port 8501 in **EC2 → Security Groups → Edit inbound rules**.

When done, stop the instance to save costs:

```bash
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region <REGION>
```

## Data source

[Cognite Open Industrial Data](https://openindustrialdata.com/) — Valhall platform,
1st-stage compressor (23-KA-9101). 10-minute average readings for pressure, temperature,
vibration, valve positions, and motor current. See `specs/data_contract.md` for the
full tag list, units, and table schemas.

## Anomaly detection

Two complementary methods, both computed on running periods only:

- **Z-score**: per-signal rolling z-score against a 24h window. Flags |z| > threshold (default 5.0).
- **Isolation Forest**: multivariate (vibration + bearing temps + pressure ratio + cooler approach),
  trained on a configurable normal period. Flags by contamination parameter (default 0.01).

Anomaly windows (consecutive flags merged within 30 min) are correlated with maintenance events
via the `v_anomalies_near_events` view.

## Text-to-SQL agent

- Prompt-based tool calling: the system prompt instructs the model to emit
  `<tool_call>` tags parsed from text responses.
- Tools: `get_schema` (table/column listing + samples) and `run_sql`
  (SELECT-only, blocked keywords, row limit, read-only connection).
- System prompt includes live database schema, domain glossary, rules, and few-shot examples.
- Every run is traced (LLM calls, tool calls, latency, tokens) in `traces.duckdb`.

## Cost notes

| Resource | Running | Stopped / idle |
|---|---|---|
| EC2 t3.medium | ~$1.00/day | $0 (EBS only: ~$1.60/month for 20 GB) |
| S3 (< 1 GB) | ~$0.02/month | Same |
| LLM calls | Per-token (varies by model) | $0 |

**Stop the instance when not in use.** Full cleanup instructions in `specs/tasks.md` (T8.2).

## What I'd do next

- **CDK/Terraform** for infrastructure-as-code instead of manual console setup
- **Containers** (Docker + ECS or App Runner) for reproducible deployment
- **HTTPS + auth** via ALB + Cognito for multi-user access
- **Scheduled pipeline** (EventBridge + Lambda or cron) for automatic data refresh
- **Streaming ingestion** from Cognite for near-real-time monitoring
- **Evaluation suite** with golden questions to measure agent accuracy across prompt versions
- **Bedrock native integration** using boto3 Converse API (bypasses the need for API keys)
