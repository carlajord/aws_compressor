# CLAUDE.md — OID Compressor Copilot

Instructions for Claude Code in this repository. Read this file and `specs/` before doing anything.

## What this project is
A toy end-to-end ML pipeline on AWS, built with spec-driven development:
Cognite Open Industrial Data (Valhall 1st-stage compressor) → Parquet → S3 → DuckDB
(features + anomaly detection) → Streamlit app on EC2 → text-to-SQL agent on Amazon Bedrock,
with agent tracing and evaluation.

The owner is an experienced data scientist (Python, statistics, O&G production analytics)
but **new to AWS**. Explain AWS concepts briefly when they come up.

## Source of truth
- `specs/requirements.md` — what to build and acceptance criteria
- `specs/design.md` — architecture, modules, agent design
- `specs/data_contract.md` — tags, units, table schemas
- `specs/tasks.md` — ordered implementation checklist

If code and specs disagree, stop and ask. Do not silently change a spec; propose the edit.

## Workflow rules
1. Work on **one task from `specs/tasks.md` at a time**, in order.
2. Before coding a task, give a short plan (files to touch, approach, how it will be tested).
3. After finishing, run the local checks you are allowed to run (see below), tick the checkbox
   in `tasks.md`, summarize what changed, and **stop for review**. Do not start the next task
   until asked.
4. Keep changes small and focused on the current task.

## ⚠️ Commands the USER runs — never execute these yourself
The user runs every command that touches the cloud, credentials, or costs money.
**Do not execute** any of the following, even if asked indirectly by a task:

- `aws ...` (any AWS CLI command), `boto3` scripts that create/modify/delete AWS resources
- anything that uploads to or downloads from S3
- anything that calls Amazon Bedrock (running the agent for real, running evals, starting the
  app in a way that triggers LLM calls)
- the Cognite extraction script (it needs an interactive browser login)
- `ssh`, `scp`, Session Manager, anything on the EC2 instance
- `systemctl`, `sudo`, installing system packages
- `git push`, or any command that publishes code

Instead, for each such step:
1. Print the exact command(s) in a fenced code block under the heading **"▶ Run this yourself"**.
2. Explain in 1–3 sentences what it does and what AWS resource it touches.
3. Say what output to expect and what to paste back to you if it fails.
4. Wait for the user to report the result before continuing.

## Commands you MAY run
- `uv sync`, `uv add <pkg>` (local Python env only)
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, `uv run mypy src`
- Local scripts that only read/write files under `data/` and use **local** Parquet/DuckDB
  (e.g. building features from Parquet already on disk)
- `git status`, `git diff`, `git add`, `git commit` (never push)

## Secrets and config
- Never read, print, or edit `.env`, `~/.aws/*`, or any credential file.
- All config comes from `config/settings.py` (pydantic-settings, env prefix `OID_`).
  No hardcoded bucket names, regions, or Bedrock model IDs anywhere else.
- `.env.example` documents variables with placeholder values only.

## Code conventions
- Python 3.11, `uv` for env management, `src/` layout, package name `oidpipe`.
- Type hints everywhere; `ruff` for lint/format.
- DuckDB SQL lives in `.sql` files or clearly named string constants, not scattered inline.
- Pure functions for feature engineering and anomaly logic so they are unit-testable.
- Tests: `pytest`, small synthetic DataFrames, **Bedrock and S3 always mocked** in tests.
- Logging via the stdlib `logging` module, no `print` in library code.
- Every module gets a short docstring stating its responsibility.

## Domain glossary (use in code comments and the agent prompt)
- 23-KA-9101: 1st-stage compressor (electric, fixed speed, centrifugal), ~3 → ~12 barg
- 23-HA-9103: 1st-stage suction cooler; 23-HA-9114/9115: discharge coolers
- 23-VG-9101: 1st-stage suction scrubber
- 23-FV-92543: anti-surge valve (ASV); 23-PV-92538: suction throttle valve (STV)
- barg = gauge pressure; bara = barg + 1.013
- Cooling medium: 70% water / 30% TEG, ρ ≈ 942 kg/m³, cp ≈ 4400 J/(kg·K)
