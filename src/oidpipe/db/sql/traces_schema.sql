-- Traces DB schema — agent spans and run summaries

CREATE TABLE IF NOT EXISTS agent_spans (
    trace_id      VARCHAR   NOT NULL,
    span_id       VARCHAR   NOT NULL,
    parent_id     VARCHAR,
    kind          VARCHAR   NOT NULL,
    name          VARCHAR   NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    latency_ms    DOUBLE,
    input_json    VARCHAR,
    output_json   VARCHAR,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    status        VARCHAR   NOT NULL DEFAULT 'ok',
    error         VARCHAR,
    PRIMARY KEY (trace_id, span_id)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    trace_id       VARCHAR   NOT NULL PRIMARY KEY,
    question       VARCHAR   NOT NULL,
    answer         VARCHAR,
    model_id       VARCHAR,
    prompt_version VARCHAR,
    steps          INTEGER,
    total_tokens   INTEGER,
    latency_ms     DOUBLE,
    created_at     TIMESTAMP NOT NULL DEFAULT current_timestamp,
    source         VARCHAR   NOT NULL DEFAULT 'app'
)
