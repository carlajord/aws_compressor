-- OID Compressor Copilot — DuckDB schema
-- Every table and column gets a COMMENT so the text-to-SQL agent
-- can discover the schema via get_schema().

-- ---------------------------------------------------------------
-- tags: CDF time series metadata
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tags (
    external_id  VARCHAR NOT NULL PRIMARY KEY,
    logical_name VARCHAR NOT NULL UNIQUE,
    name         VARCHAR,
    description  VARCHAR,
    unit         VARCHAR,
    asset_name   VARCHAR
);
COMMENT ON TABLE tags IS 'CDF time series metadata: maps external_id to logical name';
COMMENT ON COLUMN tags.external_id IS 'Cognite Data Fusion external ID for the time series';
COMMENT ON COLUMN tags.logical_name IS 'Short logical name used in code and features (e.g. p_suction_barg)';
COMMENT ON COLUMN tags.unit IS 'Engineering unit (barg, degC, %, A, kW, etc.)';
COMMENT ON COLUMN tags.asset_name IS 'Parent equipment tag (e.g. 23-KA-9101)';

-- ---------------------------------------------------------------
-- readings_10m: raw 10-minute average readings in long format
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS readings_10m (
    ts    TIMESTAMP NOT NULL,
    tag   VARCHAR   NOT NULL,
    value DOUBLE,
    PRIMARY KEY (ts, tag)
);
COMMENT ON TABLE readings_10m IS '10-minute average readings from CDF, long format (one row per timestamp per tag)';
COMMENT ON COLUMN readings_10m.ts IS 'Start of the 10-minute bucket, UTC';
COMMENT ON COLUMN readings_10m.tag IS 'Logical tag name (FK to tags.logical_name)';
COMMENT ON COLUMN readings_10m.value IS '10-minute average value in the tag''s engineering unit';

-- ---------------------------------------------------------------
-- events: maintenance and process events from CDF
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    event_id    BIGINT,
    start_time  TIMESTAMP,
    end_time    TIMESTAMP,
    type        VARCHAR,
    subtype     VARCHAR,
    description VARCHAR,
    asset_names VARCHAR
);
COMMENT ON TABLE events IS 'Maintenance and process events from CDF for compressor assets';
COMMENT ON COLUMN events.event_id IS 'CDF event ID';
COMMENT ON COLUMN events.start_time IS 'Event start time, UTC';
COMMENT ON COLUMN events.end_time IS 'Event end time, UTC (may be NULL for point events)';
COMMENT ON COLUMN events.type IS 'Event type (e.g. Maintenance, WorkOrder)';
COMMENT ON COLUMN events.subtype IS 'Event subtype (e.g. Planned, Corrective)';
COMMENT ON COLUMN events.asset_names IS 'Comma-separated asset IDs associated with the event';

-- ---------------------------------------------------------------
-- anomalies: output of anomaly detection methods
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomalies (
    ts         TIMESTAMP NOT NULL,
    signal     VARCHAR   NOT NULL,
    method     VARCHAR   NOT NULL,
    score      DOUBLE,
    is_anomaly BOOLEAN
);
COMMENT ON TABLE anomalies IS 'Anomaly detection results from z-score and Isolation Forest methods';
COMMENT ON COLUMN anomalies.ts IS 'Timestamp of the 10-minute bucket, UTC';
COMMENT ON COLUMN anomalies.signal IS 'Logical tag name, or "multivariate" for Isolation Forest';
COMMENT ON COLUMN anomalies.method IS 'Detection method: zscore or iforest';
COMMENT ON COLUMN anomalies.score IS 'Anomaly score (|z| for z-score, -decision_function for iforest)';
COMMENT ON COLUMN anomalies.is_anomaly IS 'TRUE if the score exceeds the configured threshold';
