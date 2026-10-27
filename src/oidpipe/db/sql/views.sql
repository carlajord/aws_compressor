-- Views for the OID Compressor analytics database

-- v_daily_kpis: one row per day with key performance indicators
CREATE OR REPLACE VIEW v_daily_kpis AS
WITH daily AS (
    SELECT
        CAST(ts AS DATE)              AS day,
        SUM(CASE WHEN is_running THEN 10.0 / 60.0 ELSE 0 END) AS running_hours,
        AVG(CASE WHEN is_running THEN pressure_ratio END)      AS avg_pressure_ratio,
        AVG(CASE WHEN is_running THEN cooler_duty_kw END)      AS avg_cooler_duty_kw,
        MAX(vib_mtr_de_1)             AS max_vib_mtr_de_1,
        MAX(vib_mtr_de_2)             AS max_vib_mtr_de_2,
        MAX(vib_mtr_nde_1)            AS max_vib_mtr_nde_1,
        MAX(vib_mtr_nde_2)            AS max_vib_mtr_nde_2
    FROM features_10m
    GROUP BY CAST(ts AS DATE)
),
anom_daily AS (
    SELECT CAST(ts AS DATE) AS day, count(*) AS anomaly_count
    FROM anomalies
    WHERE is_anomaly
    GROUP BY CAST(ts AS DATE)
)
SELECT d.*, COALESCE(a.anomaly_count, 0) AS anomaly_count
FROM daily d
LEFT JOIN anom_daily a ON d.day = a.day
ORDER BY d.day;

COMMENT ON VIEW v_daily_kpis IS 'Daily KPIs: running hours, avg pressure ratio, avg cooler duty, max vibration, anomaly count';

-- v_anomaly_windows: merge consecutive anomaly rows (gap ≤ 30 min) into windows
CREATE OR REPLACE VIEW v_anomaly_windows AS
WITH flagged AS (
    SELECT
        ts, method, signal, score,
        ts - LAG(ts) OVER (PARTITION BY method, signal ORDER BY ts) AS gap
    FROM anomalies
    WHERE is_anomaly
),
groups AS (
    SELECT *,
        SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '30 minutes' THEN 1 ELSE 0 END)
            OVER (PARTITION BY method, signal ORDER BY ts) AS window_id_raw
    FROM flagged
)
SELECT
    ROW_NUMBER() OVER (ORDER BY method, signal, MIN(ts)) AS window_id,
    method,
    signal,
    MIN(ts)        AS start_ts,
    MAX(ts)        AS end_ts,
    MAX(score)     AS max_score,
    COUNT(*)       AS n_points
FROM groups
GROUP BY method, signal, window_id_raw
ORDER BY start_ts;

COMMENT ON VIEW v_anomaly_windows IS 'Anomaly windows: consecutive flagged rows merged when gap ≤ 30 min';

-- v_anomalies_near_events: join anomaly windows with events within ±7 days
CREATE OR REPLACE VIEW v_anomalies_near_events AS
SELECT
    w.window_id,
    e.event_id,
    e.type AS event_type,
    ROUND(
        EXTRACT(EPOCH FROM (w.start_ts - e.start_time)) / 86400.0, 1
    ) AS days_between
FROM v_anomaly_windows w
CROSS JOIN events e
WHERE ABS(EXTRACT(EPOCH FROM (w.start_ts - e.start_time))) <= 7 * 86400
ORDER BY w.window_id, ABS(days_between);

COMMENT ON VIEW v_anomalies_near_events IS 'Anomaly windows matched with events within ±7 days';
