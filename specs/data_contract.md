# Data Contract — OID Compressor Copilot

## 1. Source
- Cognite Data Fusion, Open Industrial Data, project `publicdata` (Valhall PH, 1st-stage gas compression).
- Access: Cognite Hub account + OIDC interactive login via `cognite-sdk` (laptop only).
- Granularity: 10-minute `average` aggregates. Date range: configurable, target 3–6 months.

## 2. Tag mapping
Filled in during task T1.2 (tag discovery). Code must use the **logical name** only;
the mapping lives in `config/tags.yaml` generated from this table.

### Core compressor performance

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `p_suction_barg` | 23-KA-9101 | Compressor suction pressure | barg | `VAL_23-PT-92532:X.Value` |
| `p_discharge_barg` | 23-KA-9101 | Compressor discharge pressure | barg | `VAL_23_PT_92537:Z.X.Value` |
| `t_suction_c` | 23-KA-9101 | Compressor suction temperature | °C | `VAL_23_TT_92532:Z.X.Value` |
| `t_discharge_c` | 23-KA-9101 | Compressor discharge temperature | °C | `VAL_23_TT_92536:Z.X.Value` |
| `flow_gas` | 23-KA-9101 | Gas flow through compressor | as in CDF | `VAL_23_FT_92537_03:Z.X.Value` |

### Motor & running proxy

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `motor_current_a` | 23-KA-9101 motor | Motor current L1 (running proxy) | A | `VAL_23-KA-9101-M01_Current_L1:VALUE` |
| `motor_power_kw` | 23-KA-9101 motor | Motor active power | kW | `VAL_23-KA-9101-M01_Active_power_kW:VALUE` |

### Valves

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `asv_open_pct` | 23-FV-92543 | Anti-surge valve position | % | `VAL_23_ZT_92543:Z.X.Value` |
| `stv_open_pct` | 23-PV-92538 | Suction throttle valve position | % | `VAL_23_ZT_92538:Z.X.Value` |

### Suction cooler (23-HA-9103)

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `t_gas_in_sc_c` | 23-HA-9103 | Gas temperature into suction cooler | °C | `VAL_23-TT-92512:X.Value` |
| `t_gas_out_sc_c` | 23-HA-9103 | Gas temperature out of suction cooler | °C | `VAL_23-TIC-92504:Z.X.Value` |
| `t_cm_in_c` | 23-HA-9103 | Cooling medium inlet temperature | °C | `VAL_45-TT-92506:X.Value` |
| `t_cm_out_c` | 23-HA-9103 | Cooling medium outlet temperature | °C | `VAL_45-TT-92508:X.Value` |
| `q_cm_m3h` | 23-HA-9103 | Cooling medium flow | m³/h | `VAL_45-FT-92506:X.Value` |

### Scrubber

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `scrubber_level_pct` | 23-VG-9101 | Suction scrubber level | % | `VAL_23-LIC-92521:Z.X.Value` |

### Vibration (CMS — motor journal bearings)

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `vib_mtr_de_1` | 23-KA-9101 motor | Motor journal BRG DE probe 1 | as in CDF | `VAL_23_YT_96134_01:Z.X.Value` |
| `vib_mtr_de_2` | 23-KA-9101 motor | Motor journal BRG DE probe 2 | as in CDF | `VAL_23_YT_96134_02:Z.X.Value` |
| `vib_mtr_nde_1` | 23-KA-9101 motor | Motor journal BRG NDE probe 1 | as in CDF | `VAL_23_YT_96133_01:Z.X.Value` |
| `vib_mtr_nde_2` | 23-KA-9101 motor | Motor journal BRG NDE probe 2 | as in CDF | `VAL_23_YT_96133_02:Z.X.Value` |

### Bearing temperatures (CMS — compressor + motor journal bearings)

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `t_bearing_comp_de` | 23-KA-9101 | Compressor journal BRG DE | °C | `VAL_23_TT_96101_01:Z.X.Value` |
| `t_bearing_comp_nde` | 23-KA-9101 | Compressor journal BRG NDE | °C | `VAL_23_TT_96102_01:Z.X.Value` |
| `t_bearing_mtr_de` | 23-KA-9101 motor | Motor journal BRG DE | °C | `VAL_23_TT_96132_02:Z.X.Value` |
| `t_bearing_mtr_nde` | 23-KA-9101 motor | Motor journal BRG NDE | °C | `VAL_23_TT_96131_02:Z.X.Value` |

### Lube oil system

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `lube_oil_t_c` | lube system | Lube oil reservoir temperature | °C | `VAL_23-TT-96183:X.Value` |
| `lube_filter_dp` | lube system | Lube oil filter differential pressure | as in CDF | `VAL_23-PDT-96180:X.Value` |

### Bonus tags (CBM-derived, for cross-checks and KPIs)

| Logical name | Equipment | Description | Unit | CDF external_id |
|---|---|---|---|---|
| `pressure_ratio_cbm` | 23-KA-9101 | CBM actual pressure ratio | ratio | `VAL_23-KA-9101_APR:VALUE` |
| `shaft_power_kw` | 23-KA-9101 | CBM actual shaft power | kW | `VAL_23-KA-9101_ASP:VALUE` |
| `running_hours` | 23-KA-9101 motor | Cumulative running hours | hours | `VAL_23-KA-9101-M01_Running_hours:VALUE` |

### Rules
- If a tag is mostly empty (> 50 % missing in range), mark it `UNAVAILABLE`
  here and any dependent feature is dropped (documented in §4).
- Units marked "as in CDF" must be confirmed from sample data; record the actual unit here.

## 3. Raw files (S3 `raw/<YYYYMMDD>/` and local `data/raw/`)

### `readings.parquet` (long)
| column | type | notes |
|---|---|---|
| `ts` | TIMESTAMP (UTC) | start of 10-min bucket |
| `external_id` | VARCHAR | CDF time series external ID |
| `value` | DOUBLE | 10-min average |

### `tags.parquet`
| column | type |
|---|---|
| `external_id` | VARCHAR |
| `logical_name` | VARCHAR |
| `name`, `description`, `unit` | VARCHAR |
| `asset_name` | VARCHAR |

### `events.parquet`
| column | type | notes |
|---|---|---|
| `event_id` | BIGINT | |
| `start_time`, `end_time` | TIMESTAMP (UTC) | end may be null |
| `type`, `subtype` | VARCHAR | |
| `description` | VARCHAR | |
| `asset_names` | VARCHAR | comma-separated |

## 4. DuckDB tables (`oid.duckdb`)
Every table and column gets a `COMMENT ON` — the agent reads these via `get_schema`.

### `tags` — copy of `tags.parquet`

### `readings_10m`
`ts TIMESTAMP, tag VARCHAR (logical name), value DOUBLE` — PK (ts, tag).

### `features_10m` (wide, one row per ts)
`ts`, all logical tags as columns, plus:
`is_running BOOLEAN, pressure_ratio DOUBLE, cooler_duty_kw DOUBLE, cooler_approach_c DOUBLE,
cooler_gas_dt_c DOUBLE, asv_recycling BOOLEAN`, and for each monitored signal `s`:
`s_roll1h_mean, s_roll1h_std, s_roll24h_mean, s_roll24h_std, s_roc`.

Dropped features: _none — all tags from §2 were found in CDF (T1.2)._

### `events` — copy of `events.parquet`

### `anomalies`
| column | type |
|---|---|
| `ts` | TIMESTAMP |
| `signal` | VARCHAR (logical name, or `multivariate` for Isolation Forest) |
| `method` | VARCHAR (`zscore` / `iforest`) |
| `score` | DOUBLE |
| `is_anomaly` | BOOLEAN |

### Views
- `v_daily_kpis(day, running_hours, avg_pressure_ratio, avg_cooler_duty_kw, max_vib_*, anomaly_count)`
- `v_anomaly_windows(window_id, method, signal, start_ts, end_ts, max_score, n_points)`
- `v_anomalies_near_events(window_id, event_id, event_type, days_between)`

## 5. Traces DB (`traces.duckdb`)
Schema defined in `design.md` §7 (`agent_runs`, `agent_spans`).

## 6. Data quality checks (run in pipeline, fail loudly)
- `ts` strictly on 10-min grid, UTC, no duplicates per (ts, tag).
- Each mapped tag has ≥ 50 % non-null coverage in the range.
- Physical sanity: `p_discharge_barg > p_suction_barg` when running; pressures within 0–50 barg;
  temperatures within −20–200 °C; valve openings within 0–100 %. Violations logged with counts.
