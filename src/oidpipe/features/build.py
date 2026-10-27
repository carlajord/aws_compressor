"""Feature engineering — pure functions on DataFrames + orchestration into DuckDB."""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Cooling medium properties (70% water / 30% TEG)
CM_DENSITY_KG_M3 = 942.0
CM_CP_J_KG_K = 4400.0

# Monitored signals for rolling stats (vibration + bearing temps)
ROLLING_SIGNALS = [
    "vib_mtr_de_1",
    "vib_mtr_de_2",
    "vib_mtr_nde_1",
    "vib_mtr_nde_2",
    "t_bearing_comp_de",
    "t_bearing_comp_nde",
    "t_bearing_mtr_de",
    "t_bearing_mtr_nde",
    "cooler_approach_c",
    "cooler_duty_kw",
    "cooler_gas_dt_c",
]

ROLL_1H = 6  # 6 × 10 min
ROLL_24H = 144  # 144 × 10 min


# ---------------------------------------------------------------------------
# Pure feature functions (operate on pandas Series)
# ---------------------------------------------------------------------------


def pressure_ratio(p_suction_barg: pd.Series, p_discharge_barg: pd.Series) -> pd.Series:
    """Compute compression ratio from gauge pressures: (p_d + 1.013) / (p_s + 1.013)."""
    return (p_discharge_barg + 1.013) / (p_suction_barg + 1.013)


def cooler_duty_kw(
    q_cm_m3h: pd.Series, t_cm_in_c: pd.Series, t_cm_out_c: pd.Series
) -> pd.Series:
    """Heat duty of the suction cooler in kW."""
    m_dot = q_cm_m3h * CM_DENSITY_KG_M3 / 3600.0  # kg/s
    return m_dot * CM_CP_J_KG_K * (t_cm_out_c - t_cm_in_c) / 1000.0


def cooler_approach_c(t_gas_out_sc_c: pd.Series, t_cm_in_c: pd.Series) -> pd.Series:
    """Approach temperature: gas outlet minus cooling medium inlet."""
    return t_gas_out_sc_c - t_cm_in_c


def cooler_gas_dt_c(t_gas_in_sc_c: pd.Series, t_gas_out_sc_c: pd.Series) -> pd.Series:
    """Gas temperature drop across the suction cooler."""
    return t_gas_in_sc_c - t_gas_out_sc_c


def is_running(motor_current_a: pd.Series, threshold: float = 50.0) -> pd.Series:
    """True when motor current exceeds threshold (compressor is running)."""
    return motor_current_a > threshold


def asv_recycling(asv_open_pct: pd.Series, threshold: float = 5.0) -> pd.Series:
    """True when anti-surge valve is open beyond threshold."""
    return asv_open_pct > threshold


def rate_of_change(series: pd.Series) -> pd.Series:
    """First difference (change per 10-minute step)."""
    return series.diff()


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def compute_rolling_stats(
    df: pd.DataFrame,
    signals: list[str],
    running_mask: pd.Series,
) -> pd.DataFrame:
    """Compute 1h/24h rolling mean, std, and rate of change for monitored signals.

    Non-running periods are masked before computing rolling windows so
    shutdown transients don't corrupt the statistics.
    """
    result = pd.DataFrame(index=df.index)
    available = [s for s in signals if s in df.columns]

    for sig in available:
        masked = df[sig].where(running_mask)

        result[f"{sig}_roll1h_mean"] = masked.rolling(ROLL_1H, min_periods=1).mean()
        result[f"{sig}_roll1h_std"] = masked.rolling(ROLL_1H, min_periods=2).std()
        result[f"{sig}_roll24h_mean"] = masked.rolling(ROLL_24H, min_periods=1).mean()
        result[f"{sig}_roll24h_std"] = masked.rolling(ROLL_24H, min_periods=2).std()
        result[f"{sig}_roc"] = rate_of_change(df[sig])

    return result


# ---------------------------------------------------------------------------
# Orchestrator: readings_10m → features_10m
# ---------------------------------------------------------------------------


def pivot_readings(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pivot readings_10m from long to wide format (one column per tag)."""
    df = conn.execute(
        """
        PIVOT readings_10m
        ON tag
        USING first(value)
        GROUP BY ts
        ORDER BY ts
        """
    ).fetchdf()
    df = df.set_index("ts")
    return df


def build_features(
    conn: duckdb.DuckDBPyConnection,
    motor_current_threshold: float = 50.0,
) -> pd.DataFrame:
    """Build features_10m from readings_10m and write to DuckDB."""
    wide = pivot_readings(conn)
    logger.info("Pivoted readings: %d rows × %d columns", len(wide), len(wide.columns))

    features = wide.copy()

    # Derived features
    if "p_suction_barg" in features.columns and "p_discharge_barg" in features.columns:
        features["pressure_ratio"] = pressure_ratio(
            features["p_suction_barg"], features["p_discharge_barg"]
        )

    if all(c in features.columns for c in ["q_cm_m3h", "t_cm_in_c", "t_cm_out_c"]):
        features["cooler_duty_kw"] = cooler_duty_kw(
            features["q_cm_m3h"], features["t_cm_in_c"], features["t_cm_out_c"]
        )

    if "t_gas_out_sc_c" in features.columns and "t_cm_in_c" in features.columns:
        features["cooler_approach_c"] = cooler_approach_c(
            features["t_gas_out_sc_c"], features["t_cm_in_c"]
        )

    if "t_gas_in_sc_c" in features.columns and "t_gas_out_sc_c" in features.columns:
        features["cooler_gas_dt_c"] = cooler_gas_dt_c(
            features["t_gas_in_sc_c"], features["t_gas_out_sc_c"]
        )

    if "motor_current_a" in features.columns:
        features["is_running"] = is_running(
            features["motor_current_a"], motor_current_threshold
        )
    else:
        features["is_running"] = True

    if "asv_open_pct" in features.columns:
        features["asv_recycling"] = asv_recycling(features["asv_open_pct"])

    # Rolling stats on vibration + bearing temps (running periods only)
    running_mask = features["is_running"]
    rolling = compute_rolling_stats(features, ROLLING_SIGNALS, running_mask)
    features = pd.concat([features, rolling], axis=1)

    # Write to DuckDB
    features = features.reset_index()
    conn.execute("DROP TABLE IF EXISTS features_10m")
    conn.register("_features_df", features)
    conn.execute("CREATE TABLE features_10m AS SELECT * FROM _features_df")
    conn.unregister("_features_df")

    count = conn.execute("SELECT count(*) FROM features_10m").fetchone()[0]
    ncols = conn.execute(
        "SELECT count(*) FROM information_schema.columns WHERE table_name = 'features_10m'"
    ).fetchone()[0]
    logger.info("Built features_10m: %d rows × %d columns", count, ncols)

    return features
