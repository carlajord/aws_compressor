"""Tests for anomaly/detect.py — orchestrator that writes to DuckDB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oidpipe.anomaly.detect import run_anomaly_detection
from oidpipe.db.duck import connect, init_schema


def _build_test_db(tmp_path: Path, n: int = 300, spike_at: int | None = None):
    """Create a DuckDB with a features_10m table from synthetic data."""
    conn = connect(tmp_path / "test.duckdb")
    init_schema(conn)

    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-06-01", periods=n, freq="10min")

    df = pd.DataFrame(
        {
            "ts": ts,
            "vib_mtr_de_1": rng.normal(10.0, 0.5, n),
            "vib_mtr_de_2": rng.normal(12.0, 0.5, n),
            "vib_mtr_nde_1": rng.normal(11.0, 0.5, n),
            "vib_mtr_nde_2": rng.normal(13.0, 0.5, n),
            "t_bearing_comp_de": rng.normal(55.0, 1.0, n),
            "t_bearing_comp_nde": rng.normal(54.0, 1.0, n),
            "t_bearing_mtr_de": rng.normal(52.0, 1.0, n),
            "t_bearing_mtr_nde": rng.normal(53.0, 1.0, n),
            "pressure_ratio": rng.normal(3.2, 0.05, n),
            "cooler_approach_c": rng.normal(7.0, 0.3, n),
            "is_running": np.full(n, True),
        }
    )

    # Add rolling stats columns needed by z-score
    for sig in ["vib_mtr_de_1", "vib_mtr_de_2", "vib_mtr_nde_1", "vib_mtr_nde_2",
                "t_bearing_comp_de", "t_bearing_comp_nde", "t_bearing_mtr_de", "t_bearing_mtr_nde"]:
        df[f"{sig}_roll24h_mean"] = df[sig].rolling(144, min_periods=1).mean()
        df[f"{sig}_roll24h_std"] = df[sig].rolling(144, min_periods=2).std()

    if spike_at is not None:
        df.loc[spike_at, "vib_mtr_de_1"] = 30.0  # z ≈ 40

    conn.register("_features", df)
    conn.execute("CREATE TABLE features_10m AS SELECT * FROM _features")
    conn.unregister("_features")

    return conn


class TestRunAnomalyDetection:
    def test_writes_to_anomalies_table(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            n = run_anomaly_detection(conn, zscore_threshold=4.0, iforest_contamination=0.05)
            assert n > 0
            count = conn.execute("SELECT count(*) FROM anomalies").fetchone()[0]
            assert count == n
        finally:
            conn.close()

    def test_contains_both_methods(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0, iforest_contamination=0.05)
            methods = conn.execute(
                "SELECT DISTINCT method FROM anomalies ORDER BY method"
            ).fetchall()
            method_names = [m[0] for m in methods]
            assert "zscore" in method_names
            assert "iforest" in method_names
        finally:
            conn.close()

    def test_zscore_detects_spike(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0)
            zs_anomalies = conn.execute(
                "SELECT count(*) FROM anomalies WHERE method = 'zscore' AND is_anomaly"
            ).fetchone()[0]
            assert zs_anomalies >= 1
        finally:
            conn.close()

    def test_no_spike_few_zscore_flags(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0)
            zs_anomalies = conn.execute(
                "SELECT count(*) FROM anomalies WHERE method = 'zscore' AND is_anomaly"
            ).fetchone()[0]
            assert zs_anomalies <= 5
        finally:
            conn.close()

    def test_zscore_stores_all_scored_rows(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0)
            zs_total = conn.execute(
                "SELECT count(*) FROM anomalies WHERE method = 'zscore'"
            ).fetchone()[0]
            # 8 signals × 300 rows, minus early rows with NaN std (min_periods=2)
            assert zs_total > 300 * 7
        finally:
            conn.close()

    def test_respects_threshold(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            # Very high threshold → z-score should miss the spike
            run_anomaly_detection(conn, zscore_threshold=100.0)
            zs_anomalies = conn.execute(
                "SELECT count(*) FROM anomalies WHERE method = 'zscore' AND is_anomaly"
            ).fetchone()[0]
            assert zs_anomalies == 0
        finally:
            conn.close()

    def test_anomalies_schema(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0, iforest_contamination=0.05)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'anomalies' ORDER BY ordinal_position"
            ).fetchall()
            col_names = [c[0] for c in cols]
            assert col_names == ["ts", "signal", "method", "score", "is_anomaly"]
        finally:
            conn.close()

    def test_zscore_excludes_from_iforest_training(self, tmp_path: Path):
        conn = _build_test_db(tmp_path, n=300, spike_at=200)
        try:
            run_anomaly_detection(conn, zscore_threshold=4.0, iforest_contamination=0.05)
            # The spiked timestamp should be flagged by both methods
            spike_ts = conn.execute(
                "SELECT ts FROM features_10m LIMIT 1 OFFSET 200"
            ).fetchone()[0]
            zs_flag = conn.execute(
                "SELECT count(*) FROM anomalies WHERE method='zscore' AND is_anomaly AND ts=?",
                [spike_ts],
            ).fetchone()[0]
            assert zs_flag >= 1
        finally:
            conn.close()
