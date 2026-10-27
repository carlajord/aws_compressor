"""Tests for features/build.py — pure functions with known values + orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oidpipe.db.duck import connect, init_schema
from oidpipe.db.load import load_readings, load_tags
from oidpipe.features.build import (
    asv_recycling,
    build_features,
    compute_rolling_stats,
    cooler_approach_c,
    cooler_duty_kw,
    cooler_gas_dt_c,
    is_running,
    pivot_readings,
    pressure_ratio,
    rate_of_change,
)


# ---------------------------------------------------------------------------
# Pure function tests with known values
# ---------------------------------------------------------------------------


class TestPressureRatio:
    def test_known_value(self):
        p_s = pd.Series([3.0])  # barg
        p_d = pd.Series([12.0])  # barg
        result = pressure_ratio(p_s, p_d)
        expected = (12.0 + 1.013) / (3.0 + 1.013)
        assert result.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_ratio_always_positive(self):
        p_s = pd.Series([0.0, 1.0, 5.0])
        p_d = pd.Series([5.0, 8.0, 15.0])
        result = pressure_ratio(p_s, p_d)
        assert (result > 0).all()

    def test_zero_gauge_pressure(self):
        p_s = pd.Series([0.0])
        p_d = pd.Series([0.0])
        result = pressure_ratio(p_s, p_d)
        assert result.iloc[0] == pytest.approx(1.0, rel=1e-6)


class TestCoolerDutyKw:
    def test_known_value(self):
        # q = 100 m³/h, ΔT = 10 °C
        # m_dot = 100 * 942 / 3600 = 26.167 kg/s
        # duty = 26.167 * 4400 * 10 / 1000 = 1151.33 kW
        q = pd.Series([100.0])
        t_in = pd.Series([20.0])
        t_out = pd.Series([30.0])
        result = cooler_duty_kw(q, t_in, t_out)
        expected = (100.0 * 942.0 / 3600.0) * 4400.0 * 10.0 / 1000.0
        assert result.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_zero_flow_gives_zero_duty(self):
        result = cooler_duty_kw(pd.Series([0.0]), pd.Series([20.0]), pd.Series([30.0]))
        assert result.iloc[0] == pytest.approx(0.0)

    def test_negative_dt_gives_negative_duty(self):
        result = cooler_duty_kw(pd.Series([50.0]), pd.Series([30.0]), pd.Series([20.0]))
        assert result.iloc[0] < 0


class TestCoolerApproachC:
    def test_known_value(self):
        result = cooler_approach_c(pd.Series([25.0]), pd.Series([18.0]))
        assert result.iloc[0] == pytest.approx(7.0)


class TestCoolerGasDtC:
    def test_known_value(self):
        result = cooler_gas_dt_c(pd.Series([45.0]), pd.Series([25.0]))
        assert result.iloc[0] == pytest.approx(20.0)


class TestIsRunning:
    def test_above_threshold_is_running(self):
        result = is_running(pd.Series([100.0, 200.0, 500.0]), threshold=50.0)
        assert result.all()

    def test_below_threshold_not_running(self):
        result = is_running(pd.Series([0.0, 10.0, 49.9]), threshold=50.0)
        assert not result.any()

    def test_mixed(self):
        result = is_running(pd.Series([0.0, 100.0, 30.0, 600.0]), threshold=50.0)
        assert list(result) == [False, True, False, True]


class TestAsvRecycling:
    def test_above_threshold(self):
        result = asv_recycling(pd.Series([10.0, 50.0]), threshold=5.0)
        assert result.all()

    def test_below_threshold(self):
        result = asv_recycling(pd.Series([0.0, 3.0, 5.0]), threshold=5.0)
        assert not result.any()


class TestRateOfChange:
    def test_known_values(self):
        s = pd.Series([10.0, 12.0, 11.0, 15.0])
        result = rate_of_change(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(-1.0)
        assert result.iloc[3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Rolling stats tests
# ---------------------------------------------------------------------------


class TestComputeRollingStats:
    def test_produces_expected_columns(self):
        n = 20
        df = pd.DataFrame(
            {
                "vib_mtr_de_1": np.random.default_rng(42).random(n) * 10,
                "other_col": np.ones(n),
            }
        )
        running = pd.Series([True] * n)
        result = compute_rolling_stats(df, ["vib_mtr_de_1"], running)

        expected_cols = {
            "vib_mtr_de_1_roll1h_mean",
            "vib_mtr_de_1_roll1h_std",
            "vib_mtr_de_1_roll24h_mean",
            "vib_mtr_de_1_roll24h_std",
            "vib_mtr_de_1_roc",
        }
        assert set(result.columns) == expected_cols

    def test_masks_non_running_periods(self):
        n = 12
        values = [10.0] * 6 + [100.0] * 6  # spike at index 6
        df = pd.DataFrame({"vib_mtr_de_1": values})
        running = pd.Series([True] * 6 + [False] * 6)

        result = compute_rolling_stats(df, ["vib_mtr_de_1"], running)
        # After masking, rolling mean should stay near 10 (not see the 100s)
        mean_at_5 = result["vib_mtr_de_1_roll1h_mean"].iloc[5]
        assert mean_at_5 == pytest.approx(10.0)
        # During non-running, all values in the 1h window are NaN → rolling mean is NaN
        mean_at_11 = result["vib_mtr_de_1_roll1h_mean"].iloc[11]
        assert pd.isna(mean_at_11)

    def test_skips_missing_signals(self):
        df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
        running = pd.Series([True, True, True])
        result = compute_rolling_stats(df, ["vib_mtr_de_1"], running)
        assert result.empty or len(result.columns) == 0


# ---------------------------------------------------------------------------
# Orchestrator tests with synthetic DuckDB data
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_db(tmp_path: Path):
    """Create a DuckDB with tags and readings loaded from synthetic Parquet."""
    raw = tmp_path / "raw"
    raw.mkdir()

    tags = pd.DataFrame(
        {
            "external_id": ["EXT_PS", "EXT_PD", "EXT_MC", "EXT_VIB"],
            "logical_name": [
                "p_suction_barg",
                "p_discharge_barg",
                "motor_current_a",
                "vib_mtr_de_1",
            ],
            "name": ["ps", "pd", "mc", "vib"],
            "description": ["suction P", "discharge P", "motor I", "vibration"],
            "unit": ["barg", "barg", "A", "um"],
            "asset_name": ["KA", "KA", "KA", "KA"],
        }
    )
    tags.to_parquet(raw / "tags.parquet", index=False)

    timestamps = pd.date_range("2024-03-01", periods=24, freq="10min")
    rows = []
    for ts in timestamps:
        rows.append({"ts": ts, "external_id": "EXT_PS", "value": 3.5})
        rows.append({"ts": ts, "external_id": "EXT_PD", "value": 12.0})
        rows.append({"ts": ts, "external_id": "EXT_MC", "value": 500.0})
        rows.append({"ts": ts, "external_id": "EXT_VIB", "value": 15.0})
    pd.DataFrame(rows).to_parquet(raw / "readings.parquet", index=False)

    conn = connect(tmp_path / "test.duckdb")
    init_schema(conn)
    load_tags(conn, raw / "tags.parquet")
    load_readings(conn, raw / "readings.parquet")
    yield conn
    conn.close()


class TestPivotReadings:
    def test_produces_wide_format(self, loaded_db):
        wide = pivot_readings(loaded_db)
        assert "p_suction_barg" in wide.columns
        assert "p_discharge_barg" in wide.columns
        assert len(wide) == 24

    def test_index_is_timestamp(self, loaded_db):
        wide = pivot_readings(loaded_db)
        assert wide.index.name == "ts"


class TestBuildFeatures:
    def test_creates_features_table(self, loaded_db):
        build_features(loaded_db)
        count = loaded_db.execute("SELECT count(*) FROM features_10m").fetchone()[0]
        assert count == 24

    def test_pressure_ratio_column(self, loaded_db):
        df = build_features(loaded_db)
        assert "pressure_ratio" in df.columns
        expected = (12.0 + 1.013) / (3.5 + 1.013)
        assert df["pressure_ratio"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_is_running_column(self, loaded_db):
        df = build_features(loaded_db)
        assert "is_running" in df.columns
        assert df["is_running"].all()  # motor_current=500 > threshold=50

    def test_rolling_stats_for_vibration(self, loaded_db):
        df = build_features(loaded_db)
        assert "vib_mtr_de_1_roll1h_mean" in df.columns
        assert "vib_mtr_de_1_roll24h_std" in df.columns
        assert "vib_mtr_de_1_roc" in df.columns

    def test_table_in_duckdb(self, loaded_db):
        build_features(loaded_db)
        cols = loaded_db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'features_10m' ORDER BY ordinal_position"
        ).fetchall()
        col_names = [c[0] for c in cols]
        assert "ts" in col_names
        assert "pressure_ratio" in col_names
        assert "is_running" in col_names
