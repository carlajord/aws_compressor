"""Tests for anomaly/zscore.py — z-score detection with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oidpipe.anomaly.zscore import zscore_detect


def _make_features(n: int = 200, spike_at: int | None = None) -> pd.DataFrame:
    """Build a synthetic features DataFrame with one monitored signal.

    All values are 10.0 with roll24h_mean=10.0, roll24h_std=1.0.
    If spike_at is given, that row gets a value of 20.0 (z = 10).
    """
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-06-01", periods=n, freq="10min")
    values = np.full(n, 10.0)
    if spike_at is not None:
        values[spike_at] = 20.0

    return pd.DataFrame(
        {
            "ts": ts,
            "vib_mtr_de_1": values,
            "vib_mtr_de_1_roll24h_mean": np.full(n, 10.0),
            "vib_mtr_de_1_roll24h_std": np.full(n, 1.0),
            "is_running": np.full(n, True),
        }
    )


class TestZscoreDetect:
    def test_detects_injected_spike(self):
        df = _make_features(spike_at=100)
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 200
        flagged = result[result["is_anomaly"]]
        assert len(flagged) == 1
        assert flagged.iloc[0]["signal"] == "vib_mtr_de_1"
        assert flagged.iloc[0]["score"] == pytest.approx(10.0)
        assert flagged.iloc[0]["method"] == "zscore"

    def test_no_anomalies_on_normal_data(self):
        df = _make_features()
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 200
        assert result["is_anomaly"].sum() == 0

    def test_normal_rows_have_scores(self):
        df = _make_features()
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert (result["score"] == 0.0).all()

    def test_ignores_non_running_periods(self):
        df = _make_features(spike_at=100)
        df.loc[100, "is_running"] = False
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 199
        assert result["is_anomaly"].sum() == 0

    def test_threshold_sensitivity(self):
        df = _make_features(spike_at=50)
        # z = 10 at index 50; threshold=11 should miss it
        result = zscore_detect(df, threshold=11.0, signals=["vib_mtr_de_1"])
        assert len(result) == 200
        assert result["is_anomaly"].sum() == 0
        # threshold=9 should catch it
        result = zscore_detect(df, threshold=9.0, signals=["vib_mtr_de_1"])
        assert result["is_anomaly"].sum() == 1

    def test_handles_zero_std(self):
        df = _make_features()
        df["vib_mtr_de_1_roll24h_std"] = 0.0
        # z = inf when std=0, but (10-10)/0 = NaN → not scored
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 0

    def test_skips_missing_signals(self):
        df = _make_features()
        result = zscore_detect(df, threshold=4.0, signals=["nonexistent_signal"])
        assert len(result) == 0

    def test_output_columns_match_schema(self):
        df = _make_features(spike_at=10)
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert list(result.columns) == ["ts", "signal", "method", "score", "is_anomaly"]

    def test_multiple_spikes(self):
        df = _make_features()
        df.loc[10, "vib_mtr_de_1"] = 20.0  # z = 10
        df.loc[50, "vib_mtr_de_1"] = 25.0  # z = 15
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 200
        flagged = result[result["is_anomaly"]]
        assert len(flagged) == 2
        scores = sorted(flagged["score"].tolist())
        assert scores[0] == pytest.approx(10.0)
        assert scores[1] == pytest.approx(15.0)

    def test_uses_ts_from_index(self):
        df = _make_features(spike_at=5)
        ts_col = df["ts"].copy()
        df = df.set_index("ts")
        result = zscore_detect(df, threshold=4.0, signals=["vib_mtr_de_1"])
        assert len(result) == 200
        flagged = result[result["is_anomaly"]]
        assert len(flagged) == 1
        assert flagged.iloc[0]["ts"] == ts_col.iloc[5]

    def test_defaults_to_rolling_signals(self):
        df = _make_features(spike_at=10)
        result = zscore_detect(df, threshold=4.0)
        assert result["is_anomaly"].sum() == 1
