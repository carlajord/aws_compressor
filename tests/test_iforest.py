"""Tests for anomaly/iforest.py — Isolation Forest with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oidpipe.anomaly.iforest import iforest_detect


def _make_normal_features(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build synthetic features with two correlated signals — all normal."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-06-01", periods=n, freq="10min")
    return pd.DataFrame(
        {
            "ts": ts,
            "vib_mtr_de_1": rng.normal(10.0, 0.5, n),
            "vib_mtr_de_2": rng.normal(12.0, 0.5, n),
            "pressure_ratio": rng.normal(3.2, 0.05, n),
            "is_running": np.full(n, True),
        }
    )


def _inject_shift(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Inject a multivariate shift into a range of rows."""
    df = df.copy()
    df.loc[start:end, "vib_mtr_de_1"] += 8.0   # 16 sigma shift
    df.loc[start:end, "vib_mtr_de_2"] += 8.0
    df.loc[start:end, "pressure_ratio"] += 1.0  # 20 sigma shift
    return df


FEATURE_COLS = ["vib_mtr_de_1", "vib_mtr_de_2", "pressure_ratio"]


class TestIforestDetect:
    def test_flags_injected_shift(self):
        df = _make_normal_features()
        df = _inject_shift(df, 450, 470)
        # Mask: first 400 rows are "clean"
        normal_mask = pd.Series(True, index=df.index)
        normal_mask.iloc[450:471] = False
        result = iforest_detect(
            df,
            contamination=0.05,
            normal_mask=normal_mask,
            feature_cols=FEATURE_COLS,
        )
        anomalies = result[result["is_anomaly"]]
        shifted_ts = set(df["ts"].iloc[450:471].values)
        flagged_ts = set(anomalies["ts"].values)
        overlap = shifted_ts & flagged_ts
        assert len(overlap) >= 15, f"Expected most shifted rows flagged, got {len(overlap)}/21"

    def test_normal_data_has_few_anomalies(self):
        df = _make_normal_features()
        result = iforest_detect(
            df,
            contamination=0.01,
            feature_cols=FEATURE_COLS,
        )
        n_anomalies = result["is_anomaly"].sum()
        # contamination=0.01 on 500 rows → ~5 expected
        assert n_anomalies <= 15, f"Too many false positives: {n_anomalies}"

    def test_ignores_non_running_periods(self):
        df = _make_normal_features(n=200)
        df = _inject_shift(df, 150, 170)
        df.loc[150:170, "is_running"] = False
        result = iforest_detect(
            df,
            contamination=0.05,
            feature_cols=FEATURE_COLS,
        )
        # Non-running rows should not be scored at all
        anomaly_ts = result[result["is_anomaly"]]["ts"].values
        non_running_ts = set(df.loc[150:170, "ts"].values)
        flagged_non_running = set(anomaly_ts) & non_running_ts
        assert len(flagged_non_running) == 0

    def test_output_columns_match_schema(self):
        df = _make_normal_features(n=100)
        result = iforest_detect(df, feature_cols=FEATURE_COLS)
        assert list(result.columns) == ["ts", "signal", "method", "score", "is_anomaly"]

    def test_method_is_iforest(self):
        df = _make_normal_features(n=100)
        result = iforest_detect(df, feature_cols=FEATURE_COLS)
        assert (result["method"] == "iforest").all()

    def test_signal_is_multivariate(self):
        df = _make_normal_features(n=100)
        result = iforest_detect(df, feature_cols=FEATURE_COLS)
        assert (result["signal"] == "multivariate").all()

    def test_normal_mask_restricts_fit(self):
        df = _make_normal_features(n=300)
        df = _inject_shift(df, 200, 280)
        # Mark only the first 100 rows as clean
        normal_mask = pd.Series(False, index=df.index)
        normal_mask.iloc[:100] = True
        result = iforest_detect(
            df,
            contamination=0.05,
            normal_mask=normal_mask,
            feature_cols=FEATURE_COLS,
        )
        anomalies = result[result["is_anomaly"]]
        shifted_ts = set(df["ts"].iloc[200:281].values)
        flagged_ts = set(anomalies["ts"].values)
        overlap = shifted_ts & flagged_ts
        assert len(overlap) >= 50, f"Expected most shifted rows flagged, got {len(overlap)}/81"

    def test_skips_with_too_few_features(self):
        df = _make_normal_features(n=100)
        result = iforest_detect(df, feature_cols=["nonexistent_only"])
        assert len(result) == 0

    def test_skips_with_too_few_samples(self):
        df = _make_normal_features(n=5)
        result = iforest_detect(df, feature_cols=FEATURE_COLS)
        assert len(result) == 0

    def test_scores_are_positive_for_outliers(self):
        df = _make_normal_features()
        df = _inject_shift(df, 450, 470)
        result = iforest_detect(
            df,
            contamination=0.05,
            feature_cols=FEATURE_COLS,
        )
        anomalies = result[result["is_anomaly"]]
        assert (anomalies["score"] > 0).all()

    def test_none_normal_mask_uses_all_running(self):
        df = _make_normal_features(n=200)
        result_none = iforest_detect(df, feature_cols=FEATURE_COLS)
        result_all = iforest_detect(
            df,
            normal_mask=pd.Series(True, index=df.index),
            feature_cols=FEATURE_COLS,
        )
        assert len(result_none) == len(result_all)
