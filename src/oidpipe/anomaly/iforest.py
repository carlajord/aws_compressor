"""Isolation Forest anomaly detection on multivariate features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

IFOREST_FEATURES = [
    "vib_mtr_de_1",
    "vib_mtr_de_2",
    "vib_mtr_nde_1",
    "vib_mtr_nde_2",
    "t_bearing_comp_de",
    "t_bearing_comp_nde",
    "t_bearing_mtr_de",
    "t_bearing_mtr_nde",
    "pressure_ratio",
    "cooler_approach_c",
]


def iforest_detect(
    features: pd.DataFrame,
    contamination: float = 0.01,
    normal_mask: pd.Series | None = None,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Isolation Forest on multivariate features.

    The scaler and model are fit on the "normal period" — rows where
    is_running is True and *normal_mask* is True.  Typically *normal_mask*
    is derived from a prior z-score pass (timestamps with no z-score flag).
    If not supplied, all running rows are used for fitting.

    Returns a DataFrame with columns: ts, signal, method, score, is_anomaly
    matching the anomalies table schema. Signal is always "multivariate".
    """
    if feature_cols is None:
        feature_cols = IFOREST_FEATURES

    if "ts" in features.columns:
        ts = features["ts"].values
    else:
        ts = features.index.values

    running = features["is_running"] if "is_running" in features.columns else pd.Series(True, index=features.index)

    available = [c for c in feature_cols if c in features.columns]
    if len(available) < 2:
        logger.warning("Isolation Forest skipped: fewer than 2 feature columns available (%s)", available)
        return pd.DataFrame(columns=["ts", "signal", "method", "score", "is_anomaly"])

    X = features[available].copy()

    # Determine normal period for fitting
    fit_mask = running.copy()
    if normal_mask is not None:
        fit_mask = fit_mask & normal_mask

    X_fit = X.loc[fit_mask].dropna()
    if len(X_fit) < 10:
        logger.warning("Isolation Forest skipped: only %d normal-period samples", len(X_fit))
        return pd.DataFrame(columns=["ts", "signal", "method", "score", "is_anomaly"])

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(X_fit_scaled)

    # Score all running rows (not just normal period)
    score_mask = running & X.notna().all(axis=1)
    X_score = X.loc[score_mask]
    X_score_scaled = scaler.transform(X_score)

    raw_scores = model.decision_function(X_score_scaled)
    predictions = model.predict(X_score_scaled)

    # score = -decision_function (higher = more anomalous)
    scores = -raw_scores

    positional = score_mask.values.nonzero()[0]
    rows: list[dict] = []
    for i, (pos, score, pred) in enumerate(zip(positional, scores, predictions)):
        rows.append(
            {
                "ts": ts[pos],
                "signal": "multivariate",
                "method": "iforest",
                "score": round(float(score), 4),
                "is_anomaly": bool(pred == -1),
            }
        )

    result = pd.DataFrame(rows, columns=["ts", "signal", "method", "score", "is_anomaly"])
    n_anomalies = result["is_anomaly"].sum()
    logger.info(
        "Isolation Forest: %d anomalies out of %d scored rows (contamination=%.3f, %d features)",
        n_anomalies,
        len(result),
        contamination,
        len(available),
    )
    return result
