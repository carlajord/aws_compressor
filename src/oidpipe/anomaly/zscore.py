"""Z-score anomaly detection on monitored signals."""

from __future__ import annotations

import logging

import pandas as pd

from oidpipe.features.build import ROLLING_SIGNALS

logger = logging.getLogger(__name__)


def zscore_detect(
    features: pd.DataFrame,
    threshold: float = 4.0,
    signals: list[str] | None = None,
) -> pd.DataFrame:
    """Score every running row with |z| and flag those above *threshold*.

    Z-score is computed from the pre-built rolling 24h stats:
        z = (x - roll24h_mean) / roll24h_std

    Only running periods (is_running == True) are scored.

    Returns a DataFrame with columns: ts, signal, method, score, is_anomaly
    matching the anomalies table schema.  Every scored row is returned so
    that downstream consumers (e.g. normal-period selection for iforest) can
    distinguish flagged from unflagged timestamps.
    """
    if signals is None:
        signals = ROLLING_SIGNALS

    if "ts" in features.columns:
        ts = features["ts"].values
    else:
        ts = features.index.values

    running = features["is_running"] if "is_running" in features.columns else True

    rows: list[dict] = []

    for sig in signals:
        mean_col = f"{sig}_roll24h_mean"
        std_col = f"{sig}_roll24h_std"

        if sig not in features.columns:
            continue
        if mean_col not in features.columns or std_col not in features.columns:
            continue

        z = (features[sig] - features[mean_col]) / features[std_col]
        z_abs = z.abs()

        scored_mask = running & z_abs.notna()
        positional = scored_mask.values.nonzero()[0]
        for pos in positional:
            z_val = float(z_abs.iloc[pos])
            rows.append(
                {
                    "ts": ts[pos],
                    "signal": sig,
                    "method": "zscore",
                    "score": round(z_val, 4),
                    "is_anomaly": z_val > threshold,
                }
            )

    result = pd.DataFrame(rows, columns=["ts", "signal", "method", "score", "is_anomaly"])
    n_flagged = int(result["is_anomaly"].sum()) if len(result) > 0 else 0
    logger.info(
        "Z-score detection: %d anomalies out of %d scored rows across %d signals (threshold=%.1f)",
        n_flagged,
        len(result),
        len(signals),
        threshold,
    )
    return result
