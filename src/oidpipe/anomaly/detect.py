"""Orchestrate anomaly detection: z-score + Isolation Forest → anomalies table."""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from oidpipe.anomaly.iforest import iforest_detect
from oidpipe.anomaly.zscore import zscore_detect

logger = logging.getLogger(__name__)


def run_anomaly_detection(
    conn: duckdb.DuckDBPyConnection,
    zscore_threshold: float = 4.0,
    iforest_contamination: float = 0.01,
) -> int:
    """Run z-score then Isolation Forest and write results to the anomalies table.

    The z-score pass runs first.  Timestamps where *any* signal exceeds the
    z-score threshold are excluded from the Isolation Forest training set,
    giving a data-driven normal period instead of an arbitrary date range.

    Returns the total number of anomaly rows written.
    """
    features = conn.execute("SELECT * FROM features_10m").fetchdf()
    logger.info("Running anomaly detection on %d feature rows", len(features))

    zs = zscore_detect(features, threshold=zscore_threshold)

    # Build a normal-period mask: exclude timestamps flagged by any signal
    zs_anomalous_ts = set(zs.loc[zs["is_anomaly"], "ts"].values)
    normal_mask = ~features["ts"].isin(zs_anomalous_ts)
    n_excluded = int(normal_mask.eq(False).sum())
    logger.info(
        "Z-score excluded %d / %d timestamps from iforest training set",
        n_excluded,
        len(features),
    )

    ifo = iforest_detect(
        features,
        contamination=iforest_contamination,
        normal_mask=normal_mask,
    )

    combined = pd.concat([zs, ifo], ignore_index=True)

    if len(combined) > 0:
        conn.register("_anomalies_df", combined)
        conn.execute("INSERT INTO anomalies SELECT * FROM _anomalies_df")
        conn.unregister("_anomalies_df")

    n_zscore = len(zs[zs["is_anomaly"]]) if len(zs) > 0 else 0
    n_iforest = len(ifo[ifo["is_anomaly"]]) if len(ifo) > 0 else 0
    logger.info(
        "Anomaly detection complete: %d z-score flags, %d iforest flags, %d total rows written",
        n_zscore,
        n_iforest,
        len(combined),
    )
    return len(combined)
