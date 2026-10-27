"""CLI to extract compressor data from Cognite Open Industrial Data.

This script is run by the user on their laptop (requires client credentials).
Do NOT run this from Claude Code — see CLAUDE.md.

Usage:
    python -m oidpipe.ingest.cdf_extract --list-tags
    python -m oidpipe.ingest.cdf_extract --extract --start 2023-06-01 --end 2024-06-01
"""

from __future__ import annotations

import argparse
import logging
import ssl
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from cognite.client import ClientConfig, CogniteClient, global_config
from cognite.client.credentials import OAuthClientCredentials

from config.settings import get_settings

logger = logging.getLogger(__name__)

COMPRESSOR_ASSETS = [
    "23-KA-9101",
    "23-HA-9103",
    "23-HA-9114",
    "23-HA-9115",
    "23-VG-9101",
    "23-FV-92543",
    "23-PV-92538",
]

TAGS_YAML_PATH = Path("config/tags.yaml")


CA_BUNDLE_PATH = Path("combined-ca-bundle.pem")


def build_client(
    project: str, client_id: str, client_secret: str, tenant_id: str, cluster: str
) -> CogniteClient:
    """Build a CogniteClient with OAuth client credentials."""
    if CA_BUNDLE_PATH.exists():
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(str(CA_BUNDLE_PATH))
        global_config.ssl_context = ctx
        logger.info("Loaded custom CA bundle from %s", CA_BUNDLE_PATH)

    base_url = f"https://{cluster}.cognitedata.com"
    creds = OAuthClientCredentials(
        token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=[f"{base_url}/.default"],
    )
    return CogniteClient(
        ClientConfig(
            client_name="oid-compressor-copilot",
            base_url=base_url,
            project=project,
            credentials=creds,
        )
    )


def load_tag_mapping(tags_path: Path = TAGS_YAML_PATH) -> dict[str, dict[str, str]]:
    """Load logical-name → CDF metadata mapping from tags.yaml."""
    with open(tags_path) as f:
        return yaml.safe_load(f)


def _month_ranges(
    start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Split a date range into calendar-month chunks."""
    ranges = []
    current = start
    while current < end:
        if current.month == 12:
            next_start = datetime(current.year + 1, 1, 1)
        else:
            next_start = datetime(current.year, current.month + 1, 1)
        chunk_end = min(next_start, end)
        ranges.append((current, chunk_end))
        current = chunk_end
    return ranges


def extract_readings(
    client: CogniteClient,
    tag_mapping: dict[str, dict[str, str]],
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> pd.DataFrame:
    """Pull 10-min average readings in monthly chunks → readings.parquet (long format)."""
    external_ids = [v["external_id"] for v in tag_mapping.values()]
    chunks = _month_ranges(start, end)
    all_frames: list[pd.DataFrame] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(
            "Chunk %d/%d: %s → %s",
            i, len(chunks),
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        result = client.time_series.data.retrieve(
            external_id=external_ids,
            aggregates=["average"],
            granularity="10m",
            start=chunk_start,
            end=chunk_end,
        )
        wide = result.to_pandas()
        if wide.empty:
            logger.warning("No data for chunk %d", i)
            continue

        # Columns come back as MultiIndex tuples ("external_id", "average")
        # or pipe-joined strings "external_id|average" depending on SDK version.
        if isinstance(wide.columns, pd.MultiIndex):
            wide.columns = wide.columns.get_level_values(0)
        else:
            wide.columns = [c.replace("|average", "") for c in wide.columns]

        # Melt to long format: ts, external_id, value
        wide.index.name = "ts"
        long = wide.reset_index().melt(
            id_vars=["ts"], var_name="external_id", value_name="value"
        )
        long = long.dropna(subset=["value"])
        all_frames.append(long)

    if not all_frames:
        logger.error("No data retrieved for the entire date range.")
        sys.exit(1)

    readings = pd.concat(all_frames, ignore_index=True)
    readings = readings.sort_values(["ts", "external_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "readings.parquet"
    readings.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %d rows to %s (%d unique tags, %s → %s)",
        len(readings), out_path,
        readings["external_id"].nunique(),
        readings["ts"].min(),
        readings["ts"].max(),
    )
    return readings


def extract_tags(
    tag_mapping: dict[str, dict[str, str]], output_dir: Path
) -> pd.DataFrame:
    """Build tags.parquet from the tag mapping."""
    rows = [
        {
            "external_id": info["external_id"],
            "logical_name": logical_name,
            "name": logical_name,
            "description": info["description"],
            "unit": info["unit"],
            "asset_name": info["equipment"],
        }
        for logical_name, info in tag_mapping.items()
    ]
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "tags.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d tags to %s", len(df), out_path)
    return df


def extract_events(
    client: CogniteClient,
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> pd.DataFrame:
    """Pull maintenance/process events for compressor assets → events.parquet."""
    asset_ids: list[int] = []
    for asset_name in COMPRESSOR_ASSETS:
        assets = client.assets.search(name=asset_name, limit=1)
        if assets:
            asset_ids.append(assets[0].id)

    if not asset_ids:
        logger.warning("No assets found — writing empty events.parquet")
        df = pd.DataFrame(
            columns=["event_id", "start_time", "end_time", "type", "subtype",
                      "description", "asset_names"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "events.parquet", index=False)
        return df

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    events = client.events.list(
        asset_ids=asset_ids,
        start_time={"min": start_ms},
        end_time={"max": end_ms},
        limit=None,
    )

    rows = [
        {
            "event_id": e.id,
            "start_time": pd.Timestamp(e.start_time, unit="ms", tz="UTC") if e.start_time else None,
            "end_time": pd.Timestamp(e.end_time, unit="ms", tz="UTC") if e.end_time else None,
            "type": e.type,
            "subtype": e.subtype,
            "description": e.description,
            "asset_names": ",".join(str(a) for a in (e.asset_ids or [])),
        }
        for e in events
    ]

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "events.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d events to %s", len(df), out_path)
    return df


# ---------------------------------------------------------------------------
# Tag discovery (--list-tags)
# ---------------------------------------------------------------------------

def list_tags(client: CogniteClient, output_path: Path) -> pd.DataFrame:
    """Find time series under the compressor asset subtree and write a CSV."""
    logger.info("Searching for assets matching: %s", COMPRESSOR_ASSETS)

    all_ts_rows: list[dict[str, str | None]] = []

    for asset_name in COMPRESSOR_ASSETS:
        assets = client.assets.search(name=asset_name, limit=5)
        if not assets:
            logger.warning("No asset found for '%s'", asset_name)
            continue

        asset = assets[0]
        logger.info("Asset '%s' (id=%d) — fetching time series", asset.name, asset.id)

        time_series_list = client.time_series.list(asset_ids=[asset.id], limit=None)
        for ts in time_series_list:
            all_ts_rows.append(
                {
                    "external_id": ts.external_id,
                    "name": ts.name,
                    "description": ts.description,
                    "unit": ts.unit,
                    "asset_id": str(ts.asset_id) if ts.asset_id else None,
                    "asset_name": asset.name,
                }
            )

    if not all_ts_rows:
        logger.error("No time series found under any compressor asset.")
        sys.exit(1)

    df = pd.DataFrame(all_ts_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d candidate tags to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract data from Cognite OID")
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Discover time series under compressor assets and write tag_candidates.csv",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract readings, tags, and events to Parquet",
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Output directory for Parquet files (default: data/raw)",
    )

    args = parser.parse_args(argv)

    if not args.list_tags and not args.extract:
        parser.error("Specify --list-tags or --extract")

    settings = get_settings()
    if (
        not settings.cognite_client_id
        or not settings.cognite_client_secret
        or not settings.cognite_tenant_id
    ):
        logger.error(
            "Set OID_COGNITE_CLIENT_ID, OID_COGNITE_CLIENT_SECRET, and "
            "OID_COGNITE_TENANT_ID in .env (see .env.example)"
        )
        sys.exit(1)

    client = build_client(
        project=settings.cognite_project,
        client_id=settings.cognite_client_id,
        client_secret=settings.cognite_client_secret,
        tenant_id=settings.cognite_tenant_id,
        cluster=settings.cognite_cdf_cluster,
    )

    if args.list_tags:
        list_tags(client, Path("data/raw/tag_candidates.csv"))

    if args.extract:
        if not args.start or not args.end:
            parser.error("--extract requires --start and --end")

        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        output_dir = Path(args.output_dir)

        tag_mapping = load_tag_mapping()
        logger.info("Loaded %d tags from %s", len(tag_mapping), TAGS_YAML_PATH)

        extract_readings(client, tag_mapping, start, end, output_dir)
        extract_tags(tag_mapping, output_dir)
        extract_events(client, start, end, output_dir)

        logger.info("Extraction complete → %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
