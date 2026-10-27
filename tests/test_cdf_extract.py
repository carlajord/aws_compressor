"""Tests for the CDF extraction CLI with a mocked CogniteClient."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from oidpipe.ingest.cdf_extract import (
    COMPRESSOR_ASSETS,
    _month_ranges,
    extract_events,
    extract_readings,
    extract_tags,
    list_tags,
    load_tag_mapping,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_asset(name: str, asset_id: int) -> MagicMock:
    asset = MagicMock()
    asset.name = name
    asset.id = asset_id
    return asset


def _make_mock_ts(
    external_id: str, name: str, description: str, unit: str, asset_id: int
) -> MagicMock:
    ts = MagicMock()
    ts.external_id = external_id
    ts.name = name
    ts.description = description
    ts.unit = unit
    ts.asset_id = asset_id
    return ts


def _make_mock_event(
    event_id: int, start_time: int, end_time: int | None,
    etype: str, subtype: str, description: str, asset_ids: list[int],
) -> MagicMock:
    e = MagicMock()
    e.id = event_id
    e.start_time = start_time
    e.end_time = end_time
    e.type = etype
    e.subtype = subtype
    e.description = description
    e.asset_ids = asset_ids
    return e


def _make_wide_dataframe(external_ids: list[str], n_rows: int = 10) -> pd.DataFrame:
    """Build a wide DataFrame like cognite retrieve().to_pandas() returns."""
    ts_index = pd.date_range("2024-03-01", periods=n_rows, freq="10min", tz="UTC")
    data = {f"{eid}|average": np.random.default_rng(42).random(n_rows) * 10
            for eid in external_ids}
    return pd.DataFrame(data, index=ts_index)


@pytest.fixture
def sample_tag_mapping(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    """Write a minimal tags.yaml and return (path, mapping dict)."""
    mapping = {
        "p_suction_barg": {
            "external_id": "VAL_PT_SUCTION",
            "description": "Suction pressure",
            "unit": "barg",
            "equipment": "23-KA-9101",
        },
        "t_discharge_c": {
            "external_id": "VAL_TT_DISCHARGE",
            "description": "Discharge temperature",
            "unit": "degC",
            "equipment": "23-KA-9101",
        },
    }
    import yaml
    tags_path = tmp_path / "tags.yaml"
    with open(tags_path, "w") as f:
        yaml.dump(mapping, f)
    return tags_path, mapping


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    asset_ka = _make_mock_asset("23-KA-9101", 1001)
    asset_ha = _make_mock_asset("23-HA-9103", 1002)

    def search_side_effect(name: str, limit: int = 5) -> list[MagicMock]:
        mapping = {
            "23-KA-9101": [asset_ka],
            "23-HA-9103": [asset_ha],
        }
        return mapping.get(name, [])

    client.assets.search.side_effect = search_side_effect

    ts_suction = _make_mock_ts("PT-9101-S", "Suction pressure", "Suction P", "barg", 1001)
    ts_discharge = _make_mock_ts("PT-9101-D", "Discharge pressure", "Disch P", "barg", 1001)
    ts_cooler = _make_mock_ts("TT-9103-OUT", "Cooler outlet temp", "Gas out", "degC", 1002)

    def ts_list_side_effect(asset_ids: list[int], limit: object = None) -> list[MagicMock]:
        if asset_ids == [1001]:
            return [ts_suction, ts_discharge]
        if asset_ids == [1002]:
            return [ts_cooler]
        return []

    client.time_series.list.side_effect = ts_list_side_effect
    return client


# ---------------------------------------------------------------------------
# Tests: month ranges
# ---------------------------------------------------------------------------

class TestMonthRanges:
    def test_single_month(self):
        ranges = _month_ranges(datetime(2024, 3, 1), datetime(2024, 4, 1))
        assert len(ranges) == 1
        assert ranges[0] == (datetime(2024, 3, 1), datetime(2024, 4, 1))

    def test_three_months(self):
        ranges = _month_ranges(datetime(2024, 1, 1), datetime(2024, 4, 1))
        assert len(ranges) == 3
        assert ranges[0] == (datetime(2024, 1, 1), datetime(2024, 2, 1))
        assert ranges[2] == (datetime(2024, 3, 1), datetime(2024, 4, 1))

    def test_partial_end_month(self):
        ranges = _month_ranges(datetime(2024, 1, 1), datetime(2024, 3, 15))
        assert len(ranges) == 3
        assert ranges[2] == (datetime(2024, 3, 1), datetime(2024, 3, 15))

    def test_year_boundary(self):
        ranges = _month_ranges(datetime(2024, 11, 1), datetime(2025, 2, 1))
        assert len(ranges) == 3
        assert ranges[1] == (datetime(2024, 12, 1), datetime(2025, 1, 1))
        assert ranges[2] == (datetime(2025, 1, 1), datetime(2025, 2, 1))

    def test_twelve_months(self):
        ranges = _month_ranges(datetime(2024, 1, 1), datetime(2025, 1, 1))
        assert len(ranges) == 12


# ---------------------------------------------------------------------------
# Tests: load_tag_mapping
# ---------------------------------------------------------------------------

class TestLoadTagMapping:
    def test_loads_yaml(self, sample_tag_mapping: tuple[Path, dict]):
        tags_path, expected = sample_tag_mapping
        result = load_tag_mapping(tags_path)
        assert result == expected
        assert "p_suction_barg" in result
        assert result["p_suction_barg"]["external_id"] == "VAL_PT_SUCTION"


# ---------------------------------------------------------------------------
# Tests: extract_readings
# ---------------------------------------------------------------------------

class TestExtractReadings:
    def test_writes_parquet_long_format(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        ext_ids = [v["external_id"] for v in mapping.values()]
        wide_df = _make_wide_dataframe(ext_ids, n_rows=10)

        client = MagicMock()
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = wide_df
        client.time_series.data.retrieve.return_value = mock_result

        out = tmp_path / "raw"
        df = extract_readings(
            client, mapping,
            datetime(2024, 3, 1), datetime(2024, 4, 1),
            out,
        )

        assert (out / "readings.parquet").exists()
        assert list(df.columns) == ["ts", "external_id", "value"]
        assert set(df["external_id"]) == set(ext_ids)

    def test_monthly_chunking_calls_retrieve_per_month(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        ext_ids = [v["external_id"] for v in mapping.values()]
        wide_df = _make_wide_dataframe(ext_ids, n_rows=5)

        client = MagicMock()
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = wide_df
        client.time_series.data.retrieve.return_value = mock_result

        extract_readings(
            client, mapping,
            datetime(2024, 1, 1), datetime(2024, 4, 1),
            tmp_path / "raw",
        )

        assert client.time_series.data.retrieve.call_count == 3

    def test_exits_on_no_data(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        client = MagicMock()
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame()
        client.time_series.data.retrieve.return_value = mock_result

        with pytest.raises(SystemExit, match="1"):
            extract_readings(
                client, mapping,
                datetime(2024, 3, 1), datetime(2024, 4, 1),
                tmp_path / "raw",
            )

    def test_parquet_is_readable(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        ext_ids = [v["external_id"] for v in mapping.values()]
        wide_df = _make_wide_dataframe(ext_ids, n_rows=10)

        client = MagicMock()
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = wide_df
        client.time_series.data.retrieve.return_value = mock_result

        out = tmp_path / "raw"
        extract_readings(client, mapping, datetime(2024, 3, 1), datetime(2024, 4, 1), out)

        loaded = pd.read_parquet(out / "readings.parquet")
        assert len(loaded) > 0
        assert "ts" in loaded.columns


# ---------------------------------------------------------------------------
# Tests: extract_tags
# ---------------------------------------------------------------------------

class TestExtractTags:
    def test_writes_parquet_with_correct_columns(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        out = tmp_path / "raw"
        df = extract_tags(mapping, out)

        assert (out / "tags.parquet").exists()
        assert list(df.columns) == [
            "external_id", "logical_name", "name", "description", "unit", "asset_name"
        ]
        assert len(df) == 2

    def test_logical_names_match_keys(
        self, sample_tag_mapping: tuple[Path, dict], tmp_path: Path
    ):
        _, mapping = sample_tag_mapping
        df = extract_tags(mapping, tmp_path / "raw")
        assert set(df["logical_name"]) == set(mapping.keys())


# ---------------------------------------------------------------------------
# Tests: extract_events
# ---------------------------------------------------------------------------

class TestExtractEvents:
    def test_writes_parquet(self, tmp_path: Path):
        client = MagicMock()
        asset = _make_mock_asset("23-KA-9101", 1001)
        client.assets.search.return_value = [asset]

        mock_event = _make_mock_event(
            event_id=42,
            start_time=1709251200000,
            end_time=1709254800000,
            etype="Maintenance",
            subtype="Planned",
            description="Bearing replacement",
            asset_ids=[1001],
        )
        client.events.list.return_value = [mock_event]

        out = tmp_path / "raw"
        df = extract_events(client, datetime(2024, 1, 1), datetime(2025, 1, 1), out)

        assert (out / "events.parquet").exists()
        assert len(df) == 1
        assert df.iloc[0]["event_id"] == 42
        assert df.iloc[0]["type"] == "Maintenance"

    def test_empty_events(self, tmp_path: Path):
        client = MagicMock()
        asset = _make_mock_asset("23-KA-9101", 1001)
        client.assets.search.return_value = [asset]
        client.events.list.return_value = []

        out = tmp_path / "raw"
        df = extract_events(client, datetime(2024, 1, 1), datetime(2025, 1, 1), out)

        assert (out / "events.parquet").exists()
        assert len(df) == 0

    def test_no_assets_writes_empty_parquet(self, tmp_path: Path):
        client = MagicMock()
        client.assets.search.return_value = []

        out = tmp_path / "raw"
        df = extract_events(client, datetime(2024, 1, 1), datetime(2025, 1, 1), out)

        assert (out / "events.parquet").exists()
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: list_tags (existing)
# ---------------------------------------------------------------------------

class TestListTags:
    def test_writes_csv_with_correct_columns(self, mock_client: MagicMock, tmp_path: Path):
        output = tmp_path / "tag_candidates.csv"
        df = list_tags(mock_client, output)
        assert output.exists()
        assert list(df.columns) == [
            "external_id", "name", "description", "unit", "asset_id", "asset_name",
        ]

    def test_finds_time_series_from_multiple_assets(
        self, mock_client: MagicMock, tmp_path: Path
    ):
        output = tmp_path / "tag_candidates.csv"
        df = list_tags(mock_client, output)
        assert len(df) == 3
        assert set(df["asset_name"]) == {"23-KA-9101", "23-HA-9103"}

    def test_csv_is_readable(self, mock_client: MagicMock, tmp_path: Path):
        output = tmp_path / "tag_candidates.csv"
        list_tags(mock_client, output)
        loaded = pd.read_csv(output)
        assert len(loaded) == 3
        assert loaded.iloc[0]["external_id"] == "PT-9101-S"

    def test_creates_parent_dirs(self, mock_client: MagicMock, tmp_path: Path):
        output = tmp_path / "nested" / "deep" / "tags.csv"
        df = list_tags(mock_client, output)
        assert output.exists()
        assert len(df) == 3

    def test_exits_when_no_time_series_found(self, tmp_path: Path):
        client = MagicMock()
        client.assets.search.return_value = []
        output = tmp_path / "tags.csv"
        with pytest.raises(SystemExit, match="1"):
            list_tags(client, output)


class TestCompressorAssets:
    def test_known_assets_present(self):
        assert "23-KA-9101" in COMPRESSOR_ASSETS
        assert "23-HA-9103" in COMPRESSOR_ASSETS
        assert "23-VG-9101" in COMPRESSOR_ASSETS
        assert "23-FV-92543" in COMPRESSOR_ASSETS
        assert "23-PV-92538" in COMPRESSOR_ASSETS
