from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from housing_analytics.analytics_v1 import (
    DEFAULT_LEGACY_INPUT,
    DEFAULT_SYNTHETIC_INPUT,
    EXPECTED_CANONICAL_LINEAGE,
    LEGACY_INPUT_COLUMNS,
    REPOSITORY_ROOT,
    SchemaError,
    load_legacy_canonical,
    load_synthetic_fixture,
    prepare_analytics_v1,
    validate_legacy_schema,
    write_derived_csv,
)


def normalized_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1.0.0",
        "source_kind": "synthetic_test",
        "record_id": "ROW-1",
        "listing_title": "",
        "location_label": pd.NA,
        "city_id": "SYN_CITY_A",
        "rooms": 2,
        "halls": 1,
        "area_sqm": 80.0,
        "orientation_primary": "south",
        "furnishing": "simple",
        "floor_level": "middle",
        "total_floors": 18,
        "building_type": "slab",
        "villa_type": pd.NA,
        "follower_count": 6,
        "listing_age_days": 45,
        "asking_total_price_10k_cny": 96.0,
        "platform_unit_price_cny_sqm": 12_000.0,
        "title_mentions_park": False,
        "title_mentions_light": False,
        "title_mentions_parking": False,
        "title_mentions_water_view": False,
        "title_mentions_business": False,
        "title_mentions_transport": False,
        "title_mentions_tax_tenure": False,
    }
    row.update(overrides)
    return row


class AnalyticsV1Tests(unittest.TestCase):
    def test_legacy_schema_requires_exact_19_columns(self) -> None:
        valid = pd.DataFrame(columns=LEGACY_INPUT_COLUMNS)
        validate_legacy_schema(valid)
        with self.assertRaises(SchemaError):
            validate_legacy_schema(valid.drop(columns=["城市gdp"]))

    def test_row_filters_are_explicit(self) -> None:
        frame = pd.DataFrame(
            [
                normalized_row(record_id="VALID"),
                normalized_row(record_id="VILLA", villa_type="townhouse"),
                normalized_row(record_id="NO-FLOOR", floor_level=pd.NA),
                normalized_row(record_id="ZERO", total_floors=0),
            ]
        )
        result, lineage = prepare_analytics_v1(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(lineage.villa_rows, 1)
        self.assertEqual(lineage.invalid_floor_rows, 1)
        self.assertEqual(lineage.villa_or_invalid_floor_removed, 2)
        self.assertEqual(lineage.total_floor_zero_removed, 1)

    def test_duplicate_signature_drops_only_exact_analytical_duplicates(self) -> None:
        frame = pd.DataFrame(
            [
                normalized_row(record_id="FIRST"),
                normalized_row(record_id="DUPLICATE"),
                normalized_row(record_id="DISTINCT", title_mentions_transport=True),
            ]
        )
        result, lineage = prepare_analytics_v1(frame)
        self.assertEqual(result["record_id"].tolist(), ["FIRST", "DISTINCT"])
        self.assertEqual(lineage.exact_duplicates_removed, 1)

    def test_derived_fields_have_versioned_boundaries(self) -> None:
        result, _ = prepare_analytics_v1(pd.DataFrame([normalized_row()]))
        row = result.iloc[0]
        self.assertEqual(row["calculated_unit_price_cny_sqm"], 12_000.0)
        self.assertEqual(row["area_bucket"], "60_to_lt_90")
        self.assertEqual(row["asking_total_price_bucket"], "lt_100")

    def test_tracked_synthetic_fixture_executes_through_same_pipeline(self) -> None:
        normalized = load_synthetic_fixture(DEFAULT_SYNTHETIC_INPUT)
        result, lineage = prepare_analytics_v1(normalized)
        self.assertEqual(lineage.input_rows, 6)
        self.assertEqual(lineage.villa_or_invalid_floor_removed, 2)
        self.assertEqual(lineage.output_rows, 4)
        self.assertTrue(result["source_kind"].eq("synthetic").all())
        self.assertNotIn("city_gdp_100m_cny", result.columns)

    def test_real_output_cannot_escape_private_derived_directory(self) -> None:
        result, _ = prepare_analytics_v1(pd.DataFrame([normalized_row()]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_derived_csv(
                    result, Path(directory) / "real_data.csv", real=True
                )

    @unittest.skipUnless(
        DEFAULT_LEGACY_INPUT.is_file(), "private canonical snapshot is not available"
    )
    def test_private_canonical_lineage(self) -> None:
        normalized = load_legacy_canonical(DEFAULT_LEGACY_INPUT)
        result, lineage = prepare_analytics_v1(
            normalized, expected_lineage=EXPECTED_CANONICAL_LINEAGE
        )
        self.assertEqual(len(result), 267_305)
        self.assertEqual(lineage.exact_duplicates_removed, 30_841)
        self.assertEqual(result.drop(columns=["record_id"]).duplicated().sum(), 0)


if __name__ == "__main__":
    unittest.main()
