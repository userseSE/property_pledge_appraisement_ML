from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housing_analytics.analytics_v1 import (
    DEFAULT_SYNTHETIC_INPUT,
    load_synthetic_fixture,
    prepare_analytics_v1,
)
from housing_analytics.run_analytics import (
    FORBIDDEN_SQL_TOKENS,
    PROPERTY_SEGMENT_DIMENSIONS,
    PUBLIC_OUTPUTS,
    SQL_DIRECTORY,
    SQL_FILES,
    run_analytics,
    validate_public_summary,
    validate_sql_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SUMMARY = REPOSITORY_ROOT / "reports/summary.json"


class AnalyticsRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name)
        normalized = load_synthetic_fixture(DEFAULT_SYNTHETIC_INPUT)
        prepared, _ = prepare_analytics_v1(normalized)
        cls.input_path = root / "synthetic_analytics_v1.csv"
        prepared.to_csv(cls.input_path, index=False, lineterminator="\n")
        cls.output_directory = root / "reports"
        cls.summary = run_analytics(
            cls.input_path,
            cls.output_directory,
            source_kind="synthetic",
            min_group_size=2,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_exactly_five_fixed_sql_files(self) -> None:
        validate_sql_contract()
        self.assertEqual(
            tuple(sorted(path.name for path in SQL_DIRECTORY.glob("*.sql"))),
            SQL_FILES,
        )

    def test_sql_excludes_macro_and_title_flag_fields(self) -> None:
        combined = "\n".join(
            (SQL_DIRECTORY / filename).read_text(encoding="utf-8").lower()
            for filename in SQL_FILES
        )
        for token in FORBIDDEN_SQL_TOKENS:
            self.assertNotIn(token.lower(), combined)

    def test_synthetic_fixture_executes_full_analytics_pipeline(self) -> None:
        self.assertEqual(
            self.summary["q1_market_coverage"]["overview"]["listing_count"], 4
        )
        self.assertEqual(self.summary["metadata"]["question_count"], 5)
        self.assertEqual(
            set(self.summary["q3_property_segments"]),
            set(PROPERTY_SEGMENT_DIMENSIONS),
        )
        actual = {
            path.relative_to(self.output_directory).as_posix()
            for path in self.output_directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, set(PUBLIC_OUTPUTS))

    def test_public_summary_has_no_row_level_or_excluded_fields(self) -> None:
        validate_public_summary(self.summary, min_group_size=2)
        serialized = json.dumps(self.summary, ensure_ascii=False).lower()
        for token in (*FORBIDDEN_SQL_TOKENS, "record_id", "listing_title"):
            self.assertNotIn(token.lower(), serialized)

    def test_every_published_group_meets_threshold(self) -> None:
        threshold = self.summary["metadata"]["minimum_group_size"]
        collections = [
            self.summary["q1_market_coverage"]["top_cities"],
            self.summary["q1_market_coverage"]["room_hall_configurations"],
            self.summary["q1_market_coverage"]["area_distribution"],
            *self.summary["q3_property_segments"].values(),
            self.summary["q4_city_price_structure"]["all_listings"],
            self.summary["q4_city_price_structure"]["comparable_segment"],
            *self.summary["q5_platform_snapshot_signals"]["segment_statistics"].values(),
            self.summary["q5_platform_snapshot_signals"]["listing_age_distribution"],
        ]
        for collection in collections:
            for item in collection:
                self.assertGreaterEqual(item["sample_count"], threshold)

    @unittest.skipUnless(PUBLIC_SUMMARY.is_file(), "generated public summary is absent")
    def test_generated_real_summary_satisfies_public_contract(self) -> None:
        summary = json.loads(PUBLIC_SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["q1_market_coverage"]["overview"]["listing_count"], 267_305
        )
        self.assertEqual(summary["metadata"]["minimum_group_size"], 100)
        validate_public_summary(summary, min_group_size=100)


if __name__ == "__main__":
    unittest.main()
