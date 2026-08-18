from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from property_pledge.analytics_v1 import (
    DEFAULT_SYNTHETIC_INPUT,
    load_synthetic_fixture,
    prepare_analytics_v1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODEL_METRICS = REPOSITORY_ROOT / "reports/model_metrics.csv"
from property_pledge.run_modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    MODELING_PUBLIC_OUTPUTS,
    build_main_assignment,
    build_preprocessor,
    build_stress_assignment,
    run_modeling,
    validate_feature_contract,
)


class ModelingSlice3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        normalized = load_synthetic_fixture(DEFAULT_SYNTHETIC_INPUT)
        cls.synthetic, _ = prepare_analytics_v1(normalized)
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name)
        cls.public_output = root / "reports"
        cls.private_output = root / "private"
        cls.metadata = run_modeling(
            source_kind="synthetic",
            public_output_directory=cls.public_output,
            private_output_directory=cls.private_output,
            min_slice_size=1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_exact_feature_allowlist_excludes_target_derived_fields(self) -> None:
        validate_feature_contract()
        self.assertEqual(
            FEATURE_COLUMNS,
            (
                "city_id",
                "rooms",
                "halls",
                "area_sqm",
                "orientation_primary",
                "furnishing",
                "floor_level",
                "total_floors",
                "building_type",
            ),
        )
        self.assertFalse(set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES)
        self.assertNotIn("platform_unit_price_cny_sqm", FEATURE_COLUMNS)
        self.assertNotIn("calculated_unit_price_cny_sqm", FEATURE_COLUMNS)
        self.assertNotIn("asking_total_price_bucket", FEATURE_COLUMNS)

    def test_main_split_is_deterministic_and_disjoint(self) -> None:
        first = build_main_assignment(self.synthetic)
        second = build_main_assignment(self.synthetic.sample(frac=1, random_state=9))
        by_id_first = dict(zip(self.synthetic["record_id"], first))
        shuffled = self.synthetic.sample(frac=1, random_state=9)
        by_id_second = dict(zip(shuffled["record_id"], second.loc[shuffled.index]))
        self.assertEqual(by_id_first, by_id_second)
        self.assertEqual(first.value_counts().to_dict(), {"train": 2, "validation": 1, "frozen_test": 1})

    def test_unseen_city_split_holds_out_complete_cities(self) -> None:
        assignment, unseen = build_stress_assignment(self.synthetic)
        held = set(
            self.synthetic.loc[assignment.eq("unseen_city_test"), "city_id"].astype(str)
        )
        development = set(
            self.synthetic.loc[~assignment.eq("unseen_city_test"), "city_id"].astype(str)
        )
        self.assertEqual(held, set(unseen))
        self.assertFalse(held & development)

    def test_preprocessor_learns_categories_from_training_rows_only(self) -> None:
        train = self.synthetic.iloc[:2].copy()
        scored = self.synthetic.iloc[[2]].copy()
        scored.loc[:, "city_id"] = "NEVER_SEEN_IN_TRAINING"
        preprocessor = build_preprocessor()
        preprocessor.fit(train.loc[:, FEATURE_COLUMNS])
        encoder = preprocessor.named_transformers_["categorical"]
        city_position = list(CATEGORICAL_FEATURES).index("city_id")
        learned_cities = set(encoder.categories_[city_position].astype(str))
        self.assertNotIn("NEVER_SEEN_IN_TRAINING", learned_cities)
        transformed = preprocessor.transform(scored.loc[:, FEATURE_COLUMNS])
        self.assertEqual(transformed.shape[0], 1)

    def test_synthetic_fixture_runs_full_modeling_path(self) -> None:
        self.assertEqual(self.metadata["row_count"], 4)
        self.assertEqual(
            self.metadata["main_split_counts"],
            {"frozen_test": 1, "train": 2, "validation": 1},
        )
        actual = {
            path.relative_to(self.public_output).as_posix()
            for path in self.public_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual, set(MODELING_PUBLIC_OUTPUTS))

        metrics = pd.read_csv(self.public_output / "model_metrics.csv")
        self.assertEqual(len(metrics), 16)
        self.assertEqual(
            set(metrics["model"]),
            {"global_training_median", "city_training_median", "ridge", "xgboost"},
        )

    def test_row_level_outputs_are_private_only(self) -> None:
        self.assertTrue((self.private_output / "split_assignments.csv").is_file())
        self.assertTrue((self.private_output / "predictions.csv").is_file())
        self.assertTrue((self.private_output / "run_metadata.json").is_file())
        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                self.public_output / "model_metrics.csv",
                self.public_output / "model_error_slices.csv",
                self.public_output / "modeling_findings.md",
            )
        ).lower()
        self.assertNotIn("record_id", public_text)
        self.assertNotIn("prediction_xgboost", public_text)
        self.assertNotIn("platform_unit_price", public_text)
        self.assertNotIn("calculated_unit_price", public_text)

    def test_all_public_error_slices_include_counts(self) -> None:
        slices = pd.read_csv(self.public_output / "model_error_slices.csv")
        self.assertTrue(slices["sample_count"].ge(1).all())
        self.assertFalse(slices["sample_count"].isna().any())
        self.assertIn("median_absolute_percentage_error_pct", slices.columns)
        self.assertTrue(
            slices["median_absolute_percentage_error_pct"].ge(0).all()
        )

    @unittest.skipUnless(
        PUBLIC_MODEL_METRICS.is_file(), "generated real modeling reports are absent"
    )
    def test_generated_real_reports_preserve_frozen_test_contract(self) -> None:
        metrics = pd.read_csv(PUBLIC_MODEL_METRICS)
        frozen = metrics.loc[
            metrics["evaluation"].eq("random_split")
            & metrics["split"].eq("frozen_test")
        ]
        self.assertEqual(set(frozen["model"]), {
            "global_training_median",
            "city_training_median",
            "ridge",
            "xgboost",
        })
        self.assertTrue(frozen["sample_count"].eq(40_104).all())
        slices = pd.read_csv(REPOSITORY_ROOT / "reports/model_error_slices.csv")
        self.assertTrue(slices["sample_count"].ge(100).all())
        self.assertIn("median_absolute_percentage_error_pct", slices.columns)
        self.assertTrue(
            slices["median_absolute_percentage_error_pct"].ge(0).all()
        )


if __name__ == "__main__":
    unittest.main()
