"""Leakage-controlled regression evaluation for Modeling Slice 3.

The target is historical asking total price. Nine explicitly allowed property
fields enter the models. Split assignments, predictions, and intermediate
artifacts stay under ``private_data/``; only aggregate metrics, error slices,
findings, and figures are written to ``reports/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from housing_analytics.analytics_v1 import (
    DEFAULT_SYNTHETIC_INPUT as TRACKED_SYNTHETIC_FIXTURE,
    load_synthetic_fixture,
    prepare_analytics_v1,
)
from housing_analytics.run_analytics import (
    AREA_BUCKET_ORDER,
    BUCKET_LABELS,
    DEFAULT_REAL_INPUT,
    EXPECTED_REAL_ROW_COUNT,
    EXPECTED_REAL_SHA256,
    PRICE_BUCKET_ORDER,
    PUBLIC_OUTPUTS as ANALYTICS_PUBLIC_OUTPUTS,
    REPOSITORY_ROOT,
    sha256_file,
)


TARGET_COLUMN = "asking_total_price_10k_cny"
IDENTIFIER_COLUMN = "record_id"
FEATURE_COLUMNS = (
    "city_id",
    "rooms",
    "halls",
    "area_sqm",
    "orientation_primary",
    "furnishing",
    "floor_level",
    "total_floors",
    "building_type",
)
CATEGORICAL_FEATURES = (
    "city_id",
    "orientation_primary",
    "furnishing",
    "floor_level",
    "building_type",
)
NUMERIC_FEATURES = (
    "rooms",
    "halls",
    "area_sqm",
    "total_floors",
)
FORBIDDEN_FEATURES = frozenset(
    {
        TARGET_COLUMN,
        "platform_unit_price_cny_sqm",
        "calculated_unit_price_cny_sqm",
        "asking_total_price_bucket",
        IDENTIFIER_COLUMN,
        "location_label",
        "developer",
        "community",
        "city_gdp_100m_cny",
        "region",
        "city_green_coverage_pct",
        "follower_count",
        "listing_age_days",
        "title_mentions_park",
        "title_mentions_light",
        "title_mentions_parking",
        "title_mentions_water_view",
        "title_mentions_business",
        "title_mentions_transport",
        "title_mentions_tax_tenure",
    }
)
SLICE_COLUMNS = (
    "asking_total_price_bucket",
    "area_bucket",
    "rooms",
    "city_id",
)
REQUIRED_INPUT_COLUMNS = frozenset(
    {IDENTIFIER_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS, *SLICE_COLUMNS}
)

SPLIT_SEED = 20220817
MAIN_SPLIT_RATIOS = (
    ("train", 0.70),
    ("validation", 0.15),
    ("frozen_test", 0.15),
)
STRESS_DEVELOPMENT_RATIOS = (
    ("train", 0.70 / 0.85),
    ("validation", 0.15 / 0.85),
)
STRESS_TEST_SHARE = 0.15
TARGET_QUANTILE_BIN_COUNT = 20
DEFAULT_MIN_SLICE_SIZE = 100

RIDGE_ALPHA = 10.0
XGBOOST_PARAMETERS: Mapping[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "mae",
    "early_stopping_rounds": 40,
    "random_state": SPLIT_SEED,
    "n_jobs": 4,
    "verbosity": 0,
}

DEFAULT_PUBLIC_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "reports"
DEFAULT_PRIVATE_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "private_data/modeling_slice_3"
MODELING_PUBLIC_OUTPUTS = frozenset(
    {
        "model_metrics.csv",
        "model_error_slices.csv",
        "modeling_findings.md",
        "figures/model_comparison.png",
        "figures/model_error_slices.png",
        "figures/unseen_city_stress.png",
    }
)
ALLOWED_REPOSITORY_REPORT_OUTPUTS = ANALYTICS_PUBLIC_OUTPUTS | MODELING_PUBLIC_OUTPUTS


class ModelingError(RuntimeError):
    """Raised when the modeling, leakage, or output contract is violated."""


@dataclass
class EvaluationArtifacts:
    evaluation: str
    test_split_name: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    train_city_counts: pd.Series
    global_training_median: float
    preprocessor_output_feature_count: int
    xgboost_best_iteration: int


def validate_feature_contract(features: Sequence[str] = FEATURE_COLUMNS) -> None:
    actual = tuple(features)
    if actual != FEATURE_COLUMNS:
        raise ModelingError(
            f"model features must match the fixed ordered allowlist; got {actual}"
        )
    forbidden = sorted(set(actual) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ModelingError(f"target-derived or excluded features requested: {forbidden}")
    if set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) != set(FEATURE_COLUMNS):
        raise ModelingError("categorical/numeric feature partitions are incomplete")


def validate_input_frame(frame: pd.DataFrame) -> None:
    validate_feature_contract()
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(frame.columns))
    if missing:
        raise ModelingError(f"analytical input is missing columns: {missing}")
    if frame.empty:
        raise ModelingError("analytical input is empty")
    if frame[IDENTIFIER_COLUMN].isna().any() or frame[IDENTIFIER_COLUMN].duplicated().any():
        raise ModelingError("record_id must be non-null and unique")

    nulls = frame.loc[:, [TARGET_COLUMN, *FEATURE_COLUMNS]].isna().sum()
    failures = {name: int(count) for name, count in nulls.items() if count}
    if failures:
        raise ModelingError(f"modeling columns contain null values: {failures}")

    positive = (TARGET_COLUMN, "area_sqm", "total_floors")
    nonnegative = ("rooms", "halls")
    invalid = {
        column: int(frame[column].le(0).sum())
        for column in positive
        if frame[column].le(0).any()
    }
    invalid.update(
        {
            column: int(frame[column].lt(0).sum())
            for column in nonnegative
            if frame[column].lt(0).any()
        }
    )
    if invalid:
        raise ModelingError(f"modeling columns contain invalid values: {invalid}")


def _stable_key(value: Any, *, scope: str, seed: int = SPLIT_SEED) -> str:
    payload = f"{scope}|{seed}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _allocate_counts(
    sample_count: int, ratios: Sequence[tuple[str, float]]
) -> dict[str, int]:
    if sample_count < 1:
        return {name: 0 for name, _ in ratios}
    ratio_total = sum(ratio for _, ratio in ratios)
    if not math.isclose(ratio_total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ModelingError(f"split ratios must sum to one; got {ratio_total}")

    counts = {name: 0 for name, _ in ratios}
    remaining = sample_count
    if sample_count >= len(ratios):
        for name, _ in ratios:
            counts[name] = 1
        remaining -= len(ratios)

    raw = {name: remaining * ratio for name, ratio in ratios}
    for name, _ in ratios:
        addition = math.floor(raw[name])
        counts[name] += addition
        remaining -= addition

    priority = sorted(
        (name for name, _ in ratios),
        key=lambda name: (-(raw[name] - math.floor(raw[name])), list(counts).index(name)),
    )
    for name in priority[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != sample_count:
        raise ModelingError(f"failed to allocate {sample_count} rows: {counts}")
    return counts


def _target_quantile_bins(target: pd.Series) -> pd.Series:
    bin_count = min(
        TARGET_QUANTILE_BIN_COUNT,
        max(1, len(target) // 50),
    )
    if bin_count == 1:
        return pd.Series(0, index=target.index, dtype="int64")
    bins = pd.qcut(target, q=bin_count, labels=False, duplicates="drop")
    if bins.isna().any():
        raise ModelingError("target quantile split bins contain null assignments")
    return bins.astype("int64")


def deterministic_stratified_assignment(
    frame: pd.DataFrame,
    *,
    ratios: Sequence[tuple[str, float]],
    scope: str,
) -> pd.Series:
    """Assign rows by target quantile for splitting only.

    Quantile-bin values are discarded after assignment and are never returned as
    model features.
    """

    bins = _target_quantile_bins(frame[TARGET_COLUMN])
    assignment = pd.Series(index=frame.index, dtype="string")
    for bin_value in sorted(bins.unique()):
        indices = bins.index[bins.eq(bin_value)].tolist()
        indices.sort(
            key=lambda index: _stable_key(
                frame.at[index, IDENTIFIER_COLUMN], scope=f"{scope}|bin={bin_value}"
            )
        )
        counts = _allocate_counts(len(indices), ratios)
        cursor = 0
        for split_name, _ in ratios:
            split_count = counts[split_name]
            selected = indices[cursor : cursor + split_count]
            assignment.loc[selected] = split_name
            cursor += split_count
    if assignment.isna().any():
        raise ModelingError("split assignment left rows unassigned")
    return assignment


def build_main_assignment(frame: pd.DataFrame) -> pd.Series:
    assignment = deterministic_stratified_assignment(
        frame, ratios=MAIN_SPLIT_RATIOS, scope="main"
    )
    expected = {name for name, _ in MAIN_SPLIT_RATIOS}
    if set(assignment.unique()) != expected:
        raise ModelingError(f"main split is incomplete: {assignment.value_counts().to_dict()}")
    return assignment


def select_unseen_cities(frame: pd.DataFrame) -> tuple[str, ...]:
    city_counts = frame.groupby("city_id", sort=False).size().rename("sample_count")
    if len(city_counts) < 2:
        raise ModelingError("unseen-city stress test requires at least two cities")
    ordered = sorted(
        city_counts.index.astype(str).tolist(),
        key=lambda city: _stable_key(city, scope="unseen-city-order"),
    )
    cumulative: list[tuple[int, int]] = []
    running = 0
    for count, city in enumerate(ordered[:-1], start=1):
        running += int(city_counts.loc[city])
        cumulative.append((count, running))
    target_rows = len(frame) * STRESS_TEST_SHARE
    selected_count, _ = min(
        cumulative,
        key=lambda item: (abs(item[1] - target_rows), item[0]),
    )
    return tuple(ordered[:selected_count])


def build_stress_assignment(
    frame: pd.DataFrame,
) -> tuple[pd.Series, tuple[str, ...]]:
    unseen_cities = select_unseen_cities(frame)
    unseen_mask = frame["city_id"].astype(str).isin(unseen_cities)
    development = frame.loc[~unseen_mask]
    development_assignment = deterministic_stratified_assignment(
        development,
        ratios=STRESS_DEVELOPMENT_RATIOS,
        scope="unseen-city-development",
    )
    assignment = pd.Series("unseen_city_test", index=frame.index, dtype="string")
    assignment.loc[development.index] = development_assignment
    if frame.loc[assignment.eq("train"), "city_id"].isin(unseen_cities).any():
        raise ModelingError("held-out cities leaked into stress training data")
    if not frame.loc[assignment.eq("unseen_city_test"), "city_id"].isin(unseen_cities).all():
        raise ModelingError("stress test contains a partially held-out city")
    return assignment, unseen_cities


def build_preprocessor() -> ColumnTransformer:
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    numeric = Pipeline(
        steps=[("scale", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype="float64")
    predicted_values = np.asarray(predicted, dtype="float64")
    error = predicted_values - actual_values
    absolute_error = np.abs(error)
    squared_error = error**2
    denominator = float(np.sum((actual_values - actual_values.mean()) ** 2))
    r2 = None
    if len(actual_values) >= 2 and denominator > 0:
        r2 = 1.0 - float(np.sum(squared_error)) / denominator
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "r2": r2,
        "median_absolute_error": float(np.median(absolute_error)),
    }


def _city_median_predictions(
    train: pd.DataFrame, scored: pd.DataFrame, global_median: float
) -> np.ndarray:
    city_medians = train.groupby("city_id")[TARGET_COLUMN].median()
    return (
        scored["city_id"].map(city_medians).fillna(global_median).to_numpy(dtype="float64")
    )


def evaluate_assignment(
    frame: pd.DataFrame,
    assignment: pd.Series,
    *,
    evaluation: str,
    test_split_name: str,
) -> EvaluationArtifacts:
    train = frame.loc[assignment.eq("train")]
    validation = frame.loc[assignment.eq("validation")]
    test = frame.loc[assignment.eq(test_split_name)]
    if min(len(train), len(validation), len(test)) < 1:
        raise ModelingError(
            f"{evaluation} has an empty split: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

    preprocessor = build_preprocessor()
    train_matrix = preprocessor.fit_transform(train.loc[:, FEATURE_COLUMNS])
    validation_matrix = preprocessor.transform(validation.loc[:, FEATURE_COLUMNS])
    output_feature_count = int(train_matrix.shape[1])

    train_target = train[TARGET_COLUMN].to_numpy(dtype="float64")
    validation_target = validation[TARGET_COLUMN].to_numpy(dtype="float64")
    test_target = test[TARGET_COLUMN].to_numpy(dtype="float64")
    global_median = float(np.median(train_target))

    ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-5)
    ridge.fit(train_matrix, train_target)

    xgboost = XGBRegressor(**XGBOOST_PARAMETERS)
    xgboost.fit(
        train_matrix,
        train_target,
        eval_set=[(validation_matrix, validation_target)],
        verbose=False,
    )
    best_iteration = int(xgboost.best_iteration)
    # The frozen/held-out matrix is materialized only after all fitting and
    # validation-based stopping decisions are complete.
    test_matrix = preprocessor.transform(test.loc[:, FEATURE_COLUMNS])

    split_frames = {
        "validation": (validation, validation_matrix, validation_target),
        test_split_name: (test, test_matrix, test_target),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_name, (split_frame, matrix, actual) in split_frames.items():
        predictions: dict[str, np.ndarray] = {
            "global_training_median": np.full(len(split_frame), global_median),
            "city_training_median": _city_median_predictions(
                train, split_frame, global_median
            ),
            "ridge": ridge.predict(matrix).astype("float64"),
            "xgboost": xgboost.predict(matrix).astype("float64"),
        }
        for model_name, predicted in predictions.items():
            metric_rows.append(
                {
                    "evaluation": evaluation,
                    "split": split_name,
                    "model": model_name,
                    "sample_count": len(split_frame),
                    "city_count": int(split_frame["city_id"].nunique()),
                    **_regression_metrics(actual, predicted),
                    "best_iteration": best_iteration if model_name == "xgboost" else None,
                }
            )
        private = pd.DataFrame(
            {
                IDENTIFIER_COLUMN: split_frame[IDENTIFIER_COLUMN].to_numpy(),
                "evaluation": evaluation,
                "split": split_name,
                "actual_asking_total_price_10k_cny": actual,
                **{
                    f"prediction_{model_name}": values
                    for model_name, values in predictions.items()
                },
            }
        )
        private["error_xgboost"] = (
            private["prediction_xgboost"]
            - private["actual_asking_total_price_10k_cny"]
        )
        private["absolute_error_xgboost"] = private["error_xgboost"].abs()
        prediction_frames.append(private)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return EvaluationArtifacts(
        evaluation=evaluation,
        test_split_name=test_split_name,
        metrics=metrics,
        predictions=predictions,
        train_city_counts=train.groupby("city_id").size(),
        global_training_median=global_median,
        preprocessor_output_feature_count=output_feature_count,
        xgboost_best_iteration=best_iteration,
    )


def _city_sample_size_tier(training_count: int) -> str:
    if training_count == 0:
        return "unseen_in_training"
    if training_count < 1_000:
        return "lt_1000_train_rows"
    if training_count < 1_500:
        return "1000_to_lt_1500_train_rows"
    return "1500_plus_train_rows"


def build_error_slices(
    frame: pd.DataFrame,
    artifacts: EvaluationArtifacts,
    *,
    min_slice_size: int,
) -> pd.DataFrame:
    test_predictions = artifacts.predictions.loc[
        artifacts.predictions["split"].eq(artifacts.test_split_name)
    ].copy()
    source = frame.set_index(IDENTIFIER_COLUMN)
    joined = test_predictions.join(
        source.loc[:, list(SLICE_COLUMNS)], on=IDENTIFIER_COLUMN, validate="one_to_one"
    )
    joined["city_sample_size_tier"] = joined["city_id"].map(
        lambda city: _city_sample_size_tier(
            int(artifacts.train_city_counts.get(city, 0))
        )
    )

    slice_dimensions = (
        "asking_total_price_bucket",
        "area_bucket",
        "rooms",
        "city_sample_size_tier",
        "city_id",
    )
    rows: list[dict[str, Any]] = []
    for dimension in slice_dimensions:
        for value, group in joined.groupby(dimension, dropna=False, sort=False):
            if len(group) < min_slice_size:
                continue
            actual = group["actual_asking_total_price_10k_cny"].to_numpy(dtype="float64")
            predicted = group["prediction_xgboost"].to_numpy(dtype="float64")
            metrics = _regression_metrics(actual, predicted)
            error = predicted - actual
            median_absolute_percentage_error_pct = float(
                np.median(np.abs(error) / actual) * 100.0
            )
            rows.append(
                {
                    "evaluation": artifacts.evaluation,
                    "split": artifacts.test_split_name,
                    "model": "xgboost",
                    "slice_dimension": dimension,
                    "slice_value": str(value),
                    "sample_count": len(group),
                    **metrics,
                    "median_absolute_percentage_error_pct": median_absolute_percentage_error_pct,
                    "p90_absolute_error": float(
                        np.quantile(np.abs(error), 0.90, method="nearest")
                    ),
                    "mean_error": float(error.mean()),
                    "median_error": float(np.median(error)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "evaluation",
                "split",
                "model",
                "slice_dimension",
                "slice_value",
                "sample_count",
                "mae",
                "rmse",
                "r2",
                "median_absolute_error",
                "median_absolute_percentage_error_pct",
                "p90_absolute_error",
                "mean_error",
                "median_error",
            ]
        )
    return result.sort_values(
        ["evaluation", "slice_dimension", "slice_value"], kind="stable"
    ).reset_index(drop=True)


def _split_count_dict(assignment: pd.Series) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in assignment.value_counts().sort_index().items()
    }


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _model_metric(
    metrics: pd.DataFrame, *, evaluation: str, split: str, model: str
) -> pd.Series:
    selected = metrics.loc[
        metrics["evaluation"].eq(evaluation)
        & metrics["split"].eq(split)
        & metrics["model"].eq(model)
    ]
    if len(selected) != 1:
        raise ModelingError(
            f"expected one metric row for {evaluation}/{split}/{model}; found {len(selected)}"
        )
    return selected.iloc[0]


def _worst_slice(
    slices: pd.DataFrame,
    *,
    evaluation: str,
    dimension: str,
) -> pd.Series | None:
    selected = slices.loc[
        slices["evaluation"].eq(evaluation)
        & slices["slice_dimension"].eq(dimension)
    ]
    if selected.empty:
        return None
    return selected.sort_values(["mae", "sample_count"], ascending=[False, False]).iloc[0]


def _format_metric(value: Any, digits: int) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_findings(
    metrics: pd.DataFrame,
    slices: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    main_test = {
        model: _model_metric(
            metrics,
            evaluation="random_split",
            split="frozen_test",
            model=model,
        )
        for model in (
            "global_training_median",
            "city_training_median",
            "ridge",
            "xgboost",
        )
    }
    stress_xgb = _model_metric(
        metrics,
        evaluation="unseen_city_stress",
        split="unseen_city_test",
        model="xgboost",
    )
    worst_price = _worst_slice(
        slices, evaluation="random_split", dimension="asking_total_price_bucket"
    )
    worst_area = _worst_slice(
        slices, evaluation="random_split", dimension="area_bucket"
    )
    worst_room = _worst_slice(
        slices, evaluation="random_split", dimension="rooms"
    )
    worst_city = _worst_slice(
        slices, evaluation="random_split", dimension="city_id"
    )

    main_counts = metadata["main_split_counts"]
    stress_counts = metadata["stress_split_counts"]
    xgb = main_test["xgboost"]
    city_baseline = main_test["city_training_median"]
    baseline_difference = float(city_baseline["mae"]) - float(xgb["mae"])
    baseline_direction = "lower" if baseline_difference >= 0 else "higher"
    baseline_change = abs(baseline_difference) / float(city_baseline["mae"])
    if int(xgb["best_iteration"]) == int(XGBOOST_PARAMETERS["n_estimators"]) - 1:
        stopping_evidence = (
            f"XGBoost used all {XGBOOST_PARAMETERS['n_estimators']} configured rounds; "
            "validation did not trigger early stopping."
        )
    else:
        stopping_evidence = (
            f"Validation early stopping selected iteration {int(xgb['best_iteration'])}."
        )
    lines = [
        "# Modeling Slice 3 Findings",
        "",
        (
            "Scope: historical second-hand asking-price regression on the canonical "
            f"{metadata['row_count']:,}-row analytical snapshot. The frozen test contains "
            f"{main_counts['frozen_test']:,} rows; all reported error slices require "
            f"n ≥ {metadata['minimum_slice_size']}."
        ),
        "",
        "## Evaluation contract",
        "",
        (
            f"- Main split: {main_counts['train']:,} train / "
            f"{main_counts['validation']:,} validation / "
            f"{main_counts['frozen_test']:,} frozen test."
        ),
        (
            f"- Unseen-city diagnostic: {stress_counts['train']:,} development-train / "
            f"{stress_counts['validation']:,} development-validation / "
            f"{stress_counts['unseen_city_test']:,} rows from "
            f"{metadata['unseen_city_count']} completely held-out cities."
        ),
        (
            "- Allowed features: `city_id`, `rooms`, `halls`, `area_sqm`, "
            "`orientation_primary`, `furnishing`, `floor_level`, `total_floors`, "
            "`building_type`."
        ),
        (
            "- Baselines: global training-target median; per-city training-target median "
            "with global fallback for unseen cities; Ridge with fixed alpha 10; one fixed "
            "XGBoost configuration."
        ),
        (
            "- Train-only preprocessing: numeric standardization plus one-hot encoding "
            "with unknown categories ignored. Validation is used only for XGBoost early "
            "stopping; frozen test data is transformed and scored only after fitting."
        ),
        (
            "- Target quantile bins are used only to assign the main split and are then "
            "discarded. Price and area buckets are used only for post-prediction error slices."
        ),
        (
            "- Unit-price fields, target buckets, record identifiers, platform signals, "
            "macro fields, location-like text, and title flags are excluded from all model features."
        ),
        (
            "- No target trimming or post-split outlier removal is applied; large-area, "
            "high-price, and high-room-count observations remain in their assigned splits."
        ),
        "",
        "## 1. Frozen-test performance relative to the two median baselines",
        "",
        (
            f"**Finding.** XGBoost reaches MAE {xgb['mae']:.2f}, RMSE {xgb['rmse']:.2f}, "
            f"R² {_format_metric(xgb['r2'], 3)}, and median absolute error "
            f"{xgb['median_absolute_error']:.2f} in CNY 10,000 units."
        ),
        "",
        (
            f"**Evidence.** The city training-median baseline has MAE "
            f"{city_baseline['mae']:.2f} and RMSE {city_baseline['rmse']:.2f}; "
            f"XGBoost MAE is {baseline_change:.1%} {baseline_direction}. "
            f"{stopping_evidence} See `figures/model_comparison.png`."
        ),
        "",
        (
            "**Limitation.** This is one deterministic split and one fixed boosting "
            "configuration; it is not a multi-model benchmark or evidence about sale prices."
        ),
        "",
    ]

    if worst_price is not None and worst_area is not None:
        lines.extend(
            [
                "## 2. Absolute errors vary across asking-price and area segments",
                "",
                (
                    f"**Finding.** The highest-MAE asking-price bucket is "
                    f"`{worst_price['slice_value']}` with MAE {worst_price['mae']:.2f} "
                    f"and RMSE {worst_price['rmse']:.2f} "
                    f"(median absolute percentage error "
                    f"{worst_price['median_absolute_percentage_error_pct']:.1f}%; "
                    f"n={int(worst_price['sample_count']):,})."
                ),
                "",
                (
                    f"**Evidence.** The highest-MAE area bucket is "
                    f"`{worst_area['slice_value']}` with MAE {worst_area['mae']:.2f} "
                    f"(n={int(worst_area['sample_count']):,}). See "
                    "`figures/model_error_slices.png`."
                ),
                "",
                (
                    "**Limitation.** These are absolute errors on a strongly right-skewed "
                    "target, so higher-priced groups mechanically permit larger currency errors."
                ),
                "",
            ]
        )

    if worst_room is not None and worst_city is not None:
        lines.extend(
            [
                "## 3. Error remains heterogeneous across property and city slices",
                "",
                (
                    f"**Finding.** Among published room-count slices, `{worst_room['slice_value']}` "
                    f"rooms has the largest MAE at {worst_room['mae']:.2f} "
                    f"(n={int(worst_room['sample_count']):,})."
                ),
                "",
                (
                    f"**Evidence.** The highest-MAE published city slice is "
                    f"{worst_city['slice_value']} at {worst_city['mae']:.2f} "
                    f"(n={int(worst_city['sample_count']):,})."
                ),
                "",
                (
                    "**Limitation.** Slice rankings are descriptive and can combine target "
                    "scale, sample composition, and sparse support; they do not identify causes."
                ),
                "",
            ]
        )

    stress_ratio = float(stress_xgb["mae"]) / float(xgb["mae"])
    lines.extend(
        [
            "## 4. Complete-city holdout robustness diagnostic",
            "",
            (
                f"**Finding.** XGBoost unseen-city MAE is {stress_xgb['mae']:.2f}, "
                f"{stress_ratio:.2f}× the random frozen-test MAE; unseen-city RMSE is "
                f"{stress_xgb['rmse']:.2f} and R² is "
                f"{_format_metric(stress_xgb['r2'], 3)}."
            ),
            "",
            (
                f"**Evidence.** All {metadata['unseen_city_count']} stress-test cities are "
                "absent from preprocessing fit, model fit, and early stopping. See "
                "`figures/unseen_city_stress.png`."
            ),
            "",
            (
                "**Limitation.** This is a hash-selected city holdout robustness diagnostic. "
                "It does not prove performance for arbitrary future cities or time periods."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _configure_matplotlib() -> Any:
    cache = Path(tempfile.gettempdir()) / "housing-analytics-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    for candidate in (
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti TC",
        "Noto Sans CJK SC",
    ):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            plt.rcParams["font.family"] = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _save_figure(figure: Any, path: Path, footer: str) -> None:
    figure.text(0.01, 0.012, footer, ha="left", va="bottom", fontsize=8, color="#555")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, bbox_inches="tight", facecolor="white")
    temporary.replace(path)


def _ordered_slice(
    slices: pd.DataFrame,
    *,
    evaluation: str,
    dimension: str,
    order: Sequence[str],
) -> pd.DataFrame:
    selected = slices.loc[
        slices["evaluation"].eq(evaluation)
        & slices["slice_dimension"].eq(dimension)
    ].copy()
    positions = {value: index for index, value in enumerate(order)}
    selected["_order"] = selected["slice_value"].map(positions).fillna(len(order))
    return selected.sort_values(["_order", "slice_value"]).drop(columns="_order")


def render_figures(
    metrics: pd.DataFrame,
    slices: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_directory: Path,
) -> None:
    plt = _configure_matplotlib()
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    footer = (
        f"Canonical asking-listing snapshot; seed={SPLIT_SEED}; error slices "
        f"n≥{metadata['minimum_slice_size']}. Unit: CNY 10,000 unless noted."
    )

    main = metrics.loc[
        metrics["evaluation"].eq("random_split")
        & metrics["split"].eq("frozen_test")
    ].copy()
    model_order = (
        "global_training_median",
        "city_training_median",
        "ridge",
        "xgboost",
    )
    labels = {
        "global_training_median": "Global median",
        "city_training_median": "City median",
        "ridge": "Ridge",
        "xgboost": "XGBoost",
    }
    main["_order"] = main["model"].map({name: i for i, name in enumerate(model_order)})
    main = main.sort_values("_order")
    xlabels = [labels[name] for name in main["model"]]
    colors = ["#8C8C8C", "#6E8B9E", "#4F9D8C", "#C76D3A"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for axis, metric, title in (
        (axes[0, 0], "mae", "MAE"),
        (axes[0, 1], "rmse", "RMSE"),
        (axes[1, 0], "r2", "R²"),
        (axes[1, 1], "median_absolute_error", "Median absolute error"),
    ):
        values = pd.to_numeric(main[metric], errors="coerce").astype("float64")
        axis.bar(xlabels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        if metric != "r2":
            axis.set_ylabel("CNY 10,000")
    fig.suptitle("Frozen-test model comparison", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "model_comparison.png", footer)
    plt.close(fig)

    price = _ordered_slice(
        slices,
        evaluation="random_split",
        dimension="asking_total_price_bucket",
        order=PRICE_BUCKET_ORDER,
    )
    area = _ordered_slice(
        slices,
        evaluation="random_split",
        dimension="area_bucket",
        order=AREA_BUCKET_ORDER,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for axis, selected, title in (
        (axes[0], price, "Error by asking-price bucket"),
        (axes[1], area, "Error by area bucket"),
    ):
        if selected.empty:
            axis.text(0.5, 0.5, "No slices meet minimum n", ha="center", va="center")
            continue
        labels_for_plot = [
            BUCKET_LABELS.get(str(value), str(value)) for value in selected["slice_value"]
        ]
        positions = np.arange(len(selected))
        width = 0.36
        axis.bar(positions - width / 2, selected["mae"], width, label="MAE", color="#356A8A")
        axis.bar(positions + width / 2, selected["rmse"], width, label="RMSE", color="#C76D3A")
        axis.set_xticks(positions, labels_for_plot, rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylabel("CNY 10,000")
        axis.legend(frameon=False)
    fig.suptitle("XGBoost frozen-test error slices", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "model_error_slices.png", footer)
    plt.close(fig)

    comparison = metrics.loc[
        metrics["model"].eq("xgboost")
        & (
            (
                metrics["evaluation"].eq("random_split")
                & metrics["split"].eq("frozen_test")
            )
            | (
                metrics["evaluation"].eq("unseen_city_stress")
                & metrics["split"].eq("unseen_city_test")
            )
        )
    ].copy()
    comparison["label"] = comparison["evaluation"].map(
        {
            "random_split": "Random frozen test",
            "unseen_city_stress": "Complete-city holdout",
        }
    )
    stress_cities = slices.loc[
        slices["evaluation"].eq("unseen_city_stress")
        & slices["slice_dimension"].eq("city_id")
    ].nlargest(10, "mae")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    positions = np.arange(len(comparison))
    width = 0.36
    axes[0].bar(positions - width / 2, comparison["mae"], width, label="MAE", color="#356A8A")
    axes[0].bar(positions + width / 2, comparison["rmse"], width, label="RMSE", color="#C76D3A")
    axes[0].set_xticks(positions, comparison["label"], rotation=15, ha="right")
    axes[0].set_ylabel("CNY 10,000")
    axes[0].set_title("Random split vs unseen-city diagnostic")
    axes[0].legend(frameon=False)
    if stress_cities.empty:
        axes[1].text(0.5, 0.5, "No city slices meet minimum n", ha="center", va="center")
    else:
        stress_cities = stress_cities.sort_values("mae")
        axes[1].barh(stress_cities["slice_value"], stress_cities["mae"], color="#7A6BA6")
        axes[1].set_xlabel("MAE (CNY 10,000)")
    axes[1].set_title("Largest unseen-city MAE slices")
    fig.suptitle("Unseen-city robustness stress test", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "unseen_city_stress.png", footer)
    plt.close(fig)


def validate_public_outputs(
    metrics: pd.DataFrame,
    slices: pd.DataFrame,
    *,
    minimum_slice_size: int,
) -> None:
    required_metric_columns = {
        "evaluation",
        "split",
        "model",
        "sample_count",
        "city_count",
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "best_iteration",
    }
    if set(metrics.columns) != required_metric_columns:
        raise ModelingError(f"unexpected metrics schema: {list(metrics.columns)}")
    if not slices.empty and slices["sample_count"].lt(minimum_slice_size).any():
        raise ModelingError("a public error slice is below the minimum sample size")
    serialized = (metrics.to_csv(index=False) + slices.to_csv(index=False)).lower()
    forbidden_public_tokens = (
        "record_id",
        "actual_asking_total_price",
        "prediction_",
        "platform_unit_price",
        "calculated_unit_price",
        "title_mentions_",
        "city_gdp",
    )
    found = [token for token in forbidden_public_tokens if token in serialized]
    if found:
        raise ModelingError(f"public aggregate outputs contain private/leakage fields: {found}")


def _validate_report_manifest(output_directory: Path, *, repository_reports: bool) -> None:
    actual = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file()
    }
    allowed = ALLOWED_REPOSITORY_REPORT_OUTPUTS if repository_reports else MODELING_PUBLIC_OUTPUTS
    missing = sorted(MODELING_PUBLIC_OUTPUTS - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        raise ModelingError(
            f"model report manifest mismatch; missing={missing}, unexpected={unexpected}"
        )


def _load_source(
    *, source_kind: str, input_path: Path | None
) -> tuple[pd.DataFrame, str, str]:
    if source_kind == "real":
        source = (input_path or DEFAULT_REAL_INPUT).resolve()
        private_root = (REPOSITORY_ROOT / "private_data/derived").resolve()
        try:
            source.relative_to(private_root)
        except ValueError as exc:
            raise ModelingError(f"real row-level input must remain under {private_root}") from exc
        if not source.is_file():
            raise ModelingError(f"real analytical input does not exist: {source}")
        source_hash = sha256_file(source)
        frame = pd.read_csv(source, low_memory=False)
        if len(frame) != EXPECTED_REAL_ROW_COUNT:
            raise ModelingError(
                f"expected {EXPECTED_REAL_ROW_COUNT} rows; found {len(frame)}"
            )
        if source_hash != EXPECTED_REAL_SHA256:
            raise ModelingError(f"real snapshot SHA-256 mismatch: {source_hash}")
        return frame, source.name, source_hash

    if source_kind == "synthetic":
        if input_path is not None:
            source = input_path.resolve()
            frame = pd.read_csv(source, low_memory=False)
            return frame, source.name, sha256_file(source)
        normalized = load_synthetic_fixture(TRACKED_SYNTHETIC_FIXTURE)
        frame, _ = prepare_analytics_v1(normalized)
        return (
            frame,
            TRACKED_SYNTHETIC_FIXTURE.name,
            sha256_file(TRACKED_SYNTHETIC_FIXTURE),
        )
    raise ModelingError(f"unsupported source kind: {source_kind}")


def rebuild_reports_from_private_predictions(
    *,
    input_path: Path | None = None,
    public_output_directory: Path | None = None,
    private_output_directory: Path | None = None,
    min_slice_size: int = DEFAULT_MIN_SLICE_SIZE,
) -> dict[str, Any]:
    """Rebuild aggregate slices/reports without fitting or predicting.

    This path exists for post-prediction reporting changes. It treats the stored
    private split assignments and predictions as immutable inputs and preserves
    the already-published global model metrics and absolute slice metrics.
    """

    frame, source_name, source_hash = _load_source(
        source_kind="real", input_path=input_path
    )
    validate_input_frame(frame)
    public_output = (
        public_output_directory.resolve()
        if public_output_directory is not None
        else DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve()
    )
    if public_output != DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve():
        raise ModelingError("real aggregate reports must stay in the repository reports directory")
    private_output = (
        private_output_directory.resolve()
        if private_output_directory is not None
        else (DEFAULT_PRIVATE_OUTPUT_DIRECTORY / "real").resolve()
    )
    private_root = (REPOSITORY_ROOT / "private_data").resolve()
    try:
        private_output.relative_to(private_root)
    except ValueError as exc:
        raise ModelingError(f"private predictions must remain under {private_root}") from exc

    assignment_path = private_output / "split_assignments.csv"
    prediction_path = private_output / "predictions.csv"
    metadata_path = private_output / "run_metadata.json"
    metric_path = public_output / "model_metrics.csv"
    slice_path = public_output / "model_error_slices.csv"
    for path in (assignment_path, prediction_path, metadata_path, metric_path, slice_path):
        if not path.is_file():
            raise ModelingError(f"reports-only input is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != source_hash:
        raise ModelingError("private prediction metadata does not match the canonical source")
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ModelingError("private prediction metadata has the wrong row count")

    assignments = pd.read_csv(assignment_path, dtype={IDENTIFIER_COLUMN: "string"})
    predictions = pd.read_csv(prediction_path, dtype={IDENTIFIER_COLUMN: "string"})
    if assignments[IDENTIFIER_COLUMN].duplicated().any():
        raise ModelingError("private split assignments contain duplicate record IDs")
    assignment_by_id = assignments.set_index(IDENTIFIER_COLUMN)
    record_ids = frame[IDENTIFIER_COLUMN].astype("string")
    main_assignment = record_ids.map(assignment_by_id["random_split"])
    stress_assignment = record_ids.map(assignment_by_id["unseen_city_stress_split"])
    main_assignment.index = frame.index
    stress_assignment.index = frame.index
    if main_assignment.isna().any() or stress_assignment.isna().any():
        raise ModelingError("private split assignments do not cover the canonical frame")

    def stored_artifacts(
        *,
        evaluation: str,
        test_split_name: str,
        assignment: pd.Series,
        feature_count_key: str,
        iteration_key: str,
    ) -> EvaluationArtifacts:
        selected_predictions = predictions.loc[
            predictions["evaluation"].eq(evaluation)
        ].copy()
        if selected_predictions.empty:
            raise ModelingError(f"private predictions are missing {evaluation}")
        train = frame.loc[assignment.eq("train")]
        return EvaluationArtifacts(
            evaluation=evaluation,
            test_split_name=test_split_name,
            metrics=pd.DataFrame(),
            predictions=selected_predictions,
            train_city_counts=train.groupby("city_id").size(),
            global_training_median=float(train[TARGET_COLUMN].median()),
            preprocessor_output_feature_count=int(metadata[feature_count_key]),
            xgboost_best_iteration=int(metadata[iteration_key]),
        )

    rebuilt_slices = pd.concat(
        [
            build_error_slices(
                frame,
                stored_artifacts(
                    evaluation="random_split",
                    test_split_name="frozen_test",
                    assignment=main_assignment,
                    feature_count_key="main_preprocessor_output_feature_count",
                    iteration_key="main_xgboost_best_iteration",
                ),
                min_slice_size=min_slice_size,
            ),
            build_error_slices(
                frame,
                stored_artifacts(
                    evaluation="unseen_city_stress",
                    test_split_name="unseen_city_test",
                    assignment=stress_assignment,
                    feature_count_key="stress_preprocessor_output_feature_count",
                    iteration_key="stress_xgboost_best_iteration",
                ),
                min_slice_size=min_slice_size,
            ),
        ],
        ignore_index=True,
    )
    existing_slices = pd.read_csv(slice_path)
    relative_column = "median_absolute_percentage_error_pct"
    keys = [
        "evaluation",
        "split",
        "model",
        "slice_dimension",
        "slice_value",
        "sample_count",
    ]
    base_slices = existing_slices.drop(columns=[relative_column], errors="ignore")
    relative_values = rebuilt_slices.loc[:, [*keys, relative_column]]
    augmented_slices = base_slices.merge(
        relative_values,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(augmented_slices) != len(base_slices) or len(base_slices) != len(rebuilt_slices):
        raise ModelingError("stored predictions do not reproduce the existing slice keys/counts")

    metrics = pd.read_csv(metric_path)
    validate_public_outputs(
        metrics, augmented_slices, minimum_slice_size=min_slice_size
    )
    metadata["source_file"] = source_name
    metadata["minimum_slice_size"] = min_slice_size
    _write_csv_atomic(augmented_slices, slice_path)
    _write_text_atomic(
        public_output / "modeling_findings.md",
        render_findings(metrics, augmented_slices, metadata),
    )
    render_figures(metrics, augmented_slices, metadata, public_output)
    _validate_report_manifest(public_output, repository_reports=True)
    return metadata


def run_modeling(
    *,
    source_kind: str,
    input_path: Path | None = None,
    public_output_directory: Path | None = None,
    private_output_directory: Path | None = None,
    min_slice_size: int = DEFAULT_MIN_SLICE_SIZE,
) -> dict[str, Any]:
    if min_slice_size < 1:
        raise ModelingError("minimum slice size must be positive")
    frame, source_name, source_hash = _load_source(
        source_kind=source_kind, input_path=input_path
    )
    validate_input_frame(frame)

    if public_output_directory is None:
        public_output = (
            DEFAULT_PUBLIC_OUTPUT_DIRECTORY
            if source_kind == "real"
            else DEFAULT_PRIVATE_OUTPUT_DIRECTORY / "synthetic_reports"
        )
    else:
        public_output = public_output_directory.resolve()
    if source_kind == "synthetic" and public_output == DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve():
        raise ModelingError("synthetic runs cannot overwrite real public reports")
    if source_kind == "real" and public_output != DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve():
        raise ModelingError(
            "canonical real aggregate reports must be written to the repository reports directory"
        )

    private_output = (
        private_output_directory.resolve()
        if private_output_directory is not None
        else DEFAULT_PRIVATE_OUTPUT_DIRECTORY / source_kind
    )
    if source_kind == "real":
        private_root = (REPOSITORY_ROOT / "private_data").resolve()
        try:
            private_output.relative_to(private_root)
        except ValueError as exc:
            raise ModelingError(f"real row-level outputs must remain under {private_root}") from exc

    main_assignment = build_main_assignment(frame)
    stress_assignment, unseen_cities = build_stress_assignment(frame)
    main_artifacts = evaluate_assignment(
        frame,
        main_assignment,
        evaluation="random_split",
        test_split_name="frozen_test",
    )
    stress_artifacts = evaluate_assignment(
        frame,
        stress_assignment,
        evaluation="unseen_city_stress",
        test_split_name="unseen_city_test",
    )

    metrics = pd.concat(
        [main_artifacts.metrics, stress_artifacts.metrics], ignore_index=True
    )
    metrics = metrics.loc[
        :,
        [
            "evaluation",
            "split",
            "model",
            "sample_count",
            "city_count",
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "best_iteration",
        ],
    ].sort_values(["evaluation", "split", "model"], kind="stable").reset_index(drop=True)
    slices = pd.concat(
        [
            build_error_slices(
                frame, main_artifacts, min_slice_size=min_slice_size
            ),
            build_error_slices(
                frame, stress_artifacts, min_slice_size=min_slice_size
            ),
        ],
        ignore_index=True,
    )
    validate_public_outputs(metrics, slices, minimum_slice_size=min_slice_size)

    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "source_file": source_name,
        "source_sha256": source_hash,
        "row_count": len(frame),
        "split_seed": SPLIT_SEED,
        "target_quantile_bin_count_maximum": TARGET_QUANTILE_BIN_COUNT,
        "main_split_counts": _split_count_dict(main_assignment),
        "stress_split_counts": _split_count_dict(stress_assignment),
        "unseen_city_count": len(unseen_cities),
        "minimum_slice_size": min_slice_size,
        "feature_columns": list(FEATURE_COLUMNS),
        "main_preprocessor_output_feature_count": main_artifacts.preprocessor_output_feature_count,
        "stress_preprocessor_output_feature_count": stress_artifacts.preprocessor_output_feature_count,
        "main_xgboost_best_iteration": main_artifacts.xgboost_best_iteration,
        "stress_xgboost_best_iteration": stress_artifacts.xgboost_best_iteration,
        "ridge_alpha": RIDGE_ALPHA,
        "xgboost_parameters": dict(XGBOOST_PARAMETERS),
    }

    private_output.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(
        {
            IDENTIFIER_COLUMN: frame[IDENTIFIER_COLUMN],
            "random_split": main_assignment,
            "unseen_city_stress_split": stress_assignment,
        }
    )
    _write_csv_atomic(assignments, private_output / "split_assignments.csv")
    private_predictions = pd.concat(
        [main_artifacts.predictions, stress_artifacts.predictions], ignore_index=True
    )
    _write_csv_atomic(private_predictions, private_output / "predictions.csv")
    _write_text_atomic(
        private_output / "run_metadata.json",
        json.dumps(
            {**metadata, "unseen_cities": list(unseen_cities)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    public_output.mkdir(parents=True, exist_ok=True)
    (public_output / "figures").mkdir(parents=True, exist_ok=True)
    existing = {
        path.relative_to(public_output).as_posix()
        for path in public_output.rglob("*")
        if path.is_file()
    }
    allowed = (
        ALLOWED_REPOSITORY_REPORT_OUTPUTS
        if public_output == DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve()
        else MODELING_PUBLIC_OUTPUTS
    )
    unexpected = sorted(existing - allowed)
    if unexpected:
        raise ModelingError(f"refusing to mix unexpected public files: {unexpected}")

    _write_csv_atomic(metrics, public_output / "model_metrics.csv")
    _write_csv_atomic(slices, public_output / "model_error_slices.csv")
    _write_text_atomic(
        public_output / "modeling_findings.md",
        render_findings(metrics, slices, metadata),
    )
    render_figures(metrics, slices, metadata, public_output)
    _validate_report_manifest(
        public_output,
        repository_reports=public_output == DEFAULT_PUBLIC_OUTPUT_DIRECTORY.resolve(),
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--private-output-dir", type=Path)
    parser.add_argument("--min-slice-size", type=int, default=DEFAULT_MIN_SLICE_SIZE)
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="rebuild aggregate reports from existing private predictions without model fitting",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.reports_only:
        if args.source != "real":
            raise ModelingError("--reports-only currently requires --source real")
        metadata = rebuild_reports_from_private_predictions(
            input_path=args.input,
            public_output_directory=args.output_dir,
            private_output_directory=args.private_output_dir,
            min_slice_size=args.min_slice_size,
        )
    else:
        metadata = run_modeling(
            source_kind=args.source,
            input_path=args.input,
            public_output_directory=args.output_dir,
            private_output_directory=args.private_output_dir,
            min_slice_size=args.min_slice_size,
        )
    print(
        json.dumps(
            {
                "row_count": metadata["row_count"],
                "main_split_counts": metadata["main_split_counts"],
                "stress_split_counts": metadata["stress_split_counts"],
                "unseen_city_count": metadata["unseen_city_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
