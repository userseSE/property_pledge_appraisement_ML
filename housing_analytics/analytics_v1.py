"""Deterministic preparation for Analytics Slice 1.

The real input and output stay under ``private_data/``. The same preparation
logic also accepts the tracked synthetic JSONL fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_INPUT = (
    REPOSITORY_ROOT
    / "private_data/canonical/secondhand_legacy_pre_encoding.csv"
)
DEFAULT_REAL_OUTPUT = REPOSITORY_ROOT / "private_data/derived/secondhand_analytics_v1.csv"
DEFAULT_SYNTHETIC_INPUT = (
    REPOSITORY_ROOT / "fixtures/secondhand_analytics.synthetic.jsonl"
)
DEFAULT_SYNTHETIC_SCHEMA = (
    REPOSITORY_ROOT / "fixtures/secondhand_analytics.schema.json"
)
DEFAULT_SYNTHETIC_OUTPUT = (
    REPOSITORY_ROOT / "private_data/derived/secondhand_analytics_v1.synthetic.csv"
)

CANONICAL_SHA256 = "9b28d50cb76770299dab0747c5ff048e3e3f6067a233e401c9b4e3d330376020"
EXPECTED_CANONICAL_LINEAGE = (300_397, 298_146, 267_305)

LEGACY_INPUT_COLUMNS = (
    "标题",
    "开发商",
    "室",
    "厅",
    "面积（平米）",
    "朝向",
    "装修",
    "所在高度",
    "总楼层高",
    "建筑结构",
    "是否别墅",
    "关注人数",
    "发布时长（天）",
    "售价/万",
    "单价",
    "城市",
    "城市gdp",
    "所在地区",
    "城市绿化率(%)",
)

ORIENTATION_MAP = {"东": "east", "南": "south", "西": "west", "北": "north"}
FURNISHING_MAP = {
    "其他": "other",
    "毛坯": "unfinished",
    "简装": "simple",
    "精装": "decorated",
}
FLOOR_LEVEL_MAP = {
    "地下室": "basement",
    "底": "bottom",
    "低楼": "low",
    "中楼": "middle",
    "高楼": "high",
    "顶": "top",
    "下叠": "stacked_lower",
    "上叠": "stacked_upper",
}
BUILDING_TYPE_MAP = {
    "平房": "bungalow",
    "板楼": "slab",
    "板塔结合": "slab_tower",
    "塔楼": "tower",
}
VALID_FLOOR_LEVELS = frozenset(
    {"basement", "bottom", "low", "middle", "high", "top"}
)

# Literal title indicators used by the historical analytical duplicate
# signature. They are deterministic string flags, not semantic labels.
TITLE_SIGNAL_PATTERNS: Mapping[str, str] = {
    "title_mentions_park": r"公园",
    "title_mentions_light": r"采光|阳光好",
    "title_mentions_parking": r"车位",
    "title_mentions_water_view": r"海景|河景|湖景",
    "title_mentions_business": r"商圈|商场|CBD|商务|商贸|商业",
    "title_mentions_transport": r"交通|地铁",
    "title_mentions_tax_tenure": r"满五|满二",
}

DEDUPLICATION_COLUMNS = (
    "city_id",
    "rooms",
    "halls",
    "area_sqm",
    "orientation_primary",
    "furnishing",
    "floor_level",
    "total_floors",
    "building_type",
    "follower_count",
    "listing_age_days",
    "asking_total_price_10k_cny",
    "platform_unit_price_cny_sqm",
    *TITLE_SIGNAL_PATTERNS.keys(),
)

OUTPUT_COLUMNS = (
    "schema_version",
    "source_kind",
    "record_id",
    "city_id",
    "rooms",
    "halls",
    "area_sqm",
    "area_bucket",
    "orientation_primary",
    "furnishing",
    "floor_level",
    "total_floors",
    "building_type",
    "follower_count",
    "listing_age_days",
    "asking_total_price_10k_cny",
    "asking_total_price_bucket",
    "platform_unit_price_cny_sqm",
    "calculated_unit_price_cny_sqm",
    *TITLE_SIGNAL_PATTERNS.keys(),
)


class SchemaError(ValueError):
    """Raised when an input does not satisfy the documented contract."""


class LineageMismatchError(RuntimeError):
    """Raised when the canonical snapshot does not reproduce audited counts."""


@dataclass(frozen=True)
class AnalyticsLineage:
    source_kind: str
    input_rows: int
    villa_rows: int
    invalid_floor_rows: int
    villa_or_invalid_floor_removed: int
    after_villa_or_invalid_floor: int
    total_floor_zero_removed: int
    after_quality_filters: int
    exact_duplicates_removed: int
    output_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _strict_map(
    series: pd.Series,
    mapping: Mapping[str, str],
    field_name: str,
    *,
    allow_null: bool = False,
) -> pd.Series:
    stripped = _strip_strings(series)
    mapped = stripped.map(mapping)
    unknown = stripped.notna() & mapped.isna()
    if unknown.any():
        values = sorted(str(value) for value in stripped[unknown].unique())
        raise SchemaError(f"{field_name} contains unsupported values: {values}")
    if not allow_null and mapped.isna().any():
        raise SchemaError(f"{field_name} contains null values")
    return mapped


def _parse_platform_unit_price(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )
    parsed = pd.to_numeric(extracted, errors="coerce")
    if parsed.isna().any():
        raise SchemaError(
            f"单价 contains {int(parsed.isna().sum())} values that cannot be parsed"
        )
    return parsed.astype("float64")


def validate_legacy_schema(frame: pd.DataFrame) -> None:
    actual = tuple(str(column) for column in frame.columns)
    if actual == LEGACY_INPUT_COLUMNS:
        return
    missing = [column for column in LEGACY_INPUT_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in LEGACY_INPUT_COLUMNS]
    raise SchemaError(
        "legacy canonical columns do not match the 19-column contract; "
        f"missing={missing}, extra={extra}, actual_order={list(actual)}"
    )


def load_legacy_canonical(path: Path | str = DEFAULT_LEGACY_INPUT) -> pd.DataFrame:
    input_path = Path(path)
    frame = pd.read_csv(input_path, encoding="gb18030", low_memory=False)
    validate_legacy_schema(frame)

    title = frame["标题"].astype("string").fillna("")
    orientation_token = _strip_strings(frame["朝向"]).str[0]
    villa_type = _strip_strings(frame["是否别墅"]).replace("", pd.NA)

    normalized = pd.DataFrame(
        {
            "schema_version": "1.0.0",
            "source_kind": "legacy_real",
            "record_id": [f"LEGACY-{index:06d}" for index in range(1, len(frame) + 1)],
            "listing_title": title,
            "location_label": _strip_strings(frame["开发商"]),
            "city_id": _strip_strings(frame["城市"]),
            "rooms": pd.to_numeric(frame["室"], errors="raise").astype("int64"),
            "halls": pd.to_numeric(frame["厅"], errors="raise").astype("int64"),
            "area_sqm": pd.to_numeric(frame["面积（平米）"], errors="raise").astype(
                "float64"
            ),
            "orientation_primary": _strict_map(
                orientation_token, ORIENTATION_MAP, "朝向"
            ),
            "furnishing": _strict_map(frame["装修"], FURNISHING_MAP, "装修"),
            "floor_level": _strict_map(
                frame["所在高度"], FLOOR_LEVEL_MAP, "所在高度", allow_null=True
            ),
            "total_floors": pd.to_numeric(
                frame["总楼层高"], errors="raise"
            ).astype("int64"),
            "building_type": _strict_map(
                frame["建筑结构"], BUILDING_TYPE_MAP, "建筑结构"
            ),
            "villa_type": villa_type,
            "follower_count": pd.to_numeric(
                frame["关注人数"], errors="raise"
            ).astype("int64"),
            "listing_age_days": pd.to_numeric(
                frame["发布时长（天）"], errors="raise"
            ).astype("int64"),
            "asking_total_price_10k_cny": pd.to_numeric(
                frame["售价/万"], errors="raise"
            ).astype("float64"),
            "platform_unit_price_cny_sqm": _parse_platform_unit_price(
                frame["单价"]
            ),
        }
    )

    for field, pattern in TITLE_SIGNAL_PATTERNS.items():
        normalized[field] = title.str.contains(
            pattern, case=False, regex=True, na=False
        ).astype("bool")

    return normalized


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    return False


def validate_synthetic_records(
    records: Iterable[Mapping[str, Any]], schema: Mapping[str, Any]
) -> None:
    properties = schema["properties"]
    required = set(schema.get("required", ()))
    allow_extra = bool(schema.get("additionalProperties", True))

    for index, record in enumerate(records, start=1):
        missing = sorted(required - set(record))
        extra = sorted(set(record) - set(properties))
        if missing or (extra and not allow_extra):
            raise SchemaError(
                f"synthetic record {index}: missing={missing}, extra={extra}"
            )

        for field, rule in properties.items():
            if field not in record:
                continue
            value = record[field]
            expected = rule.get("type")
            expected_types = expected if isinstance(expected, list) else [expected]
            expected_types = [item for item in expected_types if item is not None]
            if expected_types and not any(
                _json_type_matches(value, item) for item in expected_types
            ):
                raise SchemaError(
                    f"synthetic record {index}: {field} has an invalid type"
                )
            if "const" in rule and value != rule["const"]:
                raise SchemaError(
                    f"synthetic record {index}: {field} violates const"
                )
            if "enum" in rule and value not in rule["enum"]:
                raise SchemaError(
                    f"synthetic record {index}: {field} is outside its enum"
                )
            if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
                raise SchemaError(
                    f"synthetic record {index}: {field} violates its pattern"
                )
            if "minimum" in rule and value < rule["minimum"]:
                raise SchemaError(
                    f"synthetic record {index}: {field} is below minimum"
                )
            if "maximum" in rule and value > rule["maximum"]:
                raise SchemaError(
                    f"synthetic record {index}: {field} is above maximum"
                )
            if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
                raise SchemaError(
                    f"synthetic record {index}: {field} is not above exclusiveMinimum"
                )


def load_synthetic_fixture(
    path: Path | str = DEFAULT_SYNTHETIC_INPUT,
    schema_path: Path | str = DEFAULT_SYNTHETIC_SCHEMA,
) -> pd.DataFrame:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validate_synthetic_records(records, schema)
    frame = pd.DataFrame(records)

    normalized = pd.DataFrame(
        {
            "schema_version": frame["schema_version"].astype("string"),
            "source_kind": "synthetic",
            "record_id": frame["record_id"].astype("string"),
            "listing_title": "",
            "location_label": pd.NA,
            "city_id": frame["city_code"].astype("string"),
            "rooms": frame["rooms"].astype("int64"),
            "halls": frame["halls"].astype("int64"),
            "area_sqm": frame["area_sqm"].astype("float64"),
            "orientation_primary": frame["orientation_primary"].astype("string"),
            "furnishing": frame["furnishing"].astype("string"),
            "floor_level": frame["floor_level"].astype("string"),
            "total_floors": frame["total_floors"].astype("int64"),
            "building_type": frame["building_type"].astype("string"),
            "villa_type": frame["villa_type"].astype("string"),
            "follower_count": frame["follower_count"].astype("int64"),
            "listing_age_days": frame["listing_age_days"].astype("int64"),
            "asking_total_price_10k_cny": frame[
                "asking_total_price_10k_cny"
            ].astype("float64"),
            "platform_unit_price_cny_sqm": frame[
                "asking_unit_price_cny_sqm"
            ].astype("float64"),
        }
    )
    for field in TITLE_SIGNAL_PATTERNS:
        normalized[field] = False
    return normalized


def _validate_normalized_values(frame: pd.DataFrame) -> None:
    required = {
        "record_id",
        "city_id",
        "rooms",
        "halls",
        "area_sqm",
        "orientation_primary",
        "furnishing",
        "floor_level",
        "total_floors",
        "building_type",
        "villa_type",
        "follower_count",
        "listing_age_days",
        "asking_total_price_10k_cny",
        "platform_unit_price_cny_sqm",
        *TITLE_SIGNAL_PATTERNS.keys(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SchemaError(f"normalized input is missing columns: {missing}")

    invalid = {
        "rooms": frame["rooms"].lt(0),
        "halls": frame["halls"].lt(0),
        "area_sqm": frame["area_sqm"].le(0),
        "total_floors": frame["total_floors"].lt(0),
        "follower_count": frame["follower_count"].lt(0),
        "listing_age_days": frame["listing_age_days"].lt(0),
        "asking_total_price_10k_cny": frame["asking_total_price_10k_cny"].le(0),
        "platform_unit_price_cny_sqm": frame[
            "platform_unit_price_cny_sqm"
        ].le(0),
    }
    failures = {name: int(mask.sum()) for name, mask in invalid.items() if mask.any()}
    if failures:
        raise SchemaError(f"normalized input has invalid numeric values: {failures}")


def _bucket(
    values: pd.Series, bins: list[float], labels: list[str]
) -> pd.Series:
    bucketed = pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)
    if bucketed.isna().any():
        raise SchemaError(f"bucket assignment failed for {int(bucketed.isna().sum())} rows")
    return bucketed.astype("string")


def prepare_analytics_v1(
    normalized: pd.DataFrame,
    *,
    expected_lineage: tuple[int, int, int] | None = None,
) -> tuple[pd.DataFrame, AnalyticsLineage]:
    _validate_normalized_values(normalized)
    frame = normalized.copy()

    villa_mask = frame["villa_type"].notna()
    invalid_floor_mask = frame["floor_level"].isna() | ~frame["floor_level"].isin(
        VALID_FLOOR_LEVELS
    )
    initial_filter_mask = villa_mask | invalid_floor_mask
    after_initial = frame.loc[~initial_filter_mask].copy()

    zero_floor_mask = after_initial["total_floors"].eq(0)
    quality = after_initial.loc[~zero_floor_mask].copy()

    duplicate_mask = quality.duplicated(
        subset=list(DEDUPLICATION_COLUMNS), keep="first"
    )
    prepared = quality.loc[~duplicate_mask].copy()

    prepared["calculated_unit_price_cny_sqm"] = (
        prepared["asking_total_price_10k_cny"] * 10_000 / prepared["area_sqm"]
    ).round(2)
    prepared["area_bucket"] = _bucket(
        prepared["area_sqm"],
        [0, 60, 90, 120, 144, float("inf")],
        ["lt_60", "60_to_lt_90", "90_to_lt_120", "120_to_lt_144", "144_plus"],
    )
    prepared["asking_total_price_bucket"] = _bucket(
        prepared["asking_total_price_10k_cny"],
        [0, 100, 200, 300, 500, float("inf")],
        ["lt_100", "100_to_lt_200", "200_to_lt_300", "300_to_lt_500", "500_plus"],
    )

    result = prepared.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)
    output_duplicate_count = int(
        result.drop(columns=["record_id"]).duplicated(keep=False).sum()
    )
    if output_duplicate_count:
        raise LineageMismatchError(
            f"prepared output still has {output_duplicate_count} duplicate rows"
        )

    lineage = AnalyticsLineage(
        source_kind=str(frame["source_kind"].iloc[0]) if len(frame) else "unknown",
        input_rows=len(frame),
        villa_rows=int(villa_mask.sum()),
        invalid_floor_rows=int(invalid_floor_mask.sum()),
        villa_or_invalid_floor_removed=int(initial_filter_mask.sum()),
        after_villa_or_invalid_floor=len(after_initial),
        total_floor_zero_removed=int(zero_floor_mask.sum()),
        after_quality_filters=len(quality),
        exact_duplicates_removed=int(duplicate_mask.sum()),
        output_rows=len(result),
    )

    if expected_lineage is not None:
        actual = (
            lineage.input_rows,
            lineage.after_quality_filters,
            lineage.output_rows,
        )
        if actual != expected_lineage:
            raise LineageMismatchError(
                f"expected lineage {expected_lineage}, observed {actual}"
            )

    return result, lineage


def _require_private_real_output(path: Path) -> None:
    private_root = (REPOSITORY_ROOT / "private_data/derived").resolve()
    try:
        path.resolve().relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            f"real derived output must stay under {private_root}; got {path.resolve()}"
        ) from exc


def write_derived_csv(frame: pd.DataFrame, path: Path | str, *, real: bool) -> None:
    output_path = Path(path)
    if real:
        _require_private_real_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
    )
    temporary.replace(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("legacy", "synthetic"), default="legacy")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expect-canonical-lineage",
        action="store_true",
        help="verify the audited SHA-256 and 300397 -> 298146 -> 267305 lineage",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source == "legacy":
        input_path = args.input or DEFAULT_LEGACY_INPUT
        output_path = args.output or DEFAULT_REAL_OUTPUT
        if args.expect_canonical_lineage:
            observed_hash = sha256_file(input_path)
            if observed_hash != CANONICAL_SHA256:
                raise LineageMismatchError(
                    f"expected canonical SHA-256 {CANONICAL_SHA256}, observed {observed_hash}"
                )
        normalized = load_legacy_canonical(input_path)
        result, lineage = prepare_analytics_v1(
            normalized,
            expected_lineage=(
                EXPECTED_CANONICAL_LINEAGE
                if args.expect_canonical_lineage
                else None
            ),
        )
        write_derived_csv(result, output_path, real=True)
    else:
        input_path = args.input or DEFAULT_SYNTHETIC_INPUT
        output_path = args.output or DEFAULT_SYNTHETIC_OUTPUT
        normalized = load_synthetic_fixture(input_path)
        result, lineage = prepare_analytics_v1(normalized)
        write_derived_csv(result, output_path, real=False)

    print(
        json.dumps(
            {"output": str(output_path), "lineage": lineage.to_dict()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
