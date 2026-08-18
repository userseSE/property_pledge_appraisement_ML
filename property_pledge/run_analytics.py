"""Run the fixed, listing-only Analytics v1 questions through SQLite.

The input is a private row-level Analytics Slice 1 CSV. Public outputs contain
aggregates only. This module does not train models, use macro fields, or infer
transaction outcomes from platform snapshot fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SQL_DIRECTORY = REPOSITORY_ROOT / "analytics/sql"
DEFAULT_REAL_INPUT = (
    REPOSITORY_ROOT / "private_data/derived/secondhand_analytics_v1.csv"
)
DEFAULT_SYNTHETIC_INPUT = (
    REPOSITORY_ROOT / "private_data/derived/secondhand_analytics_v1.synthetic.csv"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "reports"
EXPECTED_REAL_ROW_COUNT = 267_305
EXPECTED_REAL_SHA256 = (
    "2d8ce4f90d7e63d0dfbcfb6054f87f72a8b3da34cdf0f806d37389f25142f64a"
)
DEFAULT_MIN_GROUP_SIZE = 100

SQL_FILES = (
    "01_market_coverage.sql",
    "02_price_distribution.sql",
    "03_property_segments.sql",
    "04_city_comparison.sql",
    "05_listing_engagement.sql",
)
PUBLIC_OUTPUTS = frozenset(
    {
        "summary.json",
        "findings.md",
        "figures/market_coverage.png",
        "figures/price_distribution.png",
        "figures/price_by_area_bucket.png",
        "figures/city_price_structure.png",
        "figures/listing_engagement.png",
    }
)
FORBIDDEN_SQL_TOKENS = (
    "city_gdp",
    "城市gdp",
    "所在地区",
    "城市绿化率",
    "title_mentions_",
)
ANALYTICS_COLUMNS = (
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
)
PROPERTY_SEGMENT_DIMENSIONS = (
    "area_bucket",
    "rooms",
    "halls",
    "orientation_primary",
    "furnishing",
    "floor_level",
    "total_floors",
    "building_type",
)
AREA_BUCKET_ORDER = (
    "lt_60",
    "60_to_lt_90",
    "90_to_lt_120",
    "120_to_lt_144",
    "144_plus",
)
PRICE_BUCKET_ORDER = (
    "lt_100",
    "100_to_lt_200",
    "200_to_lt_300",
    "300_to_lt_500",
    "500_plus",
)
BUCKET_LABELS = {
    "lt_60": "<60",
    "60_to_lt_90": "60–<90",
    "90_to_lt_120": "90–<120",
    "120_to_lt_144": "120–<144",
    "144_plus": "144+",
    "lt_100": "<100",
    "100_to_lt_200": "100–<200",
    "200_to_lt_300": "200–<300",
    "300_to_lt_500": "300–<500",
    "500_plus": "500+",
}
QUERY_HEADER = re.compile(r"^-- name: ([a-z0-9_]+)\s*$", re.MULTILINE)


class AnalyticsError(RuntimeError):
    """Raised when the analytical input or public-output contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_named_queries(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    matches = list(QUERY_HEADER.finditer(text))
    if not matches:
        raise AnalyticsError(f"no named SQL queries found in {path}")

    queries: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        query = text[start:end].strip().rstrip(";")
        name = match.group(1)
        if not query or name in queries:
            raise AnalyticsError(f"invalid or duplicate SQL query {name!r} in {path}")
        queries[name] = query
    return queries


def validate_sql_contract(sql_directory: Path = SQL_DIRECTORY) -> None:
    actual = tuple(sorted(path.name for path in sql_directory.glob("*.sql")))
    if actual != SQL_FILES:
        raise AnalyticsError(
            f"expected exactly five analytics SQL files {SQL_FILES}; found {actual}"
        )
    for filename in SQL_FILES:
        text = (sql_directory / filename).read_text(encoding="utf-8").lower()
        found = [token for token in FORBIDDEN_SQL_TOKENS if token.lower() in text]
        if found:
            raise AnalyticsError(f"{filename} uses excluded fields: {found}")
        load_named_queries(sql_directory / filename)


def _validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(set(ANALYTICS_COLUMNS) - set(frame.columns))
    if missing:
        raise AnalyticsError(f"analytical input is missing columns: {missing}")
    if frame.empty:
        raise AnalyticsError("analytical input is empty")

    null_counts = frame.loc[:, ANALYTICS_COLUMNS].isna().sum()
    null_failures = {
        column: int(count) for column, count in null_counts.items() if count
    }
    if null_failures:
        raise AnalyticsError(f"analytical input contains nulls: {null_failures}")

    positive = (
        "area_sqm",
        "asking_total_price_10k_cny",
        "platform_unit_price_cny_sqm",
        "calculated_unit_price_cny_sqm",
    )
    nonnegative = (
        "rooms",
        "halls",
        "total_floors",
        "follower_count",
        "listing_age_days",
    )
    failures = {
        column: int(frame[column].le(0).sum())
        for column in positive
        if frame[column].le(0).any()
    }
    failures.update(
        {
            column: int(frame[column].lt(0).sum())
            for column in nonnegative
            if frame[column].lt(0).any()
        }
    )
    if failures:
        raise AnalyticsError(f"analytical input has invalid numeric values: {failures}")


def _connect(frame: pd.DataFrame) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.create_function("LOG10", 1, math.log10)
    connection.create_function("POW10", 1, lambda value: 10.0**value)
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -200000")
    frame.loc[:, ANALYTICS_COLUMNS].to_sql(
        "listings", connection, index=False, if_exists="replace"
    )
    connection.execute("CREATE INDEX idx_listings_city ON listings(city_id)")
    return connection


def _query(
    connection: sqlite3.Connection,
    queries: Mapping[str, str],
    name: str,
    *,
    min_group_size: int,
    substitutions: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    sql = queries[name]
    for placeholder, value in (substitutions or {}).items():
        token = "{{" + placeholder + "}}"
        sql = sql.replace(token, value)
    if "{{" in sql or "}}" in sql:
        raise AnalyticsError(f"unresolved SQL placeholder in {name}")
    return pd.read_sql_query(
        sql, connection, params={"min_group_size": min_group_size}
    )


def _number(value: Any, *, digits: int = 4) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    integer_fields = {
        "listing_count",
        "city_count",
        "sample_count",
        "min_rooms",
        "max_rooms",
        "min_halls",
        "max_halls",
        "within_1_cny_sqm",
        "within_5_cny_sqm",
        "above_100_cny_sqm",
        "within_0_1_percent",
        "above_1_percent",
    }
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        name = str(key)
        cleaned_value = _number(value)
        if name in integer_fields and cleaned_value is not None:
            cleaned_value = int(cleaned_value)
        cleaned[name] = cleaned_value
    return cleaned


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_record(row) for row in frame.to_dict(orient="records")]


def _nearest_rank_statistics(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AnalyticsError("cannot summarize an empty distribution")

    def percentile(fraction: float) -> float:
        return ordered[math.floor((len(ordered) - 1) * fraction)]

    return {
        "sample_count": len(ordered),
        "minimum": _number(ordered[0]),
        "mean": _number(sum(ordered) / len(ordered)),
        "p10": _number(percentile(0.10)),
        "p25": _number(percentile(0.25)),
        "median": _number(percentile(0.50)),
        "p75": _number(percentile(0.75)),
        "p90": _number(percentile(0.90)),
        "maximum": _number(ordered[-1]),
    }


def _metric_records(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _records(frame):
        metric = str(row.pop("metric"))
        if row.get("p25") is not None and row.get("p75") is not None:
            row["iqr"] = _number(float(row["p75"]) - float(row["p25"]))
        result[metric] = row
    return result


def _reshape_segment_statistics(
    frame: pd.DataFrame, segment_column: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment, group in frame.groupby(segment_column, sort=False):
        item: dict[str, Any] = {
            segment_column: _number(segment),
            "sample_count": int(group["sample_count"].iloc[0]),
        }
        for row in group.to_dict(orient="records"):
            metric = str(row["metric"])
            item[metric] = {
                "p25": _number(row["p25"]),
                "median": _number(row["median"]),
                "p75": _number(row["p75"]),
                "iqr": _number(float(row["p75"]) - float(row["p25"])),
            }
        output.append(item)
    return output


def _ordered_segments(
    items: list[dict[str, Any]], key: str, order: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    if order is not None:
        position = {value: index for index, value in enumerate(order)}
        return sorted(items, key=lambda item: position.get(str(item[key]), len(order)))
    try:
        return sorted(items, key=lambda item: float(item[key]))
    except (TypeError, ValueError):
        return sorted(items, key=lambda item: str(item[key]))


def build_summary(
    connection: sqlite3.Connection,
    *,
    source_name: str,
    source_sha256: str,
    min_group_size: int,
    sql_directory: Path = SQL_DIRECTORY,
) -> dict[str, Any]:
    coverage_queries = load_named_queries(sql_directory / SQL_FILES[0])
    price_queries = load_named_queries(sql_directory / SQL_FILES[1])
    segment_queries = load_named_queries(sql_directory / SQL_FILES[2])
    city_queries = load_named_queries(sql_directory / SQL_FILES[3])
    signal_queries = load_named_queries(sql_directory / SQL_FILES[4])

    overview_frame = _query(
        connection, coverage_queries, "overview", min_group_size=min_group_size
    )
    overview = _record(overview_frame.iloc[0].to_dict())
    coverage_stats = _metric_records(
        _query(
            connection,
            coverage_queries,
            "coverage_statistics",
            min_group_size=min_group_size,
        )
    )
    city_counts = _query(
        connection, coverage_queries, "city_counts", min_group_size=min_group_size
    )
    eligible_cities = city_counts.loc[
        city_counts["sample_count"].ge(min_group_size)
    ].copy()
    top_cities = eligible_cities.head(15)
    top_listing_count = int(top_cities["sample_count"].sum())
    listing_count = int(overview["listing_count"])
    long_tail_listing_count = listing_count - top_listing_count

    room_hall = _query(
        connection,
        coverage_queries,
        "room_hall_counts",
        min_group_size=min_group_size,
    )
    area_distribution = _query(
        connection,
        coverage_queries,
        "area_distribution",
        min_group_size=min_group_size,
    )

    price_stats = _metric_records(
        _query(
            connection,
            price_queries,
            "price_statistics",
            min_group_size=min_group_size,
        )
    )
    unit_sanity = _record(
        _query(
            connection,
            price_queries,
            "unit_price_sanity",
            min_group_size=min_group_size,
        ).iloc[0].to_dict()
    )
    sanity_n = int(unit_sanity["sample_count"])
    for count_key, share_key in (
        ("within_1_cny_sqm", "share_within_1_cny_sqm"),
        ("within_5_cny_sqm", "share_within_5_cny_sqm"),
        ("above_100_cny_sqm", "share_above_100_cny_sqm"),
        ("within_0_1_percent", "share_within_0_1_percent"),
        ("above_1_percent", "share_above_1_percent"),
    ):
        unit_sanity[share_key] = _number(int(unit_sanity[count_key]) / sanity_n)
    log_distribution = _query(
        connection,
        price_queries,
        "log_price_distribution",
        min_group_size=min_group_size,
    )

    property_segments: dict[str, list[dict[str, Any]]] = {}
    segment_template = segment_queries["property_segment_statistics"]
    for dimension in PROPERTY_SEGMENT_DIMENSIONS:
        query_map = {"property_segment_statistics": segment_template}
        frame = _query(
            connection,
            query_map,
            "property_segment_statistics",
            min_group_size=min_group_size,
            substitutions={"dimension": dimension},
        )
        items = _reshape_segment_statistics(frame, "segment")
        if dimension == "area_bucket":
            items = _ordered_segments(items, "segment", AREA_BUCKET_ORDER)
        elif dimension in {"rooms", "halls", "total_floors"}:
            items = _ordered_segments(items, "segment")
        else:
            items = _ordered_segments(items, "segment")
        property_segments[dimension] = items

    city_frame = _query(
        connection,
        city_queries,
        "city_price_structure",
        min_group_size=min_group_size,
    )
    city_structure = _reshape_segment_statistics(city_frame, "city_id")
    city_structure.sort(key=lambda item: (-int(item["sample_count"]), item["city_id"]))
    comparable = _records(
        _query(
            connection,
            city_queries,
            "comparable_city_segment",
            min_group_size=min_group_size,
        )
    )

    signal_frame = _query(
        connection,
        signal_queries,
        "listing_signal_segments",
        min_group_size=min_group_size,
    )
    listing_signals: dict[str, list[dict[str, Any]]] = {}
    for dimension, group in signal_frame.groupby("dimension", sort=False):
        items = _reshape_segment_statistics(group, "segment")
        order = PRICE_BUCKET_ORDER if dimension == "asking_total_price_bucket" else AREA_BUCKET_ORDER
        listing_signals[str(dimension)] = _ordered_segments(items, "segment", order)

    age_distribution = _records(
        _query(
            connection,
            signal_queries,
            "listing_age_distribution",
            min_group_size=min_group_size,
        )
    )

    summary: dict[str, Any] = {
        "metadata": {
            "schema_version": "1.0.0",
            "source_file": source_name,
            "source_sha256": source_sha256,
            "minimum_group_size": min_group_size,
            "question_count": 5,
            "price_basis": "historical asking prices, not transaction prices",
        },
        "q1_market_coverage": {
            "overview": overview,
            "coverage_statistics": coverage_stats,
            "city_sample_count_distribution": _nearest_rank_statistics(
                city_counts["sample_count"].tolist()
            ),
            "top_cities": _records(top_cities),
            "long_tail": {
                "city_count": int(overview["city_count"]) - len(top_cities),
                "listing_count": long_tail_listing_count,
                "listing_share": _number(long_tail_listing_count / listing_count),
            },
            "room_hall_configurations": _records(room_hall),
            "area_distribution": _records(area_distribution),
        },
        "q2_price_distribution": {
            "statistics": price_stats,
            "unit_price_sanity": unit_sanity,
            "log_scale_distribution": {
                metric: _records(group.drop(columns=["metric"]))
                for metric, group in log_distribution.groupby("metric", sort=False)
            },
        },
        "q3_property_segments": property_segments,
        "q4_city_price_structure": {
            "all_listings": city_structure,
            "comparable_segment_definition": {
                "area_sqm": "60 to 90 inclusive",
                "rooms": "2 or 3",
            },
            "comparable_segment": comparable,
        },
        "q5_platform_snapshot_signals": {
            "segment_statistics": listing_signals,
            "listing_age_distribution": age_distribution,
            "interpretation_boundary": (
                "Follower count and listing age are platform-observed snapshot fields; "
                "they are not verified buyer behavior or sale outcomes."
            ),
        },
    }
    validate_public_summary(summary, min_group_size=min_group_size)
    return summary


def validate_public_summary(summary: Mapping[str, Any], *, min_group_size: int) -> None:
    grouped_collections: list[list[Mapping[str, Any]]] = [
        summary["q1_market_coverage"]["top_cities"],
        summary["q1_market_coverage"]["room_hall_configurations"],
        summary["q1_market_coverage"]["area_distribution"],
        *summary["q3_property_segments"].values(),
        summary["q4_city_price_structure"]["all_listings"],
        summary["q4_city_price_structure"]["comparable_segment"],
        *summary["q5_platform_snapshot_signals"]["segment_statistics"].values(),
        summary["q5_platform_snapshot_signals"]["listing_age_distribution"],
    ]
    for collection in grouped_collections:
        for item in collection:
            if int(item["sample_count"]) < min_group_size:
                raise AnalyticsError(
                    f"public group below minimum n={min_group_size}: {item}"
                )

    serialized = json.dumps(summary, ensure_ascii=False).lower()
    forbidden = (*FORBIDDEN_SQL_TOKENS, "record_id", "listing_title")
    found = [token for token in forbidden if token.lower() in serialized]
    if found:
        raise AnalyticsError(f"public summary contains excluded row-level fields: {found}")


def _configure_matplotlib() -> Any:
    cache_directory = Path(tempfile.gettempdir()) / "property-pledge-matplotlib-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    for candidate in ("Arial Unicode MS", "PingFang SC", "Heiti TC", "Noto Sans CJK SC"):
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
    figure.text(0.01, 0.012, footer, ha="left", va="bottom", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.045, 1, 0.965))
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, bbox_inches="tight", facecolor="white")
    temporary.replace(path)


def _empty_panel(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _errorbar_buckets(
    axis: Any,
    items: list[Mapping[str, Any]],
    metric: str,
    *,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    if not items:
        _empty_panel(axis, "No groups meet the minimum sample size")
        axis.set_title(title)
        return
    labels = [BUCKET_LABELS.get(str(item["segment"]), str(item["segment"])) for item in items]
    medians = [float(item[metric]["median"]) for item in items]
    lower = [median - float(item[metric]["p25"]) for median, item in zip(medians, items)]
    upper = [float(item[metric]["p75"]) - median for median, item in zip(medians, items)]
    positions = list(range(len(items)))
    axis.errorbar(
        positions,
        medians,
        yerr=[lower, upper],
        fmt="o-",
        color=color,
        capsize=4,
        linewidth=2,
    )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def render_figures(summary: Mapping[str, Any], output_directory: Path) -> None:
    plt = _configure_matplotlib()
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    n = int(summary["q1_market_coverage"]["overview"]["listing_count"])
    minimum = int(summary["metadata"]["minimum_group_size"])
    footer = (
        f"Historical asking-listing snapshot; n={n:,}; published groups n≥{minimum}. "
        "Prices are asking prices, not transaction prices."
    )

    coverage = summary["q1_market_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    city_items = list(reversed(coverage["top_cities"]))
    axes[0].barh(
        [str(item["city_id"]) for item in city_items],
        [int(item["sample_count"]) for item in city_items],
        color="#356A8A",
    )
    long_tail = coverage["long_tail"]
    axes[0].set_title(
        "Top 15 city samples\n"
        f"Remaining {long_tail['city_count']} cities: {long_tail['listing_share']:.1%} of listings"
    )
    axes[0].set_xlabel("Listings")
    axes[0].tick_params(axis="y", labelsize=8)

    area_items = coverage["area_distribution"]
    axes[1].bar(
        [str(item["area_interval"]).replace("_to_lt_", "–<").replace("lt_", "<").replace("_plus", "+") for item in area_items],
        [int(item["sample_count"]) for item in area_items],
        color="#4F9D8C",
    )
    axes[1].set_title("Floor-area coverage")
    axes[1].set_xlabel("Area interval (m²)")
    axes[1].set_ylabel("Listings")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Q1 · Market coverage", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "market_coverage.png", footer)
    plt.close(fig)

    price = summary["q2_price_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    price_specs = (
        ("asking_total_price_10k_cny", "Asking total price", "CNY 10,000"),
        ("calculated_unit_price_cny_sqm", "Calculated unit price", "CNY/m²"),
    )
    for axis, (metric, title, xlabel) in zip(axes, price_specs):
        bins = price["log_scale_distribution"].get(metric, [])
        if bins:
            lower = [float(item["lower_bound"]) for item in bins]
            width = [float(item["upper_bound"]) - float(item["lower_bound"]) for item in bins]
            axis.bar(
                lower,
                [int(item["sample_count"]) for item in bins],
                width=width,
                align="edge",
                color="#356A8A" if metric.startswith("asking") else "#C76D3A",
            )
            axis.set_xscale("log")
        else:
            _empty_panel(axis, "No bins meet the minimum sample size")
        stats = price["statistics"][metric]
        axis.set_title(f"{title}\nmedian {stats['median']:,.1f}; mean {stats['mean']:,.1f}")
        axis.set_xlabel(f"{xlabel} (log scale)")
        axis.set_ylabel("Listings")
    sanity = price["unit_price_sanity"]
    fig.suptitle(
        "Q2 · Asking-price distributions\n"
        f"Platform vs calculated unit price: {sanity['share_within_1_cny_sqm']:.1%} within 1 CNY/m²",
        fontsize=15,
        fontweight="bold",
    )
    _save_figure(fig, figure_directory / "price_distribution.png", footer)
    plt.close(fig)

    area_segments = summary["q3_property_segments"]["area_bucket"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    _errorbar_buckets(
        axes[0],
        area_segments,
        "asking_total_price_10k_cny",
        title="Asking total price by area bucket",
        ylabel="Median and IQR (CNY 10,000)",
        color="#356A8A",
    )
    _errorbar_buckets(
        axes[1],
        area_segments,
        "calculated_unit_price_cny_sqm",
        title="Calculated unit price by area bucket",
        ylabel="Median and IQR (CNY/m²)",
        color="#C76D3A",
    )
    fig.suptitle("Q3 · Property segments and asking prices", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "price_by_area_bucket.png", footer)
    plt.close(fig)

    city = summary["q4_city_price_structure"]
    all_by_city = {str(item["city_id"]): item for item in city["all_listings"]}
    comparable = sorted(
        city["comparable_segment"],
        key=lambda item: (-int(item["sample_count"]), str(item["city_id"])),
    )[:15]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    if comparable:
        ordered_by_price = sorted(
            city["comparable_segment"],
            key=lambda item: float(item["median_unit_price_cny_sqm"]),
        )
        price_positions = list(range(len(ordered_by_price)))
        axes[0].scatter(
            [float(item["median_unit_price_cny_sqm"]) for item in ordered_by_price],
            price_positions,
            color="#C76D3A",
            s=25,
        )
        axes[0].set_yticks([])
        axes[0].set_xlabel("Median calculated unit price (CNY/m²)")
        axes[0].set_ylabel("Eligible cities, ordered by median")
        axes[0].set_title("Comparable-segment distribution")
        for item, vertical_alignment in (
            (ordered_by_price[0], "bottom"),
            (ordered_by_price[-1], "top"),
        ):
            index = ordered_by_price.index(item)
            axes[0].annotate(
                f"{item['city_id']} · {item['median_unit_price_cny_sqm']:,.0f}",
                (float(item["median_unit_price_cny_sqm"]), index),
                xytext=(8, 0),
                textcoords="offset points",
                va=vertical_alignment,
                fontsize=9,
            )

        city_names = [str(item["city_id"]) for item in comparable]
        positions = list(range(len(city_names)))
        all_medians = [
            float(all_by_city[name]["calculated_unit_price_cny_sqm"]["median"])
            for name in city_names
        ]
        comparable_medians = [float(item["median_unit_price_cny_sqm"]) for item in comparable]
        axes[1].scatter(all_medians, positions, color="#8C8C8C", label="All listings", s=46)
        axes[1].scatter(
            comparable_medians,
            positions,
            color="#C76D3A",
            label="60–90 m² and 2–3 rooms",
            s=52,
        )
        for y, left, right in zip(positions, all_medians, comparable_medians):
            axes[1].plot([left, right], [y, y], color="#C7C7C7", linewidth=1.2, zorder=0)
        axes[1].set_yticks(positions, city_names)
        axes[1].invert_yaxis()
        axes[1].legend(frameon=False)
        axes[1].set_xlabel("Median calculated unit price (CNY/m²)")
        axes[1].set_title("Largest comparable samples: all vs controlled segment")
    else:
        for axis in axes:
            _empty_panel(axis, "No city segment meets the minimum sample size")
    fig.suptitle("Q4 · City price structure", fontsize=16, fontweight="bold")
    _save_figure(fig, figure_directory / "city_price_structure.png", footer)
    plt.close(fig)

    signals = summary["q5_platform_snapshot_signals"]
    price_signal = signals["segment_statistics"].get("asking_total_price_bucket", [])
    area_signal = signals["segment_statistics"].get("area_bucket", [])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    _errorbar_buckets(
        axes[0, 0],
        price_signal,
        "follower_count",
        title="Follower count by asking-price bucket",
        ylabel="Median and IQR",
        color="#356A8A",
    )
    _errorbar_buckets(
        axes[0, 1],
        price_signal,
        "listing_age_days",
        title="Listing age by asking-price bucket",
        ylabel="Median and IQR (days)",
        color="#C76D3A",
    )
    _errorbar_buckets(
        axes[1, 0],
        area_signal,
        "follower_count",
        title="Follower count by area bucket",
        ylabel="Median and IQR",
        color="#4F9D8C",
    )
    age_distribution = signals["listing_age_distribution"]
    if age_distribution:
        axes[1, 1].bar(
            [int(item["listing_age_days"]) for item in age_distribution],
            [int(item["sample_count"]) for item in age_distribution],
            width=2.0,
            color="#7A6BA6",
        )
        axes[1, 1].set_xlabel("Encoded listing age (days)")
        axes[1, 1].set_ylabel("Listings")
    else:
        _empty_panel(axes[1, 1], "No duration values meet the minimum sample size")
    axes[1, 1].set_title("Listing-age distribution")
    fig.suptitle(
        "Q5 · Platform-observed engagement and listing-duration signals",
        fontsize=16,
        fontweight="bold",
    )
    _save_figure(fig, figure_directory / "listing_engagement.png", footer)
    plt.close(fig)


def render_findings(summary: Mapping[str, Any]) -> str:
    coverage = summary["q1_market_coverage"]
    price = summary["q2_price_distribution"]
    segments = summary["q3_property_segments"]
    city = summary["q4_city_price_structure"]
    signals = summary["q5_platform_snapshot_signals"]
    listing_count = int(coverage["overview"]["listing_count"])
    top_ten = coverage["top_cities"][:10]
    top_ten_share = sum(int(item["sample_count"]) for item in top_ten) / listing_count

    total_stats = price["statistics"]["asking_total_price_10k_cny"]
    sanity = price["unit_price_sanity"]
    area_items = segments["area_bucket"]
    comparable = city["comparable_segment"]
    comparable_sorted = sorted(
        comparable, key=lambda item: float(item["median_unit_price_cny_sqm"])
    )
    price_signals = signals["segment_statistics"].get("asking_total_price_bucket", [])

    lines = [
        "# Analytics v1 Findings",
        "",
        (
            f"Scope: {listing_count:,} historical second-hand asking listings. All results are "
            "descriptive aggregates generated from `reports/summary.json`; grouped "
            f"results require n ≥ {summary['metadata']['minimum_group_size']}."
        ),
        "",
        "## 1. City coverage and sample concentration",
        "",
        (
            f"**Finding.** The snapshot contains {listing_count:,} listings across "
            f"{coverage['overview']['city_count']} cities without being dominated by a "
            "small set of city samples."
        ),
        "",
        (
            f"**Evidence.** The ten largest city samples contain {top_ten_share:.1%} of "
            f"all listings; the median city contributes "
            f"{coverage['city_sample_count_distribution']['median']:,.0f} listings and "
            f"the largest contributes {coverage['city_sample_count_distribution']['maximum']:,.0f}. "
            "See `figures/market_coverage.png`."
        ),
        "",
        (
            "**Limitation.** City counts must not be read as market size. Their relatively "
            "even distribution may reflect collection limits or crawl design, and the "
            "snapshot is not a probability sample of city housing markets."
        ),
        "",
        "## 2. Asking total prices are strongly right-skewed",
        "",
        (
            f"**Finding.** The mean asking total price ({total_stats['mean']:,.1f} × "
            f"CNY 10,000) exceeds the median ({total_stats['median']:,.1f} × CNY 10,000)."
        ),
        "",
        (
            f"**Evidence.** P10/P90 are {total_stats['p10']:,.1f} and "
            f"{total_stats['p90']:,.1f} × CNY 10,000; the IQR is "
            f"{total_stats['iqr']:,.1f}. {sanity['share_within_1_cny_sqm']:.1%} of "
            "platform unit prices are within 1 CNY/m² of price ÷ area after rounding. "
            f"Only {sanity['above_1_percent']:,} rows differ by more than 1% in relative terms. "
            "See `figures/price_distribution.png`."
        ),
        "",
        (
            "**Limitation.** These are asking prices; the right tail does not show completed "
            "sale values, appraisal values, or eventual outcomes. The small unit-price "
            "mismatch tail remains a field-definition/data-quality exception, not proof "
            "that the two fields are identical."
        ),
        "",
    ]

    if area_items:
        low_area = area_items[0]
        high_area = area_items[-1]
        unit_low = min(
            area_items,
            key=lambda item: float(item["calculated_unit_price_cny_sqm"]["median"]),
        )
        unit_high = max(
            area_items,
            key=lambda item: float(item["calculated_unit_price_cny_sqm"]["median"]),
        )
        lines.extend(
            [
                "## 3. Area segments differ in both total and unit asking-price structure",
                "",
                (
                    f"**Finding.** Median total asking price rises from "
                    f"{low_area['asking_total_price_10k_cny']['median']:,.1f} × CNY 10,000 "
                    f"in the {BUCKET_LABELS[str(low_area['segment'])]} m² bucket to "
                    f"{high_area['asking_total_price_10k_cny']['median']:,.1f} × CNY 10,000 "
                    f"in the {BUCKET_LABELS[str(high_area['segment'])]} m² bucket."
                ),
                "",
                (
                    f"**Evidence.** Median calculated unit price is not monotonic across "
                    f"area buckets: it ranges from "
                    f"{unit_low['calculated_unit_price_cny_sqm']['median']:,.0f} CNY/m² "
                    f"for {BUCKET_LABELS[str(unit_low['segment'])]} m² to "
                    f"{unit_high['calculated_unit_price_cny_sqm']['median']:,.0f} CNY/m² "
                    f"for {BUCKET_LABELS[str(unit_high['segment'])]} m². The figure also "
                    "reports IQRs. See `figures/price_by_area_bucket.png`."
                ),
                "",
                (
                    "**Limitation.** Area buckets also differ in city and property composition; "
                    "the comparison is descriptive and not an isolated area effect."
                ),
                "",
            ]
        )

    if comparable_sorted:
        lowest = comparable_sorted[0]
        highest = comparable_sorted[-1]
        lines.extend(
            [
                "## 4. City price structure remains heterogeneous within a narrower property segment",
                "",
                (
                    "**Finding.** Among cities meeting the sample threshold for 60–90 m², "
                    "2–3-room listings, median calculated unit price still varies materially."
                ),
                "",
                (
                    f"**Evidence.** Comparable-segment medians range from "
                    f"{lowest['median_unit_price_cny_sqm']:,.0f} CNY/m² in "
                    f"{lowest['city_id']} (n={lowest['sample_count']:,}) to "
                    f"{highest['median_unit_price_cny_sqm']:,.0f} CNY/m² in "
                    f"{highest['city_id']} (n={highest['sample_count']:,}). "
                    "See `figures/city_price_structure.png`."
                ),
                "",
                (
                    "**Limitation.** Matching only on area and room count does not control "
                    "for neighborhood, building age, condition, or other unobserved composition."
                ),
                "",
            ]
        )

    if price_signals:
        first = price_signals[0]
        last = price_signals[-1]
        lines.extend(
            [
                "## 5. Platform snapshot signals vary across asking-price buckets",
                "",
                (
                    f"**Finding.** Median follower count changes from "
                    f"{first['follower_count']['median']:,.0f} in the "
                    f"{BUCKET_LABELS[str(first['segment'])]} bucket to "
                    f"{last['follower_count']['median']:,.0f} in the "
                    f"{BUCKET_LABELS[str(last['segment'])]} bucket; median encoded listing "
                    f"age changes from {first['listing_age_days']['median']:,.0f} to "
                    f"{last['listing_age_days']['median']:,.0f} days."
                ),
                "",
                (
                    "**Evidence.** The same two fields are summarized by price and area "
                    "buckets with medians and IQRs in `figures/listing_engagement.png`."
                ),
                "",
                (
                    "**Limitation.** Follower count and listing age are coarse, platform-observed "
                    "snapshot fields. They do not establish buyer behavior, sale probability, "
                    "or speed of sale."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_output_manifest(output_directory: Path) -> None:
    actual = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual - PUBLIC_OUTPUTS)
    missing = sorted(PUBLIC_OUTPUTS - actual)
    if unexpected or missing:
        raise AnalyticsError(
            f"public output manifest mismatch; missing={missing}, unexpected={unexpected}"
        )


def run_analytics(
    input_path: Path | str,
    output_directory: Path | str,
    *,
    source_kind: str,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    if source_kind not in {"real", "synthetic"}:
        raise AnalyticsError(f"unsupported source kind: {source_kind}")
    if min_group_size < 1:
        raise AnalyticsError("minimum group size must be positive")

    validate_sql_contract()
    source = Path(input_path).resolve()
    output = Path(output_directory).resolve()
    if not source.is_file():
        raise AnalyticsError(f"analytical input does not exist: {source}")
    if source_kind == "synthetic" and output == DEFAULT_OUTPUT_DIRECTORY.resolve():
        raise AnalyticsError("synthetic runs cannot overwrite the public real-data reports")
    if source_kind == "real":
        private_root = (REPOSITORY_ROOT / "private_data/derived").resolve()
        try:
            source.relative_to(private_root)
        except ValueError as exc:
            raise AnalyticsError(f"real row-level input must remain under {private_root}") from exc

    source_hash = sha256_file(source)
    frame = pd.read_csv(source, low_memory=False)
    _validate_input(frame)
    if source_kind == "real":
        if len(frame) != EXPECTED_REAL_ROW_COUNT:
            raise AnalyticsError(
                f"expected {EXPECTED_REAL_ROW_COUNT} real rows; found {len(frame)}"
            )
        if source_hash != EXPECTED_REAL_SHA256:
            raise AnalyticsError(
                f"real snapshot SHA-256 mismatch: {source_hash}"
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    existing = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(existing - PUBLIC_OUTPUTS)
    if unexpected:
        raise AnalyticsError(f"refusing to mix unexpected public files: {unexpected}")

    connection = _connect(frame)
    try:
        summary = build_summary(
            connection,
            source_name=source.name,
            source_sha256=source_hash,
            min_group_size=min_group_size,
        )
    finally:
        connection.close()

    _write_text_atomic(
        output / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(output / "findings.md", render_findings(summary))
    render_figures(summary, output)
    _validate_output_manifest(output)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input or (
        DEFAULT_REAL_INPUT if args.source == "real" else DEFAULT_SYNTHETIC_INPUT
    )
    output_directory = args.output_dir or (
        DEFAULT_OUTPUT_DIRECTORY
        if args.source == "real"
        else REPOSITORY_ROOT / "private_data/derived/synthetic_reports"
    )
    summary = run_analytics(
        input_path,
        output_directory,
        source_kind=args.source,
        min_group_size=args.min_group_size,
    )
    print(
        json.dumps(
            {
                "row_count": summary["q1_market_coverage"]["overview"]["listing_count"],
                "city_count": summary["q1_market_coverage"]["overview"]["city_count"],
                "minimum_group_size": summary["metadata"]["minimum_group_size"],
                "output_directory": str(Path(output_directory).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
