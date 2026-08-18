# Housing Data Analytics & Price Modeling

Reproducible analysis of a historical second-hand housing listing snapshot: sanitized acquisition evidence → deterministic data preparation → SQL analytics → leakage-controlled asking-price regression → error and city-holdout diagnostics.

| Scope | Frozen evaluation | Geographic stress test |
|---|---|---|
| 267,305 listings across 135 cities | XGBoost MAE **35.96**, RMSE **76.72**, R² **0.749** | 19 unseen cities: MAE **84.51** (2.35×), R² **0.176** |

Prices and errors are in CNY 10,000 units. These are historical **asking prices**, not transactions, appraisals, collateral values, or lending-risk outcomes. The complete-city holdout is a robustness diagnostic, not proof of generalization to arbitrary new cities.

## System flow

```text
Historical listing-card acquisition (sanitized evidence only)
        ↓
Legacy raw-to-canonical cleaning (partially documented boundary)
        ↓
Private 300,397-row canonical snapshot
        ↓  invalid villa/floor rules + exact analytical deduplication
Private 267,305-row analytical table
        ├── SQLite: five fixed descriptive questions
        │       └── aggregate summary, findings, and figures
        └── deterministic train / validation / frozen-test evaluation
                └── aggregate metrics, error slices, and city holdout
```

Real row-level inputs, derived tables, split assignments, and predictions stay under ignored `private_data/`. The public repository contains code, invented fixtures, and aggregate outputs only.

## Analytical results

- The snapshot covers 267,305 listings and 135 cities; city counts describe collection coverage, not market size.
- Asking prices are right-skewed: median 105.0 versus mean 145.8, with P10/P90 of 49.8/268.0.
- In a 60–90 m², 2–3-room comparison, published city medians still range from 3,822 to 54,545 CNY/m² (groups require n ≥ 100); neighborhood and building composition remain uncontrolled.
- 99.2% of platform unit prices differ from `asking price / area` by no more than 1 CNY/m². The platform and calculated unit-price fields are therefore excluded from modeling as target-derived leakage.

See the machine-generated [analytics findings](reports/findings.md), [summary JSON](reports/summary.json), and [figures](reports/figures/).

## Leakage-controlled modeling

Target: `asking_total_price_10k_cny`.

Features: `city_id`, rooms, halls, area, primary orientation, furnishing, floor category, total floors, and building type. Preprocessing is fit on training rows only. Target quantile bins are used only for deterministic split assignment; validation is used for the fixed XGBoost early-stopping boundary; the frozen test does not influence preprocessing, feature selection, or hyperparameters.

| Model | Frozen-test MAE | RMSE | R² | Median AE |
|---|---:|---:|---:|---:|
| Global training median | 75.87 | 158.40 | -0.069 | 41.20 |
| City training median | 56.63 | 124.18 | 0.343 | 30.20 |
| Ridge | 47.19 | 97.29 | 0.597 | 28.41 |
| XGBoost | **35.96** | **76.72** | **0.749** | **18.76** |

Errors are not uniform. For XGBoost, the 500+ asking-price bucket has MAE 245.28 and median absolute percentage error 27.6% (n=1,044); the 144+ m² bucket has MAE 76.14 (n=5,251). Under the separate complete-city holdout, XGBoost degrades to MAE 84.51, RMSE 199.12, and R² 0.176 across 39,967 rows.

See [modeling findings](reports/modeling_findings.md), [metrics](reports/model_metrics.csv), and [error slices](reports/model_error_slices.csv).

## Historical acquisition evidence

The original project discovered city sites, paginated `/ershoufang/pg{page}/`, parsed six listing-card fields, and produced per-city raw tables. The large notebooks and copied outputs were removed from the current tree because they mixed absolute paths, stale experiments, third-party content, and unverifiable artifacts.

[`housing_analytics/acquisition.py`](housing_analytics/acquisition.py) retains a tested offline URL/parser boundary without network access, proxy logic, TLS bypasses, or live-site reliability claims. The remaining acquisition-to-canonical gap is documented in [Historical Acquisition Boundary](docs/acquisition_history.md).

## Reproduce

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-modeling.txt
```

Run the preparation, analytics, and modeling path with invented tracked input:

```bash
python -m housing_analytics.analytics_v1 --source synthetic
python -m housing_analytics.run_analytics --source synthetic
python -m housing_analytics.run_modeling --source synthetic
python -m unittest discover -s tests -v
```

With the audited private canonical snapshot in place:

```bash
python -m housing_analytics.analytics_v1 --source legacy --expect-canonical-lineage
python -m housing_analytics.run_analytics --source real
python -m housing_analytics.run_modeling --source real
```

The frozen test has already been disclosed. It must not be used for further tuning.

## Repository structure

```text
housing_analytics/   preparation, analytics runner, modeling evaluation, acquisition evidence
analytics/sql/       five fixed descriptive SQL question sets
fixtures/            schema plus invented row-level test data
tests/               schema, lineage, leakage, split, output, and synthetic-path tests
reports/             aggregate analytics/modeling results and figures
docs/                data contract, public boundary, and acquisition limitations
private_data/        ignored real inputs, derived rows, splits, and predictions
```

Detailed preparation rules and the `300,397 → 298,146 → 267,305` lineage are in [Analytics Slice 1](docs/analytics_slice_1.md). Data-release constraints are in [Public Data Boundary](docs/public_data_boundary.md).

## Limitations

- The snapshot is historical platform listing data, not a probability sample or a time-based evaluation.
- The raw acquisition-to-19-column canonical transformation is not fully reproducible from retained evidence.
- One deterministic frozen split does not establish performance stability across time or sources.
- The city holdout is deliberately labeled a stress test; it does not prove new-city generalization.
