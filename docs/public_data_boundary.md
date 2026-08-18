# Public Data Boundary

This repository preserves legacy code, notebooks, documentation, and aggregate figures, but does not publish row-level housing records in the current tree.

## Boundary

The following are public:

- legacy notebooks and scripts;
- documentation and data dictionaries;
- aggregate/reference tables explicitly listed below;
- aggregate charts and historical model-output artifacts;
- fully synthetic fixtures under `fixtures/`.

The following are private:

- scraped listing rows;
- per-city raw exports;
- merged or cleaned listing-level tables;
- modeling matrices and train/test files derived from real listings;
- any future row-level CSV, XLSX, JSON, JSONL, Parquet, Feather, or Arrow file under `DataSet/`.

The current-tree cleanup removes 632 tracked row-level CSV/XLSX files. It does **not** rewrite Git history. Those files remain recoverable from older commits until a separately reviewed history-rewrite operation is performed.

## Private canonical input

The selected input for a future analytics pipeline is the historical file:

```text
DataSet/合并二手房量化前数据.csv
```

It is preserved locally at the ignored path:

```text
private_data/canonical/secondhand_legacy_pre_encoding.csv
```

Expected SHA-256:

```text
9b28d50cb76770299dab0747c5ff048e3e3f6067a233e401c9b4e3d330376020
```

Why this version:

- 300,397 rows across 135 cities;
- keeps city identity and human-readable categorical values;
- contains price, property attributes, listing-age/follower fields, GDP, region, and city greening rate;
- sits before removal of the remaining villa rows, the zero-floor row, and exact duplicates, so those rules can be rebuilt and tested explicitly;
- avoids the final numeric table's loss of city identity.

Known limitations:

- it is already downstream of undocumented parsing, missing-structure filtering, and missing-floor filtering;
- 1,909 rows still have missing floor level;
- 2,250 rows have an explicit villa subtype;
- one row has total floors equal to zero;
- it contains 30,944 exact duplicate rows at this stage;
- listing titles and community/location strings remain third-party row-level content and must stay private.

A future pipeline should accept this path through configuration, for example `PROPERTY_PLEDGE_CANONICAL_INPUT`, rather than hard-code a repository-relative data path.

## Public synthetic contract

`fixtures/secondhand_analytics.schema.json` defines the normalized public record contract. `fixtures/secondhand_analytics.synthetic.jsonl` contains invented rows for parser, validation, and documentation tests.

The fixture deliberately excludes free-text listing titles, community/developer strings, addresses, URLs, source IDs, and real city names. `synthetic=true` is required on every record.

Synthetic values are examples only. They must not be used as analytical evidence or model results.

## Retained aggregate/reference tables

These three files are retained because they contain city-level aggregates rather than property/listing rows:

- `DataSet/主要城市gdp.xlsx`
- `DataSet/newHouse/链家新房城市页数表.xlsx`
- `DataSet/newHouse - 2/链家新房城市页数表.xlsx`

Retention is not a blanket license determination. Before relying on them in a new pipeline, add exact source URLs, retrieval dates, definitions, and reuse terms. If that provenance cannot be established, replace them with documented public-source extracts or synthetic equivalents.

## Release checks

Before committing future changes:

1. `git status --short` must show no files under `private_data/`.
2. No real row-level table may be added under `DataSet/`.
3. Public fixtures must contain only invented values and must validate against the schema.
4. Aggregate outputs must be checked for small-cell disclosure and source/license requirements.
5. A current-tree deletion must not be described as removal from Git history.
