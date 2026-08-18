# Public Data Boundary

This repository publishes reproducible code, documentation, synthetic fixtures, and aggregate analytical/modeling outputs. It does not publish real row-level housing records in the current tree.

## Boundary

The following are public:

- the canonical preparation, analytics, and evaluation code under `housing_analytics/`;
- fixed SQL questions under `analytics/sql/`;
- generated aggregate reports and figures under `reports/`;
- acquisition-boundary documentation and a sanitized offline parser;
- fully synthetic fixtures under `fixtures/`.

The following are private:

- scraped listing rows;
- per-city raw exports;
- merged or cleaned listing-level tables;
- modeling matrices and train/test files derived from real listings;
- any real row-level CSV, XLSX, JSON, JSONL, Parquet, Feather, Arrow, split-assignment, or prediction file.

An earlier current-tree cleanup removed 632 tracked row-level CSV/XLSX paths. It did **not** rewrite Git history. Those paths remain recoverable from older commits until a separately reviewed history-rewrite operation is performed.

## Private canonical input

The selected private canonical input is preserved locally at the ignored path:

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

The current pipeline uses this explicit repository-local private boundary by default and also accepts an input path on the command line.

## Public synthetic contract

`fixtures/secondhand_analytics.schema.json` defines the normalized public record contract. `fixtures/secondhand_analytics.synthetic.jsonl` contains invented rows for parser, validation, and documentation tests.

The fixture deliberately excludes free-text listing titles, community/developer strings, addresses, URLs, source IDs, and real city names. `synthetic=true` is required on every record.

Synthetic values are examples only. They must not be used as analytical evidence or model results.

## Removed reference data

The remaining city GDP and new-house page-count workbooks were removed from the current tree. Their exact source URLs, retrieval dates, definitions, and reuse terms were not established, and the canonical listing-only pipeline does not require them.

## Release checks

Before committing future changes:

1. `git status --short` must show no files under `private_data/`.
2. No real row-level table, split assignment, or prediction may be added outside ignored `private_data/`.
3. Public fixtures must contain only invented values and must validate against the schema.
4. Aggregate outputs must be checked for small-cell disclosure and source/license requirements.
5. A current-tree deletion must not be described as removal from Git history.
