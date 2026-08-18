# Synthetic Fixtures

The files in this directory are public, invented test data. They contain no copied listing rows, real city names, addresses, listing text, platform IDs, or source URLs.

- `secondhand_analytics.schema.json`: JSON Schema for one normalized second-hand housing analytics record.
- `secondhand_analytics.synthetic.jsonl`: small fixture set for validation and future pipeline tests.

Every record must set `synthetic` to `true`. These values are not analytical findings and must not be used for model evaluation.
