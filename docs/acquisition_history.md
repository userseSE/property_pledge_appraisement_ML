# Historical Acquisition Boundary

This document preserves the useful, verifiable part of the original 2022 web-data acquisition workflow without retaining the large notebooks, captured HTML, proxy lists, absolute local paths, or third-party listing rows.

## What the historical code did

The original second-hand listing notebook:

1. requested the Lianjia city directory at `https://www.lianjia.com/city/`;
2. extracted city names and city base URLs;
3. opened each city's `/ershoufang/` page and read the page-count metadata;
4. iterated URLs following `/ershoufang/pg{page}/`;
5. selected cards under `.sellListContent li`;
6. extracted title, community/location label, house-information text, follow-information text, asking total price, and displayed unit price;
7. wrote one raw table per city before later merging and cleaning.

The sanitized implementation in `housing_analytics/acquisition.py` preserves the pagination contract and an offline HTML parser for those six raw listing-card fields. Tests run it only against invented HTML. It performs no HTTP requests and writes no data.

## Deliberately not retained

- user-agent rotation and retry loops;
- disabled TLS verification;
- hard-coded Windows paths;
- proxy/IP-pool artifacts;
- copied page HTML and third-party row-level records;
- new-house collection code, which is outside the packaged second-hand analysis;
- claims that historical selectors still work on the current live site.

## Reproducibility gap

The old raw scraper emitted six coarse text fields. The selected private canonical snapshot contains 19 parsed columns after additional legacy cleaning, enrichment, and merging. That raw-to-canonical transformation is only partially documented and is not represented as reproducible code here.

For that reason, the supported analytical pipeline begins at:

```text
private_data/canonical/secondhand_legacy_pre_encoding.csv
```

The acquisition module is evidence of historical system boundaries, not a maintained ingestion client. Any future live acquisition would require a new source contract, terms-of-use review, retrieval manifest, rate limits, and dedicated tests.
