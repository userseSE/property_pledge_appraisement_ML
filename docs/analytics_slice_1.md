# Analytics Slice 1: Data Contract and Preparation

This slice prepares a deterministic, private analytical table from the audited 300,397-row canonical snapshot. It does not train models, produce charts, or depend on the incomplete city-macro provenance.

## Canonical input columns

| # | Source column | Classification | Meaning and provenance status | Analytics v1 decision |
|---:|---|---|---|---|
| 1 | `标题` | Property/listing field | Third-party listing title captured by the scraper. Exact source URL and reuse terms are not stored. | Raw text excluded. Only seven explicitly documented literal keyword flags are retained. |
| 2 | `开发商` | Target / identifier | The values appear to be community plus subdistrict text, not a verified developer field; the legacy name is misleading. | Excluded. |
| 3 | `室` | Property/listing field | Parsed bedroom/room count. | Included as `rooms`. |
| 4 | `厅` | Property/listing field | Parsed living-room/hall count. | Included as `halls`. |
| 5 | `面积（平米）` | Property/listing field | Parsed floor area in square metres. | Included as `area_sqm`. |
| 6 | `朝向` | Property/listing field | Scraped orientation text. The legacy notes specify keeping the first direction. | Included as readable `orientation_primary`. |
| 7 | `装修` | Property/listing field | Scraped furnishing/decoration category. | Included as a readable category. |
| 8 | `所在高度` | Property/listing field | Parsed relative floor level. Missing values in this snapshot occur only on villa rows. | Included after invalid-floor filtering. |
| 9 | `总楼层高` | Property/listing field | Parsed total floors in the building. | Included after removing the one zero-floor row. |
| 10 | `建筑结构` | Property/listing field | Parsed building-structure category. | Included as a readable category. |
| 11 | `是否别墅` | Property/listing field | Villa subtype; null denotes a non-villa row in this snapshot. | Used for filtering, then excluded. |
| 12 | `关注人数` | Platform-derived field | Platform follower count at collection time, not a demonstrated liquidity outcome. | Included as `follower_count` with this limitation. |
| 13 | `发布时长（天）` | Platform-derived field | Legacy conversion of relative publish-time text to days; only 43 distinct values indicate coarse encoding. | Included as `listing_age_days`, not treated as exact time-on-market. |
| 14 | `售价/万` | Target / identifier | Listing asking price in CNY 10,000 units. It is not a transaction price, appraisal, or realized pledge value. | Included as the v1 target. |
| 15 | `单价` | Platform-derived field | Platform-displayed unit price, stored as a formatted integer CNY/sqm string. | Parsed and retained; a separate calculated unit price is added. |
| 16 | `城市` | Target / identifier | City label inferred from the per-city source file. | Included as private `city_id` for grouping and deduplication. |
| 17 | `城市gdp` | External city/macro field | Notes say 2021 GDP came from a mixture of NBS, local-government, and manually collected sources; exact row-level source URLs are absent. | Excluded from Analytics v1. |
| 18 | `所在地区` | External city/macro field | Manually assigned six-region city grouping; mapping provenance is not versioned. | Excluded from Analytics v1. |
| 19 | `城市绿化率(%)` | External city/macro field | Notes cite the 2021 China City Statistical Yearbook plus local announcements, without a complete source manifest. | Excluded from Analytics v1. |

The legacy notes incorrectly map `室` to `hall` and `厅` to `room`. Alignment against the historical pre-dedup table confirms the raw values, but Analytics v1 uses the correct meanings: `室 -> rooms`, `厅 -> halls`.

## Deterministic transformation

1. Read the 19-column CSV as `gb18030` and require the exact column order.
2. Normalize whitespace and map categorical values to readable English labels. Keep only the first orientation direction, as specified in the legacy cleaning notes.
3. Remove rows that are villas or have a missing/unsupported floor level. Villa-specific `上叠/下叠` values are parsed explicitly but are not valid ordinary-residential floor levels.
4. Remove rows where total floors equals zero. Unsupported negative values fail validation; they are not silently filtered.
5. Remove exact duplicates using the normalized analytical signature below, keeping the first source row.
6. Add calculated unit price and versioned area/asking-price buckets.
7. Write real output only under ignored `private_data/derived/`.

Audited real-data lineage:

```text
300,397 input rows
  - 2,250 villa or invalid-floor rows
  -     1 total-floor-zero row
= 298,146 quality-filtered rows
  - 30,841 exact analytical duplicates
= 267,305 Analytics v1 rows
```

All 1,909 missing-floor rows and the 341 `上叠/下叠` rows overlap the 2,250 villa rows, so the union removes 2,250 rows.

## Duplicate signature

Raw title and community/location strings are identifiers and are not used directly as equality keys. The signature contains:

- city identifier;
- rooms, halls, area, primary orientation, furnishing, floor level, total floors, and building type;
- follower count and listing age;
- asking total price and parsed platform unit price;
- seven literal title indicators:
  - park: `公园`
  - light: `采光|阳光好`
  - parking: `车位`
  - water view: `海景|河景|湖景`
  - business: `商圈|商场|CBD|商务|商贸|商业`
  - transport: `交通|地铁`
  - tax/tenure: `满五|满二`

These booleans report literal string matches only. They do not establish proximity, quality, tax eligibility, or legal status. On the canonical snapshot, this signature reproduces the historical 30,841-row duplicate mask exactly, with zero keep/drop mismatches, while replacing unverified macro fields with the city identifier.

## Derived fields

- `calculated_unit_price_cny_sqm = asking_total_price_10k_cny * 10,000 / area_sqm`, rounded to two decimals.
- Area buckets (sqm): `<60`, `60–<90`, `90–<120`, `120–<144`, `144+`.
- Asking-price buckets (CNY 10,000): `<100`, `100–<200`, `200–<300`, `300–<500`, `500+`.

The boundaries are an explicit v1 engineering contract, not empirical findings or official market segments.

## Run

Real canonical snapshot:

```bash
python -m pip install -r requirements-analytics.txt
python -m housing_analytics.analytics_v1 --source legacy --expect-canonical-lineage
```

Tracked synthetic fixture:

```bash
python -m housing_analytics.analytics_v1 --source synthetic
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Unresolved definitions and provenance

- No complete source-URL, retrieval-time, or license manifest exists for the scraped listings.
- `开发商` is mislabeled and appears to combine community and district text.
- Asking price is not transaction price, collateral appraisal, or realized pledge value.
- Followers and listing age are platform snapshot fields, not validated liquidity outcomes.
- The macro columns lack row-level source manifests and are intentionally excluded.
- The literal title flags are reproducibility aids, not verified semantic features.
