# Analytics v1 Findings

Scope: 267,305 historical second-hand asking listings. All results are descriptive aggregates generated from `reports/summary.json`; grouped results require n ≥ 100.

## 1. City coverage and sample concentration

**Finding.** The snapshot contains 267,305 listings across 135 cities without being dominated by a small set of city samples.

**Evidence.** The ten largest city samples contain 9.7% of all listings; the median city contributes 2,186 listings and the largest contributes 2,751. See `figures/market_coverage.png`.

**Limitation.** City counts must not be read as market size. Their relatively even distribution may reflect collection limits or crawl design, and the snapshot is not a probability sample of city housing markets.

## 2. Asking total prices are strongly right-skewed

**Finding.** The mean asking total price (145.8 × CNY 10,000) exceeds the median (105.0 × CNY 10,000).

**Evidence.** P10/P90 are 49.8 and 268.0 × CNY 10,000; the IQR is 94.0. 99.2% of platform unit prices are within 1 CNY/m² of price ÷ area after rounding. Only 6 rows differ by more than 1% in relative terms. See `figures/price_distribution.png`.

**Limitation.** These are asking prices; the right tail does not show completed sale values, appraisal values, or eventual outcomes. The small unit-price mismatch tail remains a field-definition/data-quality exception, not proof that the two fields are identical.

## 3. Area segments differ in both total and unit asking-price structure

**Finding.** Median total asking price rises from 52.5 × CNY 10,000 in the <60 m² bucket to 185.0 × CNY 10,000 in the 144+ m² bucket.

**Evidence.** Median calculated unit price is not monotonic across area buckets: it ranges from 9,431 CNY/m² for 120–<144 m² to 10,904 CNY/m² for <60 m². The figure also reports IQRs. See `figures/price_by_area_bucket.png`.

**Limitation.** Area buckets also differ in city and property composition; the comparison is descriptive and not an isolated area effect.

## 4. City price structure remains heterogeneous within a narrower property segment

**Finding.** Among cities meeting the sample threshold for 60–90 m², 2–3-room listings, median calculated unit price still varies materially.

**Evidence.** Comparable-segment medians range from 3,822 CNY/m² in 宝鸡 (n=426) to 54,545 CNY/m² in 上海 (n=961). See `figures/city_price_structure.png`.

**Limitation.** Matching only on area and room count does not control for neighborhood, building age, condition, or other unobserved composition.

## 5. Platform snapshot signals vary across asking-price buckets

**Finding.** Median follower count changes from 1 in the <100 bucket to 5 in the 500+ bucket; median encoded listing age changes from 180 to 90 days.

**Evidence.** The same two fields are summarized by price and area buckets with medians and IQRs in `figures/listing_engagement.png`.

**Limitation.** Follower count and listing age are coarse, platform-observed snapshot fields. They do not establish buyer behavior, sale probability, or speed of sale.
