-- name: price_statistics
WITH values_long AS (
    SELECT 'asking_total_price_10k_cny' AS metric, asking_total_price_10k_cny AS value FROM listings
    UNION ALL
    SELECT 'calculated_unit_price_cny_sqm', calculated_unit_price_cny_sqm FROM listings
),
ranked AS (
    SELECT
        metric,
        value,
        ROW_NUMBER() OVER (PARTITION BY metric ORDER BY value) AS row_number,
        COUNT(*) OVER (PARTITION BY metric) AS sample_count
    FROM values_long
)
SELECT
    metric,
    MAX(sample_count) AS sample_count,
    AVG(value) AS mean,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.10 AS INTEGER) + 1 THEN value END) AS p10,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.25 AS INTEGER) + 1 THEN value END) AS p25,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN value END) AS median,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.75 AS INTEGER) + 1 THEN value END) AS p75,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.90 AS INTEGER) + 1 THEN value END) AS p90,
    MAX(value) AS maximum
FROM ranked
GROUP BY metric
ORDER BY metric;

-- name: unit_price_sanity
WITH differences AS (
    SELECT
        ABS(platform_unit_price_cny_sqm - calculated_unit_price_cny_sqm) AS absolute_difference,
        ABS(platform_unit_price_cny_sqm - calculated_unit_price_cny_sqm)
            / calculated_unit_price_cny_sqm AS relative_difference
    FROM listings
),
ranked AS (
    SELECT
        absolute_difference,
        relative_difference,
        ROW_NUMBER() OVER (ORDER BY absolute_difference) AS row_number,
        ROW_NUMBER() OVER (ORDER BY relative_difference) AS relative_row_number,
        COUNT(*) OVER () AS sample_count
    FROM differences
)
SELECT
    MAX(sample_count) AS sample_count,
    AVG(absolute_difference) AS mean_absolute_difference_cny_sqm,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN absolute_difference END) AS median_absolute_difference_cny_sqm,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.90 AS INTEGER) + 1 THEN absolute_difference END) AS p90_absolute_difference_cny_sqm,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.99 AS INTEGER) + 1 THEN absolute_difference END) AS p99_absolute_difference_cny_sqm,
    MAX(absolute_difference) AS max_absolute_difference_cny_sqm,
    MAX(CASE WHEN relative_row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN relative_difference END) AS median_relative_difference,
    MAX(CASE WHEN relative_row_number = CAST((sample_count - 1) * 0.99 AS INTEGER) + 1 THEN relative_difference END) AS p99_relative_difference,
    MAX(relative_difference) AS max_relative_difference,
    SUM(CASE WHEN absolute_difference <= 1 THEN 1 ELSE 0 END) AS within_1_cny_sqm,
    SUM(CASE WHEN absolute_difference <= 5 THEN 1 ELSE 0 END) AS within_5_cny_sqm,
    SUM(CASE WHEN absolute_difference > 100 THEN 1 ELSE 0 END) AS above_100_cny_sqm,
    SUM(CASE WHEN relative_difference <= 0.001 THEN 1 ELSE 0 END) AS within_0_1_percent,
    SUM(CASE WHEN relative_difference > 0.01 THEN 1 ELSE 0 END) AS above_1_percent
FROM ranked;

-- name: log_price_distribution
WITH values_long AS (
    SELECT 'asking_total_price_10k_cny' AS metric, asking_total_price_10k_cny AS value FROM listings
    UNION ALL
    SELECT 'calculated_unit_price_cny_sqm', calculated_unit_price_cny_sqm FROM listings
),
binned AS (
    SELECT
        metric,
        CAST(LOG10(value) * 20 AS INTEGER) AS bin_index
    FROM values_long
    WHERE value > 0
)
SELECT
    metric,
    bin_index,
    POW10(bin_index / 20.0) AS lower_bound,
    POW10((bin_index + 1) / 20.0) AS upper_bound,
    COUNT(*) AS sample_count
FROM binned
GROUP BY metric, bin_index
HAVING COUNT(*) >= :min_group_size
ORDER BY metric, bin_index;
