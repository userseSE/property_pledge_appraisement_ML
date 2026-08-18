-- name: property_segment_statistics
WITH eligible_groups AS (
    SELECT CAST({{dimension}} AS TEXT) AS segment, COUNT(*) AS sample_count
    FROM listings
    GROUP BY CAST({{dimension}} AS TEXT)
    HAVING COUNT(*) >= :min_group_size
),
values_long AS (
    SELECT
        CAST(source.{{dimension}} AS TEXT) AS segment,
        eligible.sample_count,
        'asking_total_price_10k_cny' AS metric,
        source.asking_total_price_10k_cny AS value
    FROM listings AS source
    INNER JOIN eligible_groups AS eligible
        ON CAST(source.{{dimension}} AS TEXT) = eligible.segment
    UNION ALL
    SELECT
        CAST(source.{{dimension}} AS TEXT),
        eligible.sample_count,
        'calculated_unit_price_cny_sqm',
        source.calculated_unit_price_cny_sqm
    FROM listings AS source
    INNER JOIN eligible_groups AS eligible
        ON CAST(source.{{dimension}} AS TEXT) = eligible.segment
),
ranked AS (
    SELECT
        segment,
        sample_count,
        metric,
        value,
        ROW_NUMBER() OVER (PARTITION BY segment, metric ORDER BY value) AS row_number
    FROM values_long
)
SELECT
    segment,
    MAX(sample_count) AS sample_count,
    metric,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.25 AS INTEGER) + 1 THEN value END) AS p25,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN value END) AS median,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.75 AS INTEGER) + 1 THEN value END) AS p75
FROM ranked
GROUP BY segment, metric
ORDER BY segment, metric;
