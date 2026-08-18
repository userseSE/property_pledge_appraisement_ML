-- name: listing_signal_segments
WITH segments AS (
    SELECT 'asking_total_price_bucket' AS dimension, asking_total_price_bucket AS segment,
           follower_count, listing_age_days
    FROM listings
    UNION ALL
    SELECT 'area_bucket', area_bucket, follower_count, listing_age_days
    FROM listings
),
eligible_groups AS (
    SELECT dimension, segment, COUNT(*) AS sample_count
    FROM segments
    GROUP BY dimension, segment
    HAVING COUNT(*) >= :min_group_size
),
values_long AS (
    SELECT source.dimension, source.segment, eligible.sample_count,
           'follower_count' AS metric, CAST(source.follower_count AS REAL) AS value
    FROM segments AS source
    INNER JOIN eligible_groups AS eligible USING (dimension, segment)
    UNION ALL
    SELECT source.dimension, source.segment, eligible.sample_count,
           'listing_age_days', CAST(source.listing_age_days AS REAL)
    FROM segments AS source
    INNER JOIN eligible_groups AS eligible USING (dimension, segment)
),
ranked AS (
    SELECT
        dimension,
        segment,
        sample_count,
        metric,
        value,
        ROW_NUMBER() OVER (PARTITION BY dimension, segment, metric ORDER BY value) AS row_number
    FROM values_long
)
SELECT
    dimension,
    segment,
    MAX(sample_count) AS sample_count,
    metric,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.25 AS INTEGER) + 1 THEN value END) AS p25,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN value END) AS median,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.75 AS INTEGER) + 1 THEN value END) AS p75
FROM ranked
GROUP BY dimension, segment, metric
ORDER BY dimension, segment, metric;

-- name: listing_age_distribution
SELECT listing_age_days, COUNT(*) AS sample_count
FROM listings
GROUP BY listing_age_days
HAVING COUNT(*) >= :min_group_size
ORDER BY listing_age_days;
