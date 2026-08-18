-- name: overview
SELECT
    COUNT(*) AS listing_count,
    COUNT(DISTINCT city_id) AS city_count,
    MIN(area_sqm) AS min_area_sqm,
    MAX(area_sqm) AS max_area_sqm,
    MIN(asking_total_price_10k_cny) AS min_asking_total_price_10k_cny,
    MAX(asking_total_price_10k_cny) AS max_asking_total_price_10k_cny,
    MIN(rooms) AS min_rooms,
    MAX(rooms) AS max_rooms,
    MIN(halls) AS min_halls,
    MAX(halls) AS max_halls
FROM listings;

-- name: coverage_statistics
WITH values_long AS (
    SELECT 'area_sqm' AS metric, area_sqm AS value FROM listings
    UNION ALL
    SELECT 'asking_total_price_10k_cny', asking_total_price_10k_cny FROM listings
    UNION ALL
    SELECT 'rooms', CAST(rooms AS REAL) FROM listings
    UNION ALL
    SELECT 'halls', CAST(halls AS REAL) FROM listings
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
    MIN(value) AS minimum,
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

-- name: city_counts
SELECT city_id, COUNT(*) AS sample_count
FROM listings
GROUP BY city_id
ORDER BY sample_count DESC, city_id;

-- name: room_hall_counts
SELECT rooms, halls, COUNT(*) AS sample_count
FROM listings
GROUP BY rooms, halls
HAVING COUNT(*) >= :min_group_size
ORDER BY sample_count DESC, rooms, halls;

-- name: area_distribution
SELECT
    CASE
        WHEN area_sqm < 40 THEN 'lt_40'
        WHEN area_sqm >= 200 THEN '200_plus'
        ELSE CAST(CAST(area_sqm / 10 AS INTEGER) * 10 AS TEXT)
             || '_to_lt_'
             || CAST((CAST(area_sqm / 10 AS INTEGER) + 1) * 10 AS TEXT)
    END AS area_interval,
    CASE
        WHEN area_sqm < 40 THEN 0
        WHEN area_sqm >= 200 THEN 200
        ELSE CAST(area_sqm / 10 AS INTEGER) * 10
    END AS sort_order,
    COUNT(*) AS sample_count
FROM listings
GROUP BY area_interval, sort_order
HAVING COUNT(*) >= :min_group_size
ORDER BY sort_order;
