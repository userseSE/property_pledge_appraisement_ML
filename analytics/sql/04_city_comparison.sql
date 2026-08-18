-- name: city_price_structure
WITH city_counts AS (
    SELECT city_id, COUNT(*) AS sample_count
    FROM listings
    GROUP BY city_id
    HAVING COUNT(*) >= :min_group_size
),
values_long AS (
    SELECT source.city_id, counts.sample_count, 'area_sqm' AS metric, source.area_sqm AS value
    FROM listings AS source INNER JOIN city_counts AS counts USING (city_id)
    UNION ALL
    SELECT source.city_id, counts.sample_count, 'asking_total_price_10k_cny', source.asking_total_price_10k_cny
    FROM listings AS source INNER JOIN city_counts AS counts USING (city_id)
    UNION ALL
    SELECT source.city_id, counts.sample_count, 'calculated_unit_price_cny_sqm', source.calculated_unit_price_cny_sqm
    FROM listings AS source INNER JOIN city_counts AS counts USING (city_id)
),
ranked AS (
    SELECT
        city_id,
        sample_count,
        metric,
        value,
        ROW_NUMBER() OVER (PARTITION BY city_id, metric ORDER BY value) AS row_number
    FROM values_long
)
SELECT
    city_id,
    MAX(sample_count) AS sample_count,
    metric,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.25 AS INTEGER) + 1 THEN value END) AS p25,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN value END) AS median,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.75 AS INTEGER) + 1 THEN value END) AS p75
FROM ranked
GROUP BY city_id, metric
ORDER BY city_id, metric;

-- name: comparable_city_segment
WITH comparable AS (
    SELECT *
    FROM listings
    WHERE area_sqm >= 60
      AND area_sqm <= 90
      AND rooms IN (2, 3)
),
city_counts AS (
    SELECT city_id, COUNT(*) AS sample_count
    FROM comparable
    GROUP BY city_id
    HAVING COUNT(*) >= :min_group_size
),
ranked AS (
    SELECT
        source.city_id,
        counts.sample_count,
        source.calculated_unit_price_cny_sqm AS value,
        ROW_NUMBER() OVER (
            PARTITION BY source.city_id
            ORDER BY source.calculated_unit_price_cny_sqm
        ) AS row_number
    FROM comparable AS source
    INNER JOIN city_counts AS counts USING (city_id)
)
SELECT
    city_id,
    MAX(sample_count) AS sample_count,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.25 AS INTEGER) + 1 THEN value END) AS p25_unit_price_cny_sqm,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.50 AS INTEGER) + 1 THEN value END) AS median_unit_price_cny_sqm,
    MAX(CASE WHEN row_number = CAST((sample_count - 1) * 0.75 AS INTEGER) + 1 THEN value END) AS p75_unit_price_cny_sqm
FROM ranked
GROUP BY city_id
ORDER BY median_unit_price_cny_sqm DESC, city_id;
