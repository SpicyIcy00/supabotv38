-- Hourly sales analysis for current day
-- Name: get_hourly_sales
WITH latest_date AS (
    SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
    FROM transactions 
    WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
)
SELECT
    EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
    TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
    COALESCE(SUM(t.total), 0) as total_sales
FROM transactions t
CROSS JOIN latest_date ld
WHERE LOWER(t.transaction_type) = 'sale' 
AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
GROUP BY 
    EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'),
    TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
ORDER BY hour;

