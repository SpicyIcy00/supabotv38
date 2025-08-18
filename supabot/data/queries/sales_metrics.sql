-- Latest sales metrics for current day
-- Name: get_latest_metrics
WITH latest_date AS (
    SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
    FROM transactions 
    WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
)
SELECT 
    COALESCE(SUM(t.total), 0) as latest_sales,
    COUNT(DISTINCT t.ref_id) as latest_transactions
FROM transactions t
CROSS JOIN latest_date ld
WHERE LOWER(transaction_type) = 'sale' 
AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date;

