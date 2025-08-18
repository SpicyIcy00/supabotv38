-- Store performance analysis
-- Name: get_store_performance
WITH latest_date AS (
    SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
    FROM transactions 
    WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
)
SELECT
    s.name as store_name,
    s.id as store_id,
    COALESCE(SUM(t.total), 0) as total_sales,
    COUNT(DISTINCT t.ref_id) as transaction_count,
    COALESCE(AVG(t.total), 0) as avg_transaction_value
FROM stores s
LEFT JOIN transactions t ON s.id = t.store_id 
    AND LOWER(t.transaction_type) = 'sale'
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = (SELECT max_date FROM latest_date)
GROUP BY s.name, s.id
ORDER BY total_sales DESC;

