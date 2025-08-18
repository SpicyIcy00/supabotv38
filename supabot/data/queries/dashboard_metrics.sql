-- Dashboard metrics calculation
-- Name: get_dashboard_metrics
-- Parameters: days (interval), store_ids (optional)
WITH current_period AS (
    SELECT 
        SUM(t.total) as sales,
        COUNT(DISTINCT t.ref_id) as transactions,
        AVG(t.total) as avg_transaction_value
    FROM transactions t
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
    {store_clause}
),
previous_period AS (
    SELECT 
        SUM(t.total) as prev_sales,
        COUNT(DISTINCT t.ref_id) as prev_transactions
    FROM transactions t
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s 
    AND (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
    {store_clause}
)
SELECT 
    COALESCE(cp.sales, 0) as current_sales,
    COALESCE(cp.sales * 0.65, 0) as current_profit,
    COALESCE(cp.transactions, 0) as current_transactions,
    COALESCE(cp.avg_transaction_value, 0) as avg_transaction_value,
    COALESCE(pp.prev_sales, 0) as prev_sales,
    COALESCE(pp.prev_sales * 0.65, 0) as prev_profit,
    COALESCE(pp.prev_transactions, 0) as prev_transactions
FROM current_period cp
CROSS JOIN previous_period pp;

