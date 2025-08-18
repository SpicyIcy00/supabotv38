-- Filtered metrics with time range and store filters
-- Name: get_filtered_metrics_period
-- Parameters: start_interval, end_interval, store_ids (optional)
WITH period_transactions AS (
    SELECT ref_id, total, store_id
    FROM transactions t
    WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s
    {store_clause}
),
period_costs AS (
    SELECT
        SUM(ti.quantity * p.cost) as total_cogs
    FROM transaction_items ti
    JOIN products p ON ti.product_id = p.id
    WHERE ti.transaction_ref_id IN (SELECT ref_id FROM period_transactions)
)
SELECT
    (SELECT COALESCE(SUM(total), 0) FROM period_transactions) as sales,
    (SELECT COALESCE(SUM(total), 0) FROM period_transactions) - (SELECT COALESCE(total_cogs, 0) FROM period_costs) as profit,
    (SELECT COUNT(ref_id) FROM period_transactions) as transactions,
    (SELECT COUNT(DISTINCT store_id) FROM period_transactions) as active_stores,
    (SELECT COALESCE(SUM(total), 0) / NULLIF(COUNT(ref_id), 0) FROM period_transactions) as avg_transaction_value;

