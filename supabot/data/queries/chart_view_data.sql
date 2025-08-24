-- Chart view data for analytics
-- Name: get_chart_view_data
-- Parameters: time_agg, base_name_sql, series_name_sql, metric_calculation_sql, time_condition, store_filter_sql, metric_filter_sql, group_by_sql
SELECT
    {time_agg} AS date,
    {base_name_sql} AS base_name,
    {series_name_sql} AS series_name,
    s.name AS store_name,
    {metric_calculation_sql}
FROM transaction_items ti
JOIN transactions t ON ti.transaction_ref_id = t.ref_id
JOIN products p ON ti.product_id = p.id
JOIN stores s ON t.store_id = s.id
WHERE LOWER(t.transaction_type) = 'sale'
AND COALESCE(t.is_cancelled, false) = false
{time_condition}
{store_filter_sql}
{metric_filter_sql}
{group_by_sql}
HAVING COUNT(t.ref_id) > 0 -- Ensure there's data to avoid division by zero
ORDER BY 1, 3;

