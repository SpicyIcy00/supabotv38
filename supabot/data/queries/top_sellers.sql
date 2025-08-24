-- Top selling products analysis
-- Name: get_dashboard_top_sellers
-- Parameters: days (interval), store_ids (optional)
SELECT 
    p.name as product_name,
    p.category,
    SUM(ti.quantity) as total_quantity,
    SUM(ti.quantity * ti.price) as total_revenue,
    AVG(ti.price) as avg_price,
    COUNT(DISTINCT t.ref_id) as transaction_count
FROM transaction_items ti
JOIN transactions t ON ti.transaction_ref_id = t.ref_id
JOIN products p ON ti.product_id = p.id
WHERE LOWER(t.transaction_type) = 'sale'
AND COALESCE(t.is_cancelled, false) = false
AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
{store_clause}
GROUP BY p.name, p.category
ORDER BY total_revenue DESC
LIMIT 10;

