WITH baseline_daily AS (
    SELECT 
        p.id as product_id,
        p.name as product_name,
        s.id as store_id,
        s.name as store_name,
        AVG(daily_units) as baseline_daily_units
    FROM (
        SELECT 
            ti.product_id,
            t.store_id,
            DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as sale_date,
            SUM(ti.quantity) as daily_units
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '30 days'
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') < (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '7 days'
        GROUP BY ti.product_id, t.store_id, DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')
    ) daily_sales
    JOIN products p ON daily_sales.product_id = p.id
    JOIN stores s ON daily_sales.store_id = s.id
    GROUP BY p.id, p.name, s.id, s.name
    HAVING AVG(daily_units) > 0
),

last_7d_sales AS (
    SELECT 
        ti.product_id,
        t.store_id,
        SUM(ti.quantity) as last_7d_units
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '7 days'
    GROUP BY ti.product_id, t.store_id
),

current_inventory AS (
    SELECT 
        i.product_id,
        i.store_id,
        COALESCE(i.quantity_on_hand, 0) as store_on_hand
    FROM inventory i
    /* __WAREHOUSE_FILTER__ */
),

warehouse_inventory AS (
    SELECT 
        i.product_id,
        COALESCE(SUM(i.quantity_on_hand), 0) as warehouse_on_hand
    FROM inventory i
    JOIN stores s ON i.store_id = s.id
    WHERE s.name IN ('Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview')
    GROUP BY i.product_id
)

SELECT 
    bd.product_name,
    bd.store_name,
    bd.baseline_daily_units,
    COALESCE(l7d.last_7d_units, 0) as last_7d_units,
    (bd.baseline_daily_units * 7) - COALESCE(l7d.last_7d_units, 0) as hidden_demand_units,
    COALESCE(ci.store_on_hand, 0) as store_on_hand,
    COALESCE(wi.warehouse_on_hand, 0) as warehouse_on_hand,
    LEAST(
        (bd.baseline_daily_units * 7) - COALESCE(l7d.last_7d_units, 0),
        COALESCE(wi.warehouse_on_hand, 0)
    ) as suggested_transfer_units
FROM baseline_daily bd
LEFT JOIN last_7d_sales l7d ON bd.product_id = l7d.product_id AND bd.store_id = l7d.store_id
LEFT JOIN current_inventory ci ON bd.product_id = ci.product_id AND bd.store_id = ci.store_id
LEFT JOIN warehouse_inventory wi ON bd.product_id = wi.product_id
WHERE (bd.baseline_daily_units * 7) - COALESCE(l7d.last_7d_units, 0) > 0
ORDER BY hidden_demand_units DESC, product_name ASC
