#!/usr/bin/env python3
"""
Script to fix the malformed code in appv38.py
"""

def fix_appv38():
    # Read the file
    with open('appv38.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic section
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if 'return f"Error generating AI intelligence summary: {e}"' in line:
            start_line = i + 1
        elif line.strip() == '# Advanced Analytics Page':
            end_line = i
            break
    
    if start_line is None or end_line is None:
        print("Could not find the problematic section")
        return
    
    # Create the fixed content
    fixed_lines = lines[:start_line]
    
    # Add the correct function
    fixed_lines.extend([
        '\n',
        '@st.cache_data(ttl=3600)\n',
        'def detect_hidden_demand_advanced(lookback_days=30):\n',
        '   sql = """\n',
        '   WITH weekly_sales AS (\n',
        '       SELECT \n',
        '           p.name as product_name,\n',
        '           s.name as store_name,\n',
        '           p.category,\n',
        '           DATE_TRUNC(\'week\', t.transaction_time AT TIME ZONE \'Asia/Manila\') as week,\n',
        '           SUM(ti.quantity) as weekly_qty,\n',
        '           p.id as product_id,\n',
        '           t.store_id\n',
        '       FROM transaction_items ti\n',
        '       JOIN transactions t ON ti.transaction_ref_id = t.ref_id\n',
        '       JOIN products p ON ti.product_id = p.id\n',
        '       JOIN stores s ON t.store_id = s.id\n',
        '       WHERE LOWER(t.transaction_type) = \'sale\' \n',
        '       AND t.is_cancelled = false\n',
        '       AND t.transaction_time >= CURRENT_DATE - INTERVAL %s\n',
        '       GROUP BY p.name, s.name, p.category, week, p.id, t.store_id\n',
        '   ),\n',
        '   demand_analysis AS (\n',
        '       SELECT \n',
        '           product_name, store_name, category, product_id, store_id,\n',
        '           AVG(weekly_qty) as avg_weekly_demand,\n',
        '           COUNT(*) as weeks_with_sales,\n',
        '           EXTRACT(WEEK FROM CURRENT_DATE) - EXTRACT(WEEK FROM MAX(week)) as weeks_since_last_sale\n',
        '       FROM weekly_sales\n',
        '       GROUP BY product_name, store_name, category, product_id, store_id\n',
        '       HAVING AVG(weekly_qty) >= 1.0\n',
        '   )\n',
        '   SELECT \n',
        '       da.product_name,\n',
        '       da.store_name,\n',
        '       da.category,\n',
        '       ROUND(da.avg_weekly_demand, 2) as avg_weekly_demand,\n',
        '       da.weeks_since_last_sale,\n',
        '       COALESCE(i.quantity_on_hand, 0) as current_stock,\n',
        '       LEAST(100, GREATEST(0, \n',
        '           (da.avg_weekly_demand * 20) + \n',
        '           (CASE WHEN da.weeks_since_last_sale > 2 THEN 30 ELSE 0 END) +\n',
        '           (CASE WHEN COALESCE(i.quantity_on_hand, 0) = 0 THEN 35 ELSE 0 END) +\n',
        '           (CASE WHEN COALESCE(i.quantity_on_hand, 0) <= da.avg_weekly_demand THEN 15 ELSE 0 END)\n',
        '       )) as hidden_demand_score,\n',
        '       CASE \n',
        '           WHEN COALESCE(i.quantity_on_hand, 0) = 0 AND da.avg_weekly_demand > 2 THEN \'URGENT_RESTOCK\'\n',
        '           WHEN da.weeks_since_last_sale > 3 THEN \'INVESTIGATE_STOCKOUT\'\n',
        '           ELSE \'MONITOR\'\n',
        '       END as recommendation\n',
        '   FROM demand_analysis da\n',
        '   LEFT JOIN inventory i ON da.product_id = i.product_id AND da.store_id = i.store_id\n',
        '   WHERE da.weeks_since_last_sale >= 1\n',
        '   ORDER BY hidden_demand_score DESC\n',
        '   LIMIT 50\n',
        '   """\n',
        '   \n',
        '   return execute_query_for_dashboard(sql, (f"{lookback_days} days",))\n',
        '\n',
        '\n',
    ])
    
    # Add the rest of the file
    fixed_lines.extend(lines[end_line:])
    
    # Write the fixed file
    with open('appv38_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Fixed file created as appv38_fixed.py")
    print(f"Removed {end_line - start_line} problematic lines")
    print("Added detect_hidden_demand_advanced function")

if __name__ == "__main__":
    fix_appv38()
