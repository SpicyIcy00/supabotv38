"""
Analytics repository for advanced analytics and AI-driven insights.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from .base import BaseRepository


class AnalyticsRepository(BaseRepository):
    """Repository for analytics and AI-driven insights."""
    
    @st.cache_data(ttl=3600)
    def detect_hidden_demand(_self, days_back: int = 90) -> Optional[pd.DataFrame]:
        """Detect hidden demand using AI analytics."""
        sql = _self.load_sql_query('hidden_demand.sql')
        params = [f'{days_back} days']
        return _self.execute_query(sql, params)
    
    def get_chart_view_data(self, time_range: str, metric_type: str, granularity: str, 
                           filters: Dict, store_filters: List[str]) -> Optional[pd.DataFrame]:
        """Get data for chart view with dynamic parameters."""
        
        # Time aggregation mapping
        time_agg_map = {
            "Daily": "DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')",
            "Weekly": "DATE_TRUNC('week', t.transaction_time AT TIME ZONE 'Asia/Manila')",
            "Monthly": "DATE_TRUNC('month', t.transaction_time AT TIME ZONE 'Asia/Manila')"
        }
        
        # Metric calculation mapping
        metric_calc_map = {
            "Revenue": "SUM(ti.quantity * ti.price) as metric_value",
            "Quantity": "SUM(ti.quantity) as metric_value",
            "Transactions": "COUNT(DISTINCT t.ref_id) as metric_value",
            "Avg Transaction Value": "AVG(t.total) as metric_value"
        }
        
        # Build dynamic SQL parts
        time_agg = time_agg_map.get(granularity, time_agg_map["Daily"])
        metric_calculation_sql = metric_calc_map.get(metric_type, metric_calc_map["Revenue"])
        
        # Base and series name SQL
        base_name_sql = "p.name"
        series_name_sql = "COALESCE(p.category, 'Uncategorized')"
        
        # Time condition
        days_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
        days = days_map.get(time_range, 30)
        time_condition = f"AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '{days} days'"
        
        # Store filter SQL
        store_filter_sql = ""
        params = []
        if store_filters and "All Stores" not in store_filters:
            store_filter_sql = "AND s.name = ANY(%s)"
            params.append(store_filters)
        
        # Metric filter SQL (from filters dict)
        metric_filter_sql = ""
        if filters.get('category'):
            metric_filter_sql += "AND p.category = %s"
            params.append(filters['category'])
        
        # Group by SQL
        group_by_sql = f"GROUP BY {time_agg}, p.name, p.category, s.name"
        
        # Load and format SQL template
        sql_template = self.load_sql_query('chart_view_data.sql')
        sql = sql_template.format(
            time_agg=time_agg,
            base_name_sql=base_name_sql,
            series_name_sql=series_name_sql,
            metric_calculation_sql=metric_calculation_sql,
            time_condition=time_condition,
            store_filter_sql=store_filter_sql,
            metric_filter_sql=metric_filter_sql,
            group_by_sql=group_by_sql
        )
        
        df = self.execute_query(sql, tuple(params))
        
        # Rename metric column for consistency
        if df is not None and 'metric_value' in df.columns:
            df.rename(columns={'metric_value': 'total_revenue'}, inplace=True)
            
        return df
    
    def get_business_highlights(self, metrics: Dict, top_sellers_df: pd.DataFrame, time_filter: str) -> List[str]:
        """Generate business highlights from metrics and data."""
        highlights = []
        
        # Sales performance
        current_sales = metrics.get('current_sales', 0)
        prev_sales = metrics.get('prev_sales', 0)
        
        if prev_sales > 0:
            sales_change = ((current_sales - prev_sales) / prev_sales) * 100
            if sales_change > 10:
                highlights.append(f"📈 Sales increased by {sales_change:.1f}% vs previous period")
            elif sales_change < -10:
                highlights.append(f"📉 Sales decreased by {abs(sales_change):.1f}% vs previous period")
        
        # Top seller insights
        if not top_sellers_df.empty:
            top_product = top_sellers_df.iloc[0]
            highlights.append(f"🏆 Top seller: {top_product['product_name']} (₱{top_product['total_revenue']:,.0f})")
            
            # Category insights
            if 'category' in top_sellers_df.columns:
                top_category = top_sellers_df.groupby('category')['total_revenue'].sum().idxmax()
                highlights.append(f"📊 Leading category: {top_category}")
        
        # Transaction insights
        current_transactions = metrics.get('current_transactions', 0)
        avg_transaction = metrics.get('avg_transaction_value', 0)
        
        if avg_transaction > 0:
            highlights.append(f"💰 Average transaction: ₱{avg_transaction:.0f}")
        
        if current_transactions > 0:
            highlights.append(f"🛒 Total transactions: {current_transactions:,}")
        
        return highlights[:5]  # Limit to 5 highlights


# Global singleton instance
_analytics_repository = None

def get_analytics_repository() -> AnalyticsRepository:
    """Get the global AnalyticsRepository instance."""
    global _analytics_repository
    if _analytics_repository is None:
        _analytics_repository = AnalyticsRepository()
    return _analytics_repository

