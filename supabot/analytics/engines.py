"""
Analytics engines for advanced business intelligence and insights.
Converted to use Polars for data processing. Converts to pandas only for UI compatibility.
"""

from typing import Dict, Any, Optional, List

import polars as pl
import pandas as pd  # For UI return compatibility
import streamlit as st
from supabot.core.database import get_db_manager
from supabot.data.repositories.analytics import get_analytics_repository


class AIAnalyticsEngine:
    """Advanced analytics engine for hidden demand, stockouts, and trends."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.analytics_repo = get_analytics_repository()
    
    def _execute_query(self, sql: str, params: Optional[List] = None) -> pl.DataFrame:
        """Execute SQL query, receive pandas from DB layer, convert to Polars."""
        result_pd = self.db_manager.execute_query_for_dashboard(sql, params)
        if result_pd is None:
            return pl.DataFrame()
        try:
            return pl.from_pandas(result_pd)
        except Exception:
            # Fallback if conversion fails
            return pl.DataFrame()
    
    @st.cache_data(ttl=3600)
    def detect_hidden_demand(_self, days_back: int = 90) -> pd.DataFrame:
        """Detect hidden demand using AI analytics."""
        return _self.analytics_repo.detect_hidden_demand(days_back)
    
    @st.cache_data(ttl=1800)
    def analyze_seasonal_trends(_self, product_category: str = None) -> Dict[str, Any]:
        """Analyze seasonal trends and patterns."""
        category_filter = ""
        params = []
        if product_category:
            category_filter = "AND p.category = %s"
            params.append(product_category)
        
        sql = f"""
        WITH monthly_sales AS (
            SELECT 
                EXTRACT(MONTH FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as month,
                EXTRACT(YEAR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as year,
                p.category,
                SUM(ti.quantity * ti.price) as monthly_revenue,
                SUM(ti.quantity) as monthly_quantity
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND t.transaction_time >= NOW() - INTERVAL '2 years'
            {category_filter}
            GROUP BY EXTRACT(MONTH FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), 
                     EXTRACT(YEAR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), 
                     p.category
        )
        SELECT 
            month,
            AVG(monthly_revenue) as avg_monthly_revenue,
            AVG(monthly_quantity) as avg_monthly_quantity,
            STDDEV(monthly_revenue) as revenue_volatility
        FROM monthly_sales
        GROUP BY month
        ORDER BY month
        """
        
        df = _self._execute_query(sql, params)
        
        if df.height == 0:
            return {"trends": [], "insights": []}
        
        # Calculate seasonal insights
        # Use Polars to compute peaks
        peak_idx = df.select(pl.col("avg_monthly_revenue").arg_max()).item()
        low_idx = df.select(pl.col("avg_monthly_revenue").arg_min()).item()
        peak_month = df.row(peak_idx)[df.columns.index("month")] if df.height > 0 else None
        low_month = df.row(low_idx)[df.columns.index("month")] if df.height > 0 else None
        
        insights = []
        if peak_month:
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                          5: 'May', 6: 'June', 7: 'July', 8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            insights.append(f"Peak sales month: {month_names.get(peak_month, peak_month)}")
            insights.append(f"Lowest sales month: {month_names.get(low_month, low_month)}")
        
        return {
            "trends": df.to_dicts(),
            "insights": insights,
            "peak_month": peak_month,
            "low_month": low_month
        }
    
    @st.cache_data(ttl=1800)
    def analyze_customer_behavior(_self, days_back: int = 90) -> Dict[str, Any]:
        """Analyze customer behavior patterns."""
        sql = """
        WITH transaction_patterns AS (
            SELECT 
                EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour_of_day,
                EXTRACT(DOW FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as day_of_week,
                COUNT(DISTINCT t.ref_id) as transaction_count,
                AVG(t.total) as avg_transaction_value,
                SUM(t.total) as total_revenue
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND t.transaction_time >= NOW() - INTERVAL %s
            GROUP BY 
                EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'),
                EXTRACT(DOW FROM t.transaction_time AT TIME ZONE 'Asia/Manila')
        ),
        hourly_summary AS (
            SELECT 
                hour_of_day,
                SUM(transaction_count) as total_transactions,
                AVG(avg_transaction_value) as avg_value
            FROM transaction_patterns
            GROUP BY hour_of_day
            ORDER BY total_transactions DESC
        ),
        daily_summary AS (
            SELECT 
                day_of_week,
                SUM(transaction_count) as total_transactions,
                AVG(avg_transaction_value) as avg_value
            FROM transaction_patterns
            GROUP BY day_of_week
            ORDER BY total_transactions DESC
        )
        SELECT 
            'hourly' as pattern_type,
            hour_of_day as period,
            total_transactions,
            avg_value
        FROM hourly_summary
        UNION ALL
        SELECT 
            'daily' as pattern_type,
            day_of_week as period,
            total_transactions,
            avg_value
        FROM daily_summary
        """
        
        df = _self._execute_query(sql, [f'{days_back} days'])
        
        if df.height == 0:
            return {"patterns": [], "insights": []}
        
        # Separate hourly and daily patterns
        hourly_data = (
            df.filter(pl.col("pattern_type") == "hourly")
              .sort(pl.col("total_transactions"), descending=True)
        )
        daily_data = (
            df.filter(pl.col("pattern_type") == "daily")
              .sort(pl.col("total_transactions"), descending=True)
        )
        
        insights = []
        if hourly_data.height > 0:
            peak_hour = hourly_data.row(0)[hourly_data.columns.index("period")]
            insights.append(f"Peak hour: {int(peak_hour)}:00")
        
        if daily_data.height > 0:
            day_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                        4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
            peak_day = day_names.get(daily_data.row(0)[daily_data.columns.index("period")], 'Unknown')
            insights.append(f"Peak day: {peak_day}")
        
        return {
            "hourly_patterns": hourly_data.to_dicts(),
            "daily_patterns": daily_data.to_dicts(),
            "insights": insights
        }
    
    def generate_recommendations(self, hidden_demand_df: pd.DataFrame, 
                               seasonal_data: Dict, behavior_data: Dict) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Hidden demand recommendations
        if hidden_demand_df is not None and not hidden_demand_df.empty:
            urgent_restocks = hidden_demand_df[hidden_demand_df['recommendation'] == 'URGENT_RESTOCK']
            if urgent_restocks is not None and len(urgent_restocks) > 0:
                recommendations.append(
                    f"{len(urgent_restocks)} products need urgent restocking"
                )
        
        # Seasonal recommendations
        if seasonal_data.get('peak_month') is not None:
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                          5: 'May', 6: 'June', 7: 'July', 8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            peak_month_name = month_names.get(seasonal_data['peak_month'])
            recommendations.append(
                f"Prepare for peak season in {peak_month_name}"
            )
        
        # Behavior-based recommendations
        hourly_patterns = behavior_data.get('hourly_patterns', [])
        if hourly_patterns:
            peak_hour = hourly_patterns[0]['period']
            recommendations.append(
                f"Schedule more staff around {int(peak_hour)}:00 (peak hour)"
            )
        
        # Inventory optimization
        if hidden_demand_df is not None and not hidden_demand_df.empty:
            low_stock_count = len(hidden_demand_df[hidden_demand_df['current_stock'] == 0])
            if low_stock_count > 5:
                recommendations.append(
                    f"Review inventory management - {low_stock_count} items out of stock"
                )
        
        return recommendations[:5]  # Return top 5 recommendations


# Global singleton instance
_ai_analytics_engine = None


def get_ai_analytics_engine() -> AIAnalyticsEngine:
    """Get the global AIAnalyticsEngine instance."""
    global _ai_analytics_engine
    if _ai_analytics_engine is None:
        _ai_analytics_engine = AIAnalyticsEngine()
    return _ai_analytics_engine

