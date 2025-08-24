"""
Sales data repository for handling all sales-related database operations.
Converted to use Polars for data processing. Converts to pandas only when needed by UI.
"""

from typing import Optional, List, Dict, Any

import polars as pl
import pandas as pd  # For UI compatibility in type hints/returns
import streamlit as st
from .base import BaseRepository


class SalesRepository(BaseRepository):
    """Repository for sales-related data operations."""
    
    @st.cache_data(ttl=300)
    def get_latest_metrics(_self) -> Optional[pd.DataFrame]:
        """Get latest sales metrics for current day."""
        sql = _self.load_sql_query('sales_metrics.sql')
        # DB layer returns pandas; keep as-is for UI layers consuming tables
        return _self.execute_query(sql)
    
    @st.cache_data(ttl=300)
    def get_previous_metrics(_self) -> Optional[pd.DataFrame]:
        """Get previous day sales metrics for comparison."""
        sql = _self.load_sql_query('previous_metrics.sql')
        return _self.execute_query(sql)
    
    @st.cache_data(ttl=300)
    def get_hourly_sales(_self) -> Optional[pd.DataFrame]:
        """Get hourly sales analysis for current day."""
        sql = _self.load_sql_query('hourly_sales.sql')
        return _self.execute_query(sql)
    
    @st.cache_data(ttl=300)
    def get_filtered_metrics(_self, time_filter: str = "7D", store_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get filtered metrics with time range and store filters."""
        interval = _self.get_time_filter_interval(time_filter)
        
        def _get_metrics_for_period(start_interval: str, end_interval: str):
            sql = _self.load_sql_query('filtered_metrics.sql')
            
            # Format store clause
            store_clause = ""
            params = [start_interval, end_interval]
            if store_ids:
                store_clause = "AND t.store_id = ANY(%s)"
                params.append(store_ids)
            
            # Replace placeholder in SQL
            sql = sql.format(store_clause=store_clause)
            return _self.execute_query(sql, params)
        
        # Current and previous periods
        df_current_pd = _get_metrics_for_period(interval, '0 seconds')
        df_previous_pd = _get_metrics_for_period(f'2 * {interval}', interval)
        df_current = pl.from_pandas(df_current_pd) if df_current_pd is not None and not df_current_pd.empty else pl.DataFrame()
        df_previous = pl.from_pandas(df_previous_pd) if df_previous_pd is not None and not df_previous_pd.empty else pl.DataFrame()
        
        metrics = df_current.row(0) if df_current.height > 0 else None
        metrics = dict(zip(df_current.columns, metrics)) if metrics is not None else {}
        previous_metrics_row = df_previous.row(0) if df_previous.height > 0 else None
        previous_metrics = dict(zip(df_previous.columns, previous_metrics_row)) if previous_metrics_row is not None else {}
        
        # Calculate percentage changes
        for key in ['sales', 'profit', 'transactions']:
            current_val = metrics.get(key, 0)
            previous_val = previous_metrics.get(key, 0)
            if previous_val > 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
                metrics[f'{key}_change_pct'] = change_pct
            else:
                metrics[f'{key}_change_pct'] = 0
        
        return metrics
    
    @st.cache_data(ttl=300)
    def get_dashboard_metrics(_self, time_filter: str = "7D", store_filter_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get dashboard metrics calculation."""
        days_map = {"1D": 1, "7D": 7, "1M": 30, "6M": 180, "1Y": 365}
        days = days_map.get(time_filter, 7)
        
        sql = _self.load_sql_query('dashboard_metrics.sql')
        
        # Format store clause
        store_clause = ""
        params = [f'{days} days', f'{days*2} days', f'{days} days']
        if store_filter_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.extend([store_filter_ids, store_filter_ids])
        
        # Replace placeholder in SQL
        sql = sql.format(store_clause=store_clause)
        
        result_pd = _self.execute_query(sql, params)
        if result_pd is None or result_pd.empty:
            return {}
        result_pl = pl.from_pandas(result_pd)
        row0 = result_pl.row(0)
        return dict(zip(result_pl.columns, row0))
    
    @st.cache_data(ttl=300)
    def get_top_sellers(_self, time_filter: str = "7D", store_filter_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
        """Get top selling products analysis."""
        days_map = {"1D": 1, "7D": 7, "1M": 30, "6M": 180, "1Y": 365}
        days = days_map.get(time_filter, 7)
        
        sql = _self.load_sql_query('top_sellers.sql')
        
        # Format store clause
        store_clause = ""
        params = [f'{days} days']
        if store_filter_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        # Replace placeholder in SQL
        sql = sql.format(store_clause=store_clause)
        
        return _self.execute_query(sql, params)
    
    @st.cache_data(ttl=300)
    def get_store_performance(_self) -> Optional[pd.DataFrame]:
        """Get store performance analysis."""
        sql = _self.load_sql_query('store_performance.sql')
        return _self.execute_query(sql)


# Global singleton instance
_sales_repository = None

def get_sales_repository() -> SalesRepository:
    """Get the global SalesRepository instance."""
    global _sales_repository
    if _sales_repository is None:
        _sales_repository = SalesRepository()
    return _sales_repository

