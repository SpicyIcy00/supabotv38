"""
Base repository class with common SQL loading and execution functionality.
"""

import os
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List
from supabot.core.database import get_db_manager


class BaseRepository:
    """Base class for all data repositories."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries')
    
    def load_sql_query(self, filename: str) -> str:
        """Load SQL query from file."""
        sql_path = os.path.join(self.queries_dir, filename)
        try:
            with open(sql_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
    
    def execute_query(self, sql: str, params: Optional[List] = None) -> Optional[pd.DataFrame]:
        """Execute SQL query with parameters."""
        return self.db_manager.execute_query_for_dashboard(sql, params)
    
    def format_store_clause(self, store_ids: Optional[List[int]] = None) -> str:
        """Format store filter clause for SQL queries."""
        if store_ids:
            return "AND t.store_id = ANY(%(store_ids)s)"
        return ""
    
    def get_time_interval_mapping(self) -> Dict[str, str]:
        """Get time filter interval mapping."""
        return {
            "1D": "1 day", 
            "7D": "7 days", 
            "1M": "30 days", 
            "6M": "180 days", 
            "1Y": "365 days"
        }
    
    def get_time_filter_interval(self, time_filter: str = "7d") -> str:
        """Convert time filter to SQL interval."""
        mapping = self.get_time_interval_mapping()
        return mapping.get(time_filter, "7 days")

