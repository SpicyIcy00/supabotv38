"""
Metrics and KPI components for the dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List


class MetricsDisplay:
    """Component for displaying KPI metrics and highlights."""
    
    @staticmethod
    def render_kpi_metrics(metrics: Dict[str, Any], time_filter: str):
        """Render KPI metrics in a grid layout."""
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Create 4-column layout for KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_sales = metrics.get('current_sales', 0)
            prev_sales = metrics.get('prev_sales', 0)
            sales_change = MetricsDisplay._calculate_percentage_change(current_sales, prev_sales)
            
            st.metric(
                label=f"Sales ({time_filter})",
                value=f"₱{current_sales:,.0f}",
                delta=f"{sales_change:+.1f}%" if sales_change is not None else None
            )
        
        with col2:
            current_profit = metrics.get('current_profit', 0)
            prev_profit = metrics.get('prev_profit', 0)
            profit_change = MetricsDisplay._calculate_percentage_change(current_profit, prev_profit)
            
            st.metric(
                label=f"Profit ({time_filter})",
                value=f"₱{current_profit:,.0f}",
                delta=f"{profit_change:+.1f}%" if profit_change is not None else None
            )
        
        with col3:
            current_transactions = metrics.get('current_transactions', 0)
            prev_transactions = metrics.get('prev_transactions', 0)
            txn_change = MetricsDisplay._calculate_percentage_change(current_transactions, prev_transactions)
            
            st.metric(
                label=f"Transactions ({time_filter})",
                value=f"{current_transactions:,}",
                delta=f"{txn_change:+.1f}%" if txn_change is not None else None
            )
        
        with col4:
            avg_value = metrics.get('avg_transaction_value', 0)
            
            st.metric(
                label=f"Avg Transaction",
                value=f"₱{avg_value:.0f}",
                delta=None  # No comparison for average
            )
    
    @staticmethod
    def render_business_highlights(highlights: List[str]):
        """Render business highlights section."""
        if not highlights:
            return
        
        st.subheader("📊 Business Highlights")
        
        for highlight in highlights:
            st.markdown(f"• {highlight}")
    
    @staticmethod
    def render_top_sellers(df: pd.DataFrame, title: str = "Top Sellers"):
        """Render top sellers table."""
        if df is None or df.empty:
            st.warning("No top sellers data available")
            return
        
        st.subheader(f"🏆 {title}")
        
        # Format the dataframe for display
        display_df = df.copy()
        
        # Format currency columns
        currency_cols = ['total_revenue', 'avg_price']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"₱{x:,.0f}")
        
        # Format quantity columns
        quantity_cols = ['total_quantity', 'transaction_count']
        for col in quantity_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,}")
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def render_store_performance(df: pd.DataFrame):
        """Render store performance metrics."""
        if df is None or df.empty:
            st.warning("No store performance data available")
            return
        
        st.subheader("🏪 Store Performance")
        
        # Create columns for each store
        stores = df.head(4)  # Show top 4 stores
        cols = st.columns(len(stores))
        
        for idx, (_, store) in enumerate(stores.iterrows()):
            with cols[idx]:
                st.metric(
                    label=store['store_name'],
                    value=f"₱{store['total_sales']:,.0f}",
                    delta=f"{store['transaction_count']} txns"
                )
    
    @staticmethod
    def _calculate_percentage_change(current: float, previous: float) -> Optional[float]:
        """Calculate percentage change between two values."""
        if previous == 0 or previous is None:
            return None
        return ((current - previous) / previous) * 100


class FilterComponents:
    """Components for filters and controls."""
    
    @staticmethod
    def render_time_filter(default_value: str = "7D") -> str:
        """Render time filter selector."""
        return st.selectbox(
            "Time Range",
            options=["1D", "7D", "1M", "6M", "1Y"],
            index=1,  # Default to 7D
            key="time_filter"
        )
    
    @staticmethod
    def render_store_filter(available_stores: List[str]) -> List[str]:
        """Render store multi-select filter."""
        return st.multiselect(
            "Select Stores",
            options=available_stores,
            default=["All Stores"] if "All Stores" in available_stores else available_stores[:1],
            key="store_filter"
        )
    
    @staticmethod
    def render_page_navigation() -> str:
        """Render page navigation in sidebar."""
        st.sidebar.title("🧠 SupaBot BI")
        
        pages = [
            "Dashboard",
            "Smart Reports", 
            "Chart View",
            "AI Assistant",
            "AI Intelligence Hub",
            "Settings"
        ]
        
        # Use radio buttons for navigation
        selected_page = st.sidebar.radio(
            "Navigate to:",
            pages,
            key="navigation"
        )
        
        return selected_page


# Utility functions for legacy compatibility
def render_kpi_metrics(metrics: Dict[str, Any], time_filter: str):
    """Legacy compatibility function."""
    MetricsDisplay.render_kpi_metrics(metrics, time_filter)

def render_business_highlights(highlights: List[str]):
    """Legacy compatibility function."""
    MetricsDisplay.render_business_highlights(highlights)

