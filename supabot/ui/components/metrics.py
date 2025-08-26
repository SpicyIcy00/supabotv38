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
        """Render KPI metrics in a responsive grid layout."""
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Use responsive columns - 4 on desktop, 2 on mobile
        # Streamlit automatically handles responsive behavior
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
    def render_kpi_metrics_mobile(metrics: Dict[str, Any], time_filter: str):
        """Render KPI metrics optimized for mobile devices."""
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Mobile-optimized layout with 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales metric
            current_sales = metrics.get('current_sales', 0)
            prev_sales = metrics.get('prev_sales', 0)
            sales_change = MetricsDisplay._calculate_percentage_change(current_sales, prev_sales)
            
            st.metric(
                label=f"Sales ({time_filter})",
                value=f"₱{current_sales:,.0f}",
                delta=f"{sales_change:+.1f}%" if sales_change is not None else None
            )
            
            # Transactions metric
            current_transactions = metrics.get('current_transactions', 0)
            prev_transactions = metrics.get('prev_transactions', 0)
            txn_change = MetricsDisplay._calculate_percentage_change(current_transactions, prev_transactions)
            
            st.metric(
                label=f"Transactions ({time_filter})",
                value=f"{current_transactions:,}",
                delta=f"{txn_change:+.1f}%" if txn_change is not None else None
            )
        
        with col2:
            # Profit metric
            current_profit = metrics.get('current_profit', 0)
            prev_profit = metrics.get('prev_profit', 0)
            profit_change = MetricsDisplay._calculate_percentage_change(current_profit, prev_profit)
            
            st.metric(
                label=f"Profit ({time_filter})",
                value=f"₱{current_profit:,.0f}",
                delta=f"{profit_change:+.1f}%" if profit_change is not None else None
            )
            
            # Average transaction metric
            avg_value = metrics.get('avg_transaction_value', 0)
            
            st.metric(
                label=f"Avg Transaction",
                value=f"₱{avg_value:.0f}",
                delta=None
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
        """Render top sellers table with mobile optimization."""
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
        
        # Display table with mobile optimization
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def render_store_performance(df: pd.DataFrame):
        """Render store performance metrics with responsive layout."""
        if df is None or df.empty:
            st.warning("No store performance data available")
            return
        
        st.subheader("🏪 Store Performance")
        
        # Create responsive columns for store performance
        stores = df.head(4)  # Show top 4 stores
        
        # Use responsive columns that adapt to screen size
        if len(stores) <= 2:
            cols = st.columns(len(stores))
        else:
            # For 3-4 stores, use 2x2 grid on mobile
            cols = st.columns(2)
        
        for idx, (_, store) in enumerate(stores.iterrows()):
            col_idx = idx % 2 if len(stores) > 2 else idx
            with cols[col_idx]:
                st.metric(
                    label=store['store_name'],
                    value=f"₱{store['total_sales']:,.0f}",
                    delta=f"{store['transaction_count']} txns"
                )
    
    @staticmethod
    def render_store_performance_mobile(df: pd.DataFrame):
        """Render store performance metrics optimized for mobile."""
        if df is None or df.empty:
            st.warning("No store performance data available")
            return
        
        st.subheader("🏪 Store Performance")
        
        # Mobile-optimized: show stores in a single column
        stores = df.head(4)  # Show top 4 stores
        
        for _, store in stores.iterrows():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{store['store_name']}**")
            with col2:
                st.write(f"₱{store['total_sales']:,.0f}")
            st.write(f"*{store['transaction_count']} transactions*")
            st.markdown("---")
    
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
        """Render time filter selector with mobile optimization."""
        return st.selectbox(
            "Time Range",
            options=["1D", "7D", "1M", "6M", "1Y"],
            index=1,  # Default to 7D
            key="time_filter"
        )
    
    @staticmethod
    def render_time_filter_mobile(default_value: str = "7D") -> str:
        """Render time filter selector optimized for mobile."""
        # Use radio buttons for better mobile experience
        return st.radio(
            "Time Range",
            options=["1D", "7D", "1M", "6M", "1Y"],
            index=1,  # Default to 7D
            horizontal=True,
            key="time_filter_mobile"
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
    def render_store_filter_mobile(available_stores: List[str]) -> List[str]:
        """Render store filter optimized for mobile."""
        # Use selectbox for single selection on mobile for better UX
        selected = st.selectbox(
            "Select Store",
            options=["All Stores"] + available_stores,
            index=0,  # Default to "All Stores"
            key="store_filter_mobile"
        )
        return [selected] if selected else []
    
    @staticmethod
    def render_page_navigation() -> str:
        """Render page navigation in sidebar."""
        st.sidebar.title("🧠 SupaBot BI")
        
        pages = [
            "Dashboard",
            "Smart Reports", 
            "Chart View",
            "AI Assistant",
            "Settings"
        ]
        
        # Use radio buttons for navigation
        selected_page = st.sidebar.radio(
            "Navigate to:",
            pages,
            key="navigation"
        )
        
        return selected_page
    
    @staticmethod
    def render_page_navigation_mobile() -> str:
        """Render page navigation optimized for mobile."""
        st.sidebar.title("🧠 SupaBot BI")
        
        pages = [
            "Dashboard",
            "Smart Reports", 
            "Chart View",
            "AI Assistant",
            "Settings"
        ]
        
        # Use selectbox for mobile navigation
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            pages,
            key="navigation_mobile"
        )
        
        return selected_page


# Utility functions for legacy compatibility
def render_kpi_metrics(metrics: Dict[str, Any], time_filter: str):
    """Legacy compatibility function."""
    MetricsDisplay.render_kpi_metrics(metrics, time_filter)

def render_business_highlights(highlights: List[str]):
    """Legacy compatibility function."""
    MetricsDisplay.render_business_highlights(highlights)

