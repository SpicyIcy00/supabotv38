"""
Mobile KPI cards component for responsive dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List


class MobileKPICards:
    """Mobile-optimized KPI cards component."""
    
    @staticmethod
    def render_kpi_grid(metrics: Dict[str, Any], time_filter: str):
        """
        Render KPI metrics in a responsive grid layout.
        
        Args:
            metrics: Dictionary containing KPI data
            time_filter: Current time filter being applied
        """
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Create responsive grid layout
        with st.container():
            st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
            
            # Mobile: 2x2 grid, Tablet: 2x2, Desktop: 1x4
            if st.session_state.get('screen_size', 'desktop') == 'mobile':
                MobileKPICards._render_mobile_kpi_grid(metrics, time_filter)
            elif st.session_state.get('screen_size', 'desktop') == 'tablet':
                MobileKPICards._render_tablet_kpi_grid(metrics, time_filter)
            else:
                MobileKPICards._render_desktop_kpi_grid(metrics, time_filter)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _render_mobile_kpi_grid(metrics: Dict[str, Any], time_filter: str):
        """Render KPI grid optimized for mobile (2x2 layout)."""
        # First row: Sales and Profit
        col1, col2 = st.columns(2)
        
        with col1:
            MobileKPICards._render_kpi_card(
                "Sales", 
                f"₱{metrics.get('current_sales', 0):,.0f}",
                metrics.get('current_sales', 0),
                metrics.get('prev_sales', 0),
                time_filter
            )
        
        with col2:
            MobileKPICards._render_kpi_card(
                "Profit", 
                f"₱{metrics.get('current_profit', 0):,.0f}",
                metrics.get('current_profit', 0),
                metrics.get('prev_profit', 0),
                time_filter
            )
        
        # Second row: Transactions and Avg Value
        col3, col4 = st.columns(2)
        
        with col3:
            MobileKPICards._render_kpi_card(
                "Transactions", 
                f"{metrics.get('current_transactions', 0):,}",
                metrics.get('current_transactions', 0),
                metrics.get('prev_transactions', 0),
                time_filter
            )
        
        with col4:
            MobileKPICards._render_kpi_card(
                "Avg Transaction", 
                f"₱{metrics.get('avg_transaction_value', 0):,.0f}",
                metrics.get('avg_transaction_value', 0),
                metrics.get('prev_avg_transaction_value', 0),
                time_filter
            )
    
    @staticmethod
    def _render_tablet_kpi_grid(metrics: Dict[str, Any], time_filter: str):
        """Render KPI grid optimized for tablet (2x2 layout)."""
        # Same as mobile for now, but with larger touch targets
        MobileKPICards._render_mobile_kpi_grid(metrics, time_filter)
    
    @staticmethod
    def _render_desktop_kpi_grid(metrics: Dict[str, Any], time_filter: str):
        """Render KPI grid optimized for desktop (1x4 layout)."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            MobileKPICards._render_kpi_card(
                f"Sales ({time_filter})", 
                f"₱{metrics.get('current_sales', 0):,.0f}",
                metrics.get('current_sales', 0),
                metrics.get('prev_sales', 0),
                time_filter
            )
        
        with col2:
            MobileKPICards._render_kpi_card(
                f"Profit ({time_filter})", 
                f"₱{metrics.get('current_profit', 0):,.0f}",
                metrics.get('current_profit', 0),
                metrics.get('prev_profit', 0),
                time_filter
            )
        
        with col3:
            MobileKPICards._render_kpi_card(
                "Transactions", 
                f"{metrics.get('current_transactions', 0):,}",
                metrics.get('current_transactions', 0),
                metrics.get('prev_transactions', 0),
                time_filter
            )
        
        with col4:
            MobileKPICards._render_kpi_card(
                "Avg Transaction", 
                f"₱{metrics.get('avg_transaction_value', 0):,.0f}",
                metrics.get('avg_transaction_value', 0),
                metrics.get('prev_avg_transaction_value', 0),
                time_filter
            )
    
    @staticmethod
    def _render_kpi_card(
        title: str, 
        value: str, 
        current: float, 
        previous: float, 
        time_filter: str
    ):
        """
        Render individual KPI card with mobile-optimized styling.
        
        Args:
            title: KPI title
            value: Formatted value string
            current: Current period value
            previous: Previous period value
            time_filter: Time filter label
        """
        delta = MobileKPICards._calculate_percentage_change(current, previous)
        
        # Use custom HTML for better mobile styling
        if st.session_state.get('screen_size', 'desktop') == 'mobile':
            MobileKPICards._render_mobile_kpi_card_html(title, value, delta)
        else:
            # Use standard Streamlit metric for desktop
            st.metric(
                label=title,
                value=value,
                delta=f"{delta:+.1f}%" if delta is not None else None
            )
    
    @staticmethod
    def _render_mobile_kpi_card_html(title: str, value: str, delta: Optional[float]):
        """Render KPI card using custom HTML for mobile optimization."""
        delta_html = ""
        if delta is not None:
            arrow = "↗" if delta > 0 else "↘"
            color = "#00c853" if delta > 0 else "#ff5252"
            delta_html = f"""
            <div style="color: {color}; font-size: 0.8rem; font-weight: 500;">
                {arrow} {abs(delta):.1f}%
            </div>
            """
        
        card_html = f"""
        <div class="mobile-kpi-card" style="
            background: #1c1e26;
            border: 1px solid #2e303d;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
        ">
            <div style="
                font-size: 0.85rem;
                color: #c7c7c7;
                margin-bottom: 0.5rem;
                line-height: 1.2;
            ">{title}</div>
            <div style="
                font-size: 1.5rem;
                font-weight: bold;
                color: #00d2ff;
                margin-bottom: 0.3rem;
                line-height: 1.1;
            ">{value}</div>
            {delta_html}
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def _calculate_percentage_change(current: float, previous: float) -> Optional[float]:
        """
        Calculate percentage change between current and previous values.
        
        Args:
            current: Current period value
            previous: Previous period value
            
        Returns:
            Percentage change or None if calculation not possible
        """
        if previous is None or previous == 0:
            return None
        
        if current is None:
            current = 0
        
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def render_kpi_summary(metrics: Dict[str, Any], time_filter: str):
        """
        Render a summary view of KPIs for mobile devices.
        
        Args:
            metrics: Dictionary containing KPI data
            time_filter: Current time filter being applied
        """
        if not metrics:
            return
        
        with st.container():
            st.markdown("### 📊 Performance Summary")
            
            # Create a compact summary view
            summary_data = [
                {
                    "metric": "Sales",
                    "value": f"₱{metrics.get('current_sales', 0):,.0f}",
                    "change": MobileKPICards._calculate_percentage_change(
                        metrics.get('current_sales', 0),
                        metrics.get('prev_sales', 0)
                    )
                },
                {
                    "metric": "Profit", 
                    "value": f"₱{metrics.get('current_profit', 0):,.0f}",
                    "change": MobileKPICards._calculate_percentage_change(
                        metrics.get('current_profit', 0),
                        metrics.get('prev_profit', 0)
                    )
                },
                {
                    "metric": "Transactions",
                    "value": f"{metrics.get('current_transactions', 0):,}",
                    "change": MobileKPICards._calculate_percentage_change(
                        metrics.get('current_transactions', 0),
                        metrics.get('prev_transactions', 0)
                    )
                }
            ]
            
            for item in summary_data:
                MobileKPICards._render_summary_item(
                    item["metric"], 
                    item["value"], 
                    item["change"]
                )
    
    @staticmethod
    def _render_summary_item(metric: str, value: str, change: Optional[float]):
        """Render individual summary item."""
        change_html = ""
        if change is not None:
            arrow = "↗" if change > 0 else "↘"
            color = "#00c853" if change > 0 else "#ff5252"
            change_html = f"""
            <span style="color: {color}; font-size: 0.75rem; margin-left: 0.5rem;">
                {arrow} {abs(change):.1f}%
            </span>
            """
        
        item_html = f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: #1c1e26;
            border: 1px solid #2e303d;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        ">
            <span style="color: #c7c7c7; font-size: 0.9rem;">{metric}</span>
            <div style="text-align: right;">
                <div style="color: #00d2ff; font-weight: bold; font-size: 1rem;">
                    {value}
                </div>
                {change_html}
            </div>
        </div>
        """
        
        st.markdown(item_html, unsafe_allow_html=True)
