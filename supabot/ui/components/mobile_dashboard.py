"""
Mobile dashboard integration for responsive SupaBot BI Dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Callable

from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
from supabot.ui.components.mobile.kpi_cards import MobileKPICards
from supabot.ui.components.mobile.product_list import MobileProductList
from supabot.ui.components.mobile.charts import MobileCharts
from supabot.ui.components.mobile.navigation import MobileNavigation


class MobileDashboard:
    """Comprehensive mobile dashboard integration."""
    
    @staticmethod
    def render_responsive_dashboard(
        metrics: Dict[str, Any],
        sales_df: pd.DataFrame,
        sales_cat_df: pd.DataFrame,
        inv_cat_df: pd.DataFrame,
        top_change_df: pd.DataFrame,
        cat_change_df: pd.DataFrame,
        time_filter: str,
        selected_stores: List[str]
    ):
        """
        Render the complete dashboard with responsive design.
        
        Args:
            metrics: KPI metrics data
            sales_df: Sales trend data
            sales_cat_df: Sales by category data
            inv_cat_df: Inventory by category data
            top_change_df: Top products with change data
            cat_change_df: Categories with change data
            time_filter: Current time filter
            selected_stores: Selected stores
        """
        # Load mobile-responsive CSS
        from mobile_responsive_css import get_mobile_responsive_css
        st.markdown(get_mobile_responsive_css(), unsafe_allow_html=True)
        
        # Detect screen size and set session state
        screen_size = ResponsiveWrapper.get_screen_size()
        st.session_state.screen_size = screen_size
        
        # Render mobile header if on mobile
        if screen_size == 'mobile':
            MobileDashboard._render_mobile_header()
        
        # Render KPI section with responsive design
        MobileDashboard._render_kpi_section(metrics, time_filter)
        
        # Render charts section with responsive design
        MobileDashboard._render_charts_section(sales_df, sales_cat_df, inv_cat_df)
        
        # Render products section with responsive design
        MobileDashboard._render_products_section(top_change_df, cat_change_df)
        
        # Render mobile navigation
        if screen_size == 'mobile':
            MobileDashboard._render_mobile_navigation()
    
    @staticmethod
    def _render_mobile_header():
        """Render mobile-optimized header."""
        MobileNavigation.render_mobile_header("SupaBot BI Dashboard")
        MobileNavigation.render_pull_to_refresh()
    
    @staticmethod
    def _render_kpi_section(metrics: Dict[str, Any], time_filter: str):
        """Render KPI section with responsive layout."""
        st.markdown("## 🚀 Key Performance Indicators")
        
        # Use responsive KPI grid with CSS classes
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile KPI grid with custom CSS
            st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
            
            # Check if we have real data
            has_real_data = metrics.get('current_sales', 0) > 0 or metrics.get('current_transactions', 0) > 0
            
            if has_real_data:
                # Sales KPI
                sales_change = MobileKPICards._calculate_percentage_change(metrics.get('current_sales', 0), metrics.get('prev_sales', 0))
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Sales ({time_filter})</h3>
                    <div class="value">₱{metrics.get('current_sales', 0):,.0f}</div>
                    <div class="change {'positive' if sales_change and sales_change >= 0 else 'negative'}">
                        {f'{sales_change:+.1f}%' if sales_change is not None else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Profit KPI
                profit_change = MobileKPICards._calculate_percentage_change(metrics.get('current_profit', 0), metrics.get('prev_profit', 0))
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Profit ({time_filter})</h3>
                    <div class="value">₱{metrics.get('current_profit', 0):,.0f}</div>
                    <div class="change {'positive' if profit_change and profit_change >= 0 else 'negative'}">
                        {f'{profit_change:+.1f}%' if profit_change is not None else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Transactions KPI
                trans_change = MobileKPICards._calculate_percentage_change(metrics.get('current_transactions', 0), metrics.get('prev_transactions', 0))
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Transactions</h3>
                    <div class="value">{metrics.get('current_transactions', 0):,}</div>
                    <div class="change {'positive' if trans_change and trans_change >= 0 else 'negative'}">
                        {f'{trans_change:+.1f}%' if trans_change is not None else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Products Sold KPI
                products_change = MobileKPICards._calculate_percentage_change(metrics.get('current_products_sold', 0), metrics.get('prev_products_sold', 0))
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Products Sold</h3>
                    <div class="value">{metrics.get('current_products_sold', 0):,}</div>
                    <div class="change {'positive' if products_change and products_change >= 0 else 'negative'}">
                        {f'{products_change:+.1f}%' if products_change is not None else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show sample data when no real data is available
                sample_kpis = [
                    {"title": "Sales (7D)", "value": "₱2,450,000", "change": "+12.5%"},
                    {"title": "Profit (7D)", "value": "₱490,000", "change": "+8.7%"},
                    {"title": "Transactions", "value": "1,250", "change": "+15.2%"},
                    {"title": "Products Sold", "value": "3,750", "change": "+22.1%"}
                ]
                
                for kpi in sample_kpis:
                    change_class = 'positive' if '+' in kpi['change'] else 'negative'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <h3>{kpi['title']} (Sample)</h3>
                        <div class="value">{kpi['value']}</div>
                        <div class="change {change_class}">{kpi['change']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Desktop layout with columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"Sales ({time_filter})",
                    f"₱{metrics.get('current_sales', 0):,.0f}",
                    f"{MobileKPICards._calculate_percentage_change(metrics.get('current_sales', 0), metrics.get('prev_sales', 0)):+.1f}%" if MobileKPICards._calculate_percentage_change(metrics.get('current_sales', 0), metrics.get('prev_sales', 0)) is not None else None
                )
            
            with col2:
                st.metric(
                    f"Profit ({time_filter})",
                    f"₱{metrics.get('current_profit', 0):,.0f}",
                    f"{MobileKPICards._calculate_percentage_change(metrics.get('current_profit', 0), metrics.get('prev_profit', 0)):+.1f}%" if MobileKPICards._calculate_percentage_change(metrics.get('current_profit', 0), metrics.get('prev_profit', 0)) is not None else None
                )
            
            with col3:
                st.metric(
                    "Transactions",
                    f"{metrics.get('current_transactions', 0):,}",
                    f"{MobileKPICards._calculate_percentage_change(metrics.get('current_transactions', 0), metrics.get('prev_transactions', 0)):+.1f}%" if MobileKPICards._calculate_percentage_change(metrics.get('current_transactions', 0), metrics.get('prev_transactions', 0)) is not None else None
                )
            
            with col4:
                st.metric(
                    "Avg Transaction",
                    f"₱{metrics.get('avg_transaction_value', 0):,.0f}",
                    f"{MobileKPICards._calculate_percentage_change(metrics.get('avg_transaction_value', 0), metrics.get('prev_avg_transaction_value', 0)):+.1f}%" if MobileKPICards._calculate_percentage_change(metrics.get('avg_transaction_value', 0), metrics.get('prev_avg_transaction_value', 0)) is not None else None
                )
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_charts_section(sales_df: pd.DataFrame, sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame):
        """Render charts section with responsive layout."""
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile charts with responsive CSS
            st.markdown('<div class="charts-container">', unsafe_allow_html=True)
            
            # Sales trend chart
            if not sales_df.empty:
                st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                st.markdown('<h3>📈 Sales Trend</h3>', unsafe_allow_html=True)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                MobileCharts.render_sales_trend_chart(sales_df, "📈 Sales Trend")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sales by category chart
            if not sales_cat_df.empty:
                st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                st.markdown('<h3>📊 Sales by Category</h3>', unsafe_allow_html=True)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                MobileCharts.render_pie_charts(sales_cat_df, inv_cat_df)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Desktop layout
            st.markdown("## 📊 Analytics & Visualizations")
            
            # Sales trend chart
            MobileCharts.render_sales_trend_chart(sales_df, "📈 Sales Trend")
            
            # Pie charts
            MobileCharts.render_pie_charts(sales_cat_df, inv_cat_df)
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_products_section(top_change_df: pd.DataFrame, cat_change_df: pd.DataFrame):
        """Render products section with responsive layout."""
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile product list with custom CSS
            if not top_change_df.empty:
                st.markdown('<div class="mobile-product-list">', unsafe_allow_html=True)
                st.markdown('<h3>🏆 Top 10 Products (with % change)</h3>', unsafe_allow_html=True)
                
                for index, row in top_change_df.head(10).iterrows():
                    product_name = row.get('product_name', 'Unknown Product')
                    total_revenue = row.get('total_revenue', 0)
                    percentage_change = row.get('percentage_change', 0)
                    
                    change_class = 'positive' if percentage_change >= 0 else 'negative'
                    change_symbol = '+' if percentage_change >= 0 else ''
                    
                    st.markdown(f"""
                    <div class="product-card-mobile">
                        <div class="product-header">
                            <span class="rank">#{index + 1}</span>
                            <span class="product-name">{product_name}</span>
                        </div>
                        <div class="product-stats">
                            <span class="sales">₱{total_revenue:,.0f}</span>
                            <span class="change {change_class}">{change_symbol}{percentage_change:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Show sample data when no real data is available
                sample_products = [
                    {"name": "iPhone 15 Pro", "revenue": 1250000, "change": 12.5},
                    {"name": "Samsung Galaxy S24", "revenue": 980000, "change": -3.2},
                    {"name": "MacBook Pro M3", "revenue": 850000, "change": 8.7},
                    {"name": "iPad Air", "revenue": 720000, "change": 15.3},
                    {"name": "AirPods Pro", "revenue": 650000, "change": 22.1}
                ]
                
                st.markdown('<div class="mobile-product-list">', unsafe_allow_html=True)
                st.markdown('<h3>🏆 Top 10 Products (Sample Data)</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #888; font-size: 0.9rem; margin-bottom: 16px;">Showing sample data - no real data available for selected period/stores.</p>', unsafe_allow_html=True)
                
                for index, product in enumerate(sample_products):
                    change_class = 'positive' if product['change'] >= 0 else 'negative'
                    change_symbol = '+' if product['change'] >= 0 else ''
                    
                    st.markdown(f"""
                    <div class="product-card-mobile">
                        <div class="product-header">
                            <span class="rank">#{index + 1}</span>
                            <span class="product-name">{product['name']}</span>
                        </div>
                        <div class="product-stats">
                            <span class="sales">₱{product['revenue']:,.0f}</span>
                            <span class="change {change_class}">{change_symbol}{product['change']:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Mobile category list
            if not cat_change_df.empty:
                st.markdown('<div class="mobile-category-list">', unsafe_allow_html=True)
                st.markdown('<h3>🗂️ Categories Ranked (with % change)</h3>', unsafe_allow_html=True)
                
                for index, row in cat_change_df.head(10).iterrows():
                    category_name = row.get('category_name', 'Unknown Category')
                    total_revenue = row.get('total_revenue', 0)
                    percentage_change = row.get('percentage_change', 0)
                    
                    change_class = 'positive' if percentage_change >= 0 else 'negative'
                    change_symbol = '+' if percentage_change >= 0 else ''
                    
                    st.markdown(f"""
                    <div class="category-card-mobile">
                        <div class="category-name">{category_name}</div>
                        <div class="category-stats">
                            <span class="sales">₱{total_revenue:,.0f}</span>
                            <span class="change {change_class}">{change_symbol}{percentage_change:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Show sample data when no real data is available
                sample_categories = [
                    {"name": "Smartphones", "revenue": 2800000, "change": 15.2},
                    {"name": "Laptops", "revenue": 2100000, "change": 8.7},
                    {"name": "Accessories", "revenue": 1800000, "change": 22.5},
                    {"name": "Tablets", "revenue": 1200000, "change": -5.3},
                    {"name": "Gaming", "revenue": 950000, "change": 12.8}
                ]
                
                st.markdown('<div class="mobile-category-list">', unsafe_allow_html=True)
                st.markdown('<h3>🗂️ Categories Ranked (Sample Data)</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #888; font-size: 0.9rem; margin-bottom: 16px;">Showing sample data - no real data available for selected period/stores.</p>', unsafe_allow_html=True)
                
                for index, category in enumerate(sample_categories):
                    change_class = 'positive' if category['change'] >= 0 else 'negative'
                    change_symbol = '+' if category['change'] >= 0 else ''
                    
                    st.markdown(f"""
                    <div class="category-card-mobile">
                        <div class="category-name">{category['name']}</div>
                        <div class="category-stats">
                            <span class="sales">₱{category['revenue']:,.0f}</span>
                            <span class="change {change_class}">{change_symbol}{category['change']:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Desktop layout
            st.markdown("## 🏆 Top Performers")
            
            # Top products
            MobileProductList.render_product_list(top_change_df, "🏆 Top 10 Products")
            
            # Categories
            MobileProductList.render_category_list(cat_change_df, "🗂️ Categories Ranked")
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_mobile_navigation():
        """Render mobile navigation elements."""
        # Bottom navigation tabs
        tabs = [
            {"key": "dashboard", "label": "Dashboard", "icon": "📊"},
            {"key": "analytics", "label": "Analytics", "icon": "📈"},
            {"key": "products", "label": "Products", "icon": "🏆"},
            {"key": "settings", "label": "Settings", "icon": "⚙️"}
        ]
        
        MobileNavigation.render_bottom_navigation(tabs, "dashboard")
        
        # Mobile menu
        MobileNavigation.render_mobile_menu()
    
    @staticmethod
    def render_mobile_filters():
        """Render mobile-optimized filters."""
        MobileNavigation.render_mobile_filters_section()
    
    @staticmethod
    def render_mobile_summary(metrics: Dict[str, Any], top_change_df: pd.DataFrame):
        """Render mobile summary view."""
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        with st.container():
            st.markdown("## 📱 Mobile Summary")
            
            # KPI summary
            MobileKPICards.render_kpi_summary(metrics, "Current Period")
            
            # Top products summary
            MobileProductList.render_compact_summary(top_change_df, max_items=5)
            
            # Quick actions
            actions = [
                {
                    "icon": "📊",
                    "label": "Full Dashboard",
                    "help": "View complete dashboard",
                    "callback": lambda: st.rerun()
                },
                {
                    "icon": "📈",
                    "label": "Analytics",
                    "help": "View detailed analytics",
                    "callback": lambda: st.rerun()
                },
                {
                    "icon": "🏆",
                    "label": "Top Products",
                    "help": "View top performing products",
                    "callback": lambda: st.rerun()
                },
                {
                    "icon": "⚙️",
                    "label": "Settings",
                    "help": "Configure dashboard settings",
                    "callback": lambda: st.rerun()
                }
            ]
            
            MobileNavigation.render_quick_actions(actions)
    
    @staticmethod
    def render_responsive_layout(
        left_content: Callable,
        right_content: Callable,
        **kwargs
    ):
        """
        Render responsive layout with left and right content.
        
        Args:
            left_content: Function to render left column content
            right_content: Function to render right column content
            **kwargs: Additional arguments to pass to content functions
        """
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile: Stack vertically
            with st.container():
                st.markdown("### 📊 Analytics")
                left_content(**kwargs)
            
            with st.container():
                st.markdown("### 📈 Performance")
                right_content(**kwargs)
        else:
            # Desktop: Side by side
            left_col, right_col = st.columns([1, 1], gap="large")
            
            with left_col:
                left_content(**kwargs)
            
            with right_col:
                right_content(**kwargs)
    
    @staticmethod
    def render_responsive_tabs(
        tabs: List[Dict[str, Any]],
        default_tab: str = "dashboard"
    ):
        """
        Render responsive tabs.
        
        Args:
            tabs: List of tab configurations
            default_tab: Default active tab
        """
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile: Use selectbox for tabs
            tab_names = [tab['label'] for tab in tabs]
            selected_tab = st.selectbox(
                "Select View",
                tab_names,
                index=next((i for i, tab in enumerate(tabs) if tab['key'] == default_tab), 0)
            )
            
            # Render selected tab content
            selected_tab_config = next((tab for tab in tabs if tab['label'] == selected_tab), tabs[0])
            if 'content' in selected_tab_config:
                selected_tab_config['content']()
        else:
            # Desktop: Use standard tabs
            tab_labels = [tab['label'] for tab in tabs]
            selected_tab = st.tabs(tab_labels)
            
            for i, tab in enumerate(tabs):
                with selected_tab[i]:
                    if 'content' in tab:
                        tab['content']()
    
    @staticmethod
    def render_mobile_optimized_dataframe(df: pd.DataFrame, title: str = "Data Table"):
        """
        Render mobile-optimized data table.
        
        Args:
            df: DataFrame to display
            title: Table title
        """
        if df.empty:
            st.info("No data available.")
            return
        
        screen_size = st.session_state.get('screen_size', 'desktop')
        
        if screen_size == 'mobile':
            # Mobile: Use card-based layout
            st.markdown(f"### {title}")
            
            # Add search functionality
            search_term = st.text_input(
                "🔍 Search...",
                key=f"search_{title.lower().replace(' ', '_')}",
                placeholder="Type to filter data..."
            )
            
            # Filter data
            if search_term:
                # Simple text search across all string columns
                mask = pd.DataFrame([df[col].astype(str).str.contains(search_term, case=False, na=False) 
                                   for col in df.select_dtypes(include=['object']).columns]).any()
                filtered_df = df[mask]
            else:
                filtered_df = df
            
            # Display as cards
            for idx, row in filtered_df.head(20).iterrows():  # Limit to 20 for mobile performance
                MobileDashboard._render_data_card(row, df.columns.tolist())
        else:
            # Desktop: Use standard dataframe
            st.markdown(f"### {title}")
            st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def _render_data_card(row: pd.Series, columns: List[str]):
        """Render individual data card for mobile."""
        card_html = """
        <div style="
            background: #1c1e26;
            border: 1px solid #2e303d;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        ">
        """
        
        for col in columns[:3]:  # Show first 3 columns for mobile
            value = row[col]
            if pd.isna(value):
                value = "N/A"
            elif isinstance(value, (int, float)):
                if 'sales' in col.lower() or 'revenue' in col.lower() or 'total' in col.lower():
                    value = f"₱{value:,.0f}"
                else:
                    value = f"{value:,}"
            
            card_html += f"""
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #888; font-size: 0.8rem;">{col}:</span>
                <span style="color: white; font-size: 0.9rem; margin-left: 0.5rem;">{value}</span>
            </div>
            """
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_mobile_performance_metrics(metrics: Dict[str, Any]):
        """Render mobile-optimized performance metrics."""
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        with st.container():
            st.markdown("### 📊 Performance Overview")
            
            # Create compact metrics display
            metric_items = [
                ("Sales", f"₱{metrics.get('current_sales', 0):,.0f}", "💰"),
                ("Profit", f"₱{metrics.get('current_profit', 0):,.0f}", "💎"),
                ("Transactions", f"{metrics.get('current_transactions', 0):,}", "🛒"),
                ("Avg Value", f"₱{metrics.get('avg_transaction_value', 0):,.0f}", "📈")
            ]
            
            for label, value, icon in metric_items:
                metric_html = f"""
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
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                        <span style="color: #c7c7c7; font-size: 0.9rem;">{label}</span>
                    </div>
                    <span style="color: #00d2ff; font-weight: bold; font-size: 1rem;">{value}</span>
                </div>
                """
                
                st.markdown(metric_html, unsafe_allow_html=True)
