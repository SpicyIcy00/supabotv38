"""
Mobile Dashboard V2 - Exact Layout Specification
Follows precise mobile layout requirements while maintaining desktop visual identity.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


class MobileDashboardV2:
    """Mobile dashboard with exact layout specifications."""
    
    @staticmethod
    def render_mobile_dashboard():
        """Render mobile dashboard following exact layout specifications."""
        
        # 1. HEADER SECTION - Compact and efficient
        MobileDashboardV2._render_mobile_header()
        
        # 2. CONTROLS ROW - Date and store selectors on same line
        MobileDashboardV2._render_mobile_controls()
        
        # 3. KPI SECTION - 2x2 grid layout
        MobileDashboardV2._render_mobile_kpis()
        
        # 4. CHARTS SECTION - Exact order specified
        MobileDashboardV2._render_mobile_charts()
    
    @staticmethod
    def _render_mobile_header():
        """Render compact mobile header."""
        st.markdown(
            '<div class="mobile-header"><h2>📊 SupaBot BI Dashboard</h2></div>', 
            unsafe_allow_html=True
        )
    
    @staticmethod
    def _render_mobile_controls():
        """Render date and store selectors on same horizontal line."""
        # Get data functions
        try:
            from appv38 import get_store_list, resolve_store_ids
        except ImportError:
            st.error("Unable to import data functions.")
            return
        
        # Controls row - Date and Store selectors on same line
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Time filter - compact
            time_options = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
            time_filter = st.session_state.get('dashboard_time_filter', '7D')
            time_index = time_options.index(time_filter) if time_filter in time_options else 1
            
            selected_time = st.selectbox(
                "Time Period",
                options=time_options,
                index=time_index,
                key="mobile_v2_time_selector"
            )
        
        with col2:
            # Store filter - compact
            try:
                store_df = get_store_list()
                store_list = store_df['name'].tolist() if not store_df.empty else []
            except Exception:
                store_list = []
            
            all_stores_option = "All Stores"
            current_store_filter = st.session_state.get('dashboard_store_filter', [all_stores_option])
            
            selected_stores = st.multiselect(
                "Store(s)",
                options=[all_stores_option] + store_list,
                default=current_store_filter,
                key="mobile_v2_store_selector"
            )
        
        # Custom date range if needed
        if selected_time == "Custom":
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                custom_start = st.date_input(
                    "From",
                    value=st.session_state.get('custom_start_date', pd.Timestamp.now().date()),
                    key="mobile_v2_custom_start"
                )
            with date_col2:
                custom_end = st.date_input(
                    "To",
                    value=st.session_state.get('custom_end_date', pd.Timestamp.now().date()),
                    key="mobile_v2_custom_end"
                )
            
            st.session_state.custom_start_date = custom_start
            st.session_state.custom_end_date = custom_end
        
        # Update session state
        st.session_state.dashboard_time_filter = selected_time
        st.session_state.dashboard_store_filter = selected_stores
        
        # Process store filter
        store_filter_ids = None
        if selected_stores and "All Stores" not in selected_stores and store_list:
            try:
                store_filter_ids, unmatched_names = resolve_store_ids(store_df, selected_stores)
                for nm in unmatched_names:
                    st.warning(f"Store '{nm}' not found")
            except Exception as e:
                st.error(f"Error processing store filter: {e}")
                return
        
        # Store for use in other functions
        st.session_state.mobile_v2_store_filter_ids = store_filter_ids
        st.session_state.mobile_v2_selected_time = selected_time
    
    @staticmethod
    def _render_mobile_kpis():
        """Render KPI section in 2x2 grid layout."""
        try:
            from appv38 import get_dashboard_metrics
        except ImportError:
            st.error("Unable to import dashboard metrics function.")
            return
        
        # Get metrics
        store_filter_ids = st.session_state.get('mobile_v2_store_filter_ids')
        selected_time = st.session_state.get('mobile_v2_selected_time', '7D')
        custom_start = st.session_state.get('custom_start_date') if selected_time == "Custom" else None
        custom_end = st.session_state.get('custom_end_date') if selected_time == "Custom" else None
        
        try:
            metrics = get_dashboard_metrics(selected_time, store_filter_ids, custom_start, custom_end)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
            return
        
        # KPI Section Header
        st.markdown("### 🚀 Key Performance Indicators")
        
        # 2x2 Grid Layout
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        def format_percentage_change(current, previous):
            """Calculate and format percentage change."""
            if previous is None or previous == 0:
                if current > 0:
                    return "New ↗"
                return "No data"
            
            if current is None:
                current = 0
            
            change = ((current - previous) / previous) * 100
            
            if abs(change) > 999:
                return ">999% ↗" if change > 0 else ">999% ↘"
            
            if abs(change) < 0.1:
                return "→ 0.0%"
            
            arrow = "↗" if change > 0 else "↘"
            return f"{change:+.1f}% {arrow}"
        
        # Row 1: Sales and Profit
        with row1_col1:
            sales = metrics.get('current_sales', 0)
            prev_sales = metrics.get('prev_sales', 0)
            delta = format_percentage_change(sales, prev_sales)
            st.metric("Total Sales", f"₱{sales:,.0f}", delta)
        
        with row1_col2:
            profit = metrics.get('current_profit', 0)
            prev_profit = metrics.get('prev_profit', 0)
            delta = format_percentage_change(profit, prev_profit)
            st.metric("Total Profit", f"₱{profit:,.0f}", delta)
        
        # Row 2: Transactions and Avg Value
        with row2_col1:
            transactions = metrics.get('current_transactions', 0)
            prev_transactions = metrics.get('prev_transactions', 0)
            delta = format_percentage_change(transactions, prev_transactions)
            st.metric("Transactions", f"{transactions:,}", delta)
        
        with row2_col2:
            avg_value = metrics.get('avg_transaction_value', 0)
            prev_avg_value = metrics.get('prev_avg_transaction_value', 0)
            delta = format_percentage_change(avg_value, prev_avg_value)
            st.metric("Avg Transaction", f"₱{avg_value:,.0f}", delta)
    
    @staticmethod
    def _render_mobile_charts():
        """Render charts in exact specified order."""
        try:
            from appv38 import (
                get_store_performance_with_comparison, get_sales_by_category_pie,
                get_inventory_by_category_pie, get_categories_with_change,
                get_top_products_with_change, get_daily_trend, get_avg_sales_per_hour
            )
        except ImportError:
            st.error("Unable to import chart data functions.")
            return
        
        store_filter_ids = st.session_state.get('mobile_v2_store_filter_ids')
        selected_time = st.session_state.get('mobile_v2_selected_time', '7D')
        
        # Data loading
        with st.spinner("📱 Loading mobile dashboard data..."):
            try:
                # Get all chart data
                store_performance_df = get_store_performance_with_comparison(selected_time, store_filter_ids)
                sales_cat_df = get_sales_by_category_pie(selected_time, store_filter_ids)
                inv_cat_df = get_inventory_by_category_pie(store_filter_ids)
                cat_change_df = get_categories_with_change(selected_time, store_filter_ids)
                top_change_df = get_top_products_with_change(selected_time, store_filter_ids)
                daily_trend_df = get_daily_trend(
                    days={"1D":1, "7D":7, "1M":30, "6M":180, "1Y":365}.get(selected_time, 7), 
                    store_ids=store_filter_ids
                )
                avg_sales_hour_df = get_avg_sales_per_hour(store_filter_ids)
                
            except Exception as e:
                st.error(f"❌ Error loading chart data: {e}")
                return
        
        # Charts Section Header
        st.markdown("### 📊 Analytics")
        
        # 1. Store Performance (horizontal bar chart)
        MobileDashboardV2._render_store_performance(store_performance_df)
        
        # 2. Sales by Category (pie chart)
        MobileDashboardV2._render_sales_by_category(sales_cat_df)
        
        # 3. Inventory by Category (pie chart)
        MobileDashboardV2._render_inventory_by_category(inv_cat_df)
        
        # 4. Categories Ranked (with % change) - table/list format
        MobileDashboardV2._render_categories_ranked(cat_change_df)
        
        # 5. Top 10 Products (with % change) - table/list format
        MobileDashboardV2._render_top_products(top_change_df)
        
        # 6. Sales Trend Analysis (area chart)
        MobileDashboardV2._render_sales_trend(daily_trend_df, selected_time)
        
        # 7. Average Sales Per Hour (bar chart)
        MobileDashboardV2._render_avg_sales_per_hour(avg_sales_hour_df)
    
    @staticmethod
    def _render_store_performance(df: pd.DataFrame):
        """Render store performance horizontal bar chart."""
        with st.container(border=True):
            st.markdown("##### 🏪 Store Performance")
            if not df.empty:
                store_color_map = {
                    'Rockwell': '#E74C3C',
                    'Greenhills': '#2ECC71',
                    'Magnolia': '#F1C40F',
                    'North Edsa': '#3498DB',
                    'Fairview': '#9B59B6'
                }
                
                # Horizontal bar chart for mobile
                fig = px.bar(
                    df.head(5),
                    y='store_name',
                    x='total_sales',
                    orientation='h',
                    title='Top Stores by Sales',
                    color='store_name',
                    color_discrete_map=store_color_map
                )
                
                # Add percentage change annotations
                for idx, (_, store) in enumerate(df.head(5).iterrows()):
                    current_sales = store['total_sales']
                    pct_change = store.get('pct_change')
                    
                    if pct_change is not None:
                        if pct_change > 0:
                            annotation_text = f"↗ +{pct_change:.1f}%"
                            annotation_color = "#2ECC71"
                        elif pct_change < 0:
                            annotation_text = f"↘ {pct_change:.1f}%"
                            annotation_color = "#E74C3C"
                        else:
                            annotation_text = "→ 0.0%"
                            annotation_color = "#95A5A6"
                    else:
                        annotation_text = "New ↗"
                        annotation_color = "#F39C12"
                    
                    fig.add_annotation(
                        x=current_sales,
                        y=store['store_name'],
                        text=annotation_text,
                        showarrow=False,
                        font=dict(color=annotation_color, size=10),
                        xshift=30,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)",
                        borderwidth=0
                    )
                
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=0, l=0, r=0), 
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    showlegend=False
                )
                fig.update_xaxes(tickprefix='₱', separatethousands=True)
                fig.update_yaxes(title_text="")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No store performance data available.")
    
    @staticmethod
    def _render_sales_by_category(df: pd.DataFrame):
        """Render sales by category pie chart."""
        with st.container(border=True):
            st.markdown("##### 💰 Sales by Category")
            if not df.empty:
                df_plot = df.head(8).copy()  # Limit for mobile
                df_plot['category_canonical'] = df_plot['category'].astype(str).apply(
                    lambda x: x[:15] + "..." if len(str(x)) > 15 else str(x)
                )
                df_plot = df_plot.groupby('category_canonical', as_index=False)['total_revenue'].sum()
                
                fig = px.pie(
                    df_plot,
                    values='total_revenue',
                    names='category_canonical',
                    color='category_canonical',
                    hole=0.4,
                    color_discrete_map=MobileDashboardV2._get_fixed_category_color_map()
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=False, 
                    height=250,
                    margin=dict(t=0, b=0, l=0, r=0), 
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available.")
    
    @staticmethod
    def _render_inventory_by_category(df: pd.DataFrame):
        """Render inventory by category pie chart."""
        with st.container(border=True):
            st.markdown("##### 📦 Inventory by Category")
            if not df.empty:
                df_plot = df.head(8).copy()  # Limit for mobile
                df_plot['category_canonical'] = df_plot['category'].astype(str).apply(
                    lambda x: x[:15] + "..." if len(str(x)) > 15 else str(x)
                )
                df_plot = df_plot.groupby('category_canonical', as_index=False)['total_quantity'].sum()
                
                fig = px.pie(
                    df_plot,
                    values='total_quantity',
                    names='category_canonical',
                    color='category_canonical',
                    hole=0.4,
                    color_discrete_map=MobileDashboardV2._get_fixed_category_color_map()
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=False, 
                    height=250,
                    margin=dict(t=0, b=0, l=0, r=0), 
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No inventory data available.")
    
    @staticmethod
    def _render_categories_ranked(df: pd.DataFrame):
        """Render categories ranked with % change in table format."""
        with st.container(border=True):
            st.markdown("##### 🗂️ Categories Ranked")
            if not df.empty:
                dfc = df.head(8).copy()  # Limit for mobile
                dfc.rename(columns={'category': 'Category', 'total_revenue': 'Sales'}, inplace=True)
                
                def _pct_cell(x):
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return "New"
                    arrow = '▲' if x >= 0 else '▼'
                    return f"{arrow} {abs(x):.1f}%"
                
                dfc['Δ %'] = dfc['pct_change'].apply(_pct_cell)
                show_c = dfc[['Category', 'Sales', 'Δ %']].copy()
                
                # Truncate category names for mobile
                show_c['Category'] = show_c['Category'].apply(
                    lambda x: x[:15] + "..." if len(str(x)) > 15 else str(x)
                )
                
                def style_change(col):
                    styles = []
                    for val in col:
                        if isinstance(val, str) and val.startswith('▲'):
                            styles.append('color: #00c853; font-weight: 600')
                        elif isinstance(val, str) and val.startswith('▼'):
                            styles.append('color: #ff5252; font-weight: 600')
                        else:
                            styles.append('color: #aaaaaa')
                    return styles
                
                styled_c = (show_c.style
                    .format({'Sales': '₱{:,.0f}'})
                    .apply(style_change, subset=['Δ %']))
                
                st.write(styled_c)
            else:
                st.info("No category data available.")
    
    @staticmethod
    def _render_top_products(df: pd.DataFrame):
        """Render top 10 products with % change in table format."""
        with st.container(border=True):
            st.markdown("##### 🏆 Top 10 Products")
            if not df.empty:
                df_disp = df.head(10).copy()  # Top 10 for mobile
                df_disp.rename(columns={'product_name': 'Product', 'total_revenue': 'Sales'}, inplace=True)
                
                def _pct_cell(x):
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return "New"
                    arrow = '▲' if x >= 0 else '▼'
                    return f"{arrow} {abs(x):.1f}%"
                
                df_disp['Δ %'] = df_disp['pct_change'].apply(_pct_cell)
                show_df = df_disp[['Product', 'Sales', 'Δ %']].copy()
                
                # Truncate product names for mobile
                show_df['Product'] = show_df['Product'].apply(
                    lambda x: x[:20] + "..." if len(str(x)) > 20 else str(x)
                )
                
                # Mobile-friendly styling
                def style_change(col):
                    styles = []
                    for val in col:
                        if isinstance(val, str) and val.startswith('▲'):
                            styles.append('color: #00c853; font-weight: 600')
                        elif isinstance(val, str) and val.startswith('▼'):
                            styles.append('color: #ff5252; font-weight: 600')
                        else:
                            styles.append('color: #aaaaaa')
                    return styles
                
                styled = (show_df.style
                    .format({'Sales': '₱{:,.0f}'})
                    .apply(style_change, subset=['Δ %']))
                
                st.write(styled)
            else:
                st.info("No product data available.")
    
    @staticmethod
    def _render_sales_trend(df: pd.DataFrame, time_filter: str):
        """Render sales trend area chart."""
        with st.container(border=True):
            st.markdown("##### 📈 Sales Trend Analysis")
            if not df.empty:
                fig = px.area(
                    df, 
                    x='date', 
                    y='daily_sales',
                    title=f"Sales Trend for Last {time_filter}"
                )
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=0, l=0, r=0),
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_yaxes(tickprefix='₱', separatethousands=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales trend data available.")
    
    @staticmethod
    def _render_avg_sales_per_hour(df: pd.DataFrame):
        """Render average sales per hour bar chart."""
        with st.container(border=True):
            st.markdown("##### ⏰ Average Sales Per Hour")
            if not df.empty:
                fig = px.bar(
                    df,
                    x='hour',
                    y='avg_sales',
                    title='Average Sales by Hour of Day',
                    color='avg_sales',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=0, l=0, r=0),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                fig.update_yaxes(tickprefix='₱', separatethousands=True)
                fig.update_xaxes(title_text="Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly sales data available.")
    
    @staticmethod
    def _get_fixed_category_color_map():
        """Get consistent color map for categories."""
        return {
            'Electronics': '#FF6B6B',
            'Clothing': '#4ECDC4',
            'Home & Garden': '#45B7D1',
            'Sports & Outdoors': '#96CEB4',
            'Books': '#FFEAA7',
            'Automotive': '#DDA0DD',
            'Health & Beauty': '#98D8C8',
            'Toys & Games': '#F7DC6F',
            'Food & Beverages': '#BB8FCE',
            'Jewelry': '#85C1E9',
            'Other': '#AEB6BF'
        }
