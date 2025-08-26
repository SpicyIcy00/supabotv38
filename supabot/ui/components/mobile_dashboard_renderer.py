"""
Mobile dashboard renderer for SupaBot BI Dashboard.
Provides complete mobile-optimized dashboard experience.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from .mobile_dashboard import MobileDashboard
from .mobile_detection import MobileDetection, ResponsiveLayout
from ..styles.mobile_css import MobileStyles


class MobileDashboardRenderer:
    """Complete mobile dashboard renderer."""
    
    @staticmethod
    def render_mobile_dashboard():
        """Render the complete mobile-optimized dashboard."""
        # Load mobile styles
        MobileStyles.load_all_mobile_styles()
        
        # Inject mobile detection
        MobileDetection.inject_detection_script()
        
        # Mobile header
        st.markdown(
            '<div class="main-header"><h1>📊 SupaBot BI Dashboard</h1><p>Mobile Business Intelligence</p></div>', 
            unsafe_allow_html=True
        )
        
        # Get data functions from the main app
        try:
            # Import data functions from the main app
            from appv38 import (
                get_dashboard_metrics, get_sales_by_category_pie, get_inventory_by_category_pie,
                get_top_products_with_change, get_categories_with_change, get_daily_trend,
                get_store_performance_with_comparison, get_store_list, resolve_store_ids,
                get_intelligent_date_range, get_previous_period_dates, get_avg_sales_per_hour
            )
        except ImportError:
            st.error("Unable to import data functions. Please ensure appv38.py is available.")
            return
        
        # Mobile filters section
        st.markdown("### 📱 Dashboard Controls")
        
        # Time filter
        time_options = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
        time_filter = st.session_state.get('dashboard_time_filter', '7D')
        time_index = time_options.index(time_filter) if time_filter in time_options else 1
        
        selected_time = st.selectbox(
            "Time Period:",
            options=time_options,
            index=time_index,
            key="mobile_time_selector"
        )
        
        # Custom date range
        if selected_time == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                custom_start = st.date_input(
                    "From Date",
                    value=st.session_state.get('custom_start_date', pd.Timestamp.now().date()),
                    key="mobile_custom_start"
                )
            with col2:
                custom_end = st.date_input(
                    "To Date",
                    value=st.session_state.get('custom_end_date', pd.Timestamp.now().date()),
                    key="mobile_custom_end"
                )
            
            st.session_state.custom_start_date = custom_start
            st.session_state.custom_end_date = custom_end
            
            if custom_start > custom_end:
                st.error("Start date cannot be after end date!")
                return
        
        # Store filter
        try:
            store_df = get_store_list()
            store_list = store_df['name'].tolist() if not store_df.empty else []
        except Exception:
            store_list = []
        
        all_stores_option = "All Stores"
        current_store_filter = st.session_state.get('dashboard_store_filter', [all_stores_option])
        
        selected_stores = st.multiselect(
            "Select Store(s):",
            options=[all_stores_option] + store_list,
            default=current_store_filter,
            key="mobile_store_selector"
        )
        
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
        
        # Get custom dates if applicable
        custom_start = st.session_state.custom_start_date if selected_time == "Custom" else None
        custom_end = st.session_state.custom_end_date if selected_time == "Custom" else None
        
        # Data loading with mobile spinner
        with st.spinner(f"📱 Loading mobile dashboard data..."):
            try:
                # Get metrics
                metrics = get_dashboard_metrics(selected_time, store_filter_ids, custom_start, custom_end)
                
                if not metrics or all(v == 0 for v in [metrics.get('current_sales', 0), metrics.get('current_transactions', 0)]):
                    st.warning("No data found for selected filters")
                    return
                
                # Get chart data
                sales_cat_df = get_sales_by_category_pie(selected_time, store_filter_ids)
                inv_cat_df = get_inventory_by_category_pie(store_filter_ids)
                top_change_df = get_top_products_with_change(selected_time, store_filter_ids)
                cat_change_df = get_categories_with_change(selected_time, store_filter_ids)
                daily_trend_df = get_daily_trend(
                    days={"1D":1, "7D":7, "1M":30, "6M":180, "1Y":365}.get(selected_time, 7), 
                    store_ids=store_filter_ids
                )
                store_performance_df = get_store_performance_with_comparison(selected_time, store_filter_ids)
                
            except Exception as e:
                st.error(f"❌ Error loading dashboard data: {e}")
                return
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Mobile KPI Section (2x2 grid)
        st.subheader("🚀 Key Performance Indicators")
        MobileDashboardRenderer._render_mobile_kpis(metrics, selected_time)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Mobile Charts Section (stacked vertically)
        st.subheader("📊 Analytics")
        MobileDashboardRenderer._render_mobile_charts(
            sales_cat_df, inv_cat_df, daily_trend_df, store_performance_df, selected_time
        )
        
        # Mobile Data Tables Section
        st.subheader("📋 Data Tables")
        MobileDashboardRenderer._render_mobile_tables(top_change_df, cat_change_df)
        
        # Mobile AI Section
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🧠 AI Intelligence")
        MobileDashboardRenderer._render_mobile_ai_section()
    
    @staticmethod
    def _render_mobile_kpis(metrics: Dict[str, Any], time_filter: str):
        """Render mobile-optimized KPI metrics."""
        # 2x2 grid for mobile
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        def format_percentage_change(current, previous):
            """Calculate and format percentage change for mobile."""
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
    def _render_mobile_charts(sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame, 
                            daily_trend_df: pd.DataFrame, store_performance_df: pd.DataFrame, 
                            time_filter: str):
        """Render mobile-optimized charts."""
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Sales by Category
        with st.container(border=True):
            st.markdown("##### 💰 Sales by Category")
            if not sales_cat_df.empty:
                df_plot = sales_cat_df.head(8).copy()  # Limit for mobile
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
                    color_discrete_map=MobileDashboard._get_fixed_category_color_map()
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
        
        # Store Performance (horizontal bars for mobile)
        with st.container(border=True):
            st.markdown("##### 🏪 Store Performance")
            if not store_performance_df.empty:
                store_color_map = {
                    'Rockwell': '#E74C3C',
                    'Greenhills': '#2ECC71',
                    'Magnolia': '#F1C40F',
                    'North Edsa': '#3498DB',
                    'Fairview': '#9B59B6'
                }
                
                # Horizontal bar chart for mobile
                fig = px.bar(
                    store_performance_df.head(5),
                    y='store_name',
                    x='total_sales',
                    orientation='h',
                    title='Top Stores by Sales',
                    color='store_name',
                    color_discrete_map=store_color_map
                )
                
                # Add percentage change annotations
                for idx, (_, store) in enumerate(store_performance_df.head(5).iterrows()):
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
        
        # Sales Trend
        with st.container(border=True):
            st.markdown("##### 📈 Sales Trend")
            if not daily_trend_df.empty:
                fig = px.area(
                    daily_trend_df, 
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
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales trend data available.")
    
    @staticmethod
    def _render_mobile_tables(top_change_df: pd.DataFrame, cat_change_df: pd.DataFrame):
        """Render mobile-optimized data tables."""
        
        # Top Products
        with st.container(border=True):
            st.markdown("##### 🏆 Top Products")
            if not top_change_df.empty:
                df_disp = top_change_df.head(8).copy()  # Limit for mobile
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
        
        # Categories
        with st.container(border=True):
            st.markdown("##### 🗂️ Categories")
            if not cat_change_df.empty:
                dfc = cat_change_df.head(8).copy()  # Limit for mobile
                dfc.rename(columns={'category': 'Category', 'total_revenue': 'Sales'}, inplace=True)
                
                def _pct_cell2(x):
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return "New"
                    arrow = '▲' if x >= 0 else '▼'
                    return f"{arrow} {abs(x):.1f}%"
                
                dfc['Δ %'] = dfc['pct_change'].apply(_pct_cell2)
                show_c = dfc[['Category', 'Sales', 'Δ %']].copy()
                
                # Truncate category names for mobile
                show_c['Category'] = show_c['Category'].apply(
                    lambda x: x[:15] + "..." if len(str(x)) > 15 else str(x)
                )
                
                def style_change2(col):
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
                    .apply(style_change2, subset=['Δ %']))
                
                st.write(styled_c)
            else:
                st.info("No category data available.")
    
    @staticmethod
    def _render_mobile_ai_section():
        """Render mobile-optimized AI section."""
        st.markdown("Generate comprehensive business intelligence reports powered by AI.")
        
        if st.button("🔍 Generate AI Intelligence Report", type="primary", use_container_width=True, key="mobile_ai_generate"):
            st.info("AI Intelligence feature is available in the desktop version for full functionality.")
            st.markdown("💡 **Tip:** Switch to desktop view for advanced AI features and detailed reports.")
