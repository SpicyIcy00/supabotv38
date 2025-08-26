"""
Mobile-specific dashboard components for SupaBot BI Dashboard.
Provides mobile-optimized layouts and interactions while keeping desktop code untouched.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import streamlit.components.v1 as components


class MobileDashboard:
    """Mobile-optimized dashboard components."""
    
    @staticmethod
    def inject_mobile_detection():
        """Inject JavaScript to detect mobile devices."""
        mobile_detection_js = """
        <script>
        function isMobileDevice() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                   window.innerWidth <= 768;
        }
        
        // Set mobile flag in session storage
        if (isMobileDevice()) {
            sessionStorage.setItem('isMobile', 'true');
        } else {
            sessionStorage.setItem('isMobile', 'false');
        }
        
        // Listen for resize events
        window.addEventListener('resize', function() {
            if (isMobileDevice()) {
                sessionStorage.setItem('isMobile', 'true');
            } else {
                sessionStorage.setItem('isMobile', 'false');
            }
        });
        </script>
        """
        st.markdown(mobile_detection_js, unsafe_allow_html=True)
    
    @staticmethod
    def is_mobile_device() -> bool:
        """Check if current device is mobile using JavaScript."""
        # Inject detection script
        MobileDashboard.inject_mobile_detection()
        
        # Use JavaScript to check mobile status
        js_code = """
        <script>
        const isMobile = sessionStorage.getItem('isMobile') === 'true';
        if (isMobile) {
            document.body.setAttribute('data-mobile', 'true');
        }
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)
        
        # For now, use a simple heuristic based on screen width
        # In a real implementation, you'd use the JavaScript result
        return True  # Assume mobile for now - will be refined
    
    @staticmethod
    def render_mobile_kpi_metrics(metrics: Dict[str, Any], time_filter: str):
        """Render KPI metrics optimized for mobile (2x2 grid)."""
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Mobile: 2x2 grid instead of 4 columns
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
    def render_mobile_filters(time_filter: str, store_filter: List[str], store_list: List[str]):
        """Render mobile-optimized filters."""
        st.markdown("### 📱 Filters")
        
        # Time filter - vertical layout for mobile
        st.markdown("**Time Period:**")
        time_options = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
        time_index = time_options.index(time_filter) if time_filter in time_options else 1
        
        # Use selectbox instead of radio for better mobile UX
        selected_time = st.selectbox(
            "Select Time Period:",
            options=time_options,
            index=time_index,
            key="mobile_time_filter"
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
        
        # Store filter - mobile-friendly multiselect
        st.markdown("**Stores:**")
        all_stores_option = "All Stores"
        selected_stores = st.multiselect(
            "Select Store(s):",
            options=[all_stores_option] + store_list,
            default=store_filter if store_filter else [all_stores_option],
            key="mobile_store_filter"
        )
        
        return selected_time, selected_stores
    
    @staticmethod
    def render_mobile_charts(sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame, 
                           daily_trend_df: pd.DataFrame, store_performance_df: pd.DataFrame):
        """Render mobile-optimized charts (stacked vertically)."""
        
        # Sales by Category (mobile-optimized)
        with st.container(border=True):
            st.markdown("##### 💰 Sales by Category")
            if not sales_cat_df.empty:
                df_plot = sales_cat_df.head(8).copy()  # Limit to 8 for mobile
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
                    height=250,  # Smaller height for mobile
                    margin=dict(t=0, b=0, l=0, r=0), 
                    template="plotly_dark", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available.")
        
        # Store Performance (mobile-optimized)
        with st.container(border=True):
            st.markdown("##### 🏪 Store Performance")
            if not store_performance_df.empty:
                # Fixed store colors
                store_color_map = {
                    'Rockwell': '#E74C3C',
                    'Greenhills': '#2ECC71',
                    'Magnolia': '#F1C40F',
                    'North Edsa': '#3498DB',
                    'Fairview': '#9B59B6'
                }
                
                # Create horizontal bar chart for mobile
                fig = px.bar(
                    store_performance_df.head(5),
                    y='store_name',  # Swap x/y for horizontal bars
                    x='total_sales',
                    orientation='h',  # Horizontal bars
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
        
        # Sales Trend (mobile-optimized)
        with st.container(border=True):
            st.markdown("##### 📈 Sales Trend")
            if not daily_trend_df.empty:
                fig = px.area(
                    daily_trend_df, 
                    x='date', 
                    y='daily_sales',
                    title="Daily Sales Trend"
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
    def render_mobile_data_tables(top_change_df: pd.DataFrame, cat_change_df: pd.DataFrame):
        """Render mobile-optimized data tables."""
        
        # Top Products (mobile-optimized)
        with st.container(border=True):
            st.markdown("##### 🏆 Top Products")
            if not top_change_df.empty:
                df_disp = top_change_df.head(8).copy()  # Limit to 8 for mobile
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
        
        # Categories (mobile-optimized)
        with st.container(border=True):
            st.markdown("##### 🗂️ Categories")
            if not cat_change_df.empty:
                dfc = cat_change_df.head(8).copy()  # Limit to 8 for mobile
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
    def _get_fixed_category_color_map():
        """Get fixed color map for categories."""
        return {
            'Electronics': '#FF6B6B',
            'Clothing': '#4ECDC4',
            'Food & Beverage': '#45B7D1',
            'Home & Garden': '#96CEB4',
            'Sports & Outdoors': '#FFEAA7',
            'Books & Media': '#DDA0DD',
            'Health & Beauty': '#98D8C8',
            'Automotive': '#F7DC6F',
            'Toys & Games': '#BB8FCE',
            'Other': '#85C1E9'
        }
