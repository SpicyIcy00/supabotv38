"""
Mobile chart components for responsive dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List


class MobileCharts:
    """Mobile-optimized chart components."""
    
    @staticmethod
    def render_sales_trend_chart(sales_df: pd.DataFrame, title: str = "📈 Sales Trend"):
        """
        Render sales trend chart optimized for mobile.
        
        Args:
            sales_df: DataFrame containing sales trend data
            title: Chart title
        """
        if sales_df.empty:
            st.info("No sales trend data available.")
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            # Add swipe indicator for mobile
            if st.session_state.get('screen_size', 'desktop') == 'mobile':
                st.markdown(
                    '<div class="swipe-indicator">💡 Swipe to view full chart</div>',
                    unsafe_allow_html=True
                )
            
            # Create mobile-optimized chart
            fig = MobileCharts._create_mobile_sales_chart(sales_df)
            
            # Render chart in scrollable container
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
                st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _create_mobile_sales_chart(df: pd.DataFrame) -> go.Figure:
        """Create mobile-optimized sales trend chart."""
        # Determine column names
        if 'date' in df.columns:
            x_col = 'date'
            y_col = 'total_revenue'
        elif 'Date' in df.columns:
            x_col = 'Date'
            y_col = 'Sales'
        else:
            # Fallback to first two columns
            cols = df.columns.tolist()
            if len(cols) >= 2:
                x_col = cols[0]
                y_col = cols[1]
            else:
                st.error("Invalid sales data format")
                return go.Figure()
        
        # Create figure with mobile-optimized settings
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            line=dict(color='#00d2ff', width=3),
            marker=dict(size=6, color='#00d2ff'),
            fill='tonexty',
            fillcolor='rgba(0, 210, 255, 0.1)',
            name='Sales'
        ))
        
        # Mobile-optimized layout
        fig.update_layout(
            title=None,  # Remove title for mobile
            height=300,  # Reduced height for mobile
            margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10),
                tickangle=45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                tickfont=dict(size=10),
                tickformat=','
            ),
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def render_pie_charts(sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame):
        """
        Render pie charts optimized for mobile.
        
        Args:
            sales_cat_df: Sales by category DataFrame
            inv_cat_df: Inventory by category DataFrame
        """
        # Mobile: Stack vertically, Desktop: Side by side
        if st.session_state.get('screen_size', 'desktop') == 'mobile':
            MobileCharts._render_mobile_pie_charts(sales_cat_df, inv_cat_df)
        else:
            MobileCharts._render_desktop_pie_charts(sales_cat_df, inv_cat_df)
    
    @staticmethod
    def _render_mobile_pie_charts(sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame):
        """Render pie charts in vertical stack for mobile."""
        # Sales by Category
        with st.container():
            st.markdown("### 💰 Sales by Category")
            if not sales_cat_df.empty:
                fig = MobileCharts._create_mobile_pie_chart(
                    sales_cat_df, 
                    'category', 
                    'total_revenue',
                    'Sales by Category'
                )
                st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
            else:
                st.info("No sales data available.")
        
        # Inventory by Category
        with st.container():
            st.markdown("### 📦 Inventory by Category")
            if not inv_cat_df.empty:
                fig = MobileCharts._create_mobile_pie_chart(
                    inv_cat_df,
                    'category',
                    'total_inventory_value',
                    'Inventory by Category'
                )
                st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
            else:
                st.info("No inventory data available.")
    
    @staticmethod
    def _render_desktop_pie_charts(sales_cat_df: pd.DataFrame, inv_cat_df: pd.DataFrame):
        """Render pie charts side by side for desktop."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Sales by Category")
            if not sales_cat_df.empty:
                fig = MobileCharts._create_mobile_pie_chart(
                    sales_cat_df,
                    'category',
                    'total_revenue',
                    'Sales by Category'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available.")
        
        with col2:
            st.markdown("### 📦 Inventory by Category")
            if not inv_cat_df.empty:
                fig = MobileCharts._create_mobile_pie_chart(
                    inv_cat_df,
                    'category',
                    'total_inventory_value',
                    'Inventory by Category'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No inventory data available.")
    
    @staticmethod
    def _create_mobile_pie_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str) -> go.Figure:
        """Create mobile-optimized pie chart."""
        # Prepare data
        df_plot = df.head(8).copy()  # Limit to top 8 for mobile readability
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=df_plot[category_col],
            values=df_plot[value_col],
            hole=0.4,
            textinfo='percent+label',
            textposition='inside',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        # Mobile-optimized layout
        fig.update_layout(
            title=None,
            height=250,  # Reduced height for mobile
            margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(size=10)
        )
        
        return fig
    
    @staticmethod
    def render_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "📊 Performance Chart"):
        """
        Render horizontal bar chart optimized for mobile.
        
        Args:
            df: DataFrame containing chart data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
        """
        if df.empty:
            st.info("No data available for chart.")
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            # Create horizontal bar chart for mobile
            fig = MobileCharts._create_mobile_bar_chart(df, x_col, y_col)
            
            # Render chart
            st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
    
    @staticmethod
    def _create_mobile_bar_chart(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """Create mobile-optimized horizontal bar chart."""
        # Sort data for better mobile viewing
        df_sorted = df.sort_values(y_col, ascending=True).head(10)
        
        fig = go.Figure(data=[go.Bar(
            x=df_sorted[y_col],
            y=df_sorted[x_col],
            orientation='h',
            marker=dict(
                color='#00d2ff',
                line=dict(color='#00d2ff', width=1)
            ),
            text=df_sorted[y_col].apply(lambda x: f"₱{x:,.0f}" if isinstance(x, (int, float)) else str(x)),
            textposition='auto'
        )])
        
        # Mobile-optimized layout
        fig.update_layout(
            title=None,
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                tickfont=dict(size=10),
                tickformat=','
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10)
            ),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def render_compact_chart(df: pd.DataFrame, chart_type: str = "line", title: str = "📈 Chart"):
        """
        Render a compact chart for mobile summary views.
        
        Args:
            df: DataFrame containing chart data
            chart_type: Type of chart ('line', 'bar', 'pie')
            title: Chart title
        """
        if df.empty:
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            if chart_type == "line":
                fig = MobileCharts._create_compact_line_chart(df)
            elif chart_type == "bar":
                fig = MobileCharts._create_compact_bar_chart(df)
            elif chart_type == "pie":
                fig = MobileCharts._create_compact_pie_chart(df)
            else:
                st.error(f"Unsupported chart type: {chart_type}")
                return
            
            st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
    
    @staticmethod
    def _create_compact_line_chart(df: pd.DataFrame) -> go.Figure:
        """Create compact line chart for mobile."""
        # Use first two columns as x and y
        cols = df.columns.tolist()
        if len(cols) < 2:
            return go.Figure()
        
        x_col, y_col = cols[0], cols[1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines',
            line=dict(color='#00d2ff', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 210, 255, 0.1)'
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(t=5, b=5, l=5, r=5),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _create_compact_bar_chart(df: pd.DataFrame) -> go.Figure:
        """Create compact bar chart for mobile."""
        cols = df.columns.tolist()
        if len(cols) < 2:
            return go.Figure()
        
        x_col, y_col = cols[0], cols[1]
        
        fig = go.Figure(data=[go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color='#00d2ff'
        )])
        
        fig.update_layout(
            height=200,
            margin=dict(t=5, b=5, l=5, r=5),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _create_compact_pie_chart(df: pd.DataFrame) -> go.Figure:
        """Create compact pie chart for mobile."""
        cols = df.columns.tolist()
        if len(cols) < 2:
            return go.Figure()
        
        category_col, value_col = cols[0], cols[1]
        
        fig = go.Figure(data=[go.Pie(
            labels=df[category_col].head(5),
            values=df[value_col].head(5),
            hole=0.6,
            textinfo='none'
        )])
        
        fig.update_layout(
            height=200,
            margin=dict(t=5, b=5, l=5, r=5),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _get_mobile_chart_config() -> Dict[str, Any]:
        """Get mobile-optimized chart configuration."""
        return {
            'displayModeBar': False,  # Hide toolbar on mobile
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'scrollZoom': True  # Enable scroll zoom for mobile
        }
    
    @staticmethod
    def render_chart_with_controls(df: pd.DataFrame, chart_type: str = "line", title: str = "📊 Interactive Chart"):
        """
        Render chart with mobile-optimized controls.
        
        Args:
            df: DataFrame containing chart data
            chart_type: Type of chart
            title: Chart title
        """
        if df.empty:
            st.info("No data available for chart.")
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            # Mobile-friendly controls
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type_control = st.selectbox(
                    "Chart Type",
                    ["line", "bar", "pie"],
                    key="chart_type_control"
                )
            
            with col2:
                max_items = st.slider(
                    "Max Items",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="max_items_control"
                )
            
            # Filter data based on controls
            filtered_df = df.head(max_items)
            
            # Render chart
            if chart_type_control == "line":
                fig = MobileCharts._create_mobile_sales_chart(filtered_df)
            elif chart_type_control == "bar":
                fig = MobileCharts._create_mobile_bar_chart(filtered_df, filtered_df.columns[0], filtered_df.columns[1])
            elif chart_type_control == "pie":
                fig = MobileCharts._create_mobile_pie_chart(filtered_df, filtered_df.columns[0], filtered_df.columns[1], title)
            
            st.plotly_chart(fig, use_container_width=True, config=MobileCharts._get_mobile_chart_config())
