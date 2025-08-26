"""
Chart factory and visualization components for SupaBot BI Dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, List, Tuple


class ChartFactory:
    """Factory class for creating various chart types."""
    
    @staticmethod
    def create_smart_visualization(results_df: pd.DataFrame, question: str) -> Optional[go.Figure]:
        """Create intelligent visualization based on data characteristics and question."""
        if results_df is None or results_df.empty:
            return None
        
        # Analyze data characteristics
        numeric_cols = list(results_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)
        text_cols = list(results_df.select_dtypes(include=['object', 'string']).columns)
        date_cols = list(results_df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns)
        
        # Determine best chart type
        chart_type = ChartFactory._determine_chart_type(question.lower(), results_df, numeric_cols, text_cols, date_cols)
        
        # Create appropriate chart
        if chart_type == "pie":
            return ChartFactory.create_pie_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "treemap":
            return ChartFactory.create_treemap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "scatter":
            return ChartFactory.create_scatter_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "line":
            return ChartFactory.create_line_chart(results_df, question, numeric_cols, text_cols, date_cols)
        elif chart_type == "heatmap":
            return ChartFactory.create_heatmap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "box":
            return ChartFactory.create_box_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "area":
            return ChartFactory.create_area_chart(results_df, question, numeric_cols, text_cols, date_cols)
        else:  # Default to bar chart
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
    
    @staticmethod
    def _determine_chart_type(question_lower: str, results_df: pd.DataFrame, 
                             numeric_cols: List[str], text_cols: List[str], date_cols: List[str]) -> str:
        """Determine the best chart type based on question and data characteristics."""
        row_count = len(results_df)
        
        # Time series patterns
        if date_cols and any(word in question_lower for word in ['trend', 'over time', 'daily', 'weekly', 'monthly', 'timeline']):
            if any(word in question_lower for word in ['area', 'fill', 'cumulative']):
                return "area"
            return "line"
        
        # Distribution and comparison patterns
        if any(word in question_lower for word in ['share', 'proportion', 'percentage', 'distribution']) and row_count <= 10:
            return "pie"
        
        # Hierarchical data patterns  
        if any(word in question_lower for word in ['breakdown', 'composition', 'hierarchy']) and row_count <= 15:
            return "treemap"
        
        # Correlation patterns
        if len(numeric_cols) >= 2 and any(word in question_lower for word in ['relationship', 'correlation', 'vs', 'against']):
            return "scatter"
        
        # Matrix/heatmap patterns
        if len(text_cols) >= 2 and any(word in question_lower for word in ['heatmap', 'matrix', 'cross']):
            return "heatmap"
        
        # Distribution analysis
        if any(word in question_lower for word in ['distribution', 'variance', 'outlier', 'spread']):
            return "box"
        
        # Default to bar chart for most cases
        return "bar"
    
    @staticmethod
    def create_pie_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a pie chart visualization."""
        value_col = ChartFactory._get_best_value_column(numeric_cols)
        label_col = ChartFactory._get_best_label_column(text_cols)
        
        if not value_col or not label_col:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        # Limit to top 10 for readability
        df_sorted = results_df.nlargest(10, value_col)
        
        fig = px.pie(df_sorted, 
                     values=value_col, 
                     names=label_col,
                     title=f"Distribution: {question}",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500, showlegend=True)
        return fig
    
    @staticmethod
    def create_treemap_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a treemap chart visualization."""
        value_col = ChartFactory._get_best_value_column(numeric_cols)
        label_col = ChartFactory._get_best_label_column(text_cols)
        
        if not value_col or not label_col:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        fig = px.treemap(results_df.head(15),
                         path=[label_col],
                         values=value_col,
                         title=f"Hierarchy: {question}",
                         color=value_col,
                         color_continuous_scale='Blues')
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_scatter_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a scatter plot visualization."""
        if len(numeric_cols) < 2:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        color_col = text_cols[0] if text_cols else None
        
        fig = px.scatter(results_df,
                         x=x_col,
                         y=y_col,
                         color=color_col,
                         title=f"Relationship: {question}",
                         hover_data=numeric_cols[:3])
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_line_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], 
                         text_cols: List[str], date_cols: List[str]) -> go.Figure:
        """Create a line chart visualization."""
        value_col = ChartFactory._get_best_value_column(numeric_cols)
        
        if date_cols:
            x_col = date_cols[0]
        elif text_cols:
            x_col = text_cols[0]
        else:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        # Sort by x column for proper line progression
        df_sorted = results_df.sort_values(x_col)
        
        fig = px.line(df_sorted,
                      x=x_col,
                      y=value_col,
                      title=f"Trend: {question}",
                      markers=True)
        
        fig.update_layout(height=500)
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        return fig
    
    @staticmethod
    def create_heatmap_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a heatmap visualization."""
        if len(text_cols) < 2 or len(numeric_cols) < 1:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        # Create pivot table for heatmap
        try:
            pivot_df = results_df.pivot_table(
                values=numeric_cols[0],
                index=text_cols[0],
                columns=text_cols[1],
                aggfunc='sum',
                fill_value=0
            )
            
            fig = px.imshow(pivot_df,
                           title=f"Matrix: {question}",
                           color_continuous_scale='RdYlBu_r',
                           aspect='auto')
            
            fig.update_layout(height=500)
            return fig
        except:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
    
    @staticmethod
    def create_box_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a box plot visualization."""
        if not numeric_cols:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        value_col = numeric_cols[0]
        category_col = text_cols[0] if text_cols else None
        
        if category_col:
            fig = px.box(results_df,
                         x=category_col,
                         y=value_col,
                         title=f"Distribution: {question}")
        else:
            fig = px.box(results_df,
                         y=value_col,
                         title=f"Distribution: {question}")
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_area_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], 
                         text_cols: List[str], date_cols: List[str]) -> go.Figure:
        """Create an area chart visualization."""
        value_col = ChartFactory._get_best_value_column(numeric_cols)
        
        if date_cols:
            x_col = date_cols[0]
        elif text_cols:
            x_col = text_cols[0]
        else:
            return ChartFactory.create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        df_sorted = results_df.sort_values(x_col)
        
        fig = px.area(df_sorted,
                      x=x_col,
                      y=value_col,
                      title=f"Cumulative: {question}")
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_bar_chart(results_df: pd.DataFrame, question: str, numeric_cols: List[str], text_cols: List[str]) -> go.Figure:
        """Create a bar chart visualization (default fallback)."""
        value_col = ChartFactory._get_best_value_column(numeric_cols)
        label_col = ChartFactory._get_best_label_column(text_cols)
        
        if not value_col:
            # If no numeric columns, show first few rows as text
            fig = go.Figure()
            fig.add_annotation(text="No numeric data to visualize", 
                             xref="paper", yref="paper", x=0.5, y=0.5,
                             showarrow=False, font=dict(size=16))
            fig.update_layout(title=f"Data: {question}", height=400)
            return fig
        
        if not label_col:
            # Create index-based bar chart
            fig = px.bar(results_df.head(20),
                         y=value_col,
                         title=f"Analysis: {question}")
        else:
            # Limit to top 20 for readability
            df_display = results_df.nlargest(20, value_col)
            fig = px.bar(df_display,
                         x=label_col,
                         y=value_col,
                         title=f"Analysis: {question}")
            
            # Rotate x-axis labels if too many
            if len(df_display) > 10:
                fig.update_xaxes(tickangle=45)
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def _get_best_value_column(numeric_cols: List[str]) -> Optional[str]:
        """Select the best numeric column for values."""
        if not numeric_cols:
            return None
        
        # Prioritize common value columns
        priority_patterns = ['total', 'amount', 'revenue', 'sales', 'value', 'sum', 'count', 'quantity']
        
        for pattern in priority_patterns:
            for col in numeric_cols:
                if pattern.lower() in col.lower():
                    return col
        
        return numeric_cols[0]  # Default to first numeric column
    
    @staticmethod
    def _get_best_label_column(text_cols: List[str]) -> Optional[str]:
        """Select the best text column for labels."""
        if not text_cols:
            return None
        
        # Prioritize common label columns
        priority_patterns = ['name', 'title', 'label', 'category', 'type', 'product', 'store']
        
        for pattern in priority_patterns:
            for col in text_cols:
                if pattern.lower() in col.lower():
                    return col
        
        return text_cols[0]  # Default to first text column


# Legacy function for compatibility
def create_smart_visualization(results_df: pd.DataFrame, question: str) -> Optional[go.Figure]:
    """Legacy compatibility function."""
    return ChartFactory.create_smart_visualization(results_df, question)

