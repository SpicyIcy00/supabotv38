"""
Mobile product list component for responsive dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List


class MobileProductList:
    """Mobile-optimized product list component."""
    
    @staticmethod
    def render_product_list(products_df: pd.DataFrame, title: str = "🏆 Top Products"):
        """
        Render product list in mobile-optimized card layout.
        
        Args:
            products_df: DataFrame containing product data
            title: Section title
        """
        if products_df.empty:
            st.info("No product data available for selected period/stores.")
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            # Add search functionality for mobile
            search_term = st.text_input(
                "🔍 Search products...",
                key="product_search",
                placeholder="Type to filter products..."
            )
            
            # Filter products based on search
            filtered_df = MobileProductList._filter_products(products_df, search_term)
            
            # Render products in card layout
            MobileProductList._render_product_cards(filtered_df)
    
    @staticmethod
    def _filter_products(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
        """
        Filter products based on search term.
        
        Args:
            df: Product DataFrame
            search_term: Search query
            
        Returns:
            Filtered DataFrame
        """
        if not search_term:
            return df
        
        # Search in product name column
        if 'product_name' in df.columns:
            mask = df['product_name'].str.contains(search_term, case=False, na=False)
            return df[mask]
        elif 'Product' in df.columns:
            mask = df['Product'].str.contains(search_term, case=False, na=False)
            return df[mask]
        
        return df
    
    @staticmethod
    def _render_product_cards(df: pd.DataFrame):
        """
        Render products as mobile-optimized cards.
        
        Args:
            df: Product DataFrame
        """
        # Determine column names based on DataFrame structure
        if 'product_name' in df.columns:
            name_col = 'product_name'
            sales_col = 'total_revenue'
            change_col = 'pct_change'
        elif 'Product' in df.columns:
            name_col = 'Product'
            sales_col = 'Sales'
            change_col = 'Δ %'
        else:
            st.error("Invalid product data format")
            return
        
        # Limit to top 10 for mobile performance
        display_df = df.head(10)
        
        for idx, row in display_df.iterrows():
            MobileProductList._render_product_card(
                name=row[name_col],
                sales=row[sales_col],
                change=row.get(change_col, None),
                rank=idx + 1
            )
    
    @staticmethod
    def _render_product_card(name: str, sales: float, change: Any, rank: int):
        """
        Render individual product card.
        
        Args:
            name: Product name
            sales: Sales value
            change: Percentage change
            rank: Product rank
        """
        # Format change value
        change_text, change_class = MobileProductList._format_change(change)
        
        # Format sales value
        sales_text = f"₱{sales:,.0f}" if isinstance(sales, (int, float)) else str(sales)
        
        card_html = f"""
        <div class="mobile-product-card" style="
            background: #1c1e26;
            border: 1px solid #2e303d;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        ">
            <div class="mobile-product-header" style="
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 0.5rem;
            ">
                <div style="display: flex; align-items: center; flex: 1;">
                    <span style="
                        background: #3a47d5;
                        color: white;
                        border-radius: 50%;
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 0.75rem;
                        font-weight: bold;
                        margin-right: 0.75rem;
                        flex-shrink: 0;
                    ">#{rank}</span>
                    <div class="mobile-product-name" style="
                        font-weight: 600;
                        color: #ffffff;
                        font-size: 0.9rem;
                        line-height: 1.3;
                        flex: 1;
                    ">{name}</div>
                </div>
                <div class="mobile-product-stats" style="
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    gap: 0.25rem;
                    margin-left: 0.5rem;
                ">
                    <div class="mobile-product-sales" style="
                        font-size: 1rem;
                        font-weight: bold;
                        color: #00d2ff;
                    ">{sales_text}</div>
                    <div class="mobile-product-change {change_class}" style="
                        font-size: 0.75rem;
                        padding: 0.25rem 0.5rem;
                        border-radius: 4px;
                        font-weight: 500;
                        background: {MobileProductList._get_change_background(change_class)};
                        color: {MobileProductList._get_change_color(change_class)};
                    ">{change_text}</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def _format_change(change: Any) -> tuple:
        """
        Format change value for display.
        
        Args:
            change: Raw change value
            
        Returns:
            Tuple of (formatted_text, css_class)
        """
        if change is None or pd.isna(change):
            return "New", "neutral"
        
        # Handle string format (e.g., "▲ 15.2%")
        if isinstance(change, str):
            if change.startswith('▲'):
                return change, "positive"
            elif change.startswith('▼'):
                return change, "negative"
            elif change == "New":
                return change, "neutral"
            else:
                return change, "neutral"
        
        # Handle numeric format
        if isinstance(change, (int, float)):
            if change > 0:
                return f"▲ {abs(change):.1f}%", "positive"
            elif change < 0:
                return f"▼ {abs(change):.1f}%", "negative"
            else:
                return "→ 0.0%", "neutral"
        
        return str(change), "neutral"
    
    @staticmethod
    def _get_change_background(change_class: str) -> str:
        """Get background color for change indicator."""
        colors = {
            "positive": "rgba(0, 200, 83, 0.2)",
            "negative": "rgba(255, 82, 82, 0.2)",
            "neutral": "rgba(170, 170, 170, 0.2)"
        }
        return colors.get(change_class, "rgba(170, 170, 170, 0.2)")
    
    @staticmethod
    def _get_change_color(change_class: str) -> str:
        """Get text color for change indicator."""
        colors = {
            "positive": "#00c853",
            "negative": "#ff5252",
            "neutral": "#aaaaaa"
        }
        return colors.get(change_class, "#aaaaaa")
    
    @staticmethod
    def render_category_list(categories_df: pd.DataFrame, title: str = "🗂️ Categories"):
        """
        Render category list in mobile-optimized format.
        
        Args:
            categories_df: DataFrame containing category data
            title: Section title
        """
        if categories_df.empty:
            st.info("No category data available.")
            return
        
        with st.container():
            st.markdown(f"### {title}")
            
            # Determine column names
            if 'category' in categories_df.columns:
                name_col = 'category'
                sales_col = 'total_revenue'
                change_col = 'pct_change'
            elif 'Category' in categories_df.columns:
                name_col = 'Category'
                sales_col = 'Sales'
                change_col = 'Δ %'
            else:
                st.error("Invalid category data format")
                return
            
            # Display categories in compact list
            for idx, row in categories_df.head(10).iterrows():
                MobileProductList._render_category_item(
                    name=row[name_col],
                    sales=row[sales_col],
                    change=row.get(change_col, None),
                    rank=idx + 1
                )
    
    @staticmethod
    def _render_category_item(name: str, sales: float, change: Any, rank: int):
        """
        Render individual category item.
        
        Args:
            name: Category name
            sales: Sales value
            change: Percentage change
            rank: Category rank
        """
        change_text, change_class = MobileProductList._format_change(change)
        sales_text = f"₱{sales:,.0f}" if isinstance(sales, (int, float)) else str(sales)
        
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
            transition: background-color 0.2s ease;
        ">
            <div style="display: flex; align-items: center;">
                <span style="
                    background: #00d2ff;
                    color: #1c1e26;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.7rem;
                    font-weight: bold;
                    margin-right: 0.5rem;
                ">#{rank}</span>
                <span style="color: #c7c7c7; font-size: 0.9rem;">{name}</span>
            </div>
            <div style="text-align: right;">
                <div style="color: #00d2ff; font-weight: bold; font-size: 0.9rem;">
                    {sales_text}
                </div>
                <div style="
                    font-size: 0.7rem;
                    color: {MobileProductList._get_change_color(change_class)};
                ">{change_text}</div>
            </div>
        </div>
        """
        
        st.markdown(item_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_compact_summary(products_df: pd.DataFrame, max_items: int = 5):
        """
        Render a compact summary of top products for mobile.
        
        Args:
            products_df: Product DataFrame
            max_items: Maximum number of items to display
        """
        if products_df.empty:
            return
        
        with st.container():
            st.markdown("### 🏆 Top Performers")
            
            # Get top products
            top_products = products_df.head(max_items)
            
            for idx, row in top_products.iterrows():
                name = row.get('product_name', row.get('Product', 'Unknown'))
                sales = row.get('total_revenue', row.get('Sales', 0))
                sales_text = f"₱{sales:,.0f}" if isinstance(sales, (int, float)) else str(sales)
                
                summary_html = f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.5rem;
                    background: #1c1e26;
                    border: 1px solid #2e303d;
                    border-radius: 6px;
                    margin-bottom: 0.25rem;
                ">
                    <span style="color: #c7c7c7; font-size: 0.8rem;">{name}</span>
                    <span style="color: #00d2ff; font-weight: bold; font-size: 0.8rem;">
                        {sales_text}
                    </span>
                </div>
                """
                
                st.markdown(summary_html, unsafe_allow_html=True)
