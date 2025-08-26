#!/usr/bin/env python3
"""
Simple test to force mobile mode and verify mobile components work
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_mock_data():
    """Create mock data for testing"""
    # Mock metrics
    metrics = {
        'total_sales': 1250000,
        'total_transactions': 1250,
        'average_order_value': 1000,
        'total_products_sold': 3750
    }
    
    # Mock sales trend
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sales_df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(30000, 50000, len(dates)),
        'transactions': np.random.randint(100, 200, len(dates))
    })
    
    # Mock top products
    top_df = pd.DataFrame({
        'product_name': [f'Product {i}' for i in range(1, 11)],
        'total_revenue': np.random.randint(50000, 200000, 10),
        'total_quantity': np.random.randint(100, 500, 10),
        'percentage_change': np.random.uniform(-20, 50, 10)
    })
    
    # Mock categories
    cat_df = pd.DataFrame({
        'category_name': [f'Category {i}' for i in range(1, 6)],
        'total_revenue': np.random.randint(100000, 500000, 5),
        'percentage_change': np.random.uniform(-15, 30, 5)
    })
    
    return metrics, sales_df, top_df, cat_df

def main():
    st.set_page_config(page_title="Mobile Force Test", page_icon="📱")
    
    st.title("📱 Mobile Force Test")
    st.markdown("Testing mobile components with forced mobile mode")
    
    # Force mobile mode
    st.session_state.screen_size = 'mobile'
    
    # Create mock data
    metrics, sales_df, top_df, cat_df = create_mock_data()
    
    # Test mobile components
    try:
        from supabot.ui.components.mobile.kpi_cards import MobileKPICards
        from supabot.ui.components.mobile.charts import MobileCharts
        from supabot.ui.components.mobile.product_list import MobileProductList
        
        st.success("✅ Mobile components imported successfully")
        
        # Test KPI Cards
        st.header("📊 Mobile KPI Cards")
        MobileKPICards.render_kpi_grid(metrics)
        
        # Test Charts
        st.header("📈 Mobile Charts")
        MobileCharts.render_sales_trend_chart(sales_df)
        
        # Test Product List
        st.header("🏆 Mobile Product List")
        MobileProductList.render_product_list(top_df)
        
        # Test Category List
        st.header("📂 Mobile Category List")
        MobileProductList.render_category_list(cat_df)
        
        st.success("🎉 All mobile components working!")
        
    except Exception as e:
        st.error(f"❌ Mobile components failed: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    # Show current mode
    st.header("🎯 Current Mode")
    st.info(f"Screen size: {st.session_state.get('screen_size', 'not set')}")
    
    if st.button("Switch to Desktop Mode"):
        st.session_state.screen_size = 'desktop'
        st.success("✅ Switched to desktop mode - refresh to see changes")
        st.rerun()
    
    if st.button("Switch to Mobile Mode"):
        st.session_state.screen_size = 'mobile'
        st.success("✅ Switched to mobile mode - refresh to see changes")
        st.rerun()

if __name__ == "__main__":
    main()
