#!/usr/bin/env python3
"""
Test script for mobile-responsive SupaBot BI Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create test data for mobile responsive testing"""
    
    # Test metrics
    metrics = {
        'current_sales': 1250000,
        'prev_sales': 1100000,
        'current_profit': 250000,
        'prev_profit': 220000,
        'current_transactions': 1250,
        'prev_transactions': 1100,
        'current_products_sold': 3750,
        'prev_products_sold': 3300,
        'avg_transaction_value': 1000,
        'prev_avg_transaction_value': 950
    }
    
    # Test sales trend data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sales_df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(30000, 50000, len(dates)),
        'transactions': np.random.randint(100, 200, len(dates))
    })
    
    # Test top products data
    top_df = pd.DataFrame({
        'product_name': [
            'iPhone 15 Pro', 'Samsung Galaxy S24', 'MacBook Pro M3', 
            'iPad Air', 'AirPods Pro', 'Apple Watch Series 9',
            'Sony WH-1000XM5', 'Nintendo Switch OLED', 'DJI Mini 3 Pro',
            'GoPro Hero 12'
        ],
        'total_revenue': np.random.randint(50000, 200000, 10),
        'total_quantity': np.random.randint(100, 500, 10),
        'percentage_change': np.random.uniform(-20, 50, 10)
    })
    
    # Test categories data
    cat_df = pd.DataFrame({
        'category_name': ['Smartphones', 'Laptops', 'Tablets', 'Accessories', 'Gaming'],
        'total_revenue': np.random.randint(100000, 500000, 5),
        'percentage_change': np.random.uniform(-15, 30, 5)
    })
    
    return metrics, sales_df, top_df, cat_df

def main():
    st.set_page_config(
        page_title="Mobile Responsive Test",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Mobile-Responsive Dashboard Test")
    st.markdown("Testing the new mobile-responsive implementation")
    
    # Create test data
    metrics, sales_df, top_df, cat_df = create_test_data()
    
    # Force mobile mode for testing
    st.session_state.screen_size = 'mobile'
    
    # Test mobile dashboard
    try:
        from supabot.ui.components.mobile_dashboard import MobileDashboard
        
        st.success("✅ Mobile dashboard components imported successfully")
        
        # Test the responsive dashboard
        st.markdown("---")
        st.markdown("### 🎯 Testing Mobile-Responsive Dashboard")
        
        MobileDashboard.render_responsive_dashboard(
            metrics=metrics,
            sales_df=sales_df,
            sales_cat_df=top_df,  # Using top_df as sales_cat_df for testing
            inv_cat_df=cat_df,
            top_change_df=top_df,
            cat_change_df=cat_df,
            time_filter="7D",
            selected_stores=["Test Store"]
        )
        
        st.success("🎉 Mobile-responsive dashboard rendered successfully!")
        
    except Exception as e:
        st.error(f"❌ Error testing mobile dashboard: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    # Test controls
    st.markdown("---")
    st.markdown("### 🧪 Test Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Force Mobile Mode"):
            st.session_state.screen_size = 'mobile'
            st.success("✅ Mobile mode forced")
            st.rerun()
    
    with col2:
        if st.button("Force Desktop Mode"):
            st.session_state.screen_size = 'desktop'
            st.success("✅ Desktop mode forced")
            st.rerun()
    
    # Show current mode
    current_mode = st.session_state.get('screen_size', 'not set')
    st.info(f"Current mode: {current_mode}")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📋 Testing Instructions")
    
    st.markdown("""
    #### 🎯 What to Test:
    
    1. **KPI Cards**: Should be in a 2x2 grid on mobile
    2. **Charts**: Should be stacked vertically and scrollable
    3. **Product Lists**: Should be card-based with proper spacing
    4. **Responsive Design**: Should adapt to different screen sizes
    
    #### 📱 Mobile Features:
    - Touch-friendly buttons (44px minimum)
    - Scrollable chart containers
    - Card-based product lists
    - Proper spacing and typography
    
    #### 🖥️ Desktop Features:
    - 4-column KPI grid
    - Side-by-side charts
    - Standard table layouts
    - Full-width utilization
    """)
    
    # CSS Test
    st.markdown("---")
    st.markdown("### 🎨 CSS Test")
    
    try:
        from mobile_responsive_css import get_mobile_responsive_css
        st.success("✅ Mobile-responsive CSS loaded successfully")
        
        # Show a sample of the CSS
        css = get_mobile_responsive_css()
        with st.expander("View CSS"):
            st.code(css[:1000] + "..." if len(css) > 1000 else css)
            
    except Exception as e:
        st.error(f"❌ Error loading CSS: {e}")

if __name__ == "__main__":
    main()
