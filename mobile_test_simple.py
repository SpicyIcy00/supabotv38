#!/usr/bin/env python3
"""
Simple mobile test for SupaBot BI Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="Mobile Test", page_icon="📱", layout="wide")
    
    st.title("📱 Mobile UI Test")
    
    # Test 1: Check if mobile components can be imported
    st.header("🔍 Import Test")
    
    try:
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        st.success("✅ ResponsiveWrapper imported successfully")
        
        from supabot.ui.components.mobile.kpi_cards import MobileKPICards
        st.success("✅ MobileKPICards imported successfully")
        
        from supabot.ui.components.mobile.charts import MobileCharts
        st.success("✅ MobileCharts imported successfully")
        
        from supabot.ui.components.mobile.product_list import MobileProductList
        st.success("✅ MobileProductList imported successfully")
        
        from supabot.ui.components.mobile.navigation import MobileNavigation
        st.success("✅ MobileNavigation imported successfully")
        
        from supabot.ui.components.mobile_dashboard import MobileDashboard
        st.success("✅ MobileDashboard imported successfully")
        
    except ImportError as e:
        st.error(f"❌ Import failed: {e}")
        return
    
    # Test 2: Check screen size detection
    st.header("📱 Screen Size Detection")
    
    try:
        screen_size = ResponsiveWrapper.get_screen_size()
        st.info(f"Detected screen size: {screen_size}")
        
        # Force mobile for testing
        st.session_state.screen_size = 'mobile'
        st.success("✅ Screen size detection working")
        
    except Exception as e:
        st.error(f"❌ Screen size detection failed: {e}")
    
    # Test 3: Test mobile KPI cards
    st.header("📊 Mobile KPI Test")
    
    try:
        # Create mock metrics
        metrics = {
            'current_sales': 1250000,
            'prev_sales': 1100000,
            'current_profit': 375000,
            'prev_profit': 330000,
            'current_transactions': 1250,
            'prev_transactions': 1100,
            'avg_transaction_value': 1000,
            'prev_avg_transaction_value': 1000
        }
        
        MobileKPICards.render_kpi_grid(metrics, "Test Period")
        st.success("✅ Mobile KPI cards rendered successfully")
        
    except Exception as e:
        st.error(f"❌ Mobile KPI test failed: {e}")
    
    # Test 4: Test mobile charts
    st.header("📈 Mobile Charts Test")
    
    try:
        # Create mock sales data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sales_df = pd.DataFrame({
            'date': dates,
            'total_revenue': np.random.randint(30000, 50000, len(dates))
        })
        
        MobileCharts.render_sales_trend_chart(sales_df, "Test Sales Trend")
        st.success("✅ Mobile charts rendered successfully")
        
    except Exception as e:
        st.error(f"❌ Mobile charts test failed: {e}")
    
    # Test 5: Mobile navigation
    st.header("🧭 Mobile Navigation Test")
    
    try:
        MobileNavigation.render_mobile_header("Test Dashboard")
        st.success("✅ Mobile navigation rendered successfully")
        
    except Exception as e:
        st.error(f"❌ Mobile navigation test failed: {e}")
    
    # Instructions for mobile testing
    st.sidebar.markdown("### 📱 Mobile Testing Instructions")
    st.sidebar.info("""
    1. **On Desktop**: Use browser dev tools (F12)
    2. **Click device simulation icon** (mobile/tablet icon)
    3. **Select a mobile device** (e.g., iPhone 12)
    4. **Refresh the page**
    5. **Check if mobile layout appears**
    """)
    
    st.sidebar.markdown("### 🔧 Troubleshooting")
    st.sidebar.warning("""
    If mobile components don't work:
    1. Check import paths
    2. Verify all files exist
    3. Restart Streamlit server
    4. Clear browser cache
    """)

if __name__ == "__main__":
    main()
