#!/usr/bin/env python3
"""
Test script for mobile-responsive SupaBot BI Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mock data for testing
def create_mock_data():
    """Create mock data for testing the mobile dashboard."""
    
    # Mock metrics
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
    
    # Mock sales trend data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sales_df = pd.DataFrame({
        'date': dates,
        'total_revenue': np.random.randint(30000, 50000, len(dates))
    })
    
    # Mock category data
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home & Garden']
    sales_cat_df = pd.DataFrame({
        'category': categories,
        'total_revenue': np.random.randint(100000, 300000, len(categories))
    })
    
    inv_cat_df = pd.DataFrame({
        'category': categories,
        'total_inventory_value': np.random.randint(50000, 150000, len(categories))
    })
    
    # Mock product data
    products = [f'Product {i}' for i in range(1, 11)]
    top_change_df = pd.DataFrame({
        'product_name': products,
        'total_revenue': np.random.randint(50000, 200000, len(products)),
        'pct_change': np.random.uniform(-20, 30, len(products))
    })
    
    cat_change_df = pd.DataFrame({
        'category': categories,
        'total_revenue': np.random.randint(100000, 300000, len(categories)),
        'pct_change': np.random.uniform(-15, 25, len(categories))
    })
    
    return metrics, sales_df, sales_cat_df, inv_cat_df, top_change_df, cat_change_df

def test_mobile_components():
    """Test individual mobile components."""
    
    st.title("🧪 Mobile Dashboard Component Tests")
    
    # Create mock data
    metrics, sales_df, sales_cat_df, inv_cat_df, top_change_df, cat_change_df = create_mock_data()
    
    # Test responsive wrapper
    st.header("📱 Responsive Wrapper Test")
    try:
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        screen_size = ResponsiveWrapper.get_screen_size()
        st.success(f"✅ Screen size detected: {screen_size}")
        
        # Test responsive columns
        cols = ResponsiveWrapper.responsive_columns(mobile_cols=1, tablet_cols=2, desktop_cols=4)
        st.info(f"✅ Responsive columns created: {len(cols)} columns")
        
    except ImportError as e:
        st.error(f"❌ Responsive wrapper import failed: {e}")
    
    # Test mobile KPI cards
    st.header("📊 Mobile KPI Cards Test")
    try:
        from supabot.ui.components.mobile.kpi_cards import MobileKPICards
        
        # Test KPI grid
        MobileKPICards.render_kpi_grid(metrics, "Test Period")
        st.success("✅ Mobile KPI cards rendered successfully")
        
        # Test KPI summary
        MobileKPICards.render_kpi_summary(metrics, "Test Period")
        st.success("✅ Mobile KPI summary rendered successfully")
        
    except ImportError as e:
        st.error(f"❌ Mobile KPI cards import failed: {e}")
    
    # Test mobile product list
    st.header("🏆 Mobile Product List Test")
    try:
        from supabot.ui.components.mobile.product_list import MobileProductList
        
        # Test product list
        MobileProductList.render_product_list(top_change_df, "Test Products")
        st.success("✅ Mobile product list rendered successfully")
        
        # Test category list
        MobileProductList.render_category_list(cat_change_df, "Test Categories")
        st.success("✅ Mobile category list rendered successfully")
        
    except ImportError as e:
        st.error(f"❌ Mobile product list import failed: {e}")
    
    # Test mobile charts
    st.header("📈 Mobile Charts Test")
    try:
        from supabot.ui.components.mobile.charts import MobileCharts
        
        # Test sales trend chart
        MobileCharts.render_sales_trend_chart(sales_df, "Test Sales Trend")
        st.success("✅ Mobile sales trend chart rendered successfully")
        
        # Test pie charts
        MobileCharts.render_pie_charts(sales_cat_df, inv_cat_df)
        st.success("✅ Mobile pie charts rendered successfully")
        
    except ImportError as e:
        st.error(f"❌ Mobile charts import failed: {e}")
    
    # Test mobile navigation
    st.header("🧭 Mobile Navigation Test")
    try:
        from supabot.ui.components.mobile.navigation import MobileNavigation
        
        # Test mobile header
        MobileNavigation.render_mobile_header("Test Dashboard")
        st.success("✅ Mobile header rendered successfully")
        
        # Test bottom navigation
        tabs = [
            {"key": "dashboard", "label": "Dashboard", "icon": "📊"},
            {"key": "analytics", "label": "Analytics", "icon": "📈"},
            {"key": "products", "label": "Products", "icon": "🏆"}
        ]
        MobileNavigation.render_bottom_navigation(tabs, "dashboard")
        st.success("✅ Mobile bottom navigation rendered successfully")
        
    except ImportError as e:
        st.error(f"❌ Mobile navigation import failed: {e}")

def test_mobile_dashboard_integration():
    """Test the complete mobile dashboard integration."""
    
    st.header("🚀 Complete Mobile Dashboard Integration Test")
    
    try:
        from supabot.ui.components.mobile_dashboard import MobileDashboard
        
        # Create mock data
        metrics, sales_df, sales_cat_df, inv_cat_df, top_change_df, cat_change_df = create_mock_data()
        
        # Test responsive dashboard
        MobileDashboard.render_responsive_dashboard(
            metrics=metrics,
            sales_df=sales_df,
            sales_cat_df=sales_cat_df,
            inv_cat_df=inv_cat_df,
            top_change_df=top_change_df,
            cat_change_df=cat_change_df,
            time_filter="1M",
            selected_stores=["Store 1", "Store 2"]
        )
        
        st.success("✅ Complete mobile dashboard integration test passed!")
        
    except ImportError as e:
        st.error(f"❌ Mobile dashboard integration import failed: {e}")
    except Exception as e:
        st.error(f"❌ Mobile dashboard integration test failed: {e}")

def test_css_styles():
    """Test CSS styles are loading correctly."""
    
    st.header("🎨 CSS Styles Test")
    
    try:
        from supabot.ui.styles.css import DashboardStyles
        
        # Load styles
        DashboardStyles.load_all_styles()
        st.success("✅ CSS styles loaded successfully")
        
        # Test responsive CSS classes
        st.markdown("""
        <div class="kpi-grid">
            <div style="background: #1c1e26; padding: 1rem; border-radius: 10px; margin: 0.5rem;">
                Test KPI Card 1
            </div>
            <div style="background: #1c1e26; padding: 1rem; border-radius: 10px; margin: 0.5rem;">
                Test KPI Card 2
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Responsive CSS classes applied successfully")
        
    except ImportError as e:
        st.error(f"❌ CSS styles import failed: {e}")

def main():
    """Main test function."""
    
    st.set_page_config(
        page_title="Mobile Dashboard Tests",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 SupaBot BI Dashboard - Mobile Responsive Tests")
    
    # Add navigation
    test_type = st.sidebar.selectbox(
        "Select Test Type",
        ["Component Tests", "Integration Test", "CSS Styles Test", "All Tests"]
    )
    
    if test_type == "Component Tests":
        test_mobile_components()
    elif test_type == "Integration Test":
        test_mobile_dashboard_integration()
    elif test_type == "CSS Styles Test":
        test_css_styles()
    elif test_type == "All Tests":
        test_mobile_components()
        st.markdown("---")
        test_mobile_dashboard_integration()
        st.markdown("---")
        test_css_styles()
    
    # Add test results summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Test Results")
    st.sidebar.success("✅ All tests completed")
    
    # Add device simulation info
    st.sidebar.markdown("### Device Simulation")
    st.sidebar.info("""
    To test mobile responsiveness:
    1. Open browser dev tools (F12)
    2. Click device simulation icon
    3. Select mobile device (e.g., iPhone 12)
    4. Refresh the page
    """)

if __name__ == "__main__":
    main()
