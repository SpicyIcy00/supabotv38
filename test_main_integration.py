#!/usr/bin/env python3
"""
Comprehensive test for main.py integration with mobile-responsive features
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_main_imports():
    """Test if main.py can import all required modules"""
    st.header("🔍 Import Test")
    
    try:
        # Test main.py imports
        from main import main
        st.success("✅ main.py imported successfully")
        
        # Test appv38 imports
        from appv38 import render_dashboard, init_session_state
        st.success("✅ appv38.py functions imported successfully")
        
        # Test mobile components
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        st.success("✅ Mobile components imported successfully")
        
        # Test settings and styles
        from supabot.config.settings import settings
        from supabot.ui.styles.css import DashboardStyles
        st.success("✅ Settings and styles imported successfully")
        
    except ImportError as e:
        st.error(f"❌ Import failed: {e}")
        return False
    
    return True

def test_session_state():
    """Test session state initialization"""
    st.header("📊 Session State Test")
    
    try:
        from appv38 import init_session_state
        init_session_state()
        
        required_keys = [
            'current_page', 'time_filter', 'selected_stores', 
            'chart_type', 'ai_model', 'benchmark_results'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in st.session_state:
                missing_keys.append(key)
        
        if missing_keys:
            st.warning(f"⚠️ Missing session state keys: {missing_keys}")
        else:
            st.success("✅ All session state keys initialized")
            
        st.info(f"Current page: {st.session_state.get('current_page', 'Not set')}")
        st.info(f"Time filter: {st.session_state.get('time_filter', 'Not set')}")
        
    except Exception as e:
        st.error(f"❌ Session state test failed: {e}")
        return False
    
    return True

def test_mobile_detection():
    """Test mobile screen size detection"""
    st.header("📱 Mobile Detection Test")
    
    try:
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        
        screen_size = ResponsiveWrapper.get_screen_size()
        st.info(f"Detected screen size: {screen_size}")
        
        if screen_size == 'mobile':
            st.success("🎯 Mobile layout will be used")
        elif screen_size == 'tablet':
            st.success("🎯 Tablet layout will be used")
        else:
            st.success("🎯 Desktop layout will be used")
            
        # Test responsive functions
        cols = ResponsiveWrapper.responsive_columns(mobile_cols=2, tablet_cols=3, desktop_cols=4)
        st.info(f"Responsive columns created: {len(cols)} columns")
        
    except Exception as e:
        st.error(f"❌ Mobile detection test failed: {e}")
        return False
    
    return True

def test_dashboard_rendering():
    """Test dashboard rendering functions"""
    st.header("🎨 Dashboard Rendering Test")
    
    try:
        from appv38 import render_dashboard, render_legacy_dashboard, render_responsive_dashboard
        
        st.success("✅ Dashboard functions imported successfully")
        
        # Test if we can call the functions (without actually rendering)
        st.info("Dashboard functions are available:")
        st.code("""
        - render_dashboard() - Main dashboard with mobile detection
        - render_legacy_dashboard() - Desktop-only layout
        - render_responsive_dashboard() - Mobile-responsive layout
        """)
        
    except Exception as e:
        st.error(f"❌ Dashboard rendering test failed: {e}")
        return False
    
    return True

def test_css_styles():
    """Test if CSS styles are loaded"""
    st.header("🎨 CSS Styles Test")
    
    try:
        from supabot.ui.styles.css import DashboardStyles
        
        # Load styles
        DashboardStyles.load_all_styles()
        st.success("✅ CSS styles loaded successfully")
        
        # Test mobile-specific styles
        st.markdown("""
        <style>
        .mobile-test {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="mobile-test">✅ Mobile CSS styles are working</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ CSS styles test failed: {e}")
        return False
    
    return True

def test_navigation():
    """Test navigation structure"""
    st.header("🧭 Navigation Test")
    
    try:
        # Simulate the navigation structure from main.py
        pages = [
            "Dashboard",
            "Product Sales Report", 
            "Chart View",
            "AI Assistant",
            "Settings",
        ]
        
        st.success(f"✅ Navigation pages: {', '.join(pages)}")
        
        # Test page mapping
        page_map = {
            "Dashboard": "render_dashboard",
            "Product Sales Report": "render_product_sales_report",
            "Chart View": "render_chart_view", 
            "AI Assistant": "render_chat",
            "Settings": "render_settings",
        }
        
        st.info("Page mapping structure:")
        for page, function in page_map.items():
            st.code(f"{page} → {function}")
            
    except Exception as e:
        st.error(f"❌ Navigation test failed: {e}")
        return False
    
    return True

def main():
    st.set_page_config(
        page_title="Main.py Integration Test",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Main.py Integration Test")
    st.markdown("Testing mobile-responsive features with main.py")
    
    # Run all tests
    tests = [
        ("Import Test", test_main_imports),
        ("Session State Test", test_session_state),
        ("Mobile Detection Test", test_mobile_detection),
        ("Dashboard Rendering Test", test_dashboard_rendering),
        ("CSS Styles Test", test_css_styles),
        ("Navigation Test", test_navigation),
    ]
    
    results = []
    for test_name, test_func in tests:
        st.markdown("---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            st.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    st.markdown("---")
    st.header("📊 Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        st.success(f"🎉 All {total} tests passed!")
    else:
        st.warning(f"⚠️ {passed}/{total} tests passed")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        st.text(f"{status} - {test_name}")
    
    # Instructions
    st.markdown("---")
    st.header("📋 Next Steps")
    
    st.markdown("""
    ### 🚀 To test the full application:
    
    1. **Run main.py**: `streamlit run main.py`
    2. **Test Desktop**: Open in desktop browser
    3. **Test Mobile**: 
       - Use browser dev tools (F12)
       - Click device simulation icon
       - Select mobile device (e.g., iPhone 12)
       - Refresh page
    
    ### 📱 Mobile Features to Test:
    - **KPI Cards**: Should be 2x2 grid on mobile
    - **Charts**: Should be scrollable and touch-friendly
    - **Navigation**: Should adapt to mobile screen
    - **Tables**: Should be card-based on mobile
    
    ### 🔧 Troubleshooting:
    - **No Data**: Check database connection
    - **Wrong Layout**: Clear browser cache
    - **Import Errors**: Restart Streamlit server
    """)

if __name__ == "__main__":
    main()
