#!/usr/bin/env python3
"""
Test script to verify desktop and mobile layouts work correctly
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="Desktop/Mobile Test", page_icon="📱", layout="wide")
    
    st.title("🖥️ Desktop vs 📱 Mobile Layout Test")
    
    # Test 1: Check if mobile components can be imported
    st.header("🔍 Import Test")
    
    try:
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        st.success("✅ ResponsiveWrapper imported successfully")
        
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
        
        # Show current layout mode
        if screen_size == 'mobile':
            st.success("🎯 Currently using MOBILE layout")
        else:
            st.success("🎯 Currently using DESKTOP layout")
        
    except Exception as e:
        st.error(f"❌ Screen size detection failed: {e}")
    
    # Test 3: Force mobile mode for testing
    st.header("🧪 Force Mobile Mode Test")
    
    if st.button("Force Mobile Mode"):
        st.session_state.screen_size = 'mobile'
        st.success("✅ Mobile mode forced - refresh to see mobile layout")
        st.rerun()
    
    if st.button("Force Desktop Mode"):
        st.session_state.screen_size = 'desktop'
        st.success("✅ Desktop mode forced - refresh to see desktop layout")
        st.rerun()
    
    # Test 4: Show current session state
    st.header("📊 Current Session State")
    st.info(f"Screen size in session: {st.session_state.get('screen_size', 'not set')}")
    
    # Test 5: Instructions
    st.header("📋 Testing Instructions")
    
    st.markdown("""
    ### 🖥️ Desktop Testing:
    1. **Current View**: You should see the desktop layout
    2. **Check**: All data should be visible and properly formatted
    3. **Verify**: KPI cards should be in a 4-column layout
    
    ### 📱 Mobile Testing:
    1. **On Desktop**: Use browser dev tools (F12)
    2. **Click device simulation icon** (mobile/tablet icon)
    3. **Select a mobile device** (e.g., iPhone 12)
    4. **Refresh the page**
    5. **Check**: You should see mobile-optimized layout
    
    ### 🔧 Troubleshooting:
    - **No Data**: Check database connection
    - **Wrong Layout**: Clear browser cache and refresh
    - **Import Errors**: Restart Streamlit server
    """)
    
    # Test 6: Quick layout preview
    st.header("👀 Layout Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖥️ Desktop Layout")
        st.markdown("""
        - 4-column KPI grid
        - Side-by-side charts
        - Full-width tables
        - Standard navigation
        """)
    
    with col2:
        st.markdown("### 📱 Mobile Layout")
        st.markdown("""
        - 2x2 KPI grid
        - Stacked charts
        - Card-based lists
        - Bottom navigation
        """)

if __name__ == "__main__":
    main()
