#!/usr/bin/env python3
"""
Debug script to test data loading and mobile detection
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def main():
    st.set_page_config(page_title="Data Loading Debug", page_icon="🔍")
    
    st.title("🔍 Data Loading Debug")
    
    # Test 1: Session State
    st.header("📊 Session State Check")
    
    if "dashboard_time_filter" not in st.session_state:
        st.error("❌ dashboard_time_filter not in session state")
        st.session_state.dashboard_time_filter = "7D"
    else:
        st.success(f"✅ dashboard_time_filter: {st.session_state.dashboard_time_filter}")
    
    if "dashboard_store_filter" not in st.session_state:
        st.error("❌ dashboard_store_filter not in session state")
        st.session_state.dashboard_store_filter = ["Rockwell", "Greenhills", "Magnolia", "North Edsa", "Fairview"]
    else:
        st.success(f"✅ dashboard_store_filter: {st.session_state.dashboard_store_filter}")
    
    # Test 2: Mobile Detection
    st.header("📱 Mobile Detection Test")
    
    try:
        from supabot.ui.components.mobile.responsive_wrapper import ResponsiveWrapper
        screen_size = ResponsiveWrapper.get_screen_size()
        st.success(f"✅ Screen size detected: {screen_size}")
        
        if screen_size == 'mobile':
            st.info("🎯 Mobile layout should be used")
        else:
            st.info("💻 Desktop layout should be used")
            
    except Exception as e:
        st.error(f"❌ Mobile detection failed: {e}")
    
    # Test 3: Data Loading Functions
    st.header("📈 Data Loading Test")
    
    try:
        from appv38 import (
            get_dashboard_metrics, 
            get_sales_trend_data, 
            get_sales_by_category_data,
            get_inventory_by_category_data,
            get_top_products_with_change,
            get_categories_with_change
        )
        
        st.success("✅ Data loading functions imported")
        
        # Test metrics
        st.subheader("Metrics Test")
        try:
            metrics = get_dashboard_metrics("7D", None)
            if metrics:
                st.success(f"✅ Metrics loaded: {len(metrics)} items")
                st.json(metrics)
            else:
                st.warning("⚠️ No metrics data returned")
        except Exception as e:
            st.error(f"❌ Metrics loading failed: {e}")
        
        # Test sales trend
        st.subheader("Sales Trend Test")
        try:
            sales_df = get_sales_trend_data("7D", None)
            if sales_df is not None and not sales_df.empty:
                st.success(f"✅ Sales trend loaded: {len(sales_df)} rows")
                st.dataframe(sales_df.head())
            else:
                st.warning("⚠️ No sales trend data returned")
        except Exception as e:
            st.error(f"❌ Sales trend loading failed: {e}")
        
        # Test top products
        st.subheader("Top Products Test")
        try:
            top_df = get_top_products_with_change("7D", None)
            if top_df is not None and not top_df.empty:
                st.success(f"✅ Top products loaded: {len(top_df)} rows")
                st.dataframe(top_df.head())
            else:
                st.warning("⚠️ No top products data returned")
        except Exception as e:
            st.error(f"❌ Top products loading failed: {e}")
        
        # Test categories
        st.subheader("Categories Test")
        try:
            cat_df = get_categories_with_change("7D", None)
            if cat_df is not None and not cat_df.empty:
                st.success(f"✅ Categories loaded: {len(cat_df)} rows")
                st.dataframe(cat_df.head())
            else:
                st.warning("⚠️ No categories data returned")
        except Exception as e:
            st.error(f"❌ Categories loading failed: {e}")
        
    except ImportError as e:
        st.error(f"❌ Failed to import data functions: {e}")
    
    # Test 4: Database Connection
    st.header("🗄️ Database Connection Test")
    
    try:
        from supabot.core.database import get_db_manager
        db_manager = get_db_manager()
        status = db_manager.get_pool_status()
        
        if status.get("status") == "healthy":
            st.success("✅ Database connection healthy")
        else:
            st.error(f"❌ Database connection issues: {status}")
            
        st.json(status)
        
    except Exception as e:
        st.error(f"❌ Database connection test failed: {e}")
    
    # Test 5: Force Mobile Mode
    st.header("🧪 Force Mobile Mode Test")
    
    if st.button("Force Mobile Mode"):
        st.session_state.screen_size = 'mobile'
        st.success("✅ Mobile mode forced - refresh to see mobile layout")
        st.rerun()
    
    if st.button("Force Desktop Mode"):
        st.session_state.screen_size = 'desktop'
        st.success("✅ Desktop mode forced - refresh to see desktop layout")
        st.rerun()
    
    # Test 6: Current Layout Mode
    st.header("🎨 Current Layout Mode")
    
    current_screen = st.session_state.get('screen_size', 'not set')
    st.info(f"Current screen size in session: {current_screen}")
    
    if current_screen == 'mobile':
        st.success("🎯 Currently using MOBILE layout")
    elif current_screen == 'desktop':
        st.success("🎯 Currently using DESKTOP layout")
    else:
        st.info("🎯 Screen size not set - will be detected automatically")

if __name__ == "__main__":
    main()
