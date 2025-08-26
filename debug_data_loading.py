#!/usr/bin/env python3
"""
Debug script to test data loading for the mobile dashboard
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def main():
    st.set_page_config(
        page_title="Data Loading Debug",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Data Loading Debug")
    st.markdown("Testing data loading to identify why dashboard is empty")
    
    # Test database connection
    st.markdown("## 1. Database Connection Test")
    try:
        from supabot.core.database import get_db_manager
        db_manager = get_db_manager()
        conn = db_manager.create_connection()
        
        if conn:
            st.success("✅ Database connection successful")
            
            # Test basic query
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as total_stores FROM stores")
            result = cur.fetchone()
            st.info(f"📊 Total stores in database: {result[0] if result else 'Unknown'}")
            
            cur.execute("SELECT COUNT(*) as total_transactions FROM transactions")
            result = cur.fetchone()
            st.info(f"📊 Total transactions in database: {result[0] if result else 'Unknown'}")
            
            cur.execute("SELECT COUNT(*) as total_products FROM products")
            result = cur.fetchone()
            st.info(f"📊 Total products in database: {result[0] if result else 'Unknown'}")
            
            conn.close()
        else:
            st.error("❌ Database connection failed")
            return
            
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return
    
    # Test data fetching functions
    st.markdown("## 2. Data Fetching Test")
    
    try:
        from appv38 import (
            get_store_list, 
            get_dashboard_metrics, 
            get_top_products_with_change,
            get_categories_with_change,
            get_sales_by_category_pie,
            get_inventory_by_category_pie,
            get_daily_trend,
            resolve_store_ids
        )
        
        st.success("✅ Data fetching functions imported successfully")
        
        # Test store list
        st.markdown("### Store List Test")
        store_df = get_store_list()
        if store_df is not None and not store_df.empty:
            st.success(f"✅ Found {len(store_df)} stores")
            st.dataframe(store_df.head())
        else:
            st.error("❌ No stores found")
            return
        
        # Test with different time filters
        time_filters = ["1D", "7D", "1M", "6M", "1Y"]
        
        for time_filter in time_filters:
            st.markdown(f"### Testing Time Filter: {time_filter}")
            
            # Test metrics
            try:
                metrics = get_dashboard_metrics(time_filter, None)
                if metrics and any(v > 0 for v in [metrics.get('current_sales', 0), metrics.get('current_transactions', 0)]):
                    st.success(f"✅ Metrics found for {time_filter}")
                    st.json(metrics)
                else:
                    st.warning(f"⚠️ No metrics data for {time_filter}")
            except Exception as e:
                st.error(f"❌ Error getting metrics for {time_filter}: {e}")
            
            # Test top products
            try:
                top_df = get_top_products_with_change(time_filter, None)
                if not top_df.empty:
                    st.success(f"✅ Found {len(top_df)} top products for {time_filter}")
                    st.dataframe(top_df.head())
                else:
                    st.warning(f"⚠️ No top products data for {time_filter}")
            except Exception as e:
                st.error(f"❌ Error getting top products for {time_filter}: {e}")
            
            # Test categories
            try:
                cat_df = get_categories_with_change(time_filter, None)
                if not cat_df.empty:
                    st.success(f"✅ Found {len(cat_df)} categories for {time_filter}")
                    st.dataframe(cat_df.head())
                else:
                    st.warning(f"⚠️ No categories data for {time_filter}")
            except Exception as e:
                st.error(f"❌ Error getting categories for {time_filter}: {e}")
            
            # Test sales by category
            try:
                sales_cat_df = get_sales_by_category_pie(time_filter, None)
                if not sales_cat_df.empty:
                    st.success(f"✅ Found {len(sales_cat_df)} sales categories for {time_filter}")
                    st.dataframe(sales_cat_df.head())
                else:
                    st.warning(f"⚠️ No sales by category data for {time_filter}")
            except Exception as e:
                st.error(f"❌ Error getting sales by category for {time_filter}: {e}")
            
            # Test daily trend
            try:
                days_map = {"1D":1, "7D":7, "1M":30, "6M":180, "1Y":365}
                daily_df = get_daily_trend(days_map.get(time_filter, 7), None)
                if not daily_df.empty:
                    st.success(f"✅ Found {len(daily_df)} daily trend records for {time_filter}")
                    st.dataframe(daily_df.head())
                else:
                    st.warning(f"⚠️ No daily trend data for {time_filter}")
            except Exception as e:
                st.error(f"❌ Error getting daily trend for {time_filter}: {e}")
            
            st.markdown("---")
        
        # Test with specific store filters
        st.markdown("## 3. Store Filter Test")
        
        # Get first few store IDs
        if store_df is not None and not store_df.empty:
            test_store_ids = store_df['id'].head(3).tolist()
            st.info(f"Testing with store IDs: {test_store_ids}")
            
            # Test with store filter
            try:
                metrics = get_dashboard_metrics("7D", test_store_ids)
                if metrics and any(v > 0 for v in [metrics.get('current_sales', 0), metrics.get('current_transactions', 0)]):
                    st.success("✅ Metrics found with store filter")
                    st.json(metrics)
                else:
                    st.warning("⚠️ No metrics data with store filter")
            except Exception as e:
                st.error(f"❌ Error getting metrics with store filter: {e}")
        
    except Exception as e:
        st.error(f"❌ Error importing data fetching functions: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    # Test mobile dashboard rendering
    st.markdown("## 4. Mobile Dashboard Test")
    
    try:
        from supabot.ui.components.mobile_dashboard import MobileDashboard
        
        st.success("✅ Mobile dashboard components imported successfully")
        
        # Create test data
        test_metrics = {
            'current_sales': 1000000,
            'prev_sales': 900000,
            'current_profit': 200000,
            'prev_profit': 180000,
            'current_transactions': 1000,
            'prev_transactions': 900,
            'current_products_sold': 3000,
            'prev_products_sold': 2700
        }
        
        # Create test DataFrames
        test_sales_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'sales': [100000, 120000, 110000, 130000, 140000, 150000, 160000]
        })
        
        test_top_df = pd.DataFrame({
            'product_name': ['Product A', 'Product B', 'Product C'],
            'total_revenue': [50000, 40000, 30000],
            'percentage_change': [10.5, -5.2, 15.8]
        })
        
        test_cat_df = pd.DataFrame({
            'category_name': ['Category A', 'Category B'],
            'total_revenue': [200000, 150000],
            'percentage_change': [8.3, -2.1]
        })
        
        # Test mobile dashboard rendering
        st.markdown("### Testing Mobile Dashboard with Test Data")
        
        MobileDashboard.render_responsive_dashboard(
            metrics=test_metrics,
            sales_df=test_sales_df,
            sales_cat_df=test_top_df,
            inv_cat_df=test_cat_df,
            top_change_df=test_top_df,
            cat_change_df=test_cat_df,
            time_filter="7D",
            selected_stores=["Test Store"]
        )
        
        st.success("✅ Mobile dashboard rendered successfully with test data")
        
    except Exception as e:
        st.error(f"❌ Error testing mobile dashboard: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
