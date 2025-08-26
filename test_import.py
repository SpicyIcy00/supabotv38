#!/usr/bin/env python3
"""
Test file to isolate import issues
"""

import sys
import traceback

def test_imports():
    """Test each import step individually."""
    
    print("Testing imports step by step...")
    
    # Test 1: Basic imports
    try:
        import streamlit as st
        print("✅ streamlit import successful")
    except Exception as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    # Test 2: supabot.config.settings
    try:
        from supabot.config.settings import settings
        print("✅ supabot.config.settings import successful")
    except Exception as e:
        print(f"❌ supabot.config.settings import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: supabot.ui.styles.css
    try:
        from supabot.ui.styles.css import DashboardStyles
        print("✅ supabot.ui.styles.css import successful")
    except Exception as e:
        print(f"❌ supabot.ui.styles.css import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: appv38 functions
    try:
        from appv38 import (
            init_session_state,
            render_dashboard,
            render_product_sales_report,
            render_chart_view,
            render_chat,
            render_settings,
            render_ai_intelligence_hub,
            run_benchmarks,
        )
        print("✅ appv38 function imports successful")
    except Exception as e:
        print(f"❌ appv38 function imports failed: {e}")
        traceback.print_exc()
        return False
    
    print("✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1)
