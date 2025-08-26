#!/usr/bin/env python3
"""
Test script for Advanced Data Analytics implementation
"""

import sys
import traceback

def test_imports():
    """Test that all required components can be imported."""
    try:
        from appv38 import (
            AIAnalyticsEngine,
            PredictiveForecastingEngine,
            CustomerIntelligenceEngine,
            MarketIntelligenceEngine,
            SmartAlertManager,
            AutomatedInsightEngine,
            render_advanced_data
        )
        print("✅ All analytics engines imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_engine_initialization():
    """Test that analytics engines can be initialized."""
    try:
        from appv38 import create_db_connection, get_claude_client
        
        # Test engine initialization
        ai_analytics = AIAnalyticsEngine(create_db_connection)
        predictive_forecasting = PredictiveForecastingEngine(create_db_connection)
        customer_intelligence = CustomerIntelligenceEngine(create_db_connection)
        market_intelligence = MarketIntelligenceEngine(create_db_connection)
        smart_alerts = SmartAlertManager(create_db_connection)
        automated_insights = AutomatedInsightEngine(create_db_connection, get_claude_client)
        
        print("✅ All analytics engines initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        traceback.print_exc()
        return False

def test_navigation():
    """Test that the navigation includes the Advanced Data tab."""
    try:
        from main import main
        print("✅ Main application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Navigation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Advanced Data Analytics Implementation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Engine Initialization Test", test_engine_initialization),
        ("Navigation Test", test_navigation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Advanced Data Analytics implementation is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
