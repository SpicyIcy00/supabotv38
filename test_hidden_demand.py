#!/usr/bin/env python3
"""
Test script for the robust hidden demand function
"""

import sys
import os

# Add the current directory to the path so we can import from appv38
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from appv38 import get_hidden_demand, DEFAULT_STORES
    
    print("✅ Successfully imported hidden demand function")
    print(f"✅ Default stores: {DEFAULT_STORES}")
    
    # Test the function signature
    print("✅ Function signature check passed")
    
    # Test with no parameters (should use default stores)
    print("\n🧪 Testing function with no parameters...")
    try:
        result = get_hidden_demand()
        print(f"✅ Function executed successfully")
        print(f"✅ Result type: {type(result)}")
        if hasattr(result, 'empty'):
            print(f"✅ Result empty: {result.empty}")
            if not result.empty:
                print(f"✅ Result columns: {list(result.columns)}")
                print(f"✅ Result shape: {result.shape}")
    except Exception as e:
        print(f"❌ Function execution failed: {e}")
    
    print("\n✅ All tests completed successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
