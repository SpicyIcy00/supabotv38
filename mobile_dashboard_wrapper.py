"""
Mobile Dashboard Wrapper for SupaBot BI Dashboard.
Simple integration point for mobile-responsive dashboard.
"""

import streamlit as st
from supabot.ui.components.mobile_detection import MobileDetection
from supabot.ui.components.mobile_dashboard_renderer import MobileDashboardRenderer


def render_responsive_dashboard():
    """
    Render dashboard with mobile detection and conditional rendering.
    This function automatically detects mobile devices and renders the appropriate version.
    """
    
    # Add mobile detection toggle in sidebar for testing
    with st.sidebar:
        st.markdown("### 📱 Mobile Testing")
        force_mobile = st.checkbox("Force Mobile View", value=False, help="Override mobile detection for testing")
        MobileDetection.set_mobile_override(force_mobile)
        
        # Show current device status
        is_mobile = MobileDetection.is_mobile()
        device_status = "📱 Mobile" if is_mobile else "🖥️ Desktop"
        st.info(f"Current View: {device_status}")
    
    # Conditional rendering based on device type
    if MobileDetection.is_mobile():
        # Render mobile dashboard
        MobileDashboardRenderer.render_mobile_dashboard()
    else:
        # Render desktop dashboard (original code)
        try:
            from appv38 import render_dashboard
            render_dashboard()
        except ImportError:
            st.error("Desktop dashboard not available. Please ensure appv38.py is available.")
            st.info("Falling back to mobile dashboard...")
            MobileDashboardRenderer.render_mobile_dashboard()


def render_mobile_only_dashboard():
    """
    Force mobile dashboard rendering (for testing or mobile-only deployment).
    """
    MobileDetection.set_mobile_override(True)
    MobileDashboardRenderer.render_mobile_dashboard()


def render_desktop_only_dashboard():
    """
    Force desktop dashboard rendering (for testing or desktop-only deployment).
    """
    MobileDetection.set_mobile_override(False)
    try:
        from appv38 import render_dashboard
        render_dashboard()
    except ImportError:
        st.error("Desktop dashboard not available. Please ensure appv38.py is available.")


# Utility functions for easy integration
def is_mobile_device() -> bool:
    """Check if current device is mobile."""
    return MobileDetection.is_mobile()

def get_device_info() -> dict:
    """Get current device information."""
    return MobileDetection.get_screen_info()

def set_mobile_override(is_mobile: bool):
    """Override mobile detection for testing."""
    MobileDetection.set_mobile_override(is_mobile)
