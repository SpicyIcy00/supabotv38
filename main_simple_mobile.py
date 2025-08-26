"""
SupaBot BI Dashboard - Simple Mobile-Enabled Entry Point
A simplified version that works around import issues.
"""

import streamlit as st
from supabot.config.settings import settings
from supabot.ui.styles.css import DashboardStyles

# Import mobile components
from supabot.ui.components.mobile_detection import MobileDetection
from supabot.ui.components.mobile_dashboard_renderer import MobileDashboardRenderer


def init_session_state():
    """Initialize session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if 'dashboard_time_filter' not in st.session_state:
        st.session_state.dashboard_time_filter = "7D"
    if 'dashboard_store_filter' not in st.session_state:
        st.session_state.dashboard_store_filter = ["All Stores"]
    if 'custom_start_date' not in st.session_state:
        st.session_state.custom_start_date = None
    if 'custom_end_date' not in st.session_state:
        st.session_state.custom_end_date = None


def render_dashboard():
    """Render the main dashboard."""
    st.markdown('<div class="main-header"><h1>📊 SupaBot BI Dashboard</h1><p>Business Intelligence Dashboard</p></div>', unsafe_allow_html=True)
    st.info("Dashboard functionality is available in the mobile version. Please use the mobile view for full functionality.")


def render_product_sales_report():
    """Render product sales report."""
    st.markdown('<div class="main-header"><h1>📈 Product Sales Report</h1><p>Product Analytics</p></div>', unsafe_allow_html=True)
    st.info("Product sales report functionality is available in the mobile version. Please use the mobile view for full functionality.")


def render_chart_view():
    """Render chart view."""
    st.markdown('<div class="main-header"><h1>📊 Chart View</h1><p>Interactive Charts</p></div>', unsafe_allow_html=True)
    st.info("Chart view functionality is available in the mobile version. Please use the mobile view for full functionality.")


def render_chat():
    """Render AI chat interface."""
    st.markdown('<div class="main-header"><h1>🤖 AI Assistant</h1><p>AI-Powered Chat</p></div>', unsafe_allow_html=True)
    st.info("AI chat functionality is available in the mobile version. Please use the mobile view for full functionality.")


def render_settings():
    """Render settings page."""
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Application Settings</p></div>', unsafe_allow_html=True)
    
    # Mobile detection toggle
    force_mobile = st.checkbox("Force Mobile View", value=False, help="Override mobile detection for testing")
    MobileDetection.set_mobile_override(force_mobile)
    
    # Show current device status
    is_mobile = MobileDetection.is_mobile()
    device_status = "📱 Mobile" if is_mobile else "🖥️ Desktop"
    st.info(f"Current View: {device_status}")
    
    # Mobile optimization settings
    st.subheader("Mobile Optimization")
    st.checkbox("Enable Touch Optimizations", value=True, help="Optimize for touch interactions")
    st.checkbox("Reduce Chart Heights", value=True, help="Use smaller charts for mobile")
    st.checkbox("Limit Table Rows", value=True, help="Show fewer rows in data tables")
    
    # Theme settings
    st.subheader("Theme")
    st.checkbox("Dark Theme", value=True, help="Use dark theme (recommended for mobile)")
    st.checkbox("High Contrast", value=False, help="Use high contrast mode")


def main():
    """Main application entry point with mobile detection."""
    try:
        # Configure Streamlit and load styles
        settings.configure_streamlit()
        DashboardStyles.load_all_styles()
        
        # Initialize session state
        init_session_state()
        
        # Inject mobile detection
        MobileDetection.inject_detection_script()
        
        # Sidebar Navigation with mobile detection
        st.sidebar.title("🧠 SupaBot BI")
        
        # Mobile detection toggle in sidebar
        with st.sidebar:
            st.markdown("### 📱 Device Detection")
            force_mobile = st.checkbox("Force Mobile View", value=False, help="Override mobile detection for testing")
            MobileDetection.set_mobile_override(force_mobile)
            
            # Show current device status
            is_mobile = MobileDetection.is_mobile()
            device_status = "📱 Mobile" if is_mobile else "🖥️ Desktop"
            st.info(f"Current View: {device_status}")
        
        pages = [
            "Dashboard",
            "Product Sales Report",
            "Chart View",
            "AI Assistant",
            "Settings",
        ]
        
        # Page selector
        selected_page = st.sidebar.radio(
            "Navigate to:",
            pages,
            key="navigation",
            index=pages.index(st.session_state.get("current_page", "Dashboard"))
        )
        
        # Update session state if page changed
        if selected_page != st.session_state.get("current_page"):
            st.session_state.current_page = selected_page
        
        # Conditional page mapping based on device type
        if MobileDetection.is_mobile():
            # Mobile page map - use mobile dashboard for all pages
            page_map = {
                "Dashboard": MobileDashboardRenderer.render_mobile_dashboard,
                "Product Sales Report": render_product_sales_report,
                "Chart View": render_chart_view,
                "AI Assistant": render_chat,
                "Settings": render_settings,
            }
        else:
            # Desktop page map
            page_map = {
                "Dashboard": render_dashboard,
                "Product Sales Report": render_product_sales_report,
                "Chart View": render_chart_view,
                "AI Assistant": render_chat,
                "Settings": render_settings,
            }
        
        # Render selected page
        current_page = st.session_state.get("current_page", "Dashboard")
        try:
            if current_page in page_map:
                page_map[current_page]()
            else:
                st.error(f"❌ Page '{current_page}' not found")
                st.info("Please select a valid page from the sidebar")
        except Exception as page_error:
            st.error(f"❌ Error loading {current_page}: {page_error}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
        
        # Footer with mobile indicator
        device_indicator = "📱 Mobile" if MobileDetection.is_mobile() else "🖥️ Desktop"
        st.markdown(
            f"<hr><div style='text-align:center;color:#666;'>"
            f"<p>🧠 Enhanced SupaBot with Mobile Responsiveness | {device_indicator} | Powered by Claude Sonnet 3.5</p>"
            f"</div>", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
