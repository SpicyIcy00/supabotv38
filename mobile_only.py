"""
SupaBot BI Dashboard - Mobile-Only Version
A complete mobile-optimized dashboard that doesn't rely on appv38 imports.
"""

import streamlit as st
from supabot.config.settings import settings
from supabot.ui.styles.css import DashboardStyles
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


def render_mobile_dashboard():
    """Render mobile-optimized dashboard."""
    MobileDashboardRenderer.render_mobile_dashboard()


def render_mobile_product_sales_report():
    """Render mobile-optimized product sales report."""
    st.markdown('<div class="main-header"><h1>📈 Product Sales Report</h1><p>Mobile Business Intelligence</p></div>', unsafe_allow_html=True)
    st.info("📱 Mobile-optimized product sales report coming soon. Please use desktop view for full functionality.")


def render_mobile_chart_view():
    """Render mobile-optimized chart view."""
    st.markdown('<div class="main-header"><h1>📊 Chart View</h1><p>Mobile Business Intelligence</p></div>', unsafe_allow_html=True)
    st.info("📱 Mobile-optimized chart view coming soon. Please use desktop view for full functionality.")


def render_mobile_chat():
    """Render mobile-optimized chat interface."""
    st.markdown('<div class="main-header"><h1>🤖 AI Assistant</h1><p>Mobile Business Intelligence</p></div>', unsafe_allow_html=True)
    st.info("📱 Mobile-optimized AI assistant coming soon. Please use desktop view for full functionality.")


def render_mobile_settings():
    """Render mobile-optimized settings."""
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Mobile Business Intelligence</p></div>', unsafe_allow_html=True)
    
    # Mobile-specific settings
    st.subheader("📱 Mobile Settings")
    
    # Mobile detection toggle
    force_mobile = st.checkbox("Force Mobile View", value=True, help="Override mobile detection for testing")
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
    """Main application entry point - Mobile Only."""
    try:
        # Configure Streamlit and load styles
        settings.configure_streamlit()
        DashboardStyles.load_all_styles()
        
        # Initialize session state
        init_session_state()
        
        # Force mobile view
        MobileDetection.set_mobile_override(True)
        
        # Inject mobile detection
        MobileDetection.inject_detection_script()
        
        # Sidebar Navigation
        st.sidebar.title("🧠 SupaBot BI Mobile")
        
        # Mobile detection toggle in sidebar
        with st.sidebar:
            st.markdown("### 📱 Mobile Mode")
            force_mobile = st.checkbox("Force Mobile View", value=True, help="Override mobile detection for testing")
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
        
        # Mobile page map
        page_map = {
            "Dashboard": render_mobile_dashboard,
            "Product Sales Report": render_mobile_product_sales_report,
            "Chart View": render_mobile_chart_view,
            "AI Assistant": render_mobile_chat,
            "Settings": render_mobile_settings,
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
        st.markdown(
            f"<hr><div style='text-align:center;color:#666;'>"
            f"<p>🧠 SupaBot BI Mobile Dashboard | 📱 Mobile Optimized | Powered by Claude Sonnet 3.5</p>"
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
