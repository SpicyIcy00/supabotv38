"""
SupaBot BI Dashboard - Production-Ready Entry Point
A modular, scalable business intelligence dashboard built with Streamlit.
"""

import streamlit as st
from supabot.config.settings import settings
from supabot.ui.styles.css import DashboardStyles
from supabot.ui.components.responsive import ResponsiveLayout

# Import page renderers from the original app
from appv38 import (
    init_session_state,
    render_dashboard,
    render_product_sales_report,
    render_chart_view,
    render_chat,
    render_settings,
    render_ai_intelligence_hub,
    render_advanced_analytics,
    run_benchmarks,
)


def main():
    """Main application entry point."""
    try:
        # Configure Streamlit and load styles
        settings.configure_streamlit()
        DashboardStyles.load_all_styles()
        
        # Initialize session state
        init_session_state()
        
        # Define pages
        pages = [
            "Dashboard",
            "Product Sales Report",
            "Chart View",
            "Advanced Analytics",
            "AI Assistant",
            "Settings",
        ]
        
        # Use responsive navigation
        selected_page = ResponsiveLayout.responsive_navigation(
            pages=pages,
            current_page=st.session_state.get("current_page", "Dashboard")
        )
        
        # Update session state if page changed
        if selected_page != st.session_state.get("current_page"):
            st.session_state.current_page = selected_page
        
        # Page navigation with error handling
        page_map = {
            "Dashboard": render_dashboard,
            "Product Sales Report": render_product_sales_report,
            "Chart View": render_chart_view,
            "Advanced Analytics": render_advanced_analytics,
            "AI Assistant": render_chat,
            "Settings": render_settings,
        }
        
        # Render selected page
        current_page = st.session_state.get("current_page", "Dashboard")
        try:
            if current_page in page_map:
                page_map[current_page]()
            else:
                st.error(f"Page '{current_page}' not found")
                st.info("Please select a valid page from the sidebar")
        except Exception as page_error:
            st.error(f"Error loading {current_page}: {page_error}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


        
        # Footer
        st.markdown(
            "<hr><div style='text-align:center;color:#666;'>"
            "<p>Enhanced SupaBot with Smart Visualizations | Powered by Claude Sonnet 3.5</p>"
            "</div>", 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")
        import traceback
        with st.expander("Full Error Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
