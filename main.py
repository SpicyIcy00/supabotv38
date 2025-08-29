"""
SupaBot BI Dashboard - Production-Ready Entry Point
A modular, scalable business intelligence dashboard built with Streamlit.
"""

import streamlit as st
from supabot.config.settings import settings
from supabot.ui.styles.css import DashboardStyles

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
        
        # Inject mobile sidebar JavaScript
        st.markdown("""
        <script>
        // Mobile Sidebar Toggle Functionality
        (function() {
            'use strict';
            
            function toggleMobileSidebar() {
                const sidebar = document.querySelector('.css-1d391kg');
                if (sidebar) {
                    sidebar.classList.toggle('expanded');
                    
                    const mainContent = document.querySelector('.main .block-container');
                    if (mainContent) {
                        if (sidebar.classList.contains('expanded')) {
                            mainContent.style.paddingLeft = '1rem';
                        } else {
                            mainContent.style.paddingLeft = '0.5rem';
                        }
                    }
                }
            }
            
            function initMobileSidebar() {
                if (window.innerWidth <= 767) {
                    const toggleButton = document.querySelector('button[aria-label*="sidebar"]');
                    if (toggleButton) {
                        toggleButton.addEventListener('click', toggleMobileSidebar);
                        
                        toggleButton.style.position = 'fixed';
                        toggleButton.style.top = '1rem';
                        toggleButton.style.right = '1rem';
                        toggleButton.style.zIndex = '1001';
                        toggleButton.style.background = '#00d2ff';
                        toggleButton.style.color = 'white';
                        toggleButton.style.border = 'none';
                        toggleButton.style.borderRadius = '50%';
                        toggleButton.style.width = '48px';
                        toggleButton.style.height = '48px';
                        toggleButton.style.display = 'flex';
                        toggleButton.style.alignItems = 'center';
                        toggleButton.style.justifyContent = 'center';
                        toggleButton.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
                    }
                    
                    const sidebar = document.querySelector('.css-1d391kg');
                    if (sidebar) {
                        sidebar.style.transform = 'translateX(-100%)';
                        sidebar.style.transition = 'transform 0.3s ease-in-out';
                        sidebar.style.position = 'fixed';
                        sidebar.style.top = '0';
                        sidebar.style.left = '0';
                        sidebar.style.height = '100vh';
                        sidebar.style.zIndex = '1000';
                        sidebar.style.background = '#1e1e1e';
                        sidebar.style.overflowY = 'auto';
                    }
                }
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initMobileSidebar);
            } else {
                initMobileSidebar();
            }
            
            window.addEventListener('resize', initMobileSidebar);
        })();
        </script>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.title("SupaBot BI")
        pages = [
            "Dashboard",
            "Product Sales Report",
            "Chart View",
            "Advanced Analytics",
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
