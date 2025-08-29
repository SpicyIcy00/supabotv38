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
        
        # Inject mobile-responsive CSS directly
        st.markdown("""
        <style>
        /* Mobile-specific styles that will work immediately */
        @media (max-width: 767px) {
            /* Force mobile layout */
            .stApp {
                padding: 0.5rem !important;
            }
            
            /* Fix sidebar on mobile */
            .css-1d391kg {
                width: 100% !important;
                max-width: 100% !important;
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                height: 100vh !important;
                z-index: 1000 !important;
                background: #1e1e1e !important;
                transform: translateX(-100%) !important;
                transition: transform 0.3s ease-in-out !important;
            }
            
            /* Show sidebar when expanded */
            .css-1d391kg.expanded {
                transform: translateX(0) !important;
            }
            
            /* Fix time period selectors */
            .stRadio > div {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            .stRadio > div > label {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                padding: 0.5rem !important;
                margin: 0.25rem 0 !important;
                border: 1px solid #ddd !important;
                border-radius: 4px !important;
                background: #f8f9fa !important;
                min-height: 48px !important;
                font-size: 0.9rem !important;
            }
            
            /* Fix form controls */
            .stSelectbox, .stMultiselect, .stDateInput {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix charts on mobile */
            .plotly-graph-div {
                width: 100% !important;
                height: 350px !important;
                max-width: 100vw !important;
                overflow: hidden !important;
            }
            
            [data-testid="stChart"] {
                width: 100% !important;
                max-width: 100% !important;
                overflow: hidden !important;
            }
            
            /* Fix containers */
            [data-testid="stContainer"] {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
                padding: 0.5rem !important;
            }
            
            /* Fix columns */
            [data-testid="column"] {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix buttons */
            button, [role="button"] {
                width: 100% !important;
                min-height: 48px !important;
            }
            
            /* Fix sidebar toggle button */
            .css-1d391kg .css-1v0mbdj button[aria-label*="sidebar"] {
                position: fixed !important;
                top: 1rem !important;
                right: 1rem !important;
                z-index: 1001 !important;
                background: #00d2ff !important;
                color: white !important;
                border: none !important;
                border-radius: 50% !important;
                width: 48px !important;
                height: 48px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
            }
            
            /* Ensure main content takes full width */
            .main .block-container {
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            /* Fix time period selector specific styling */
            .time-period-selector {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix filter containers */
            .filter-container {
                width: 100% !important;
            }
            
            .filter-container [data-testid="column"] {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix chart view filters */
            .chart-view-filters [data-testid="column"] {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix comparison sets */
            [data-testid="stContainer"] {
                width: 100% !important;
                margin: 0 0 0.5rem 0 !important;
                padding: 0.5rem !important;
            }
            
            /* Ensure all radio buttons are properly styled */
            [data-baseweb="radio"] {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.5rem !important;
                width: 100% !important;
            }
            
            [data-baseweb="radio"] label {
                width: 100% !important;
                text-align: left !important;
                padding: 0.5rem !important;
                margin: 0 !important;
                border: 1px solid #ddd !important;
                border-radius: 4px !important;
                background: #f8f9fa !important;
                min-height: 48px !important;
                display: flex !important;
                align-items: center !important;
            }
            
            /* Fix any remaining form alignment issues */
            .stRadio, .stSelectbox, .stMultiselect, .stDateInput {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Ensure consistent spacing */
            .stMarkdown, .stText {
                margin-bottom: 0.5rem !important;
            }
            
            /* Fix mobile chart layout */
            .chart-container {
                width: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            .chart-container [data-testid="column"] {
                width: 100% !important;
                margin: 0 !important;
                min-width: 100% !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
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
