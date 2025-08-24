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
        
        # Sidebar Navigation
        st.sidebar.title("🧠 SupaBot BI")
        pages = [
            "Dashboard",
            "AI Intelligence Hub",
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
        
        # Page navigation with error handling
        page_map = {
            "Dashboard": render_dashboard,
            "AI Intelligence Hub": render_ai_intelligence_hub,
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

        # Sidebar: Performance metrics (if available)
        perf = st.session_state.get("perf", {})
        if perf or True:
            with st.sidebar.expander("⏱️ Performance", expanded=False):
                last_ms = perf.get("last_query_time_ms")
                last_kb = perf.get("last_query_mem_kb")
                col_a, col_b = st.columns(2)
                if isinstance(last_ms, (int, float)):
                    col_a.metric("Last query (ms)", f"{last_ms:.1f}")
                if isinstance(last_kb, (int, float)):
                    col_b.metric("Peak mem (KB)", f"{last_kb:.0f}")

                db_pool = perf.get("db_pool")
                if isinstance(db_pool, dict) and db_pool:
                    st.caption("DB pool status")
                    st.write(db_pool)

                # Benchmarks UI
                st.markdown("---")
                if st.button("Run Benchmarks", use_container_width=True):
                    try:
                        bench_df = run_benchmarks()
                        st.session_state["last_benchmarks"] = bench_df
                    except Exception as e:
                        st.warning(f"Benchmark failed: {e}")
                bench_df = st.session_state.get("last_benchmarks")
                if bench_df is not None:
                    st.dataframe(bench_df, use_container_width=True)
        
        # Footer
        st.markdown(
            "<hr><div style='text-align:center;color:#666;'>"
            "<p>🧠 Enhanced SupaBot with Smart Visualizations | Powered by Claude Sonnet 3.5</p>"
            "</div>", 
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
