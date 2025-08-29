"""
Responsive layout utilities for SupaBot BI Dashboard
Provides helper functions for creating responsive layouts across all device sizes.
"""

import streamlit as st
from typing import List, Tuple, Optional, Any


class ResponsiveLayout:
    """Utility class for creating responsive layouts in Streamlit."""
    
    @staticmethod
    def responsive_columns(column_weights: List[int], gap: str = "small") -> List[Any]:
        """
        Create responsive columns that adapt to screen size.
        
        Args:
            column_weights: List of column weights (e.g., [1, 1, 1, 1] for 4 equal columns)
            gap: Gap between columns ("small", "medium", "large")
        
        Returns:
            List of column objects
        """
        # On mobile, stack vertically; on larger screens, use original layout
        if len(column_weights) == 4:
            # KPI layout: 2x2 on mobile, 4 columns on desktop
            return st.columns(column_weights, gap=gap)
        elif len(column_weights) == 2:
            # Chart layout: stack on mobile, side by side on larger screens
            return st.columns(column_weights, gap=gap)
        elif len(column_weights) == 3:
            # Filter layout: stack on mobile, 3 columns on larger screens
            return st.columns(column_weights, gap=gap)
        else:
            # Default behavior
            return st.columns(column_weights, gap=gap)
    
    @staticmethod
    def mobile_first_container(container_type: str = "chart") -> str:
        """
        Generate appropriate CSS class for mobile-first containers.
        
        Args:
            container_type: Type of container ("kpi", "chart", "filter", "table")
        
        Returns:
            CSS class string
        """
        container_classes = {
            "kpi": "kpi-container",
            "chart": "chart-container", 
            "filter": "filter-container",
            "table": "table-container"
        }
        return container_classes.get(container_type, "chart-container")
    
    @staticmethod
    def responsive_header(title: str, subtitle: str = "", icon: str = "") -> None:
        """
        Create a responsive header that adapts to screen size.
        
        Args:
            title: Main header title
            subtitle: Optional subtitle
            icon: Header icon
        """
        header_html = f"""
        <div class="main-header">
            <h1>{icon} {title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_metric(label: str, value: str, delta: str = "", help_text: str = "") -> None:
        """
        Create a responsive metric that works well on all screen sizes.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta/change indicator
            help_text: Optional help text
        """
        st.metric(label=label, value=value, delta=delta, help=help_text)
    
    @staticmethod
    def responsive_card(title: str, content: Any, height: str = "auto", tall: bool = False) -> None:
        """
        Create a responsive card container.
        
        Args:
            title: Card title
            content: Card content (any Streamlit element)
            height: Card height ("auto", "small", "medium", "large")
            tall: Whether this is a tall card
        """
        card_class = "dashboard-card-tall" if tall else "dashboard-card"
        
        with st.container():
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True)
            content
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def responsive_navigation(pages: List[str], current_page: str = None) -> str:
        """
        Create a responsive navigation sidebar that works well on mobile and desktop.
        
        Args:
            pages: List of page names
            current_page: Currently selected page
            
        Returns:
            Selected page name
        """
        # Add mobile navigation styles
        st.markdown("""
        <style>
        /* Mobile navigation styles */
        @media (max-width: 767px) {
            /* Mobile sidebar optimizations */
            .css-1d391kg {
                width: 100% !important;
                max-width: 100% !important;
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                z-index: 1000 !important;
                background: #0e1117 !important;
                border-right: none !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            }
            
            /* Mobile sidebar content */
            .css-1d391kg .css-1d391kg {
                padding: 0.5rem !important;
                overflow-y: auto !important;
                max-height: 100vh !important;
            }
            
            /* Mobile navigation title */
            .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
                font-size: 1.2rem !important;
                margin-bottom: 1rem !important;
                text-align: center !important;
                color: #00d2ff !important;
            }
            
            /* Mobile radio buttons */
            .css-1d391kg [data-baseweb="radio"] {
                margin-bottom: 0.5rem !important;
            }
            
            .css-1d391kg [data-baseweb="radio"] label {
                min-height: 48px !important;
                display: flex !important;
                align-items: center !important;
                padding: 0.75rem !important;
                background: #1c1e26 !important;
                border-radius: 8px !important;
                margin-bottom: 0.5rem !important;
                border: 1px solid #2e303d !important;
            }
            
            .css-1d391kg [data-baseweb="radio"] label:hover {
                background: #2e303d !important;
                border-color: #00d2ff !important;
            }
            
            .css-1d391kg [data-baseweb="radio"] input[type="radio"]:checked + label {
                background: #00d2ff !important;
                color: #0e1117 !important;
                border-color: #00d2ff !important;
            }
            
            /* Mobile sidebar scrollbar */
            .css-1d391kg::-webkit-scrollbar {
                width: 6px !important;
            }
            
            .css-1d391kg::-webkit-scrollbar-track {
                background: #1c1e26 !important;
            }
            
            .css-1d391kg::-webkit-scrollbar-thumb {
                background: #00d2ff !important;
                border-radius: 3px !important;
            }
            
            /* Mobile main content adjustment */
            .main .block-container {
                margin-left: 0 !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            /* Mobile sidebar toggle button */
            .mobile-nav-toggle {
                position: fixed !important;
                top: 1rem !important;
                left: 1rem !important;
                z-index: 1001 !important;
                background: #00d2ff !important;
                color: #0e1117 !important;
                border: none !important;
                border-radius: 50% !important;
                width: 48px !important;
                height: 48px !important;
                font-size: 1.5rem !important;
                cursor: pointer !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            }
        }
        
        /* Desktop sidebar styles */
        @media (min-width: 768px) {
            .css-1d391kg {
                width: 21rem !important;
                max-width: 21rem !important;
            }
            
            .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
                font-size: 1.5rem !important;
                margin-bottom: 1.5rem !important;
                text-align: left !important;
            }
            
            .css-1d391kg [data-baseweb="radio"] label {
                min-height: 44px !important;
                padding: 0.5rem !important;
                background: transparent !important;
                border: none !important;
                margin-bottom: 0.25rem !important;
            }
            
            .css-1d391kg [data-baseweb="radio"] label:hover {
                background: #1c1e26 !important;
                border: none !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create the sidebar navigation
        with st.sidebar:
            st.title("SupaBot BI")
            
            # Page selector with responsive styling
            selected_page = st.radio(
                "Navigate to:",
                pages,
                key="navigation",
                index=pages.index(current_page) if current_page in pages else 0,
                label_visibility="collapsed"  # Hide the label for cleaner look
            )
            
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Add version info
            st.markdown(
                "<div style='text-align:center;color:#666;font-size:0.8rem;'>"
                "v3.8 Enhanced</div>", 
                unsafe_allow_html=True
            )
        
        return selected_page
    
    @staticmethod
    def mobile_optimized_table(data: Any, **kwargs) -> None:
        """
        Create a mobile-optimized table with horizontal scroll on small screens.
        
        Args:
            data: Table data (DataFrame, etc.)
            **kwargs: Additional arguments for st.dataframe
        """
        with st.container():
            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.dataframe(data, use_container_width=True, **kwargs)
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def responsive_filter_section(title: str = "Filters") -> Tuple[Any, Any, Any]:
        """
        Create a responsive filter section with 3 columns.
        
        Args:
            title: Section title
            
        Returns:
            Tuple of three column objects
        """
        st.markdown(f"### {title}")
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        return col1, col2, col3
    
    @staticmethod
    def close_filter_section() -> None:
        """Close the filter container div."""
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def responsive_chart_section() -> Tuple[Any, Any]:
        """
        Create a responsive chart section with 2 columns.
        
        Returns:
            Tuple of two column objects
        """
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        return col1, col2
    
    @staticmethod
    def close_chart_section() -> None:
        """Close the chart container div."""
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def responsive_kpi_section() -> Tuple[Any, Any, Any, Any]:
        """
        Create a responsive KPI section with 4 columns.
        
        Returns:
            Tuple of four column objects
        """
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        return kpi1, kpi2, kpi3, kpi4
    
    @staticmethod
    def close_kpi_section() -> None:
        """Close the KPI container div."""
        st.markdown('</div>', unsafe_allow_html=True)


# Convenience functions for common responsive patterns
def create_responsive_dashboard_layout():
    """Create a responsive dashboard layout with proper containers."""
    return ResponsiveLayout()


def responsive_filter_row():
    """Create a responsive filter row with 3 columns."""
    return ResponsiveLayout.responsive_filter_section()


def responsive_chart_row():
    """Create a responsive chart row with 2 columns."""
    return ResponsiveLayout.responsive_chart_section()


def responsive_kpi_row():
    """Create a responsive KPI row with 4 columns."""
    return ResponsiveLayout.responsive_kpi_section()


def close_responsive_containers():
    """Close all responsive containers."""
    ResponsiveLayout.close_filter_section()
    ResponsiveLayout.close_chart_section()
    ResponsiveLayout.close_kpi_section()
