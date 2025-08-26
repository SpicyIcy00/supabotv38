"""
Mobile-specific CSS styles for SupaBot BI Dashboard.
Provides mobile-optimized styling while keeping desktop styles untouched.
"""

import streamlit as st


class MobileStyles:
    """Mobile-optimized CSS styles for the dashboard."""
    
    # Mobile-specific CSS
    MOBILE_CSS = """
    <style>
    /* Mobile-specific styles - only applied on mobile devices */
    @media (max-width: 768px) {
        /* Mobile header adjustments */
        .main-header h1 { 
            font-size: 1.8rem !important; 
            margin-bottom: 0.5rem;
        }
        .main-header p { 
            font-size: 0.9rem !important; 
        }
        
        /* Mobile KPI cards - ensure proper spacing and sizing */
        div[data-testid="stMetric"] {
            height: 100px !important;
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stMetric"] > div:nth-child(1) {
            font-size: 0.8rem !important;
            margin-bottom: 0.3rem !important;
        }
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.4rem !important;
            margin-bottom: 0.2rem !important;
        }
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.7rem !important;
        }
        
        /* Mobile chart containers */
        .dashboard-card {
            height: auto !important;
            min-height: 280px !important;
            padding: 0.75rem !important;
            margin-bottom: 1rem !important;
        }
        .dashboard-card h5 {
            font-size: 1rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        /* Mobile filter controls */
        .stSelectbox, .stMultiselect {
            margin-bottom: 1rem !important;
        }
        
        /* Mobile data tables */
        .dataframe {
            font-size: 0.8rem !important;
        }
        .dataframe th, .dataframe td {
            padding: 0.5rem 0.25rem !important;
        }
        
        /* Mobile sidebar adjustments */
        .css-1d391kg {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Mobile button sizing */
        .stButton > button {
            min-height: 44px !important;
            font-size: 1rem !important;
            padding: 0.75rem 1rem !important;
        }
        
        /* Mobile container spacing */
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* Mobile chart responsiveness */
        .js-plotly-plot {
            max-width: 100% !important;
        }
        
        /* Mobile text sizing */
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.3rem !important; }
        h4 { font-size: 1.1rem !important; }
        h5 { font-size: 1rem !important; }
        
        /* Mobile spacing */
        .stMarkdown {
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile container borders */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > div {
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile radio buttons */
        .stRadio > div {
            flex-direction: column !important;
        }
        .stRadio > div > label {
            margin-bottom: 0.5rem !important;
            min-height: 44px !important;
            display: flex !important;
            align-items: center !important;
        }
        
        /* Mobile multiselect */
        .stMultiSelect > div {
            min-height: 44px !important;
        }
        
        /* Mobile date inputs */
        .stDateInput > div {
            min-height: 44px !important;
        }
        
        /* Mobile progress bars */
        .stProgress > div {
            height: 8px !important;
        }
        
        /* Mobile expander */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 0.75rem !important;
        }
        
        /* Mobile tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Mobile alerts and messages */
        .stAlert {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile info boxes */
        .stInfo {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile warning boxes */
        .stWarning {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile error boxes */
        .stError {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile success boxes */
        .stSuccess {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile spinner */
        .stSpinner {
            margin: 1rem auto !important;
        }
        
        /* Mobile caption text */
        .caption {
            font-size: 0.75rem !important;
            margin-top: 0.25rem !important;
        }
        
        /* Mobile horizontal rules */
        hr {
            margin: 1rem 0 !important;
        }
        
        /* Mobile container borders */
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            border: 1px solid #2e303d !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Mobile plotly chart responsiveness */
        .plotly-graph-div {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Mobile table responsiveness */
        .stDataFrame {
            max-width: 100% !important;
            overflow-x: auto !important;
        }
        
        /* Mobile metric responsiveness */
        [data-testid="stMetric"] {
            max-width: 100% !important;
        }
        
        /* Mobile column spacing */
        [data-testid="column"] {
            padding: 0 0.25rem !important;
        }
        
        /* Mobile row spacing */
        [data-testid="row"] {
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* Touch-friendly interactions for mobile */
    @media (max-width: 768px) {
        /* Ensure minimum touch target size */
        button, [role="button"], .stButton > button {
            min-height: 44px !important;
            min-width: 44px !important;
        }
        
        /* Improve touch targets for interactive elements */
        .stSelectbox > div, .stMultiselect > div, .stDateInput > div {
            min-height: 44px !important;
        }
        
        /* Better spacing for touch interactions */
        .stRadio > div > label, .stCheckbox > div > label {
            min-height: 44px !important;
            padding: 0.5rem !important;
        }
        
        /* Mobile-friendly hover states */
        button:hover, [role="button"]:hover {
            transform: scale(1.02) !important;
            transition: transform 0.2s ease !important;
        }
    }
    
    /* Mobile-specific dark theme adjustments */
    @media (max-width: 768px) {
        /* Ensure proper contrast on mobile */
        .stApp {
            background-color: #0e1117 !important;
        }
        
        /* Mobile text contrast */
        .stMarkdown, p, span, div {
            color: #ffffff !important;
        }
        
        /* Mobile link colors */
        a {
            color: #00d2ff !important;
        }
        
        /* Mobile selection colors */
        ::selection {
            background-color: #00d2ff !important;
            color: #000000 !important;
        }
    }
    </style>
    """
    
    @staticmethod
    def load_mobile_styles():
        """Load mobile-specific CSS styles."""
        st.markdown(MobileStyles.MOBILE_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def inject_mobile_viewport():
        """Inject mobile viewport meta tag."""
        viewport_meta = """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        """
        st.markdown(viewport_meta, unsafe_allow_html=True)
    
    @staticmethod
    def load_all_mobile_styles():
        """Load all mobile-specific styles and viewport settings."""
        MobileStyles.inject_mobile_viewport()
        MobileStyles.load_mobile_styles()


# Legacy function for compatibility
def load_mobile_css():
    """Legacy compatibility function."""
    MobileStyles.load_mobile_styles()
