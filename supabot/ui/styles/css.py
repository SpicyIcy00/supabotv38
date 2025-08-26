"""
CSS styling for SupaBot BI Dashboard
Centralizes all application styling and UI theming.
"""

import streamlit as st


class DashboardStyles:
    """CSS styles for the SupaBot BI Dashboard."""
    
    # Main dark theme CSS
    CUSTOM_DARK_CSS = """
    <style>
    /* KPI cards */
    div[data-testid="stMetric"] {
      background: linear-gradient(135deg,#0f172a 0%, #111827 100%);
      padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);
    }
    /* Headings */
    .main-header h1 {letter-spacing:0.3px;}
    /* Horizontal radio pills */
    [data-baseweb="button-group"] button {border-radius: 999px !important;}
    /* Container padding */
    .block-container {padding-top: 1.0rem;}
    </style>
    """
    
    # Comprehensive dashboard styling
    DASHBOARD_CSS = """
    <style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        padding: 1.5rem; border-radius: 10px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: bold; }
    
    /* KPI Metric Boxes - Ensure Equal Height */
    div[data-testid="stMetric"] {
        background-color: #1c1e26; 
        border: 1px solid #2e303d;
        padding: 1rem; 
        border-radius: 10px;
        height: 130px; /* Increased height for more content */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.8rem; /* Adjusted for space */
        font-weight: bold; 
        color: #00d2ff;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stMetric"] > div:nth-child(3) {
        font-size: 0.8rem;
    }

    /* Custom card styling */
    .dashboard-card {
        background-color: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 1rem;
        height: 350px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
    }
    .dashboard-card-tall {
        height: 450px;
    }
    .dashboard-card h5 {
        font-size: 1.1rem;
        color: #c7c7c7;
        margin-bottom: 1rem;
        border-bottom: 1px solid #3a47d5;
        padding-bottom: 0.5rem;
    }
    
    .insight-box {
        background: #16a085; padding: 1rem; border-radius: 8px;
        color: white; margin-top: 1rem; text-align: center;
    }
    .user-message{
        background:linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        padding:1rem 1.5rem; border-radius:20px 20px 0 20px;
        margin:1rem 0; color:white; font-weight:500;
    }
    .ai-message{
        background: #262730; border: 1px solid #3d3d3d;
        padding:1rem 1.5rem; border-radius:20px 20px 20px 0;
        margin:1rem 0; color:white;
    }
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 2px solid transparent;
        font-size: 1.1rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #00d2ff;
        color: #00d2ff;
    }
    </style>
    """
    
    # Mobile-responsive CSS
    MOBILE_CSS = """
    <style>
    /* Mobile-specific styles */
    @media (max-width: 768px) {
        /* Header adjustments for mobile */
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            font-size: 1.8rem !important;
            line-height: 1.2;
        }
        .main-header p {
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        /* KPI cards mobile optimization */
        div[data-testid="stMetric"] {
            height: auto !important;
            min-height: 100px;
            padding: 0.75rem !important;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stMetric"] > div:nth-child(1) {
            font-size: 0.8rem !important;
            margin-bottom: 0.3rem;
        }
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.4rem !important;
            margin-bottom: 0.2rem;
        }
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.7rem !important;
        }
        
        /* Dashboard cards mobile optimization */
        .dashboard-card {
            height: auto !important;
            min-height: 300px;
            padding: 0.75rem;
            margin-bottom: 1rem;
        }
        .dashboard-card-tall {
            height: auto !important;
            min-height: 400px;
        }
        .dashboard-card h5 {
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }
        
        /* Filter controls mobile optimization */
        [data-testid="stRadio"] {
            font-size: 0.9rem;
        }
        [data-testid="stSelectbox"] {
            font-size: 0.9rem;
        }
        [data-testid="stMultiselect"] {
            font-size: 0.9rem;
        }
        
        /* Chart containers mobile optimization */
        .js-plotly-plot {
            height: 350px !important;
        }
        
        /* Table mobile optimization */
        .dataframe {
            font-size: 0.8rem;
        }
        .dataframe th,
        .dataframe td {
            padding: 0.3rem 0.5rem;
        }
        
        /* Sidebar mobile optimization */
        .css-1d391kg {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Container padding adjustments */
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* Button and input mobile optimization */
        button {
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
        }
        
        input, textarea, select {
            font-size: 0.9rem !important;
        }
        
        /* Subheader mobile optimization */
        h3 {
            font-size: 1.2rem !important;
            margin-bottom: 0.75rem;
        }
        
        /* Expandable sections mobile optimization */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
            padding: 0.5rem !important;
        }
        
        /* Message boxes mobile optimization */
        .user-message,
        .ai-message {
            padding: 0.75rem 1rem !important;
            font-size: 0.9rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Insight box mobile optimization */
        .insight-box {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
            margin-top: 0.75rem !important;
        }
        
        /* Tab navigation mobile optimization */
        button[data-baseweb="tab"] {
            font-size: 0.9rem !important;
            padding: 0.5rem 0.75rem !important;
        }
        
        /* Progress bars mobile optimization */
        .stProgress > div > div {
            height: 0.5rem !important;
        }
        
        /* Alert messages mobile optimization */
        .stAlert {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Spinner mobile optimization */
        .stSpinner {
            font-size: 0.9rem !important;
        }
        
        /* Horizontal rule mobile optimization */
        hr {
            margin: 1rem 0 !important;
        }
        
        /* Column gap adjustments for mobile */
        [data-testid="column"] {
            margin-bottom: 0.5rem;
        }
        
        /* Chart legend mobile optimization */
        .legend {
            font-size: 0.8rem !important;
        }
        
        /* Tooltip mobile optimization */
        .tooltip {
            font-size: 0.8rem !important;
        }
    }
    
    /* Tablet-specific adjustments */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        div[data-testid="stMetric"] {
            height: 120px !important;
        }
        
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.6rem !important;
        }
        
        .dashboard-card {
            height: 320px !important;
        }
        
        .dashboard-card-tall {
            height: 420px !important;
        }
    }
    
    /* Touch-friendly interactions for mobile */
    @media (max-width: 768px) {
        /* Increase touch targets */
        button, 
        [role="button"],
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            min-height: 44px !important;
            min-width: 44px !important;
        }
        
        /* Improve scrolling on mobile */
        .main .block-container {
            overflow-x: auto;
        }
        
        /* Better spacing for mobile */
        .element-container {
            margin-bottom: 1rem;
        }
        
        /* Optimize chart interactions */
        .js-plotly-plot .plotly {
            touch-action: manipulation;
        }
    }
    </style>
    """
    
    @staticmethod
    def load_custom_dark_css():
        """Load the custom dark theme CSS."""
        st.markdown(DashboardStyles.CUSTOM_DARK_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def load_dashboard_css():
        """Load the comprehensive dashboard CSS."""
        st.markdown(DashboardStyles.DASHBOARD_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def load_mobile_css():
        """Load mobile-responsive CSS."""
        st.markdown(DashboardStyles.MOBILE_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def load_all_styles():
        """Load all CSS styles for the application."""
        DashboardStyles.load_custom_dark_css()
        DashboardStyles.load_dashboard_css()
        DashboardStyles.load_mobile_css()


# Legacy function for compatibility
def load_css():
    """Legacy compatibility function."""
    DashboardStyles.load_dashboard_css()


# Auto-load styles when imported
DashboardStyles.load_custom_dark_css()

