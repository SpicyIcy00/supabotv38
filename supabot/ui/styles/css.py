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
    
    @staticmethod
    def load_custom_dark_css():
        """Load the custom dark theme CSS."""
        st.markdown(DashboardStyles.CUSTOM_DARK_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def load_dashboard_css():
        """Load the comprehensive dashboard CSS."""
        st.markdown(DashboardStyles.DASHBOARD_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def load_all_styles():
        """Load all CSS styles for the application."""
        DashboardStyles.load_custom_dark_css()
        DashboardStyles.load_dashboard_css()


# Legacy function for compatibility
def load_css():
    """Legacy compatibility function."""
    DashboardStyles.load_dashboard_css()


# Auto-load styles when imported
DashboardStyles.load_custom_dark_css()

