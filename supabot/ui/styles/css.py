"""
CSS styling for SupaBot BI Dashboard
Centralizes all application styling and UI theming with responsive design.
"""

import streamlit as st
import os


class DashboardStyles:
    """CSS styles for the SupaBot BI Dashboard with responsive design."""
    
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
    
    # Comprehensive dashboard styling with responsive design
    DASHBOARD_CSS = """
    <style>
    /* ===== BASE MOBILE-FIRST STYLES ===== */
    .stApp { 
        background-color: #0e1117; 
        min-width: 320px;
    }
    
    /* Main header - responsive */
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        padding: 1rem;
        border-radius: 10px; 
        text-align: center;
        color: white; 
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 { 
        font-size: 1.8rem; 
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* KPI Metric Boxes - Mobile first */
    div[data-testid="stMetric"] {
        background-color: #1c1e26; 
        border: 1px solid #2e303d;
        padding: 0.75rem; 
        border-radius: 10px;
        height: auto;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.4rem;
        font-weight: bold; 
        color: #00d2ff;
        margin-bottom: 0.3rem;
    }
    
    div[data-testid="stMetric"] > div:nth-child(3) {
        font-size: 0.7rem;
    }

    /* Custom card styling - Mobile first */
    .dashboard-card {
        background-color: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 0.75rem;
        height: auto;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        margin-bottom: 1rem;
    }
    
    .dashboard-card-tall {
        min-height: 400px;
    }
    
    .dashboard-card h5 {
        font-size: 1rem;
        color: #c7c7c7;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #3a47d5;
        padding-bottom: 0.5rem;
    }
    
    /* Filter controls - Mobile responsive */
    .filter-controls {
        margin-bottom: 1.5rem;
    }
    
    /* Radio buttons - Mobile friendly */
    [data-baseweb="radio"] {
        margin-bottom: 0.5rem;
    }
    
    /* Multiselect - Mobile optimized */
    [data-baseweb="select"] {
        margin-bottom: 0.5rem;
    }
    
    /* Date inputs - Mobile friendly */
    [data-baseweb="date-input"] {
        margin-bottom: 0.5rem;
    }
    
    /* Insight boxes */
    .insight-box {
        background: #16a085; 
        padding: 0.75rem; 
        border-radius: 8px;
        color: white; 
        margin-top: 0.75rem; 
        text-align: center;
    }
    
    /* Chat messages */
    .user-message{
        background:linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        padding:0.75rem 1rem; 
        border-radius:20px 20px 0 20px;
        margin:0.75rem 0; 
        color:white; 
        font-weight:500;
    }
    
    .ai-message{
        background: #262730; 
        border: 1px solid #3d3d3d;
        padding:0.75rem 1rem; 
        border-radius:20px 20px 20px 0;
        margin:0.75rem 0; 
        color:white;
    }
    
    /* Tab buttons */
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 2px solid transparent;
        font-size: 1rem;
        padding: 0.5rem 0.75rem;
        min-height: 44px; /* Touch-friendly */
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #00d2ff;
        color: #00d2ff;
    }
    
    /* ===== TABLET STYLES (768px - 1023px) ===== */
    @media (min-width: 768px) {
        .main-header {
            padding: 1.25rem;
            margin-bottom: 2rem;
        }
        
        .main-header h1 { 
            font-size: 2.2rem; 
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        div[data-testid="stMetric"] {
            padding: 1rem;
            min-height: 120px;
            margin-bottom: 0.75rem;
        }
        
        div[data-testid="stMetric"] > div:nth-child(1) {
            font-size: 0.85rem;
        }
        
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.6rem;
        }
        
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.75rem;
        }
        
        .dashboard-card {
            padding: 1rem;
            min-height: 350px;
            margin-bottom: 1.5rem;
        }
        
        .dashboard-card h5 {
            font-size: 1.05rem;
        }
        
        .filter-controls {
            margin-bottom: 2rem;
        }
    }
    
    /* ===== DESKTOP STYLES (1024px+) ===== */
    @media (min-width: 1024px) {
        .main-header {
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .main-header h1 { 
            font-size: 2.5rem; 
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
        
        div[data-testid="stMetric"] {
            padding: 1rem;
            height: 130px;
            margin-bottom: 0;
        }
        
        div[data-testid="stMetric"] > div:nth-child(1) {
            font-size: 0.9rem;
        }
        
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.8rem;
        }
        
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.8rem;
        }
        
        .dashboard-card {
            padding: 1rem;
            height: 350px;
            margin-bottom: 0;
        }
        
        .dashboard-card-tall {
            height: 450px;
        }
        
        .dashboard-card h5 {
            font-size: 1.1rem;
        }
        
        .filter-controls {
            margin-bottom: 2rem;
        }
    }
    
    /* ===== RESPONSIVE LAYOUT UTILITIES ===== */
    
    /* Mobile: Stack all columns vertically */
    @media (max-width: 767px) {
        /* Force single column layout for all st.columns */
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* KPI grid - 2x2 on mobile */
        .kpi-container [data-testid="column"] {
            width: 50% !important;
            float: left;
        }
        
        .kpi-container [data-testid="column"]:nth-child(odd) {
            clear: left;
        }
        
        /* Chart containers - full width on mobile */
        .chart-container {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* Filter controls - stack vertically */
        .filter-container [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 0.5rem;
        }
        
        /* Tables - horizontal scroll on mobile */
        .table-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        /* Touch-friendly buttons */
        button, [role="button"] {
            min-height: 44px;
            min-width: 44px;
        }
        
        /* Increase spacing between interactive elements */
        [data-testid="stMetric"], .dashboard-card, button {
            margin-bottom: 0.75rem;
        }
    }
    
    /* Tablet: 2-column layout for some sections */
    @media (min-width: 768px) and (max-width: 1023px) {
        /* KPI grid - 2x2 on tablet */
        .kpi-container [data-testid="column"] {
            width: 50% !important;
        }
        
        /* Chart layout - side by side when space allows */
        .chart-container [data-testid="column"] {
            width: 50% !important;
        }
        
        /* Filter controls - 3 columns on tablet */
        .filter-container [data-testid="column"] {
            width: 33.33% !important;
        }
    }
    
    /* Desktop: Full multi-column layout */
    @media (min-width: 1024px) {
        /* Restore original column layouts */
        .kpi-container [data-testid="column"] {
            width: 25% !important;
        }
        
        .chart-container [data-testid="column"] {
            width: 50% !important;
        }
        
        .filter-container [data-testid="column"] {
            width: 33.33% !important;
        }
    }
    
    /* ===== PERFORMANCE OPTIMIZATIONS ===== */
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Optimize for mobile networks */
    img, svg {
        max-width: 100%;
        height: auto;
    }
    
    /* Touch-friendly scrolling */
    .stApp {
        -webkit-overflow-scrolling: touch;
    }
    
    /* ===== ACCESSIBILITY IMPROVEMENTS ===== */
    
    /* Focus indicators */
    button:focus, [role="button"]:focus {
        outline: 2px solid #00d2ff;
        outline-offset: 2px;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .dashboard-card {
            border-width: 2px;
        }
        
        div[data-testid="stMetric"] {
            border-width: 2px;
        }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
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
    def load_responsive_css():
        """Load the responsive CSS file."""
        css_file_path = os.path.join(os.path.dirname(__file__), 'responsive.css')
        if os.path.exists(css_file_path):
            with open(css_file_path, 'r') as f:
                responsive_css = f.read()
                st.markdown(f'<style>{responsive_css}</style>', unsafe_allow_html=True)
        else:
            st.warning("Responsive CSS file not found. Some responsive features may not work properly.")
    
    @staticmethod
    def load_all_styles():
        """Load all CSS styles for the application."""
        DashboardStyles.load_custom_dark_css()
        DashboardStyles.load_dashboard_css()
        DashboardStyles.load_responsive_css()


# Legacy function for compatibility
def load_css():
    """Legacy compatibility function."""
    DashboardStyles.load_dashboard_css()


# Auto-load styles when imported
DashboardStyles.load_custom_dark_css()

