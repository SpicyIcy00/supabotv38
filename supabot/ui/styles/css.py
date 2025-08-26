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
    
    # Comprehensive dashboard styling with mobile responsiveness
    DASHBOARD_CSS = """
    <style>
    /* Mobile-First Base Styles */
    .stApp { 
        background-color: #0e1117; 
        min-height: 100vh;
    }
    
    /* Mobile Navigation */
    .mobile-nav-toggle {
        display: none;
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: #3a47d5;
        border: none;
        border-radius: 8px;
        padding: 0.5rem;
        color: white;
        font-size: 1.5rem;
    }
    
    .mobile-nav-drawer {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 280px;
        height: 100vh;
        background: #1c1e26;
        border-right: 1px solid #2e303d;
        z-index: 999;
        transform: translateX(-100%);
        transition: transform 0.3s ease;
        overflow-y: auto;
        padding: 1rem;
    }
    
    .mobile-nav-drawer.open {
        transform: translateX(0);
    }
    
    /* Mobile Header */
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        padding: 1rem; 
        border-radius: 10px; 
        text-align: center;
        color: white; 
        margin-bottom: 1rem;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .main-header h1 { 
        font-size: 1.8rem; 
        font-weight: bold; 
        margin: 0;
    }
    
    /* Mobile KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* KPI Metric Boxes - Mobile First */
    div[data-testid="stMetric"] {
        background-color: #1c1e26; 
        border: 1px solid #2e303d;
        padding: 1rem; 
        border-radius: 10px;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stMetric"]:active {
        transform: scale(0.98);
    }
    
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.5rem;
        font-weight: bold; 
        color: #00d2ff;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }
    
    div[data-testid="stMetric"] > div:nth-child(3) {
        font-size: 0.75rem;
        line-height: 1.2;
    }

    /* Mobile Chart Containers */
    .chart-container {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border-radius: 10px;
        background: #1c1e26;
        border: 1px solid #2e303d;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .chart-container canvas {
        max-width: 100%;
        height: auto;
    }

    /* Custom card styling - Mobile */
    .dashboard-card {
        background-color: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 0.75rem;
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
    
    /* Mobile Data Tables */
    .mobile-table {
        background: #1c1e26;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .mobile-table-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        border-bottom: 1px solid #2e303d;
        transition: background-color 0.2s ease;
    }
    
    .mobile-table-row:last-child {
        border-bottom: none;
    }
    
    .mobile-table-row:hover {
        background-color: #262730;
    }
    
    .mobile-table-row:active {
        background-color: #3a47d5;
    }
    
    /* Mobile Product Cards */
    .mobile-product-card {
        background: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .mobile-product-card:active {
        transform: scale(0.98);
    }
    
    .mobile-product-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.5rem;
    }
    
    .mobile-product-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 0.9rem;
        line-height: 1.3;
        flex: 1;
        margin-right: 0.5rem;
    }
    
    .mobile-product-stats {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.25rem;
    }
    
    .mobile-product-sales {
        font-size: 1rem;
        font-weight: bold;
        color: #00d2ff;
    }
    
    .mobile-product-change {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .mobile-product-change.positive {
        background: rgba(0, 200, 83, 0.2);
        color: #00c853;
    }
    
    .mobile-product-change.negative {
        background: rgba(255, 82, 82, 0.2);
        color: #ff5252;
    }
    
    .mobile-product-change.neutral {
        background: rgba(170, 170, 170, 0.2);
        color: #aaaaaa;
    }
    
    /* Mobile Filter Controls */
    .mobile-filter-container {
        background: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .mobile-filter-row {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .mobile-filter-row:last-child {
        margin-bottom: 0;
    }
    
    /* Mobile Buttons */
    .mobile-button {
        background: linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .mobile-button:active {
        transform: scale(0.95);
    }
    
    /* Mobile Search */
    .mobile-search {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #2e303d;
        border-radius: 8px;
        background: #262730;
        color: white;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .mobile-search::placeholder {
        color: #888;
    }
    
    /* Mobile Pull to Refresh */
    .pull-to-refresh {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.8rem;
    }
    
    /* Mobile Swipe Indicators */
    .swipe-indicator {
        text-align: center;
        padding: 0.5rem;
        color: #888;
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    /* Insight and Message Styles */
    .insight-box {
        background: #16a085; 
        padding: 1rem; 
        border-radius: 8px;
        color: white; 
        margin-top: 1rem; 
        text-align: center;
    }
    
    .user-message{
        background:linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        padding:1rem 1.5rem; 
        border-radius:20px 20px 0 20px;
        margin:1rem 0; 
        color:white; 
        font-weight:500;
    }
    
    .ai-message{
        background: #262730; 
        border: 1px solid #3d3d3d;
        padding:1rem 1.5rem; 
        border-radius:20px 20px 20px 0;
        margin:1rem 0; 
        color:white;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 2px solid transparent;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        min-height: 44px;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #00d2ff;
        color: #00d2ff;
    }
    
    /* Mobile Container Padding */
    .block-container {
        padding: 0.75rem;
        max-width: 100%;
    }
    
    /* Tablet Styles (768px - 1024px) */
    @media (min-width: 768px) {
        .main-header {
            padding: 1.25rem;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .dashboard-card {
            padding: 1rem;
            min-height: 350px;
        }
        
        .dashboard-card h5 {
            font-size: 1.05rem;
        }
        
        .block-container {
            padding: 1rem;
        }
        
        div[data-testid="stMetric"] {
            min-height: 120px;
        }
        
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.6rem;
        }
    }
    
    /* Desktop Styles (1024px+) */
    @media (min-width: 1024px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(4, 1fr);
            gap: 1.25rem;
        }
        
        .dashboard-card {
            padding: 1rem;
            height: 350px;
        }
        
        .dashboard-card-tall {
            height: 450px;
        }
        
        .block-container {
            padding: 1.5rem;
        }
        
        div[data-testid="stMetric"] {
            height: 130px;
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
        
        /* Hide mobile-specific elements on desktop */
        .mobile-nav-toggle,
        .mobile-nav-drawer,
        .swipe-indicator {
            display: none !important;
        }
    }
    
    /* Large Desktop Styles (1200px+) */
    @media (min-width: 1200px) {
        .block-container {
            padding: 2rem;
        }
        
        .dashboard-card {
            height: 400px;
        }
        
        .dashboard-card-tall {
            height: 500px;
        }
    }
    
    /* Accessibility Improvements */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* High Contrast Mode Support */
    @media (prefers-contrast: high) {
        div[data-testid="stMetric"] {
            border: 2px solid #ffffff;
        }
        
        .dashboard-card {
            border: 2px solid #ffffff;
        }
        
        .mobile-product-card {
            border: 2px solid #ffffff;
        }
    }
    
    /* Dark Mode Enhancements */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0a0a0a;
        }
        
        .dashboard-card,
        div[data-testid="stMetric"],
        .mobile-product-card {
            background-color: #1a1a1a;
            border-color: #333;
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

