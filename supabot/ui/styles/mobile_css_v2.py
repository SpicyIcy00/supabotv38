"""
Mobile CSS V2 - Exact Visual Identity Preservation
Maintains desktop visual identity while optimizing for mobile layout.
"""

import streamlit as st


class MobileStylesV2:
    """Mobile-optimized CSS styles that preserve desktop visual identity."""
    
    # Mobile-specific CSS with exact visual identity preservation
    MOBILE_CSS_V2 = """
    <style>
    /* Mobile-specific styles - preserves desktop visual identity */
    @media (max-width: 768px) {
        /* Mobile header - compact but maintains visual identity */
        .mobile-header h2 { 
            font-size: 1.5rem !important; 
            margin-bottom: 0.5rem !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Mobile KPI cards - 2x2 grid with exact desktop styling */
        div[data-testid="stMetric"] {
            height: 120px !important;
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stMetric"] > div:nth-child(1) {
            font-size: 0.85rem !important;
            margin-bottom: 0.4rem !important;
            color: #cccccc !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 1.6rem !important;
            margin-bottom: 0.3rem !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] > div:nth-child(3) {
            font-size: 0.75rem !important;
            color: #00d2ff !important;
            font-weight: 600 !important;
        }
        
        /* Mobile chart containers - exact desktop styling */
        .dashboard-card, [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: 300px !important;
            padding: 1rem !important;
            margin-bottom: 1.5rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        }
        .dashboard-card h5, [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] h5 {
            font-size: 1.1rem !important;
            margin-bottom: 1rem !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Mobile filter controls - compact but touch-friendly */
        .stSelectbox, .stMultiselect {
            margin-bottom: 1rem !important;
        }
        .stSelectbox > div, .stMultiselect > div {
            min-height: 44px !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        .stSelectbox > div > div, .stMultiselect > div > div {
            color: #ffffff !important;
            font-size: 0.9rem !important;
        }
        
        /* Mobile data tables - exact desktop styling */
        .dataframe {
            font-size: 0.85rem !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        .dataframe th {
            background: #2a2a2a !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            padding: 0.75rem 0.5rem !important;
            border-bottom: 1px solid #333333 !important;
        }
        .dataframe td {
            padding: 0.6rem 0.5rem !important;
            color: #cccccc !important;
            border-bottom: 1px solid #333333 !important;
        }
        .dataframe tr:hover {
            background: #2a2a2a !important;
        }
        
        /* Mobile sidebar adjustments */
        .css-1d391kg {
            width: 100% !important;
            max-width: 100% !important;
            background: #0a0a0a !important;
        }
        
        /* Mobile button sizing - touch-friendly */
        .stButton > button {
            min-height: 44px !important;
            font-size: 1rem !important;
            padding: 0.75rem 1rem !important;
            background: linear-gradient(135deg, #00d2ff 0%, #0099cc 100%) !important;
            border: none !important;
            border-radius: 6px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 4px rgba(0, 210, 255, 0.3) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #00b8e6 0%, #0088b3 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0, 210, 255, 0.4) !important;
        }
        
        /* Mobile container spacing */
        .block-container {
            padding-top: 0.75rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            background: #0a0a0a !important;
        }
        
        /* Mobile chart responsiveness */
        .js-plotly-plot {
            max-width: 100% !important;
        }
        .plotly-graph-div {
            background: transparent !important;
        }
        
        /* Mobile text sizing - maintains hierarchy */
        h1 { font-size: 1.8rem !important; color: #ffffff !important; font-weight: 700 !important; }
        h2 { font-size: 1.5rem !important; color: #ffffff !important; font-weight: 600 !important; }
        h3 { font-size: 1.3rem !important; color: #ffffff !important; font-weight: 600 !important; }
        h4 { font-size: 1.1rem !important; color: #ffffff !important; font-weight: 600 !important; }
        h5 { font-size: 1rem !important; color: #ffffff !important; font-weight: 600 !important; }
        
        /* Mobile spacing - maintains visual rhythm */
        .stMarkdown {
            margin-bottom: 0.75rem !important;
        }
        
        /* Mobile container borders - exact desktop styling */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > div {
            margin-bottom: 0.75rem !important;
        }
        
        /* Mobile radio buttons - touch-friendly */
        .stRadio > div {
            flex-direction: column !important;
        }
        .stRadio > div > label {
            margin-bottom: 0.75rem !important;
            min-height: 44px !important;
            display: flex !important;
            align-items: center !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
            padding: 0.75rem !important;
            color: #ffffff !important;
        }
        .stRadio > div > label:hover {
            background: #2a2a2a !important;
            border-color: #00d2ff !important;
        }
        
        /* Mobile multiselect - touch-friendly */
        .stMultiSelect > div {
            min-height: 44px !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        
        /* Mobile date inputs - touch-friendly */
        .stDateInput > div {
            min-height: 44px !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        
        /* Mobile progress bars */
        .stProgress > div {
            height: 8px !important;
            background: #1a1a1a !important;
            border-radius: 4px !important;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #00d2ff 0%, #0099cc 100%) !important;
            border-radius: 4px !important;
        }
        
        /* Mobile expander */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 0.75rem !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            background: #1a1a1a !important;
            border-radius: 6px !important;
            padding: 0.5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem !important;
            font-size: 0.9rem !important;
            background: #2a2a2a !important;
            border: 1px solid #333333 !important;
            border-radius: 4px !important;
            color: #cccccc !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #00d2ff 0%, #0099cc 100%) !important;
            color: #ffffff !important;
            border-color: #00d2ff !important;
        }
        
        /* Mobile alerts and messages - exact desktop styling */
        .stAlert {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile info boxes */
        .stInfo {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #00d2ff !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile warning boxes */
        .stWarning {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #ff9800 !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile error boxes */
        .stError {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #f44336 !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile success boxes */
        .stSuccess {
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
            border: 1px solid #4caf50 !important;
            border-radius: 6px !important;
            color: #ffffff !important;
        }
        
        /* Mobile spinner */
        .stSpinner {
            margin: 1.5rem auto !important;
        }
        .stSpinner > div {
            border: 3px solid #1a1a1a !important;
            border-top: 3px solid #00d2ff !important;
            border-radius: 50% !important;
        }
        
        /* Mobile caption text */
        .caption {
            font-size: 0.75rem !important;
            margin-top: 0.5rem !important;
            color: #cccccc !important;
        }
        
        /* Mobile horizontal rules */
        hr {
            margin: 1.5rem 0 !important;
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent 0%, #333333 50%, transparent 100%) !important;
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
            padding: 0 0.5rem !important;
        }
        
        /* Mobile row spacing */
        [data-testid="row"] {
            margin-bottom: 0.75rem !important;
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
            padding: 0.75rem !important;
        }
        
        /* Mobile-friendly hover states */
        button:hover, [role="button"]:hover {
            transform: translateY(-1px) !important;
            transition: transform 0.2s ease !important;
        }
    }
    
    /* Mobile-specific dark theme adjustments - exact desktop colors */
    @media (max-width: 768px) {
        /* Ensure proper contrast on mobile */
        .stApp {
            background: #0a0a0a !important;
        }
        
        /* Mobile text contrast - exact desktop colors */
        .stMarkdown, p, span, div {
            color: #ffffff !important;
        }
        
        /* Mobile link colors - exact desktop colors */
        a {
            color: #00d2ff !important;
        }
        a:hover {
            color: #0099cc !important;
        }
        
        /* Mobile selection colors - exact desktop colors */
        ::selection {
            background-color: #00d2ff !important;
            color: #000000 !important;
        }
        
        /* Mobile scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a1a !important;
        }
        ::-webkit-scrollbar-thumb {
            background: #333333 !important;
            border-radius: 4px !important;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #00d2ff !important;
        }
    }
    </style>
    """
    
    @staticmethod
    def load_mobile_styles_v2():
        """Load mobile-specific CSS styles with exact visual identity preservation."""
        st.markdown(MobileStylesV2.MOBILE_CSS_V2, unsafe_allow_html=True)
    
    @staticmethod
    def inject_mobile_viewport():
        """Inject mobile viewport meta tag."""
        viewport_meta = """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        """
        st.markdown(viewport_meta, unsafe_allow_html=True)
    
    @staticmethod
    def load_all_mobile_styles_v2():
        """Load all mobile-specific styles and viewport settings."""
        MobileStylesV2.inject_mobile_viewport()
        MobileStylesV2.load_mobile_styles_v2()


# Legacy function for compatibility
def load_mobile_css_v2():
    """Legacy compatibility function."""
    MobileStylesV2.load_mobile_styles_v2()
