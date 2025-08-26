"""
Mobile-Responsive CSS for SupaBot BI Dashboard
"""

def get_mobile_responsive_css():
    """Get comprehensive mobile-responsive CSS"""
    
    css = """
    <style>
    /* ===== MOBILE-FIRST RESPONSIVE DESIGN ===== */
    
    /* Base Mobile Styles (320px - 767px) */
    .main-header {
        padding: 12px;
        margin-bottom: 16px;
    }
    
    .main-header h1 {
        font-size: 1.5rem;
        margin-bottom: 8px;
    }
    
    .main-header p {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* KPI Cards - Mobile First */
    .kpi-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid #333;
    }
    
    .kpi-card h3 {
        font-size: 0.9rem;
        margin-bottom: 8px;
        color: #ccc;
    }
    
    .kpi-card .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4aa;
    }
    
    .kpi-card .change {
        font-size: 0.8rem;
        margin-top: 4px;
    }
    
    /* Charts Container - Mobile First */
    .charts-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .chart-wrapper {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #333;
        min-height: 300px;
    }
    
    .chart-wrapper h3 {
        font-size: 1.1rem;
        margin-bottom: 12px;
        color: #fff;
    }
    
    /* Mobile Product List */
    .mobile-product-list {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    
    .mobile-product-list h3 {
        font-size: 1.1rem;
        margin-bottom: 16px;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .product-card-mobile {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #333;
    }
    
    .product-card-mobile:last-child {
        border-bottom: none;
    }
    
    .product-header {
        display: flex;
        align-items: center;
        gap: 8px;
        flex: 1;
    }
    
    .rank {
        background: #333;
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
        min-width: 24px;
        text-align: center;
    }
    
    .product-name {
        font-size: 14px;
        color: #fff;
        flex: 1;
        word-break: break-word;
    }
    
    .product-stats {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
        min-width: 80px;
    }
    
    .sales {
        font-weight: bold;
        color: #00d4aa;
        font-size: 14px;
    }
    
    .change.positive {
        color: #00d4aa;
        font-size: 12px;
    }
    
    .change.negative {
        color: #ff6b6b;
        font-size: 12px;
    }
    
    /* Mobile Category List */
    .mobile-category-list {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    
    .mobile-category-list h3 {
        font-size: 1.1rem;
        margin-bottom: 16px;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .category-card-mobile {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #333;
    }
    
    .category-card-mobile:last-child {
        border-bottom: none;
    }
    
    .category-name {
        font-size: 14px;
        color: #fff;
        flex: 1;
    }
    
    .category-stats {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
        min-width: 80px;
    }
    
    /* Filter Controls - Mobile */
    .filter-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 20px;
        padding: 16px;
        background: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #333;
    }
    
    .filter-row {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        color: #ccc;
        margin-bottom: 4px;
    }
    
    /* Touch-friendly buttons */
    .mobile-button {
        min-height: 44px;
        min-width: 44px;
        padding: 12px 16px;
        background: #333;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .mobile-button:hover {
        background: #444;
    }
    
    .mobile-button:active {
        background: #555;
    }
    
    /* Scrollable chart containers */
    .chart-container {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    .chart-container .js-plotly-plot {
        min-width: 400px;
    }
    
    /* Responsive tables */
    .mobile-table {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    .mobile-table table {
        min-width: 500px;
    }
    
    /* ===== TABLET STYLES (768px - 1023px) ===== */
    @media (min-width: 768px) {
        .main-header {
            padding: 16px 24px;
            margin-bottom: 24px;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        
        .filter-container {
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
        }
        
        .filter-row {
            flex-direction: row;
            align-items: center;
            gap: 12px;
        }
    }
    
    /* ===== DESKTOP STYLES (1024px+) ===== */
    @media (min-width: 1024px) {
        .main-header {
            padding: 20px 32px;
            margin-bottom: 32px;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }
        
        .charts-container {
            grid-template-columns: 1fr 1fr 1fr;
            gap: 32px;
        }
        
        .chart-wrapper {
            min-height: 350px;
        }
    }
    
    /* ===== LARGE DESKTOP STYLES (1200px+) ===== */
    @media (min-width: 1200px) {
        .charts-container {
            grid-template-columns: 1fr 1fr 1fr 1fr;
        }
    }
    
    /* ===== UTILITY CLASSES ===== */
    .mobile-only {
        display: block;
    }
    
    .desktop-only {
        display: none;
    }
    
    @media (min-width: 768px) {
        .mobile-only {
            display: none;
        }
        
        .desktop-only {
            display: block;
        }
    }
    
    /* ===== DARK THEME ENHANCEMENTS ===== */
    .mobile-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .mobile-section-title {
        font-size: 1.1rem;
        color: #fff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .mobile-divider {
        height: 1px;
        background: #333;
        margin: 20px 0;
    }
    
    /* ===== ACCESSIBILITY ===== */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* ===== HIGH CONTRAST MODE ===== */
    @media (prefers-contrast: high) {
        .kpi-card,
        .chart-wrapper,
        .mobile-product-list,
        .mobile-category-list,
        .filter-container {
            border: 2px solid #fff;
        }
    }
    
    </style>
    """
    
    return css
