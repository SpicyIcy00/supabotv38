"""
Test script to verify mobile optimization features.
"""

import streamlit as st
from supabot.ui.styles.css import DashboardStyles
from supabot.ui.components.mobile_utils import MobileUtils, is_mobile, responsive_columns
from supabot.ui.components.metrics import MetricsDisplay
from supabot.ui.components.charts import ChartFactory
import pandas as pd
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Mobile Test - SupaBot BI",
    page_icon="📱",
    layout="wide"
)

# Load styles
DashboardStyles.load_all_styles()

# Mobile mode toggle
if st.sidebar.checkbox("📱 Mobile Mode (for testing)", value=False):
    MobileUtils.set_mobile_mode(True)
else:
    MobileUtils.set_mobile_mode(False)

# Show current mode
st.sidebar.write(f"**Current Mode:** {'📱 Mobile' if is_mobile() else '🖥️ Desktop'}")

# Header
st.markdown('<div class="main-header"><h1>📱 Mobile Optimization Test</h1><p>Testing responsive layouts and mobile-optimized components</p></div>', unsafe_allow_html=True)

# Test 1: Responsive KPI Metrics
st.subheader("🚀 Test 1: Responsive KPI Metrics")

# Sample metrics data
test_metrics = {
    'current_sales': 1250000,
    'prev_sales': 1100000,
    'current_profit': 375000,
    'prev_profit': 330000,
    'current_transactions': 1250,
    'prev_transactions': 1100,
    'avg_transaction_value': 1000,
    'prev_avg_transaction_value': 1000
}

if is_mobile():
    MetricsDisplay.render_kpi_metrics_mobile(test_metrics, "7D")
else:
    MetricsDisplay.render_kpi_metrics(test_metrics, "7D")

# Test 2: Responsive Charts
st.subheader("📊 Test 2: Responsive Charts")

# Sample data
sample_data = pd.DataFrame({
    'category': ['Electronics', 'Clothing', 'Food', 'Books', 'Sports'],
    'sales': [50000, 30000, 25000, 15000, 10000]
})

# Create chart
if is_mobile():
    fig = ChartFactory.create_bar_chart_mobile(sample_data, "Sales by Category", ['sales'], ['category'])
else:
    fig = ChartFactory.create_bar_chart(sample_data, "Sales by Category", ['sales'], ['category'])

st.plotly_chart(fig, use_container_width=True)

# Test 3: Responsive Layout
st.subheader("📐 Test 3: Responsive Layout")

# Test responsive columns
cols = responsive_columns(4, 2)  # 4 columns on desktop, 2 on mobile

with cols[0]:
    st.metric("Metric 1", "100", "+5%")
with cols[1]:
    st.metric("Metric 2", "200", "+10%")
with cols[2]:
    st.metric("Metric 3", "300", "-2%")
with cols[3]:
    st.metric("Metric 4", "400", "+15%")

# Test 4: Mobile-optimized Tables
st.subheader("📋 Test 4: Mobile-optimized Tables")

# Sample table data
table_data = pd.DataFrame({
    'Product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    'Sales': [50000, 30000, 25000, 15000, 10000],
    'Change': [5.2, -2.1, 8.7, 1.3, -0.5]
})

if is_mobile():
    # Mobile-optimized table display
    for _, row in table_data.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{row['Product']}**")
        with col2:
            st.write(f"₱{row['Sales']:,.0f}")
        with col3:
            if row['Change'] > 0:
                st.write(f"🟢 +{row['Change']:.1f}%")
            else:
                st.write(f"🔴 {row['Change']:.1f}%")
else:
    # Desktop table display
    st.dataframe(table_data, use_container_width=True)

# Test 5: Mobile-optimized Filters
st.subheader("🔧 Test 5: Mobile-optimized Filters")

if is_mobile():
    # Mobile filters
    time_filter = st.radio("Time Period", ["1D", "7D", "1M", "6M", "1Y"], horizontal=True)
    store_filter = st.selectbox("Store", ["All Stores", "Store A", "Store B", "Store C"])
else:
    # Desktop filters
    col1, col2 = st.columns(2)
    with col1:
        time_filter = st.selectbox("Time Period", ["1D", "7D", "1M", "6M", "1Y"])
    with col2:
        store_filter = st.multiselect("Stores", ["All Stores", "Store A", "Store B", "Store C"])

st.write(f"**Selected:** {time_filter} | {store_filter}")

# Test 6: CSS Responsiveness
st.subheader("🎨 Test 6: CSS Responsiveness")

st.info("""
This section tests the CSS responsiveness. 
On mobile devices, you should see:
- Smaller fonts and padding
- Adjusted chart heights
- Touch-friendly button sizes
- Optimized spacing
""")

# Test buttons
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Button 1")
with col2:
    st.button("Button 2")
with col3:
    st.button("Button 3")

# Test expandable sections
with st.expander("📱 Mobile Optimization Features"):
    st.write("""
    **Mobile Optimizations Implemented:**
    
    1. **Responsive Layouts**: Automatic switching between desktop and mobile layouts
    2. **Mobile CSS**: Optimized styling for small screens
    3. **Touch-friendly Controls**: Larger touch targets and simplified interactions
    4. **Optimized Charts**: Smaller heights and simplified legends for mobile
    5. **Mobile Tables**: Card-based layouts instead of traditional tables
    6. **Responsive Filters**: Simplified filter controls for mobile
    7. **Mobile KPIs**: 2x2 grid layout instead of 4 columns
    """)

st.success("✅ Mobile optimization test completed! Toggle the mobile mode checkbox in the sidebar to see the differences.")
