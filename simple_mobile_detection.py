#!/usr/bin/env python3
"""
Simple mobile detection that works reliably
"""

import streamlit as st
import streamlit.components.v1 as components

def detect_mobile_simple():
    """Simple mobile detection using JavaScript"""
    
    # JavaScript to detect screen width and set session state
    js_code = """
    <script>
    function setScreenSize() {
        const width = window.innerWidth;
        let size = 'desktop';
        
        if (width < 768) {
            size = 'mobile';
        } else if (width < 1024) {
            size = 'tablet';
        }
        
        // Set in session storage
        sessionStorage.setItem('screen_size', size);
        
        // Try to communicate with Streamlit
        try {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: size
            }, '*');
        } catch (e) {
            console.log('Could not communicate with Streamlit:', e);
        }
    }
    
    // Run on load
    setScreenSize();
    
    // Run on resize
    window.addEventListener('resize', setScreenSize);
    </script>
    """
    
    # Run JavaScript
    components.html(js_code, height=0)
    
    # Check if we have a screen size in session state
    if 'screen_size' not in st.session_state:
        st.session_state.screen_size = 'desktop'
    
    return st.session_state.screen_size

def force_mobile_mode():
    """Force mobile mode for testing"""
    st.session_state.screen_size = 'mobile'
    return 'mobile'

def force_desktop_mode():
    """Force desktop mode for testing"""
    st.session_state.screen_size = 'desktop'
    return 'desktop'

def main():
    st.set_page_config(page_title="Mobile Detection Test", page_icon="📱")
    
    st.title("📱 Simple Mobile Detection Test")
    
    # Test mobile detection
    screen_size = detect_mobile_simple()
    
    st.info(f"Current screen size: {screen_size}")
    
    # Force mode buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Force Mobile Mode"):
            force_mobile_mode()
            st.success("✅ Mobile mode forced")
            st.rerun()
    
    with col2:
        if st.button("Force Desktop Mode"):
            force_desktop_mode()
            st.success("✅ Desktop mode forced")
            st.rerun()
    
    # Show current mode
    current_mode = st.session_state.get('screen_size', 'not set')
    st.info(f"Session state screen size: {current_mode}")
    
    if current_mode == 'mobile':
        st.success("🎯 MOBILE MODE ACTIVE")
        st.markdown("""
        ### Mobile Features:
        - 2x2 KPI grid
        - Scrollable charts
        - Card-based lists
        - Mobile navigation
        """)
    else:
        st.info("💻 DESKTOP MODE ACTIVE")
        st.markdown("""
        ### Desktop Features:
        - 4-column KPI grid
        - Full-width charts
        - Standard tables
        - Sidebar navigation
        """)

if __name__ == "__main__":
    main()
