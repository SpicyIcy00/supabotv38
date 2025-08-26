"""
Mobile navigation component for responsive dashboard.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Callable


class MobileNavigation:
    """Mobile-optimized navigation component."""
    
    @staticmethod
    def render_mobile_header(title: str = "SupaBot BI Dashboard"):
        """
        Render mobile-optimized header with navigation.
        
        Args:
            title: Dashboard title
        """
        # Mobile header with hamburger menu
        header_html = f"""
        <div class="mobile-header" style="
            background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <h1 style="
                font-size: 1.5rem;
                font-weight: bold;
                color: white;
                margin: 0;
            ">{title}</h1>
            <button class="mobile-nav-toggle" onclick="toggleMobileNav()" style="
                background: rgba(255,255,255,0.2);
                border: none;
                border-radius: 6px;
                padding: 0.5rem;
                color: white;
                font-size: 1.2rem;
                cursor: pointer;
            ">☰</button>
        </div>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
        
        # Add JavaScript for mobile navigation
        MobileNavigation._add_mobile_nav_script()
    
    @staticmethod
    def render_bottom_navigation(
        tabs: List[Dict[str, Any]],
        active_tab: str = "dashboard"
    ):
        """
        Render bottom tab navigation for mobile.
        
        Args:
            tabs: List of tab configurations
            active_tab: Currently active tab
        """
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        # Create bottom navigation
        nav_html = """
        <div class="mobile-bottom-nav" style="
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #1c1e26;
            border-top: 1px solid #2e303d;
            padding: 0.5rem;
            z-index: 1000;
            display: flex;
            justify-content: space-around;
        ">
        """
        
        for tab in tabs:
            is_active = tab['key'] == active_tab
            active_style = "color: #00d2ff; border-top: 2px solid #00d2ff;" if is_active else "color: #888;"
            
            nav_html += f"""
            <button onclick="switchTab('{tab['key']}')" style="
                background: none;
                border: none;
                padding: 0.5rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.25rem;
                cursor: pointer;
                {active_style}
                font-size: 0.8rem;
                min-height: 44px;
                justify-content: center;
            ">
                <span style="font-size: 1.2rem;">{tab['icon']}</span>
                <span>{tab['label']}</span>
            </button>
            """
        
        nav_html += "</div>"
        
        # Add padding to main content to account for bottom nav
        st.markdown("""
        <div style="padding-bottom: 80px;"></div>
        """, unsafe_allow_html=True)
        
        st.markdown(nav_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_filter_panel(
        filters: List[Dict[str, Any]],
        on_filter_change: Optional[Callable] = None
    ):
        """
        Render mobile filter panel.
        
        Args:
            filters: List of filter configurations
            on_filter_change: Callback function for filter changes
        """
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        with st.expander("🔧 Filters", expanded=False):
            st.markdown("### Filter Options")
            
            for filter_config in filters:
                MobileNavigation._render_filter_control(filter_config)
            
            # Filter actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Filters", key="apply_filters"):
                    if on_filter_change:
                        on_filter_change()
            
            with col2:
                if st.button("Reset", key="reset_filters"):
                    # Reset all filters
                    for filter_config in filters:
                        if 'key' in filter_config:
                            if filter_config['key'] in st.session_state:
                                del st.session_state[filter_config['key']]
    
    @staticmethod
    def _render_filter_control(filter_config: Dict[str, Any]):
        """Render individual filter control."""
        filter_type = filter_config.get('type', 'selectbox')
        label = filter_config.get('label', 'Filter')
        key = filter_config.get('key', 'filter')
        options = filter_config.get('options', [])
        default = filter_config.get('default', None)
        
        if filter_type == 'selectbox':
            st.selectbox(label, options, key=key, index=default if default is not None else 0)
        elif filter_type == 'multiselect':
            st.multiselect(label, options, key=key, default=default or [])
        elif filter_type == 'slider':
            min_val = filter_config.get('min', 0)
            max_val = filter_config.get('max', 100)
            st.slider(label, min_val, max_val, default or min_val, key=key)
        elif filter_type == 'date_input':
            st.date_input(label, key=key, value=default)
    
    @staticmethod
    def render_quick_actions(actions: List[Dict[str, Any]]):
        """
        Render quick action buttons for mobile.
        
        Args:
            actions: List of action configurations
        """
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        with st.container():
            st.markdown("### ⚡ Quick Actions")
            
            # Create action buttons in a grid
            cols = st.columns(2)
            
            for idx, action in enumerate(actions):
                col_idx = idx % 2
                with cols[col_idx]:
                    if st.button(
                        f"{action['icon']} {action['label']}",
                        key=f"quick_action_{idx}",
                        help=action.get('help', '')
                    ):
                        if 'callback' in action:
                            action['callback']()
    
    @staticmethod
    def render_mobile_menu():
        """Render mobile slide-out menu."""
        menu_html = """
        <div id="mobileMenu" class="mobile-nav-drawer" style="
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
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #2e303d;
            ">
                <h3 style="color: white; margin: 0;">Menu</h3>
                <button onclick="closeMobileNav()" style="
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.5rem;
                    cursor: pointer;
                ">×</button>
            </div>
            
            <div class="mobile-menu-items">
                <a href="#" onclick="navigateTo('dashboard')" style="
                    display: block;
                    padding: 1rem;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: background-color 0.2s ease;
                ">📊 Dashboard</a>
                
                <a href="#" onclick="navigateTo('analytics')" style="
                    display: block;
                    padding: 1rem;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: background-color 0.2s ease;
                ">📈 Analytics</a>
                
                <a href="#" onclick="navigateTo('reports')" style="
                    display: block;
                    padding: 1rem;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: background-color 0.2s ease;
                ">📋 Reports</a>
                
                <a href="#" onclick="navigateTo('settings')" style="
                    display: block;
                    padding: 1rem;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: background-color 0.2s ease;
                ">⚙️ Settings</a>
            </div>
        </div>
        """
        
        st.markdown(menu_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_pull_to_refresh():
        """Render pull-to-refresh indicator."""
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        refresh_html = """
        <div class="pull-to-refresh" style="
            text-align: center;
            padding: 1rem;
            color: #888;
            font-size: 0.8rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin-bottom: 1rem;
        ">
            🔄 Pull down to refresh
        </div>
        """
        
        st.markdown(refresh_html, unsafe_allow_html=True)
    
    @staticmethod
    def _add_mobile_nav_script():
        """Add JavaScript for mobile navigation functionality."""
        script_html = """
        <script>
        function toggleMobileNav() {
            const menu = document.getElementById('mobileMenu');
            if (menu) {
                menu.style.transform = menu.style.transform === 'translateX(0px)' 
                    ? 'translateX(-100%)' 
                    : 'translateX(0px)';
            }
        }
        
        function closeMobileNav() {
            const menu = document.getElementById('mobileMenu');
            if (menu) {
                menu.style.transform = 'translateX(-100%)';
            }
        }
        
        function switchTab(tabKey) {
            // Send tab change to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: tabKey
            }, '*');
        }
        
        function navigateTo(page) {
            // Handle navigation
            console.log('Navigating to:', page);
            closeMobileNav();
        }
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menu = document.getElementById('mobileMenu');
            const toggle = document.querySelector('.mobile-nav-toggle');
            
            if (menu && !menu.contains(event.target) && !toggle.contains(event.target)) {
                menu.style.transform = 'translateX(-100%)';
            }
        });
        
        // Handle pull to refresh
        let startY = 0;
        let currentY = 0;
        let pullDistance = 0;
        
        document.addEventListener('touchstart', function(e) {
            startY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchmove', function(e) {
            currentY = e.touches[0].clientY;
            pullDistance = currentY - startY;
            
            if (pullDistance > 50 && window.scrollY === 0) {
                // Show pull to refresh indicator
                const indicator = document.querySelector('.pull-to-refresh');
                if (indicator) {
                    indicator.style.display = 'block';
                }
            }
        });
        
        document.addEventListener('touchend', function(e) {
            if (pullDistance > 100 && window.scrollY === 0) {
                // Trigger refresh
                window.location.reload();
            }
            
            pullDistance = 0;
            
            // Hide pull to refresh indicator
            const indicator = document.querySelector('.pull-to-refresh');
            if (indicator) {
                indicator.style.display = 'none';
            }
        });
        </script>
        """
        
        st.markdown(script_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_mobile_filters_section():
        """Render mobile-optimized filters section."""
        if st.session_state.get('screen_size', 'desktop') != 'mobile':
            return
        
        with st.container():
            st.markdown("### 🔧 Quick Filters")
            
            # Time period filter
            time_period = st.selectbox(
                "Time Period",
                ["Today", "This Week", "This Month", "This Quarter", "This Year"],
                key="mobile_time_period"
            )
            
            # Store filter
            stores = ["All Stores", "Store 1", "Store 2", "Store 3"]
            selected_stores = st.multiselect(
                "Stores",
                stores,
                default=["All Stores"],
                key="mobile_stores"
            )
            
            # Category filter
            categories = ["All Categories", "Electronics", "Clothing", "Food", "Books"]
            selected_categories = st.multiselect(
                "Categories",
                categories,
                default=["All Categories"],
                key="mobile_categories"
            )
            
            # Apply filters button
            if st.button("Apply Filters", key="mobile_apply_filters"):
                st.success("Filters applied!")
                st.rerun()
