"""
Mobile detection utilities for SupaBot BI Dashboard.
Provides reliable mobile device detection for conditional rendering.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Optional


class MobileDetection:
    """Mobile device detection utilities."""
    
    @staticmethod
    def inject_detection_script():
        """Inject JavaScript for mobile detection."""
        detection_script = """
        <script>
        // Mobile detection function
        function detectMobileDevice() {
            const userAgent = navigator.userAgent || navigator.vendor || window.opera;
            const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i;
            const isMobileDevice = mobileRegex.test(userAgent) || window.innerWidth <= 768;
            
            // Store detection result
            sessionStorage.setItem('isMobileDevice', isMobileDevice.toString());
            sessionStorage.setItem('screenWidth', window.innerWidth.toString());
            
            // Add mobile class to body if mobile
            if (isMobileDevice) {
                document.body.classList.add('mobile-device');
                document.body.setAttribute('data-mobile', 'true');
            } else {
                document.body.classList.remove('mobile-device');
                document.body.setAttribute('data-mobile', 'false');
            }
            
            return isMobileDevice;
        }
        
        // Initial detection
        detectMobileDevice();
        
        // Listen for resize events
        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(function() {
                detectMobileDevice();
            }, 250);
        });
        
        // Listen for orientation changes
        window.addEventListener('orientationchange', function() {
            setTimeout(function() {
                detectMobileDevice();
            }, 500);
        });
        </script>
        """
        st.markdown(detection_script, unsafe_allow_html=True)
    
    @staticmethod
    def get_mobile_status() -> bool:
        """Get mobile status using JavaScript."""
        # Inject the detection script
        MobileDetection.inject_detection_script()
        
        # Create a component to get the mobile status
        mobile_status_js = """
        <script>
        const isMobile = sessionStorage.getItem('isMobileDevice') === 'true';
        const screenWidth = parseInt(sessionStorage.getItem('screenWidth') || '1024');
        
        // Send data to Streamlit
        if (window.parent && window.parent.postMessage) {
            window.parent.postMessage({
                type: 'mobile-detection',
                isMobile: isMobile,
                screenWidth: screenWidth
            }, '*');
        }
        </script>
        """
        
        # For now, use a simple heuristic based on session state
        # In a full implementation, you'd use the JavaScript result
        return st.session_state.get('force_mobile', False)
    
    @staticmethod
    def set_mobile_override(is_mobile: bool):
        """Override mobile detection for testing."""
        st.session_state.force_mobile = is_mobile
    
    @staticmethod
    def is_mobile() -> bool:
        """Check if current device is mobile."""
        # Check for mobile override first
        if 'force_mobile' in st.session_state:
            return st.session_state.force_mobile
        
        # Use detection result
        return MobileDetection.get_mobile_status()
    
    @staticmethod
    def get_screen_info() -> dict:
        """Get screen information."""
        return {
            'is_mobile': MobileDetection.is_mobile(),
            'screen_width': st.session_state.get('screen_width', 1024),
            'user_agent': st.session_state.get('user_agent', '')
        }


class ResponsiveLayout:
    """Responsive layout utilities."""
    
    @staticmethod
    def get_column_config(is_mobile: bool) -> dict:
        """Get column configuration based on device type."""
        if is_mobile:
            return {
                'kpi_columns': 2,  # 2x2 grid for mobile
                'main_columns': 1,  # Single column layout
                'chart_height': 250,  # Smaller charts for mobile
                'table_rows': 8,  # Fewer rows for mobile
            }
        else:
            return {
                'kpi_columns': 4,  # 4 columns for desktop
                'main_columns': 2,  # Two column layout
                'chart_height': 350,  # Larger charts for desktop
                'table_rows': 10,  # More rows for desktop
            }
    
    @staticmethod
    def get_spacing_config(is_mobile: bool) -> dict:
        """Get spacing configuration based on device type."""
        if is_mobile:
            return {
                'container_padding': '0.5rem',
                'element_margin': '0.5rem',
                'section_margin': '1rem',
                'button_padding': '0.75rem 1rem',
            }
        else:
            return {
                'container_padding': '1rem',
                'element_margin': '1rem',
                'section_margin': '2rem',
                'button_padding': '0.5rem 1rem',
            }


class MobileOptimization:
    """Mobile optimization utilities."""
    
    @staticmethod
    def optimize_text_for_mobile(text: str, max_length: int = 20) -> str:
        """Truncate text for mobile display."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def optimize_dataframe_for_mobile(df: 'pd.DataFrame', max_rows: int = 8) -> 'pd.DataFrame':
        """Optimize dataframe for mobile display."""
        if len(df) <= max_rows:
            return df
        return df.head(max_rows)
    
    @staticmethod
    def get_mobile_chart_config() -> dict:
        """Get mobile-optimized chart configuration."""
        return {
            'height': 250,
            'margin': dict(t=20, b=0, l=0, r=0),
            'font_size': 10,
            'legend_font_size': 9,
            'show_legend': False,
        }
    
    @staticmethod
    def get_desktop_chart_config() -> dict:
        """Get desktop-optimized chart configuration."""
        return {
            'height': 350,
            'margin': dict(t=30, b=0, l=0, r=0),
            'font_size': 12,
            'legend_font_size': 11,
            'show_legend': True,
        }


# Utility functions for easy access
def is_mobile_device() -> bool:
    """Check if current device is mobile."""
    return MobileDetection.is_mobile()

def get_responsive_config() -> dict:
    """Get responsive configuration for current device."""
    is_mobile = is_mobile_device()
    return ResponsiveLayout.get_column_config(is_mobile)

def get_spacing_config() -> dict:
    """Get spacing configuration for current device."""
    is_mobile = is_mobile_device()
    return ResponsiveLayout.get_spacing_config(is_mobile)

def optimize_for_mobile(text: str, max_length: int = 20) -> str:
    """Optimize text for mobile display."""
    return MobileOptimization.optimize_text_for_mobile(text, max_length)
