"""
Responsive wrapper component for detecting screen size and rendering appropriate layouts.
"""

import streamlit as st
from typing import Callable, Any, Dict, Optional
import streamlit.components.v1 as components


class ResponsiveWrapper:
    """Responsive wrapper that adapts layout based on screen size."""
    
    @staticmethod
    def get_screen_size() -> str:
        """
        Detect screen size using JavaScript and return size category.
        Returns: 'mobile', 'tablet', or 'desktop'
        """
        # Simple JavaScript to detect screen width
        js_code = """
        <script>
        function detectScreenSize() {
            const width = window.innerWidth;
            let size = 'desktop';
            
            if (width < 768) {
                size = 'mobile';
            } else if (width < 1024) {
                size = 'tablet';
            }
            
            // Store in session storage for Streamlit to access
            sessionStorage.setItem('screen_size', size);
            
            // Also try to communicate with Streamlit
            try {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: size
                }, '*');
            } catch (e) {
                console.log('Could not communicate with Streamlit:', e);
            }
        }
        
        // Run on load and resize
        detectScreenSize();
        window.addEventListener('resize', detectScreenSize);
        </script>
        """
        
        # Create a hidden component to run JavaScript
        components.html(js_code, height=0)
        
        # Use session state with fallback
        if 'screen_size' not in st.session_state:
            st.session_state.screen_size = 'desktop'  # Default fallback
        
        return st.session_state.screen_size
    
    @staticmethod
    def responsive_layout(
        mobile_layout: Callable,
        tablet_layout: Optional[Callable] = None,
        desktop_layout: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Render responsive layout based on screen size.
        
        Args:
            mobile_layout: Function to render mobile layout
            tablet_layout: Function to render tablet layout (optional)
            desktop_layout: Function to render desktop layout (optional)
            **kwargs: Arguments to pass to layout functions
        """
        screen_size = ResponsiveWrapper.get_screen_size()
        
        if screen_size == 'mobile':
            return mobile_layout(**kwargs)
        elif screen_size == 'tablet' and tablet_layout:
            return tablet_layout(**kwargs)
        elif screen_size == 'desktop' and desktop_layout:
            return desktop_layout(**kwargs)
        else:
            # Fallback to mobile layout if tablet/desktop not provided
            return mobile_layout(**kwargs)
    
    @staticmethod
    def responsive_columns(
        mobile_cols: int = 1,
        tablet_cols: int = 2,
        desktop_cols: int = 4,
        gap: str = "small"
    ) -> list:
        """
        Create responsive column layout.
        
        Args:
            mobile_cols: Number of columns for mobile
            tablet_cols: Number of columns for tablet
            desktop_cols: Number of columns for desktop
            gap: Gap between columns ('small', 'medium', 'large')
        
        Returns:
            List of column objects
        """
        screen_size = ResponsiveWrapper.get_screen_size()
        
        if screen_size == 'mobile':
            cols = mobile_cols
        elif screen_size == 'tablet':
            cols = tablet_cols
        else:
            cols = desktop_cols
        
        return st.columns(cols, gap=gap)
    
    @staticmethod
    def responsive_container(
        mobile_class: str = "",
        tablet_class: str = "",
        desktop_class: str = "",
        **kwargs
    ):
        """
        Create responsive container with appropriate CSS classes.
        
        Args:
            mobile_class: CSS class for mobile
            tablet_class: CSS class for tablet
            desktop_class: CSS class for desktop
            **kwargs: Additional container arguments
        """
        screen_size = ResponsiveWrapper.get_screen_size()
        
        if screen_size == 'mobile':
            container_class = mobile_class
        elif screen_size == 'tablet':
            container_class = tablet_class
        else:
            container_class = desktop_class
        
        # Add responsive CSS classes
        responsive_css = f"""
        <style>
        .responsive-container {{
            {container_class}
        }}
        </style>
        """
        st.markdown(responsive_css, unsafe_allow_html=True)
        
        return st.container(**kwargs)
    
    @staticmethod
    def is_mobile() -> bool:
        """Check if current screen size is mobile."""
        return ResponsiveWrapper.get_screen_size() == 'mobile'
    
    @staticmethod
    def is_tablet() -> bool:
        """Check if current screen size is tablet."""
        return ResponsiveWrapper.get_screen_size() == 'tablet'
    
    @staticmethod
    def is_desktop() -> bool:
        """Check if current screen size is desktop."""
        return ResponsiveWrapper.get_screen_size() == 'desktop'
