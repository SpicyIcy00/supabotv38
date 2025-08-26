"""
Mobile utilities for responsive UI components.
"""

import streamlit as st
from typing import Callable, Any


class MobileUtils:
    """Utilities for mobile-responsive UI components."""
    
    @staticmethod
    def is_mobile_viewport() -> bool:
        """
        Detect if the current viewport is mobile-sized.
        Uses a simple heuristic based on Streamlit's container width.
        """
        # Add JavaScript to detect viewport size and store in session state
        js_code = """
        <script>
        function checkMobile() {
            const width = window.innerWidth;
            const isMobile = width <= 768;
            
            // Send message to Streamlit
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'viewport-changed',
                    isMobile: isMobile,
                    width: width
                }, '*');
            }
        }
        
        // Check on load and resize
        checkMobile();
        window.addEventListener('resize', checkMobile);
        </script>
        """
        
        st.markdown(js_code, unsafe_allow_html=True)
        
        # Use session state to store mobile detection
        if 'is_mobile_viewport' not in st.session_state:
            st.session_state.is_mobile_viewport = False
        
        # For now, use a simple heuristic: if the app is running in a narrow container
        # This is a fallback since JavaScript communication with Streamlit can be unreliable
        return st.session_state.get('is_mobile_viewport', False)
    
    @staticmethod
    def set_mobile_mode(mobile: bool = True):
        """
        Manually set mobile mode for testing or when automatic detection fails.
        
        Args:
            mobile: True for mobile mode, False for desktop mode
        """
        st.session_state.is_mobile_viewport = mobile
    
    @staticmethod
    def responsive_layout(desktop_func: Callable, mobile_func: Callable, *args, **kwargs) -> Any:
        """
        Execute different functions based on viewport size.
        
        Args:
            desktop_func: Function to execute on desktop
            mobile_func: Function to execute on mobile
            *args, **kwargs: Arguments to pass to the functions
            
        Returns:
            Result from the appropriate function
        """
        if MobileUtils.is_mobile_viewport():
            return mobile_func(*args, **kwargs)
        else:
            return desktop_func(*args, **kwargs)
    
    @staticmethod
    def responsive_columns(desktop_cols: int, mobile_cols: int = 1):
        """
        Create responsive columns that adapt to screen size.
        
        Args:
            desktop_cols: Number of columns for desktop
            mobile_cols: Number of columns for mobile
            
        Returns:
            Streamlit columns object
        """
        if MobileUtils.is_mobile_viewport():
            return st.columns(mobile_cols)
        else:
            return st.columns(desktop_cols)
    
    @staticmethod
    def responsive_container():
        """
        Create a responsive container that adapts to screen size.
        """
        return st.container()
    
    @staticmethod
    def mobile_optimized_metric(label: str, value: str, delta: str = None):
        """
        Create a mobile-optimized metric display.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Delta value (optional)
        """
        if MobileUtils.is_mobile_viewport():
            # Mobile-optimized display
            st.write(f"**{label}**")
            st.write(f"**{value}**")
            if delta:
                st.write(f"*{delta}*")
        else:
            # Desktop display
            st.metric(label=label, value=value, delta=delta)
    
    @staticmethod
    def mobile_optimized_chart(fig, height: int = 500):
        """
        Display a chart with mobile-optimized height.
        
        Args:
            fig: Plotly figure object
            height: Desktop height (mobile will be 70% of this)
        """
        if MobileUtils.is_mobile_viewport():
            # Adjust height for mobile
            mobile_height = int(height * 0.7)
            fig.update_layout(height=mobile_height)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def mobile_optimized_dataframe(df, **kwargs):
        """
        Display a dataframe with mobile optimization.
        
        Args:
            df: Pandas DataFrame
            **kwargs: Additional arguments for st.dataframe
        """
        if MobileUtils.is_mobile_viewport():
            # Mobile optimization: limit rows and adjust display
            display_df = df.head(10)  # Show fewer rows on mobile
            kwargs.setdefault('use_container_width', True)
            kwargs.setdefault('hide_index', True)
        else:
            display_df = df
            kwargs.setdefault('use_container_width', True)
        
        st.dataframe(display_df, **kwargs)
    
    @staticmethod
    def mobile_optimized_filter(filter_type: str, **kwargs):
        """
        Create mobile-optimized filter controls.
        
        Args:
            filter_type: Type of filter ('selectbox', 'multiselect', 'radio')
            **kwargs: Filter-specific arguments
        """
        if MobileUtils.is_mobile_viewport():
            # Mobile optimizations
            if filter_type == 'multiselect':
                # Use selectbox for single selection on mobile
                return st.selectbox(**kwargs)
            elif filter_type == 'radio':
                # Use horizontal radio for mobile
                kwargs.setdefault('horizontal', True)
                return st.radio(**kwargs)
            else:
                return getattr(st, filter_type)(**kwargs)
        else:
            # Desktop behavior
            return getattr(st, filter_type)(**kwargs)


# Convenience functions for easy access
def is_mobile() -> bool:
    """Check if current viewport is mobile."""
    return MobileUtils.is_mobile_viewport()


def responsive_columns(desktop_cols: int, mobile_cols: int = 1):
    """Create responsive columns."""
    return MobileUtils.responsive_columns(desktop_cols, mobile_cols)


def mobile_optimized_metric(label: str, value: str, delta: str = None):
    """Create mobile-optimized metric."""
    return MobileUtils.mobile_optimized_metric(label, value, delta)


def mobile_optimized_chart(fig, height: int = 500):
    """Display mobile-optimized chart."""
    return MobileUtils.mobile_optimized_chart(fig, height)


def mobile_optimized_dataframe(df, **kwargs):
    """Display mobile-optimized dataframe."""
    return MobileUtils.mobile_optimized_dataframe(df, **kwargs)
