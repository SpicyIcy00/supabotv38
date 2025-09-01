"""
Configuration settings for SupaBot BI Dashboard
Centralizes all configuration, secrets, and app settings.
This module maintains backward compatibility while leveraging enhanced settings.
"""

import streamlit as st
from typing import Optional, Dict, Any

# Import enhanced settings for new functionality
from .enhanced_settings import enhanced_settings, EnhancedAppSettings


class AppSettings:
    """Application configuration and settings manager with enhanced backend."""
    
    def __init__(self):
        self._enhanced = enhanced_settings
    
    # Page configuration - use enhanced settings
    @property
    def PAGE_CONFIG(self):
        config = self._enhanced.app_config
        return {
            "page_title": config.page_title,
            "page_icon": config.page_icon,
            "layout": config.layout,
            "initial_sidebar_state": config.initial_sidebar_state
        }
    
    # Cache TTL settings (in seconds) - improved values
    CACHE_TTL = {
        "database_schema": 3600,  # 1 hour
        "latest_metrics": 1800,   # 30 minutes (improved from 60 seconds)
        "default": 1800          # 30 minutes (improved from 5 minutes)
    }
    
    # Query timeout settings
    QUERY_TIMEOUT = "30s"
    
    # Application pages
    PAGES = [
        "Dashboard",
        "Product Sales Report", 
        "Chart View",
        "Advanced Analytics",
        "AI Assistant",
        "Settings"
    ]
    
    # Default time filters
    TIME_FILTERS = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
    DEFAULT_TIME_FILTER = "7D"
    
    # AI Training file
    TRAINING_FILE = "supabot_training.json"
    
    def get_database_config(self) -> Optional[Dict[str, str]]:
        """Get database connection configuration from secrets with validation."""
        return self._enhanced.get_database_config()
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key from secrets with validation."""
        return self._enhanced.get_anthropic_api_key()
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from secrets with validation."""
        return self._enhanced.get_openai_api_key()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state with default values."""
        self._enhanced.initialize_session_state()
    
    def configure_streamlit(self):
        """Configure Streamlit page settings."""
        self._enhanced.configure_streamlit()


# Global settings instance
settings = AppSettings()

