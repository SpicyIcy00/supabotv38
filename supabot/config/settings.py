"""
Configuration settings for SupaBot BI Dashboard
Centralizes all configuration, secrets, and app settings.
"""

import streamlit as st
from typing import Optional, Dict, Any


class AppSettings:
    """Application configuration and settings manager."""
    
    # Page configuration
    PAGE_CONFIG = {
        "page_title": "SupaBot BI Dashboard",
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Cache TTL settings (in seconds)
    CACHE_TTL = {
        "database_schema": 3600,  # 1 hour
        "latest_metrics": 300,    # 5 minutes
        "default": 300           # 5 minutes default
    }
    
    # Query timeout settings
    QUERY_TIMEOUT = "30s"
    
    # Application pages
    PAGES = [
        "Dashboard",
        "Product Sales Report", 
        "Chart View",
        "AI Assistant",

        "Settings"
    ]
    
    # Default time filters
    TIME_FILTERS = ["1D", "7D", "30D", "90D", "1Y"]
    DEFAULT_TIME_FILTER = "7D"
    
    # AI Training file
    TRAINING_FILE = "supabot_training.json"
    
    @staticmethod
    def get_database_config() -> Optional[Dict[str, str]]:
        """Get database connection configuration from secrets."""
        try:
            if "postgres" in st.secrets:
                # Format 1: [postgres] section
                return {
                    "host": st.secrets["postgres"]["host"],
                    "database": st.secrets["postgres"]["database"],
                    "user": st.secrets["postgres"]["user"],
                    "password": st.secrets["postgres"]["password"],
                    "port": st.secrets["postgres"]["port"]
                }
            else:
                # Format 2: Individual keys (fallback)
                return {
                    "host": st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                    "database": st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                    "user": st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                    "password": st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                    "port": st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432"))
                }
        except KeyError as e:
            st.error(f"Missing database credential: {e}")
            st.info("Please add your database credentials to .streamlit/secrets.toml")
            return None
    
    @staticmethod
    def get_anthropic_api_key() -> Optional[str]:
        """Get Anthropic API key from secrets."""
        try:
            if "anthropic" in st.secrets:
                return st.secrets["anthropic"]["api_key"]
            else:
                return st.secrets.get("ANTHROPIC_API_KEY")
        except KeyError:
            st.error("Missing Anthropic API key")
            st.info("Please add your Anthropic API key to .streamlit/secrets.toml")
            return None
    
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state with default values."""
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "time_filter" not in st.session_state:
            st.session_state.time_filter = AppSettings.DEFAULT_TIME_FILTER
        
        if "selected_stores" not in st.session_state:
            st.session_state.selected_stores = []
    
    @staticmethod
    def configure_streamlit():
        """Configure Streamlit page settings."""
        st.set_page_config(**AppSettings.PAGE_CONFIG)


# Global settings instance
settings = AppSettings()

