"""
Enhanced configuration management for SupaBot BI Dashboard.
Provides type-safe configuration with validation.
"""

import streamlit as st
from typing import Optional, Dict, Any
from dataclasses import dataclass
from supabot.core.validators import ConfigValidator


@dataclass
class DatabaseConfig:
    """Database configuration with validation."""
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    min_pool: int = 5
    max_pool: int = 20
    
    @classmethod
    def from_secrets(cls) -> Optional['DatabaseConfig']:
        """Create database config from Streamlit secrets with validation."""
        try:
            if "postgres" in st.secrets:
                # Format 1: [postgres] section
                config_dict = {
                    "host": st.secrets["postgres"]["host"],
                    "database": st.secrets["postgres"]["database"],
                    "user": st.secrets["postgres"]["user"],
                    "password": st.secrets["postgres"]["password"],
                    "port": int(st.secrets["postgres"].get("port", 5432))
                }
            else:
                # Format 2: Individual keys (fallback)
                config_dict = {
                    "host": st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                    "database": st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                    "user": st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                    "password": st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                    "port": int(st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432")))
                }
            
            # Add pool settings
            config_dict["min_pool"] = int(st.secrets.get("DB_MIN_POOL", 5))
            config_dict["max_pool"] = int(st.secrets.get("DB_MAX_POOL", 20))
            
            # Validate configuration
            is_valid, error_msg = ConfigValidator.validate_database_config(config_dict)
            if not is_valid:
                st.error(f"Invalid database configuration: {error_msg}")
                return None
            
            return cls(**config_dict)
            
        except (KeyError, ValueError) as e:
            st.error(f"Database configuration error: {e}")
            st.info("Please check your database credentials in .streamlit/secrets.toml")
            return None


@dataclass 
class AIConfig:
    """AI service configuration."""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    @classmethod
    def from_secrets(cls) -> 'AIConfig':
        """Create AI config from Streamlit secrets."""
        anthropic_key = None
        openai_key = None
        
        # Get Anthropic key
        try:
            if "anthropic" in st.secrets:
                anthropic_key = st.secrets["anthropic"]["api_key"]
            else:
                anthropic_key = st.secrets.get("ANTHROPIC_API_KEY")
            
            # Validate Anthropic key if present
            if anthropic_key:
                is_valid, error_msg = ConfigValidator.validate_api_key(anthropic_key, "Anthropic")
                if not is_valid:
                    st.warning(f"Anthropic API key validation failed: {error_msg}")
                    anthropic_key = None
        except KeyError:
            pass
        
        # Get OpenAI key
        try:
            if "openai" in st.secrets:
                openai_key = st.secrets["openai"]["api_key"]
            else:
                openai_key = st.secrets.get("OPENAI_API_KEY", st.secrets.get("openai_api_key"))
            
            # Validate OpenAI key if present
            if openai_key:
                is_valid, error_msg = ConfigValidator.validate_api_key(openai_key, "OpenAI")
                if not is_valid:
                    st.warning(f"OpenAI API key validation failed: {error_msg}")
                    openai_key = None
        except KeyError:
            pass
        
        return cls(anthropic_api_key=anthropic_key, openai_api_key=openai_key)


@dataclass
class AppConfig:
    """Main application configuration."""
    page_title: str = "SupaBot BI Dashboard"
    page_icon: str = "🤖"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Cache settings
    default_cache_ttl: int = 1800  # 30 minutes
    metrics_cache_ttl: int = 1800  # 30 minutes  
    reference_cache_ttl: int = 3600  # 1 hour
    
    # Query settings
    query_timeout: str = "30s"
    max_query_results: int = 10000
    
    # UI settings
    time_filters: list = None
    default_time_filter: str = "7D"
    
    def __post_init__(self):
        if self.time_filters is None:
            self.time_filters = ["1D", "7D", "1M", "6M", "1Y", "Custom"]


class EnhancedAppSettings:
    """Enhanced application settings manager with configuration objects."""
    
    def __init__(self):
        self._db_config = None
        self._ai_config = None
        self._app_config = AppConfig()
    
    @property
    def database_config(self) -> Optional[DatabaseConfig]:
        """Get database configuration."""
        if self._db_config is None:
            self._db_config = DatabaseConfig.from_secrets()
        return self._db_config
    
    @property
    def ai_config(self) -> AIConfig:
        """Get AI configuration."""
        if self._ai_config is None:
            self._ai_config = AIConfig.from_secrets()
        return self._ai_config
    
    @property
    def app_config(self) -> AppConfig:
        """Get application configuration."""
        return self._app_config
    
    def get_database_config(self) -> Optional[Dict[str, str]]:
        """Legacy compatibility method."""
        config = self.database_config
        if config:
            return {
                "host": config.host,
                "database": config.database,
                "user": config.user,
                "password": config.password,
                "port": str(config.port)
            }
        return None
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Legacy compatibility method."""
        return self.ai_config.anthropic_api_key
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.ai_config.openai_api_key
    
    def configure_streamlit(self):
        """Configure Streamlit page settings."""
        config = self.app_config
        st.set_page_config(
            page_title=config.page_title,
            page_icon=config.page_icon,
            layout=config.layout,
            initial_sidebar_state=config.initial_sidebar_state
        )
    
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state with default values."""
        defaults = {
            "current_page": "Dashboard",
            "chat_history": [],
            "time_filter": "7D",
            "selected_stores": [],
            "debug_mode": False,
            "performance_metrics": {},
            "user_actions": []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


# Global enhanced settings instance
enhanced_settings = EnhancedAppSettings()

# Backward compatibility - keep original settings for existing code
class LegacyAppSettings:
    """Legacy settings class for backward compatibility."""
    
    # Page configuration
    PAGE_CONFIG = {
        "page_title": "SupaBot BI Dashboard",
        "page_icon": "🤖",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Cache TTL settings (in seconds) - improved values
    CACHE_TTL = {
        "database_schema": 3600,  # 1 hour
        "latest_metrics": 1800,   # 30 minutes (was 60 seconds)
        "default": 1800          # 30 minutes (was 5 minutes)
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
    TIME_FILTERS = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
    DEFAULT_TIME_FILTER = "7D"
    
    # AI Training file
    TRAINING_FILE = "supabot_training.json"
    
    @staticmethod
    def get_database_config() -> Optional[Dict[str, str]]:
        """Legacy method using enhanced settings."""
        return enhanced_settings.get_database_config()
    
    @staticmethod
    def get_anthropic_api_key() -> Optional[str]:
        """Legacy method using enhanced settings."""
        return enhanced_settings.get_anthropic_api_key()
    
    @staticmethod
    def initialize_session_state():
        """Legacy method using enhanced settings."""
        enhanced_settings.initialize_session_state()
    
    @staticmethod
    def configure_streamlit():
        """Legacy method using enhanced settings."""
        enhanced_settings.configure_streamlit()


# Global settings instance for backward compatibility
settings = LegacyAppSettings()
