"""
Input validation utilities for SupaBot BI Dashboard.
Ensures data safety and prevents injection attacks.
"""

import re
from typing import List, Optional, Any, Dict
import pandas as pd


class QueryValidator:
    """SQL query validation and safety checks."""
    
    # Dangerous SQL keywords that should not be in user queries
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
        'TRUNCATE', 'REPLACE', 'MERGE', 'EXEC', 'EXECUTE'
    ]
    
    @classmethod
    def validate_sql_safety(cls, sql: str) -> tuple[bool, str]:
        """
        Validate that SQL query is safe for read-only operations.
        
        Returns:
            tuple: (is_safe, error_message)
        """
        if not sql or not isinstance(sql, str):
            return False, "SQL query cannot be empty"
        
        sql_upper = sql.upper()
        
        # Check for dangerous keywords
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Dangerous keyword '{keyword}' not allowed"
        
        # Must contain SELECT
        if 'SELECT' not in sql_upper:
            return False, "Query must contain SELECT statement"
        
        # Basic injection pattern checks
        injection_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT)',
            r'--\s*[^\r\n]*',  # SQL comments
            r'/\*.*?\*/',       # Multi-line comments
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE | re.DOTALL):
                return False, "Potentially unsafe SQL pattern detected"
        
        return True, ""
    
    @classmethod
    def sanitize_sql_params(cls, params: List[Any]) -> List[Any]:
        """Sanitize SQL parameters to prevent injection."""
        if not params:
            return []
        
        sanitized = []
        for param in params:
            if isinstance(param, str):
                # Remove potentially dangerous characters
                sanitized_str = re.sub(r'[;\'"\\]', '', param)
                sanitized.append(sanitized_str)
            elif isinstance(param, (int, float)):
                sanitized.append(param)
            elif isinstance(param, list):
                # For array parameters like store IDs
                sanitized_list = [p for p in param if isinstance(p, (int, float))]
                sanitized.append(sanitized_list)
            else:
                sanitized.append(str(param))
        
        return sanitized


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_store_ids(store_ids: Optional[List[int]]) -> Optional[List[int]]:
        """Validate and sanitize store IDs."""
        if not store_ids:
            return None
        
        if not isinstance(store_ids, list):
            return None
        
        validated_ids = []
        for store_id in store_ids:
            if isinstance(store_id, int) and store_id > 0:
                validated_ids.append(store_id)
            elif isinstance(store_id, str) and store_id.isdigit():
                validated_ids.append(int(store_id))
        
        return validated_ids if validated_ids else None
    
    @staticmethod
    def validate_time_filter(time_filter: str) -> str:
        """Validate time filter parameter."""
        valid_filters = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
        return time_filter if time_filter in valid_filters else "7D"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> tuple[bool, str]:
        """Validate DataFrame structure and content."""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
        
        return True, ""
    
    @staticmethod
    def validate_date_range(start_date, end_date) -> tuple[bool, str]:
        """Validate date range parameters."""
        try:
            if start_date and end_date:
                if start_date > end_date:
                    return False, "Start date must be before end date"
                
                # Check for reasonable date range (not more than 2 years)
                from datetime import timedelta
                if (end_date - start_date) > timedelta(days=730):
                    return False, "Date range cannot exceed 2 years"
            
            return True, ""
        except Exception as e:
            return False, f"Invalid date format: {e}"


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_database_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate database configuration."""
        required_keys = ['host', 'database', 'user', 'password']
        
        for key in required_keys:
            if key not in config or not config[key]:
                return False, f"Missing required database config: {key}"
        
        # Validate port
        port = config.get('port', 5432)
        if not isinstance(port, int) or port < 1 or port > 65535:
            return False, f"Invalid port number: {port}"
        
        return True, ""
    
    @staticmethod
    def validate_api_key(api_key: Optional[str], service_name: str) -> tuple[bool, str]:
        """Validate API key format."""
        if not api_key:
            return False, f"Missing {service_name} API key"
        
        if not isinstance(api_key, str):
            return False, f"Invalid {service_name} API key format"
        
        if len(api_key) < 10:
            return False, f"{service_name} API key too short"
        
        return True, ""


# Utility functions for backward compatibility
def validate_store_ids(store_ids: Optional[List[int]]) -> Optional[List[int]]:
    """Backward compatibility function."""
    return DataValidator.validate_store_ids(store_ids)

def validate_sql_safety(sql: str) -> tuple[bool, str]:
    """Backward compatibility function."""
    return QueryValidator.validate_sql_safety(sql)
