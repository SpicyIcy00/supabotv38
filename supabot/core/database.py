"""
Database Manager for SupaBot BI Dashboard
Centralizes all database connection and query execution functionality.
"""

import pandas as pd
import psycopg2
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple


class DatabaseManager:
    """Centralized database connection and query execution manager."""
    
    def __init__(self):
        self._connection = None
    
    def create_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Create and return a database connection."""
        try:
            # Try multiple possible secret configurations
            if "postgres" in st.secrets:
                # Format 1: [postgres] section
                return psycopg2.connect(
                    host=st.secrets["postgres"]["host"],
                    database=st.secrets["postgres"]["database"],
                    user=st.secrets["postgres"]["user"],
                    password=st.secrets["postgres"]["password"],
                    port=st.secrets["postgres"]["port"]
                )
            else:
                # Format 2: Individual keys (fallback)
                return psycopg2.connect(
                    host=st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                    database=st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                    user=st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                    password=st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                    port=st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432"))
                )
        except KeyError as e:
            st.error(f"Missing database credential: {e}")
            st.info("Please add your database credentials to .streamlit/secrets.toml")
            return None
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    @st.cache_data(ttl=3600)
    def get_database_schema(_self) -> Optional[Dict[str, Dict]]:
        """Fetch the complete database schema including sample data."""
        conn = _self.create_connection()
        if not conn:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
            tables = cursor.fetchall()
            schema_info = {}
            for (table_name,) in tables:
                cursor.execute(f"SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name = '{table_name}' ORDER BY ordinal_position")
                columns = cursor.fetchall()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                schema_info[table_name] = {
                    'columns': columns, 
                    'row_count': row_count, 
                    'sample_data': sample_data
                }
            return schema_info
        except Exception as e:
            st.error(f"Schema fetch failed: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def execute_query_for_assistant(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute query for AI assistant with proper error handling and formatting."""
        conn = self.create_connection()
        if not conn: 
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SET statement_timeout = '30s'")
            df = pd.read_sql(sql, conn)
            # Format datetime columns properly
            for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
            return df
        except psycopg2.errors.QueryCanceled:
            st.error("Query took too long to execute. Try a simpler question.")
            return None
        except Exception as e:
            error_msg = str(e)
            st.error(f"Query execution failed: {error_msg}")
            if "does not exist" in error_msg: 
                st.info("💡 The query references a table or column that doesn't exist.")
            elif "syntax error" in error_msg: 
                st.info("💡 There's a syntax error in the SQL.")
            return None
        finally:
            if conn: 
                conn.close()
    
    def execute_query_for_dashboard(self, sql: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Execute query for dashboard with silent error handling."""
        conn = self.create_connection()
        if not conn: 
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SET statement_timeout = '30s'")
            cursor.close()
            # Use pd.read_sql with parameters to prevent SQL injection
            df = pd.read_sql(sql, conn, params=params)
            return df
        except Exception as e:
            # Silently handle errors for dashboard queries to avoid breaking the UI
            print(f"Dashboard query error: {e}")
            return pd.DataFrame()  # Return empty dataframe on error
        finally:
            if conn: 
                conn.close()
    
    def get_column_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dynamic formatting for dataframes."""
        config = {}
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['revenue', 'profit', 'price', 'cost', 'total', 'amount', 'value', 'sales']):
                config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="₱%d")
            elif any(keyword in col_lower for keyword in ['quantity', 'count', 'sold', 'items', 'transactions']):
                 config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="%,d")
            else:
                config[col] = st.column_config.TextColumn(label=col.replace("_", " ").title())
        return config


# Global singleton instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Legacy function compatibility - these maintain the original function signatures
def create_db_connection():
    """Legacy compatibility function."""
    return get_db_manager().create_connection()

def get_database_schema():
    """Legacy compatibility function."""
    return get_db_manager().get_database_schema()

def execute_query_for_assistant(sql: str):
    """Legacy compatibility function."""
    return get_db_manager().execute_query_for_assistant(sql)

def execute_query_for_dashboard(sql: str, params=None):
    """Legacy compatibility function."""
    return get_db_manager().execute_query_for_dashboard(sql, params)

def get_column_config(df: pd.DataFrame):
    """Legacy compatibility function."""
    return get_db_manager().get_column_config(df)

