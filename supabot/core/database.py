"""
Database Manager for SupaBot BI Dashboard
Centralizes all database connection and query execution functionality.
Now optimized to use Polars for data handling and a psycopg2 connection pool.
"""

import time
import tracemalloc
from typing import Optional, Dict, Any, List, Tuple

import polars as pl
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import streamlit as st
import pandas as pd  # Returned to UI for compatibility


class DatabaseManager:
    """Centralized database connection and query execution manager."""
    
    def __init__(self):
        self._pool: Optional[SimpleConnectionPool] = None
    
    def _init_pool(self) -> Optional[SimpleConnectionPool]:
        """Initialize a global connection pool if not already created."""
        if self._pool is not None:
            return self._pool
        try:
            if "postgres" in st.secrets:
                db_conf = st.secrets["postgres"]
            else:
                db_conf = {
                    "host": st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                    "database": st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                    "user": st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                    "password": st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                    "port": st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432")),
                }
            min_conn = int(st.secrets.get("DB_MIN_POOL", 1))
            max_conn = int(st.secrets.get("DB_MAX_POOL", 8))
            self._pool = SimpleConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=db_conf["host"],
                database=db_conf["database"],
                user=db_conf["user"],
                password=db_conf["password"],
                port=db_conf["port"],
                application_name="supabot"
            )
            # Track pool status
            st.session_state.setdefault("perf", {})
            st.session_state["perf"]["db_pool"] = {
                "min": min_conn,
                "max": max_conn,
            }
            return self._pool
        except KeyError as e:
            st.error(f"Missing database credential: {e}")
            st.info("Please add your database credentials to .streamlit/secrets.toml")
            return None
        except Exception as e:
            st.error(f"Database pool init failed: {e}")
            return None

    def create_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get a connection from the pool."""
        pool = self._init_pool()
        if pool is None:
            return None
        try:
            return pool.getconn()
        except Exception as e:
            st.error(f"Failed to get DB connection from pool: {e}")
            return None

    def release_connection(self, conn: Optional[psycopg2.extensions.connection]):
        """Release a connection back to the pool."""
        try:
            if conn and self._pool:
                self._pool.putconn(conn)
            elif conn:
                conn.close()
        except Exception:
            # Best-effort release
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
    
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
            _self.release_connection(conn)
    
    def execute_query_for_assistant(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute query for AI assistant using Polars. Returns pandas for UI compatibility."""
        conn = self.create_connection()
        if not conn:
            return None
        start = time.perf_counter()
        tracemalloc.start()
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = '30s'")
            # Prefer polars.read_database when possible
            try:
                pl_df = pl.read_database(query=sql, connection=conn)
            except Exception:
                # Fallback: execute via cursor and build Polars DataFrame
                cur.execute(sql)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                pl_df = pl.DataFrame(rows, schema=columns)

            # Format datetime columns
            for name, dtype in zip(pl_df.columns, pl_df.dtypes):
                if dtype in (pl.Datetime,):
                    pl_df = pl_df.with_columns(pl.col(name).dt.strftime("%Y-%m-%d %H:%M").alias(name))

            return pl_df.to_pandas()
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
            current, peak = tracemalloc.get_traced_memory()
            duration = (time.perf_counter() - start) * 1000
            tracemalloc.stop()
            st.session_state.setdefault("perf", {})
            st.session_state["perf"]["assistant_query_ms"] = duration
            st.session_state["perf"]["assistant_mem_peak_kb"] = int(peak / 1024)
            self.release_connection(conn)
    
    def execute_query_for_dashboard(self, sql: str, params: Optional[List] = None) -> Optional[pd.DataFrame]:
        """Execute query for dashboard with Polars. Returns pandas for UI compatibility."""
        conn = self.create_connection()
        if not conn:
            return None
        start = time.perf_counter()
        tracemalloc.start()
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = '30s'")
            # Polars does not support DB-API params directly in read_database reliably; use cursor when params
            if params:
                cur.execute(sql, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                pl_df = pl.DataFrame(rows, schema=columns)
            else:
                pl_df = pl.read_database(query=sql, connection=conn)
            return pl_df.to_pandas()
        except Exception as e:
            # Silently handle errors for dashboard queries to avoid breaking the UI
            print(f"Dashboard query error: {e}")
            return pd.DataFrame()
        finally:
            current, peak = tracemalloc.get_traced_memory()
            duration = (time.perf_counter() - start) * 1000
            tracemalloc.stop()
            st.session_state.setdefault("perf", {})
            st.session_state["perf"]["dashboard_query_ms"] = duration
            st.session_state["perf"]["dashboard_mem_peak_kb"] = int(peak / 1024)
            self.release_connection(conn)
    
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

