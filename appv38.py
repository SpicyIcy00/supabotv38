import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta, date
import time
import anthropic
import openai
import re
import json
import os
import numpy as np
from typing import List, Dict, Optional, Any

# Import modular components
from supabot.config.settings import settings
from supabot.core.database import (
    get_db_manager, create_db_connection, get_database_schema, 
    execute_query_for_assistant, execute_query_for_dashboard, get_column_config
)
from supabot.ui.styles.css import DashboardStyles

# Configure Streamlit
settings.configure_streamlit()

# Load CSS styles
DashboardStyles.load_all_styles()

# Database Pool Monitoring Functions
def show_pool_status():
    """Display connection pool status for debugging."""
    db_manager = get_db_manager()
    status = db_manager.get_pool_status()
    
    if st.sidebar.checkbox("Show DB Pool Status"):
        st.sidebar.write("### Connection Pool Status")
        st.sidebar.json(status)
        
        if status.get("status") == "exhausted":
            st.sidebar.error("⚠️ Connection pool exhausted!")
            st.sidebar.info("Consider increasing DB_MAX_POOL in secrets.toml")
        
        if st.sidebar.button("Reset Pool"):
            if db_manager.reset_pool():
                st.sidebar.success("Pool reset successfully")
            else:
                st.sidebar.error("Failed to reset pool")

def debug_connection_issues():
    """Debug connection pool issues."""
    if st.sidebar.checkbox("Debug DB Connections"):
        db_manager = get_db_manager()
        status = db_manager.get_pool_status()
        
        st.sidebar.write("### Connection Pool Status")
        st.sidebar.json(status)
        
        if st.sidebar.button("Reset Pool"):
            if db_manager.reset_pool():
                st.sidebar.success("Pool reset successfully")
            else:
                st.sidebar.error("Failed to reset pool")

def monitor_connection_health():
    """Comprehensive connection health monitoring."""
    if st.sidebar.checkbox("🔍 Connection Health Monitor"):
        db_manager = get_db_manager()
        status = db_manager.get_pool_status()
        
        st.sidebar.write("### 🔧 Connection Pool Health")
        
        # Status indicator
        if status.get("status") == "healthy":
            st.sidebar.success("✅ Pool Healthy")
        elif status.get("status") == "exhausted":
            st.sidebar.error("❌ Pool Exhausted")
        else:
            st.sidebar.warning("⚠️ Pool Issues")
        
        # Connection details
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Used", status.get("current_connections", 0))
        with col2:
            st.metric("Available", status.get("available_connections", 0))
        
        st.sidebar.write(f"**Max Connections:** {status.get('max_connections', 0)}")
        st.sidebar.write(f"**Min Connections:** {status.get('min_connections', 0)}")
        
        # Usage percentage
        max_conn = status.get("max_connections", 1)
        used_conn = status.get("current_connections", 0)
        usage_pct = (used_conn / max_conn) * 100 if max_conn > 0 else 0
        
        st.sidebar.progress(usage_pct / 100, text=f"Usage: {usage_pct:.1f}%")
        
        # Recommendations
        if usage_pct > 80:
            st.sidebar.warning("⚠️ High connection usage. Consider increasing DB_MAX_POOL.")
        elif usage_pct > 60:
            st.sidebar.info("ℹ️ Moderate connection usage.")
        else:
            st.sidebar.success("✅ Low connection usage.")
        
        # Action buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Reset Pool", help="Reset the connection pool"):
                if db_manager.reset_pool():
                    st.success("Pool reset!")
                    st.rerun()
                else:
                    st.error("Reset failed!")
        
        with col2:
            if st.button("📊 Refresh", help="Refresh pool status"):
                st.rerun()

# Enhanced Training System from v1justsupabot.py
class EnhancedTrainingSystem:
    def __init__(self, training_file="supabot_training.json"):
        self.training_file = training_file
        self.training_data = self.load_training_data()

    def load_training_data(self) -> List[Dict]:
        """Load training examples from JSON file"""
        if os.path.exists(self.training_file):
            try:
                with open(self.training_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_training_data(self):
        """Save training examples to JSON file"""
        try:
            with open(self.training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save training data: {e}")
            return False

    def add_training_example(self, question: str, sql: str, feedback: str = "correct", explanation: str = ""):
        """Add a new training example with optional explanation"""
        example = {
            "question": question.lower().strip(),
            "sql": sql.strip(),
            "feedback": feedback,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(example)
        return self.save_training_data()

    def find_similar_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar training examples using enhanced similarity"""
        question = question.lower().strip()
        scored_examples = []
        
        business_terms = {
            'sales': ['revenue', 'income', 'earnings', 'total'],
            'hour': ['time', 'hourly', 'per hour'],
            'store': ['location', 'branch', 'shop'],
            'total': ['sum', 'aggregate', 'combined', 'all'],
            'date': ['day', 'daily', 'time period']
        }
        
        for example in self.training_data:
            if example["feedback"] in ["correct", "corrected"]:
                q1_words = set(question.split())
                q2_words = set(example["question"].split())
                
                if len(q1_words | q2_words) > 0:
                    basic_similarity = len(q1_words & q2_words) / len(q1_words | q2_words)
                    
                    business_score = 0
                    for term, synonyms in business_terms.items():
                        if any(syn in question for syn in [term] + synonyms):
                            if any(syn in example["question"] for syn in [term] + synonyms):
                                business_score += 0.3
                    
                    final_score = basic_similarity + business_score
                    scored_examples.append((final_score, example))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:limit] if score > 0.2]

    def get_training_context(self, question: str) -> str:
        """Get relevant training examples formatted as context"""
        similar_examples = self.find_similar_examples(question)
        if not similar_examples:
            return ""
        
        context = "RELEVANT TRAINING EXAMPLES:\n\n"
        for i, example in enumerate(similar_examples, 1):
            context += f"Example {i}:\n"
            context += f"Question: {example['question']}\n"
            context += f"SQL: {example['sql']}\n"
            if example.get('explanation'):
                context += f"Note: {example['explanation']}\n"
            context += "\n"
        
        return context

def get_training_system():
    """Initialize training system with default examples"""
    training_system = EnhancedTrainingSystem()
    
    if len(training_system.training_data) == 0:
        default_examples = [
            {
                "question": "sales per hour total of all stores and all dates",
                "sql": """
                WITH hourly_sales AS (
                    SELECT 
                        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
                        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
                        SUM(t.total) as total_sales
                    FROM transactions t
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND COALESCE(t.is_cancelled, false) = false
                    GROUP BY EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
                    ORDER BY hour
                )
                SELECT hour, hour_label, COALESCE(total_sales, 0) as total_sales FROM hourly_sales;
                """,
                "feedback": "correct",
                "explanation": "Groups by hour only across ALL stores and dates. Different from per-store breakdown.",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        for example in default_examples:
            training_system.training_data.append(example)
        training_system.save_training_data()
    
    return training_system

# Database functions now imported from supabot.core.database

def get_claude_client():
    api_key = settings.get_anthropic_api_key()
    if api_key:
        try:
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize Claude client: {e}")
            return None
    return None

def get_openai_client():
    """Get OpenAI client with API key from secrets"""
    try:
        if "openai" in st.secrets:
            api_key = st.secrets["openai"]["api_key"]
        else:
            api_key = st.secrets.get("OPENAI_API_KEY", st.secrets.get("openai_api_key"))
        
        if api_key:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        return None
    except Exception:
        return None

# AI Assistant Core Functions
def generate_smart_sql(question, schema_info=None, training_system=None):
    """Ultimate AI SQL generator with training system integration"""
    client = get_claude_client()
    if not client: return None
    
    schema_context = "DATABASE SCHEMA:\n\n"
    if schema_info:
        for table_name, info in schema_info.items():
            schema_context += f"TABLE: {table_name} ({info['row_count']} rows)\nColumns:\n"
            for col_name, data_type, nullable, default in info['columns']:
                nullable_str = "NULL" if nullable == 'YES' else "NOT NULL"
                schema_context += f"  - {col_name}: {data_type} {nullable_str}\n"
            schema_context += "\n"

    training_context = training_system.get_training_context(question) if training_system else ""

    prompt = f"""{schema_context}{training_context}

BUSINESS CONTEXT:
- This is a retail business database tracking sales, inventory, products, and stores.
- Valid sales transactions have: transaction_type = 'sale' AND (is_cancelled IS NULL OR is_cancelled = false).
- For product-level revenue and quantity, use SUM(transactions.total) and SUM(transaction_items.quantity).
- For transaction-level revenue (e.g., total sales, sales by store), use SUM(transactions.total) to match POS recorded totals.
- TIMEZONE: Data is in Philippines timezone (UTC+8)
- TIME FORMAT: Always format time as 12-hour format (1:00 PM, 7:00 PM, etc.)

CRITICAL AGGREGATION RULES:
1. When user asks for "total across all stores" or "total of all stores" - GROUP BY time/category ONLY, do NOT group by store
2. When user asks for "per store" or "by store" - GROUP BY both store AND time/category
3. Pay attention to the level of aggregation requested.

USER QUESTION: {question}

INSTRUCTIONS:
1. Generate a PostgreSQL query that matches the EXACT aggregation level requested.
2. Use the training examples as reference.
3. For "total across all stores and all dates by hour": GROUP BY hour only. Use SUM(transactions.total)
4. For "sales per store per hour": GROUP BY store AND hour. Use SUM(transactions.total)
5. For product-related queries (e.g., 'top products by sales'), use SUM(transactions.total) for revenue and SUM(transaction_items.quantity) for units sold.
6. Use CTEs for readability. Use COALESCE for NULLs. Include meaningful aliases.
7. For time-based queries use AT TIME ZONE 'Asia/Manila' and format time as 12-hour with AM/PM.
8. Order results descending by the main metric.

Generate ONLY the SQL query, no explanations:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        sql = response.content[0].text.strip()
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql).strip()
        if not sql.endswith(';'): sql += ';'
        return sql
    except Exception as e:
        st.error(f"AI query generation failed: {e}")
        return None

def interpret_results(question, results_df, sql_query):
    client = get_claude_client()
    if not client or results_df.empty: return "The query returned no results."
    
    results_summary = f"Query returned {len(results_df)} rows. Columns: {', '.join(results_df.columns)}\n\n"
    results_summary += "First 10 rows:\n" + results_df.head(10).to_string()
    
    prompt = f"""You are a business intelligence expert. The user asked: "{question}"

SQL Query executed:
{sql_query}

Results:
{results_summary}

Please provide a clear, concise, conversational but professional answer to the user's question, followed by key insights and actionable recommendations. Use bullet points. Interpret the data, don't just repeat it. Format monetary amounts as ₱X,XXX (no decimals)."""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception:
        return "Could not interpret results."

# ENHANCED SMART VISUALIZATION with 8+ Chart Types
def create_smart_visualization(results_df, question):
    """Enhanced visualization function that automatically selects the best chart type"""
    
    if results_df.empty:
        return None
    
    # Preprocess: coerce likely numeric/date columns from object dtype
    df = results_df.copy()
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric coercion
                coerced_num = pd.to_numeric(df[col], errors='coerce')
                if coerced_num.notna().sum() >= max(1, int(0.6 * len(df))):
                    df[col] = coerced_num
                    continue
                # Try datetime coercion based on column name hints or parsability
                if any(k in col.lower() for k in ['date', 'time', 'timestamp']):
                    coerced_dt = pd.to_datetime(df[col], errors='coerce')
                    if coerced_dt.notna().sum() >= max(1, int(0.6 * len(df))):
                        df[col] = coerced_dt
    except Exception:
        pass

    # Get column types after coercion
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetime']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    # Clean the question for analysis
    question_lower = question.lower()
    
    # Determine chart type based on question keywords and data structure
    chart_type = determine_chart_type(question_lower, df, numeric_cols, text_cols, date_cols)
    
    try:
        fig = None
        
        if chart_type == "pie":
            fig = create_pie_chart(df, question, numeric_cols, text_cols)
        elif chart_type == "treemap":
            fig = create_treemap_chart(df, question, numeric_cols, text_cols)
        elif chart_type == "scatter":
            fig = create_scatter_chart(df, question, numeric_cols, text_cols)
        elif chart_type == "line":
            fig = create_line_chart(df, question, numeric_cols, text_cols, date_cols)
        elif chart_type == "heatmap":
            fig = create_heatmap_chart(df, question, numeric_cols, text_cols)
        elif chart_type == "box":
            fig = create_box_chart(df, question, numeric_cols, text_cols)
        elif chart_type == "area":
            fig = create_area_chart(df, question, numeric_cols, text_cols, date_cols)
        else:  # Default to bar chart
            fig = create_bar_chart(df, question, numeric_cols, text_cols)
        
        # Apply consistent styling
        if fig:
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500,
                title_x=0.5
            )
            
        return fig
        
    except Exception as e:
        # Fallback to bar chart if anything fails
        return create_bar_chart(results_df, question, numeric_cols, text_cols)

def determine_chart_type(question_lower, results_df, numeric_cols, text_cols, date_cols):
    """Intelligently determine the best chart type based on question and data"""
    
    # Keywords for different chart types
    pie_keywords = ['distribution', 'breakdown', 'percentage', 'proportion', 'share', 'composition', 'part of']
    treemap_keywords = ['hierarchy', 'treemap', 'nested', 'structure', 'composition', 'size comparison']
    scatter_keywords = ['correlation', 'relationship', 'vs', 'against', 'compare', 'scatter', 'relationship between']
    line_keywords = ['trend', 'over time', 'timeline', 'progression', 'change', 'growth', 'decline']
    heatmap_keywords = ['pattern', 'heatmap', 'intensity', 'by hour and day', 'activity']
    box_keywords = ['outlier', 'distribution', 'quartile', 'median', 'range', 'variance']
    area_keywords = ['cumulative', 'stacked', 'total over time', 'accumulated']
    
    # Check for specific chart type keywords
    if any(keyword in question_lower for keyword in pie_keywords) and len(text_cols) >= 1:
        return "pie"
    
    if any(keyword in question_lower for keyword in treemap_keywords) and len(text_cols) >= 1:
        return "treemap"
    
    if any(keyword in question_lower for keyword in scatter_keywords) and len(numeric_cols) >= 2:
        return "scatter"
    
    if any(keyword in question_lower for keyword in line_keywords) and (date_cols or 'hour' in question_lower or 'day' in question_lower):
        return "line"
    
    if any(keyword in question_lower for keyword in heatmap_keywords):
        return "heatmap"
    
    if any(keyword in question_lower for keyword in box_keywords):
        return "box"
    
    if any(keyword in question_lower for keyword in area_keywords) and date_cols:
        return "area"
    
    # Data-driven decisions
    row_count = len(results_df)
    
    # For small datasets with categories, prefer pie charts for distribution questions
    if row_count <= 10 and len(text_cols) >= 1 and ('category' in question_lower or 'type' in question_lower):
        return "pie"
    
    # For datasets with multiple numeric columns, prefer scatter
    if len(numeric_cols) >= 2 and row_count >= 10:
        return "scatter"
    
    # For time-based data, prefer line charts
    if date_cols or any(col for col in results_df.columns if 'hour' in col.lower() or 'time' in col.lower()):
        return "line"
    
    # Default to bar chart
    return "bar"

def get_best_value_column(numeric_cols):
    """Select the best numeric column for values"""
    priority_terms = ['revenue', 'sales', 'total', 'amount', 'value', 'price', 'cost']
    
    for term in priority_terms:
        for col in numeric_cols:
            if term in col.lower():
                return col
    
    return numeric_cols[0]  # Fallback to first numeric column

def get_best_label_column(text_cols):
    """Select the best text column for labels"""
    priority_terms = ['name', 'category', 'type', 'store', 'product']
    
    for term in priority_terms:
        for col in text_cols:
            if term in col.lower() and 'id' not in col.lower():
                return col
    
    # Return first non-ID column
    for col in text_cols:
        if 'id' not in col.lower():
            return col
    
    return text_cols[0]  # Fallback to first text column

def create_pie_chart(results_df, question, numeric_cols, text_cols):
    """Create a pie chart for distribution/breakdown questions"""
    if not text_cols or not numeric_cols:
        return None
    
    # Select best columns
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    # Filter and prepare data
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 10:  # Limit to top 10 for readability
        df_clean = df_clean.nlargest(10, value_col)
    
    # Canonicalize labels and aggregate to avoid duplicates
    canonical_col = f"{label_col}_canonical"
    # Create canonical category label using shared helper
    if 'canonicalize_category_label' in globals():
        df_plot = df_clean.copy()
        df_plot[canonical_col] = df_plot[label_col].astype(str).apply(canonicalize_category_label)
        df_plot = df_plot.groupby(canonical_col, as_index=False)[value_col].sum()
        color_map_full = get_fixed_category_color_map()
        color_map = {cat: color_map_full.get(cat, color_map_full['n/a']) for cat in df_plot[canonical_col].unique()}
        fig = px.pie(
                    df_plot,
                    values=value_col,
                    names=canonical_col,
                    color=canonical_col,
                    title=f"Distribution: {question}",
                    color_discrete_map=color_map)
    else:
        # Fallback to original behavior if helper not loaded
        color_map = get_pie_chart_colors(df_clean, label_col)
        fig = px.pie(
                    df_clean,
                    values=value_col,
                    names=label_col,
                    color=label_col,
                    title=f"Distribution: {question}",
                    color_discrete_map=color_map)
    
    return fig

def run_benchmarks() -> pd.DataFrame:
    """Run simple timing benchmarks for key data functions.
    Returns a pandas DataFrame with function name, duration_ms, and row_count.
    """
    import time
    results = []
    cases = [
        ("get_latest_metrics", lambda: get_latest_metrics()),
        ("get_previous_metrics", lambda: get_previous_metrics()),
        ("get_daily_trend_30d", lambda: get_daily_trend(30)),
        ("get_hourly_sales_average_7d", lambda: get_hourly_sales_average("7D")),
        ("get_hourly_sales_latest_day", lambda: get_hourly_sales_latest_day()),
    ]
    for name, fn in cases:
        try:
            t0 = time.perf_counter()
            df = fn()
            dur_ms = (time.perf_counter() - t0) * 1000.0
            rows = 0
            if df is not None:
                try:
                    rows = len(df)
                except Exception:
                    rows = 0
            results.append({"function": name, "duration_ms": round(dur_ms, 2), "rows": rows})
        except Exception as e:
            results.append({"function": name, "duration_ms": None, "rows": None, "error": str(e)})
    return pd.DataFrame(results)

def create_treemap_chart(results_df, question, numeric_cols, text_cols):
    """Create a treemap for hierarchical data"""
    if not text_cols or not numeric_cols:
        return None
    
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 20:
        df_clean = df_clean.nlargest(20, value_col)
    
    fig = px.treemap(df_clean, path=[label_col], values=value_col,
                    title=f"Treemap: {question}")
    
    return fig

def create_scatter_chart(results_df, question, numeric_cols, text_cols):
    """Create a scatter plot for correlation analysis"""
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # If there's a third numeric column, use it for size
    size_col = numeric_cols[2] if len(numeric_cols) > 2 else None
    
    # If there's a text column, use it for color
    color_col = text_cols[0] if text_cols else None
    
    fig = px.scatter(results_df, x=x_col, y=y_col,
                    size=size_col, color=color_col,
                    title=f"Relationship: {question}",
                    hover_data=text_cols[:2] if text_cols else None)
    
    return fig

def create_line_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create a line chart for time series data"""
    if not numeric_cols:
        return None
    
    # Determine x-axis (time-based)
    x_col = None
    if date_cols:
        x_col = date_cols[0]
    else:
        # Look for time-related columns
        for col in results_df.columns:
            if any(time_word in col.lower() for time_word in ['hour', 'time', 'date', 'day']):
                x_col = col
                break
    
    if not x_col:
        x_col = results_df.columns[0]  # Fallback to first column
    
    y_col = get_best_value_column(numeric_cols)
    
    # Sort by x-axis for proper line connection
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.line(df_sorted, x=x_col, y=y_col,
                 title=f"Trend: {question}",
                 markers=True)
    
    return fig

def create_heatmap_chart(results_df, question, numeric_cols, text_cols):
    """Create a heatmap for pattern analysis"""
    if len(results_df.columns) < 3:
        return None
    
    # Try to create a pivot table for heatmap
    if len(text_cols) >= 2 and len(numeric_cols) >= 1:
        try:
            pivot_df = results_df.pivot_table(
                index=text_cols[0],
                columns=text_cols[1],
                values=numeric_cols[0],
                fill_value=0
            )
            
            fig = px.imshow(pivot_df,
                           title=f"Pattern Analysis: {question}",
                           aspect="auto",
                           color_continuous_scale="Blues")
            
            return fig
        except:
            pass
    
    # Fallback to correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 3:
        corr_matrix = results_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix,
                       title=f"Correlation: {question}",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        return fig
    
    return None

def create_box_chart(results_df, question, numeric_cols, text_cols):
    """Create box plots for distribution analysis"""
    if not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = text_cols[0] if text_cols else None
    
    if x_col:
        fig = px.box(results_df, x=x_col, y=y_col,
                    title=f"Distribution: {question}")
    else:
        fig = px.box(results_df, y=y_col,
                    title=f"Distribution: {question}")
    
    return fig

def create_area_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create area chart for cumulative data"""
    if not numeric_cols:
        return None
    
    x_col = date_cols[0] if date_cols else results_df.columns[0]
    y_col = get_best_value_column(numeric_cols)
    
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.area(df_sorted, x=x_col, y=y_col,
                 title=f"Cumulative: {question}")
    
    return fig

def create_bar_chart(results_df, question, numeric_cols, text_cols):
    """Create bar chart (original functionality)"""
    if not text_cols or not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = get_best_label_column(text_cols)
    
    df_filtered = results_df[results_df[y_col] > 0].copy()
    if df_filtered.empty:
        return None
        
    df_sorted = df_filtered.sort_values(by=y_col, ascending=False).head(25)
    
    # Determine orientation based on label length
    chart_type = 'h' if any(len(str(s)) > 15 for s in df_sorted[x_col]) else 'v'
    
    if chart_type == 'h':
        fig = px.bar(df_sorted, x=y_col, y=x_col, orientation='h',
                    title=f"Analysis: {question}")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        fig = px.bar(df_sorted, x=x_col, y=y_col,
                    title=f"Analysis: {question}")
    
    return fig

# Execute query for AI Assistant
# Database query functions now imported from supabot.core.database

@st.cache_data(ttl=60)
def get_latest_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        COALESCE(SUM(t.total), 0) as latest_sales,
        COUNT(DISTINCT t.ref_id) as latest_transactions
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=60)
def get_previous_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        COALESCE(SUM(t.total), 0) as previous_sales,
        COUNT(DISTINCT t.ref_id) as previous_transactions
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date - INTERVAL '1 day'
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=60)
def get_hourly_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
        COALESCE(SUM(t.total), 0) as sales
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    GROUP BY 
        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'),
        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
    ORDER BY hour
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_store_performance(time_filter: str = "7D", store_ids: Optional[List[int]] = None):
    """Top stores by total sales over the selected period using transaction-level totals.
    Uses intelligent date ranges, Manila timezone, and filters out cancelled transactions.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    params: List[Any] = [str(current_start), str(current_end)]
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT 
        s.name AS store_name,
        SUM(t.total) AS total_sales
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale'
      AND COALESCE(t.is_cancelled, false) = false
      AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
      AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
      {store_clause}
    GROUP BY s.name
    HAVING SUM(t.total) > 0
    ORDER BY total_sales DESC
    """
    return execute_query_for_dashboard(sql, params=params)

@st.cache_data(ttl=300)
def get_store_performance_with_comparison(time_filter: str = "7D", store_ids: Optional[List[int]] = None):
    """Get store performance with period-over-period comparison for percentage changes.
    Returns current period sales and previous period sales for each store.
    """
    # Get intelligent date ranges for current period
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    # Get previous period dates for comparison
    prev_dates = get_previous_period_dates(current_start, current_end, time_filter)
    prev_start = prev_dates['start_date']
    prev_end = prev_dates['end_date']
    
    # Build store clause and params
    store_clause = ""
    params: List[Any] = []
    
    # Parameters in SQL order: current_start, current_end, store_ids, prev_start, prev_end, store_ids
    params.extend([str(current_start), str(current_end)])
    
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)  # For current period
    
    params.extend([str(prev_start), str(prev_end)])
    
    if store_ids:
        params.append(store_ids)  # For previous period

    sql = f"""
    WITH current_period AS (
        SELECT 
            s.name AS store_name,
            s.id AS store_id,
            COALESCE(SUM(t.total), 0) AS current_sales
        FROM transactions t
        JOIN stores s ON t.store_id = s.id
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY s.name, s.id
    ),
    previous_period AS (
        SELECT 
            s.id AS store_id,
            COALESCE(SUM(t.total), 0) AS previous_sales
        FROM transactions t
        JOIN stores s ON t.store_id = s.id
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY s.id
    )
    SELECT 
        cp.store_name,
        cp.current_sales AS total_sales,
        COALESCE(pp.previous_sales, 0) AS previous_sales,
        CASE 
            WHEN COALESCE(pp.previous_sales, 0) = 0 THEN NULL
            ELSE ((cp.current_sales - pp.previous_sales) / pp.previous_sales) * 100.0
        END AS pct_change
    FROM current_period cp
    LEFT JOIN previous_period pp ON cp.store_id = pp.store_id
    WHERE cp.current_sales > 0
    ORDER BY cp.current_sales DESC
    """
    
    return execute_query_for_dashboard(sql, params=params)

@st.cache_data(ttl=300)
def get_daily_trend(days=30, store_ids: Optional[List[int]] = None):
    """Fetches daily sales trend, with optional store filter.
    Updated to use intelligent date ranges for better period selection.
    """
    # For backward compatibility, we'll keep the days parameter but use intelligent date ranges
    # when called from dashboard functions
    # TODO: Update this function to use intelligent date ranges when called from dashboard
    
    # Build parameters in the exact order they appear in SQL
    params = []
    store_clause = ""
    
    # First %s: INTERVAL for date_series
    params.append(f'{days} days')
    
    # Second %s: INTERVAL for daily_sales filter
    params.append(f'{days} days')
    
    # Third %s: store_ids for store_clause (if present)
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)
        
    sql = f"""
    WITH date_series AS (
        SELECT generate_series(
            (NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL %s,
            (NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL '1 day',
            INTERVAL '1 day'
        )::date as date
    ),
    daily_sales AS (
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        COALESCE(SUM(t.total), 0) as daily_sales
    FROM transactions t
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ((NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL %s)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <  (NOW() AT TIME ZONE 'Asia/Manila')::date
    {store_clause}
    GROUP BY 1
    )
    SELECT 
        ds.date,
        COALESCE(ds2.daily_sales, 0) as daily_sales
    FROM date_series ds
    LEFT JOIN daily_sales ds2 ON ds.date = ds2.date
    ORDER BY ds.date
    """
    df = execute_query_for_dashboard(sql, params=params)
    if df is not None and not df.empty:
        try:
            import polars as pl
            pl_df = pl.from_pandas(df)
            pl_df = pl_df.with_columns(
                pl.col('daily_sales').cum_sum().alias('cumulative_sales')
            )
            return pl_df.to_pandas()
        except Exception:
            # Fallback to pandas if Polars fails
            df['cumulative_sales'] = df['daily_sales'].cumsum()
            return df
    return df

@st.cache_data(ttl=300)
def get_avg_sales_per_hour(time_filter: str, store_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Average sales per hour across the selected full-day window in Asia/Manila.
    Uses intelligent date ranges and ensures hours 0..23 present; averages by distinct calendar days in the window.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
    
    sql = f"""
    WITH params AS (
      SELECT 
        %s::date AS start_date,
        %s::date AS end_date
    ), hours AS (
      SELECT generate_series(0,23) AS hour
    ), window_days AS (
      SELECT COUNT(DISTINCT DATE(t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS n_days
      FROM transactions t, params p
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= p.start_date
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= p.end_date
        {store_clause}
    ), sums AS (
      SELECT 
        EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS hour,
        SUM(COALESCE(t.total,0))::numeric AS sum_sales
      FROM transactions t, params p
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= p.start_date
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= p.end_date
        {store_clause}
      GROUP BY 1
    )
    SELECT 
      h.hour,
      TO_CHAR(make_time(h.hour,0,0),'HH12:MI AM') AS hour_label,
      (COALESCE(s.sum_sales,0) / NULLIF((SELECT n_days FROM window_days),0))::numeric AS avg_sales_per_hour
    FROM hours h
    LEFT JOIN sums s ON s.hour = h.hour
    ORDER BY h.hour;
    """
    # Build parameters in the exact order they appear in SQL
    params: List[Any] = []
    
    # Parameters in SQL order:
    # 1. start_date (line 3)
    # 2. end_date (line 4)
    # 3. store_ids for window_days (line 13) - if present
    # 4. store_ids for sums (line 26) - if present
    
    params.extend([str(current_start), str(current_end)])
    
    if store_ids:
        params.append(store_ids)  # For window_days CTE
        params.append(store_ids)  # For sums CTE
    df = execute_query_for_dashboard(sql, params=params)
    if df is None or df.empty:
        return pd.DataFrame({
            'hour': list(range(24)),
            'hour_label': [f"{((h%12) or 12)}:00 {'AM' if h<12 else 'PM'}" for h in range(24)],
            'avg_sales_per_hour': [0]*24,
        })
    return df

@st.cache_data(ttl=300)
def get_store_count():
    sql = "SELECT COUNT(DISTINCT id) as store_count FROM stores"
    result = execute_query_for_dashboard(sql)
    return result.iloc[0]['store_count'] if result is not None and len(result) > 0 else 0

@st.cache_data(ttl=300)
def get_store_list():
    """Get list of all stores with id and name columns."""
    sql = "SELECT id, name FROM stores ORDER BY name"
    result = execute_query_for_dashboard(sql)
    return result if result is not None else pd.DataFrame(columns=['id', 'name'])

@st.cache_data(ttl=300)
def get_product_performance():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        p.name as product_name,
        SUM(ti.quantity) as total_quantity_sold,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.name
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_analysis():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        t.ref_id,
        s.name as store_name,
        SUM(ti.quantity) as items_per_transaction,
        t.total as total_value,
        AVG(ti.item_total / ti.quantity) as avg_item_value
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    JOIN stores s ON t.store_id = s.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY t.ref_id, s.name, t.total
    HAVING SUM(ti.quantity) > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_daily_sales_by_store():
    sql = """
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        s.name as store_name,
        COALESCE(SUM(t.total), 0) as daily_sales
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '30 days'
    GROUP BY 1,2
    ORDER BY date DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_values_by_store():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        s.name as store_name,
        t.total as total_value
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    AND t.total > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_category_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
    )
    SELECT 
        p.category as product_category,
        SUM(ti.quantity) as total_quantity_sold,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.category
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

# --- NEW DASHBOARD DATA FUNCTIONS START ---
def get_time_filter_interval(time_filter="7d"):
    """Helper function to convert time filter string to SQL interval.
    Updated to use intelligent date ranges instead of rolling periods.
    """
    # For intelligent date ranges, we return the period type rather than SQL interval
    # The actual date logic is handled in the individual query functions
    mapping = {"1D": "1D", "7D": "7D", "1M": "1M", "6M": "6M", "1Y": "1Y"}
    return mapping.get(time_filter, "7D")

def get_intelligent_date_range(time_filter: str) -> Dict[str, Any]:
    """Get intelligent date ranges based on the selected time period.
    
    Returns:
        Dict with 'start_date' and 'end_date' in Asia/Manila timezone
        - 1D: Yesterday only
        - 7D: Current week to date (Monday to today)
        - 1M: Current month to date (1st of month to today)
        - 6M: Current month to date (1st of month to today) - for backward compatibility
        - 1Y: Current year to date (1st of year to today)
    """
    from datetime import datetime, timedelta, date
    
    # Get current date in Asia/Manila timezone
    now_mnl = datetime.now()  # Assuming server is in Asia/Manila or using proper timezone handling
    
    if time_filter == "1D":
        # Last 1 day (yesterday to today for better data availability)
        start_date = (now_mnl - timedelta(days=1)).date()
        end_date = now_mnl.date()
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': f'Last 1 day ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        }
    
    elif time_filter == "7D":
        # Current week to date (Monday to yesterday) - always use yesterday for fair comparison
        yesterday = (now_mnl - timedelta(days=1)).date()  # Always use yesterday as end point
        days_since_monday = yesterday.weekday()  # Monday = 0, Sunday = 6
        start_date = yesterday - timedelta(days=days_since_monday)  # This week's Monday
        end_date = yesterday  # Always end on yesterday
        
        # If it's Monday (yesterday was Sunday), show last 7 days instead
        if days_since_monday == 6:  # Yesterday was Sunday
            start_date = yesterday - timedelta(days=6)  # Show last 7 days instead
            description = f'Last 7 days ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        else:
            description = f'Week to date ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': description
        }
    
    elif time_filter == "1M":
        # Current month to date (1st of month to today), but if it's very early in month, extend to ensure data
        start_date = now_mnl.replace(day=1).date()
        end_date = now_mnl.date()
        
        # If it's the 1st or 2nd of the month and we might not have much data, include previous days
        if now_mnl.day <= 2:
            start_date = (now_mnl - timedelta(days=29)).date()  # Show last 30 days instead
            description = f'Last 30 days ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        else:
            description = f'Month to date ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': description
        }
    
    elif time_filter == "6M":
        # For backward compatibility, use current month to date
        start_date = now_mnl.replace(day=1).date()
        end_date = now_mnl.date()
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': f'Month to date ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        }
    
    elif time_filter == "1Y":
        # Current year to date (1st of year to today)
        start_date = now_mnl.replace(month=1, day=1).date()
        end_date = now_mnl.date()
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': f'Year to date ({start_date.strftime("%b %d, %Y")} to {end_date.strftime("%b %d, %Y")})'
        }
    
    else:
        # Default to current week to date
        days_since_monday = now_mnl.weekday()
        start_date = (now_mnl - timedelta(days=days_since_monday)).date()
        end_date = now_mnl.date()
        return {
            'start_date': start_date,
            'end_date': end_date,
            'description': f'Week to date ({start_date.strftime("%b %d")} to {end_date.strftime("%b %d")})'
        }

def test_comparison_logic():
    """Test function to verify comparison logic is working correctly"""
    from datetime import datetime, timedelta, date
    
    print("🧪 TESTING COMPARISON LOGIC")
    print("=" * 50)
    
    # Test 1D logic
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    prev_week_same_day = yesterday - timedelta(days=7)
    
    print(f"1D Test:")
    print(f"  Today: {today}")
    print(f"  Yesterday: {yesterday}")
    print(f"  Previous week same day: {prev_week_same_day}")
    print(f"  Should compare: {yesterday} vs {prev_week_same_day}")
    
    # Test 7D logic
    days_since_monday = today.weekday()
    this_monday = today - timedelta(days=days_since_monday)
    prev_monday = this_monday - timedelta(days=7)
    prev_end = today - timedelta(days=7)
    
    print(f"\n7D Test:")
    print(f"  Days since Monday: {days_since_monday}")
    print(f"  This Monday: {this_monday}")
    print(f"  Previous Monday: {prev_monday}")
    print(f"  Previous end: {prev_end}")
    print(f"  Should compare: {this_monday} to {today} vs {prev_monday} to {prev_end}")
    
    # Test 1M logic
    this_month_start = today.replace(day=1)
    if today.month == 1:
        prev_month = today.replace(year=today.year - 1, month=12)
    else:
        prev_month = today.replace(month=today.month - 1)
    prev_month_start = prev_month.replace(day=1)
    prev_month_end = prev_month.replace(day=today.day)
    
    print(f"\n1M Test:")
    print(f"  This month start: {this_month_start}")
    print(f"  Previous month: {prev_month}")
    print(f"  Previous month start: {prev_month_start}")
    print(f"  Previous month end: {prev_month_end}")
    print(f"  Should compare: {this_month_start} to {today} vs {prev_month_start} to {prev_month_end}")
    
    print("\n" + "=" * 50)

def get_previous_period_dates(current_start: date, current_end: date, time_filter: str = "7D") -> Dict[str, date]:
    """Get the previous period dates for comparison using the new period-to-date logic.
    
    Args:
        current_start: Start date of current period
        current_end: End date of current period
        time_filter: The time filter used to determine comparison logic
    
    Returns:
        Dict with 'start_date' and 'end_date' for the previous period
    """
    from datetime import datetime, timedelta, date
    import calendar
    
    # Get current date for calculations (Asia/Manila timezone)
    now = datetime.now()
    today = now.date()
    
    # Debug output (only if needed)
    if False:  # Set to True for debugging
        print(f"🔍 DEBUG COMPARISON for {time_filter}:")
        print(f"   Input current_start: {current_start}")
        print(f"   Input current_end: {current_end}")
        print(f"   Today: {today}")
    
    if time_filter == "1D":
        # 1D (Yesterday) Comparison: Yesterday vs same weekday from previous week
        # Current: Yesterday only
        # Previous: Same weekday from previous week
        yesterday = today - timedelta(days=1)
        days_since_monday = yesterday.weekday()  # Monday = 0, Sunday = 6
        
        # Calculate same weekday from previous week
        prev_start = yesterday - timedelta(days=7)
        prev_end = yesterday - timedelta(days=7)
        
        print(f"   1D Logic:")
        print(f"     Yesterday: {yesterday}")
        print(f"     Days since Monday: {days_since_monday}")
        print(f"     Previous same weekday: {prev_start}")
        
        return {'start_date': prev_start, 'end_date': prev_end}
        
    elif time_filter == "7D":
        # 7D (Week to Date) Comparison: Monday to yesterday vs same span from previous week
        # Always use yesterday as end point to ensure fair comparison with completed business days
        yesterday = today - timedelta(days=1)  # Always use yesterday as end point
        
        # Current week: Monday to yesterday
        current_start = yesterday - timedelta(days=yesterday.weekday())  # This week's Monday
        current_end = yesterday  # Always end on yesterday
        
        # Previous week: Monday to same weekday as yesterday
        previous_start = current_start - timedelta(days=7)  # Last week's Monday
        previous_end = yesterday - timedelta(days=7)  # Same weekday last week as yesterday
        
        # Debug output (only if needed)
        if False:  # Set to True for debugging
            print(f"🔍 7D ALWAYS-YESTERDAY LOGIC:")
            print(f"   Today: {today}")
            print(f"   Yesterday: {yesterday}")
            print(f"   Current week: {current_start} to {current_end}")
            print(f"   Previous week: {previous_start} to {previous_end}")
            print(f"   Days compared: {(current_end - current_start).days + 1}")
        
        return {'start_date': previous_start, 'end_date': previous_end}
        
    elif time_filter == "1M":
        # 1M (Month to Date) Comparison: 1st to today vs same span from previous month
        # Current: 1st of month to today
        # Previous: 1st of previous month to same day number
        this_month_start = today.replace(day=1)
        
        # Calculate previous month
        if today.month == 1:
            prev_month = today.replace(year=today.year - 1, month=12)
        else:
            prev_month = today.replace(month=today.month - 1)
        
        prev_month_start = prev_month.replace(day=1)
        
        # Same day number in previous month (handle month-end edge cases)
        try:
            prev_month_end = prev_month.replace(day=today.day)
        except ValueError:
            # Handle cases like Jan 31 vs Feb (Feb doesn't have 31 days)
            last_day_of_prev_month = calendar.monthrange(prev_month.year, prev_month.month)[1]
            prev_month_end = prev_month.replace(day=last_day_of_prev_month)
        
        prev_start = prev_month_start
        prev_end = prev_month_end
        
        print(f"   1M Logic:")
        print(f"     This month start: {this_month_start}")
        print(f"     Previous month: {prev_month}")
        print(f"     Previous month start: {prev_month_start}")
        print(f"     Previous month end: {prev_month_end}")
        
        return {'start_date': prev_start, 'end_date': prev_end}
        
    elif time_filter == "1Y":
        # 1Y (Year to Date) Comparison: 1st of year to today vs same span from previous year
        # Current: 1st of year to today
        # Previous: 1st of previous year to same date
        this_year_start = today.replace(month=1, day=1)
        prev_year_start = this_year_start.replace(year=this_year_start.year - 1)
        prev_year_end = today.replace(year=today.year - 1)
        
        prev_start = prev_year_start
        prev_end = prev_year_end
        
        print(f"   1Y Logic:")
        print(f"     This year start: {this_year_start}")
        print(f"     Previous year start: {prev_year_start}")
        print(f"     Previous year end: {prev_year_end}")
        
        return {'start_date': prev_start, 'end_date': prev_end}
        
    else:
        # Default: Use rolling period (backward compatibility)
        period_length = (current_end - current_start).days + 1
        prev_end = current_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=period_length - 1)
        
        print(f"   Default Logic:")
        print(f"     Period length: {period_length}")
        print(f"     Previous end: {prev_end}")
        print(f"     Previous start: {prev_start}")
        
        return {'start_date': prev_start, 'end_date': prev_end}
    
    # CRITICAL: Add debugging for final result
    print(f"   FINAL RESULT:")
    print(f"     Current period: {current_start} to {current_end} ({(current_end - current_start).days + 1} days)")
    print(f"     Previous period: {prev_start} to {prev_end} ({(prev_end - prev_start).days + 1} days)")
    print(f"     Periods should have same length: {(current_end - current_start).days + 1} == {(prev_end - prev_start).days + 1}")
    
    return {
        'start_date': prev_start,
        'end_date': prev_end
    }

def double_interval_str(interval_str: str) -> str:
    """Return a doubled SQL interval string. Examples: '7 days' -> '14 days', '1 month' -> '2 months'."""
    try:
        parts = interval_str.split()
        if not parts:
            return interval_str
        num = int(parts[0])
        unit = parts[1] if len(parts) > 1 else 'days'
        doubled = num * 2
        # normalize pluralization
        if unit.endswith('y') and unit != 'day':
            unit = unit  # e.g., 'year' handled below
        if unit in ['day', 'days']:
            unit = 'days'
        elif unit in ['month', 'months']:
            unit = 'months'
        elif unit in ['year', 'years']:
            unit = 'years'
        return f"{doubled} {unit}"
    except Exception:
        return interval_str

def resolve_time_bounds(time_filter: str, custom_start: Optional[date] = None, custom_end: Optional[date] = None) -> Dict[str, Any]:
    """Resolve time window for SQL. Returns dict with mode and values.
    - For presets: {'mode': 'preset', 'interval': '7 days'}
    - For custom:  {'mode': 'custom', 'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}
    The DB will convert to Asia/Manila using AT TIME ZONE in the query.
    """
    if time_filter == "Custom" and custom_start and custom_end:
        return {
            'mode': 'custom',
            'start_date': str(custom_start),
            'end_date': str(custom_end),
        }
    return {
        'mode': 'preset',
        'interval': get_time_filter_interval(time_filter)
    }

@st.cache_data(ttl=300)
def get_filtered_metrics(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Metrics based on time filter selection, with correct profit calculation.
    Uses intelligent date ranges instead of rolling periods.
    """
    # Get intelligent date ranges for current period
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    # Get previous period dates
    prev_dates = get_previous_period_dates(current_start, current_end, time_filter)
    prev_start = prev_dates['start_date']
    prev_end = prev_dates['end_date']
    
    def _get_metrics_for_period(start_date, end_date):
        params = [str(start_date), str(end_date)]
        store_clause = ""
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.append(store_ids)
            
        sql = f"""
        WITH period_transactions AS (
            SELECT ref_id, total, store_id
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
            AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
            AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
            {store_clause}
        ),
        period_costs AS (
            SELECT
                SUM(ti.quantity * p.cost) as total_cogs
            FROM transaction_items ti
            JOIN products p ON ti.product_id = p.id
            WHERE ti.transaction_ref_id IN (SELECT ref_id FROM period_transactions)
        )
        SELECT
            (SELECT COALESCE(SUM(total), 0) FROM period_transactions) as sales,
            (SELECT COALESCE(SUM(total), 0) FROM period_transactions) - (SELECT COALESCE(total_cogs, 0) FROM period_costs) as profit,
            (SELECT COUNT(ref_id) FROM period_transactions) as transactions,
            (SELECT COUNT(DISTINCT store_id) FROM period_transactions) as active_stores,
            (SELECT COALESCE(SUM(total), 0) / NULLIF(COUNT(ref_id), 0) FROM period_transactions) as avg_transaction_value
        """
        return execute_query_for_dashboard(sql, params=params)

    # Current and previous periods
    df_current = _get_metrics_for_period(current_start, current_end)
    df_previous = _get_metrics_for_period(prev_start, prev_end)
    
    metrics = df_current.iloc[0].to_dict() if not df_current.empty else {}
    prev_metrics = df_previous.iloc[0].to_dict() if not df_previous.empty else {}
    
    # Rename previous metrics for clarity
    metrics['prev_sales'] = prev_metrics.get('sales', 0)
    metrics['prev_profit'] = prev_metrics.get('profit', 0)
    metrics['prev_transactions'] = prev_metrics.get('transactions', 0)

    # YoY comparison for 1Y filter
    if time_filter == "1Y":
        # For YoY, we need to get the same period last year
        yoy_start = current_start.replace(year=current_start.year - 1)
        yoy_end = current_end.replace(year=current_end.year - 1)
        df_yoy = _get_metrics_for_period(yoy_start, yoy_end)
        yoy_metrics = df_yoy.iloc[0].to_dict() if not df_yoy.empty else {}
        metrics['yoy_sales'] = yoy_metrics.get('sales', 0)
        metrics['yoy_profit'] = yoy_metrics.get('profit', 0)

    return metrics

@st.cache_data(ttl=300)
def get_total_sales_for_period(time_filter: str = "7D", store_ids: Optional[List[int]] = None,
                               custom_start: Optional[date] = None, custom_end: Optional[date] = None) -> float:
    """Return total sales for the selected period in Asia/Manila time.
    Uses intelligent date ranges and handles multi-store filtering with ANY(%s).
    - For Custom, includes the full days (start 00:00 to end 23:59:59).
    """
    store_clause = ""
    params: List[Any] = []
    if time_filter == "Custom" and custom_start and custom_end:
        # Inclusive of end date: [start, end + 1 day)
        sql = f"""
        SELECT COALESCE(SUM(t.total), 0) AS total_sales
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale'
        AND COALESCE(t.is_cancelled, false) = false
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s::date
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < (%s::date + INTERVAL '1 day')
        {{store_clause}}
        """
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params = [str(custom_start), str(custom_end), store_ids]
        else:
            params = [str(custom_start), str(custom_end)]
        sql = sql.replace("{store_clause}", store_clause)
    else:
        # Get intelligent date ranges
        date_range = get_intelligent_date_range(time_filter)
        current_start = date_range['start_date']
        current_end = date_range['end_date']
        
        sql = f"""
        SELECT COALESCE(SUM(t.total), 0) AS total_sales
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale'
        AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
        {{store_clause}}
        """
        params = [str(current_start), str(current_end)]
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.append(store_ids)
        sql = sql.replace("{store_clause}", store_clause)

    df = execute_query_for_dashboard(sql, params=params)
    if df is not None and not df.empty:
        return float(df.iloc[0]['total_sales'] or 0)
    return 0.0

@st.cache_data(ttl=300)
def get_sales_per_hour_average(time_filter: str = "7D", store_ids: Optional[List[int]] = None,
                               custom_start: Optional[date] = None, custom_end: Optional[date] = None) -> pd.DataFrame:
    """Return average sales per hour-of-day (0-23) over the selected period in Asia/Manila.
    Uses ANY(%s) for multi-store. Ensures all 24 hours are present.
    """
    store_clause = ""
    params: List[Any] = []
    if time_filter == "Custom" and custom_start and custom_end:
        sql = f"""
        WITH base AS (
            SELECT (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d,
                   EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS hr,
                   SUM(t.total) AS sales
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale'
              AND COALESCE(t.is_cancelled, false) = false
              AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s::date
              AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < (%s::date + INTERVAL '1 day')
              {store_clause}
            GROUP BY 1,2
        ), hours AS (
            SELECT generate_series(0,23) AS hr
        ), stats AS (
            SELECT hr, AVG(sales) AS avg_sales
            FROM base
            GROUP BY hr
        )
        SELECT h.hr AS hour, COALESCE(s.avg_sales, 0) AS avg_sales
        FROM hours h
        LEFT JOIN stats s ON s.hr = h.hr
        ORDER BY h.hr
        """
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params = [str(custom_start), str(custom_end), store_ids]
        else:
            params = [str(custom_start), str(custom_end)]
        sql = sql.replace("{store_clause}", store_clause)
    else:
        interval = get_time_filter_interval(time_filter)
        sql = f"""
        WITH base AS (
            SELECT (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d,
                   EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS hr,
                   SUM(t.total) AS sales
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale'
              AND COALESCE(t.is_cancelled, false) = false
              AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
              {store_clause}
            GROUP BY 1,2
        ), hours AS (
            SELECT generate_series(0,23) AS hr
        ), stats AS (
            SELECT hr, AVG(sales) AS avg_sales
            FROM base
            GROUP BY hr
        )
        SELECT h.hr AS hour, COALESCE(s.avg_sales, 0) AS avg_sales
        FROM hours h
        LEFT JOIN stats s ON s.hr = h.hr
        ORDER BY h.hr
        """
        params = [interval]
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.append(store_ids)
        sql = sql.replace("{store_clause}", store_clause)

    df = execute_query_for_dashboard(sql, params=params)
    if df is None:
        return pd.DataFrame({"hour": list(range(24)), "avg_sales": [0]*24})
    return df

@st.cache_data(ttl=300)
def get_hourly_sales_pattern(time_filter: str = "7D", store_filter_ids: Optional[List[int]] = None,
                             custom_start: Optional[date] = None, custom_end: Optional[date] = None) -> pd.DataFrame:
    """Get average sales per hour across all days in the selected period.
    Uses intelligent date ranges and ensures all hours 0-23 are present and adds friendly hour labels (12-hour clock).
    """
    df = get_sales_per_hour_average(time_filter, store_filter_ids, custom_start, custom_end)
    if df is None:
        return pd.DataFrame({
            "hour": list(range(24)),
            "hour_label": [f"{((h%12) or 12)}:00 {'AM' if h<12 else 'PM'}" for h in range(24)],
            "avg_sales": [0]*24,
        })
    try:
        import polars as pl
        pl_df = pl.from_pandas(df)
        # Ensure complete 0..23 coverage via left join
        hours = pl.DataFrame({"hour": list(range(24))})
        pl_df = hours.join(pl_df, on="hour", how="left")
        # Fill nulls
        pl_df = pl_df.with_columns(
            pl.col("avg_sales").fill_null(0)
        )
        # Build hour label in 12-hour format
        h_mod = (pl.col("hour") % 12)
        hour12 = pl.when(h_mod.eq(0)).then(12).otherwise(h_mod)
        ampm = pl.when(pl.col("hour") < 12).then(pl.lit("AM")).otherwise(pl.lit("PM"))
        pl_df = pl_df.with_columns(
            pl.format("{}:00 {}", hour12, ampm).alias("hour_label")
        )
        # Ensure sorted by hour
        pl_df = pl_df.sort("hour")
        return pl_df.to_pandas()
    except Exception:
        # Fallback to pandas path
        all_hours = pd.DataFrame({"hour": list(range(24))})
        df = all_hours.merge(df, on="hour", how="left").fillna({"avg_sales": 0})
        df["hour_label"] = df["hour"].apply(lambda h: f"{((int(h)%12) or 12)}:00 {'AM' if int(h)<12 else 'PM'}")
        return df

def create_hourly_sales_chart(hourly_data: pd.DataFrame):
    """Create professional hourly sales pattern chart with dark theme and PHP formatting."""
    if hourly_data is None or hourly_data.empty:
        return go.Figure()
    # Maintain hour order 0..23
    hourly_data = hourly_data.sort_values("hour").copy()
    fig = px.bar(hourly_data, x="hour_label", y="avg_sales",
                 labels={"hour_label": "Hour of Day", "avg_sales": "Average Sales"},
                 title="Average Sales by Hour")
    fig.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0), template="plotly_dark",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(tickprefix='₱', separatethousands=True)
    fig.update_traces(hovertemplate='%{x}: ₱%{y:,.0f}<extra></extra>')
    return fig

@st.cache_data(ttl=300)
def get_hourly_sales_latest_day(store_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Total sales per hour for the latest available calendar day in Asia/Manila.
    Ensures hours 0..23 are present. Respects optional store filter via ANY(%s).
    Updated to use intelligent date ranges for better period selection.
    """
    store_clause = ""
    params: List[Any] = []
    # Store filter must apply both in latest_date CTE and in joined transactions
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
    sql = f"""
    WITH latest_date AS (
      SELECT MAX((t.transaction_time AT TIME ZONE 'Asia/Manila')::date) AS max_date
      FROM transactions t
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled, false) = false
      {store_clause}
    ), hours AS (
      SELECT generate_series(0,23) AS hour
    )
    SELECT 
      h.hour,
      TO_CHAR(make_time(h.hour,0,0),'HH12:MI AM') AS hour_label,
      COALESCE(SUM(t.total),0) AS sales_latest_day
    FROM hours h
    LEFT JOIN transactions t
      ON EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int = h.hour
      AND (t.transaction_time AT TIME ZONE 'Asia/Manila')::date = (SELECT max_date FROM latest_date)
      AND LOWER(t.transaction_type)='sale'
      AND COALESCE(t.is_cancelled, false) = false
      {store_clause}
    GROUP BY h.hour
    ORDER BY h.hour;
    """
    if store_ids:
        # one for latest_date, one for join
        params = [store_ids, store_ids]
    df = execute_query_for_dashboard(sql, params=params if params else None)
    if df is None or df.empty:
        return pd.DataFrame({"hour": list(range(24)), "hour_label": [f"{((h%12) or 12)}:00 {'AM' if h<12 else 'PM'}" for h in range(24)], "sales_latest_day": [0]*24})
    return df

@st.cache_data(ttl=300)
def get_hourly_sales_average(time_filter: str = "7D", store_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Average sales per hour across all calendar days within the selected window in Asia/Manila.
    Uses intelligent date ranges. Average = sum(sales at hour over window) / number of distinct calendar days in the window.
    Ensures hours 0..23 are present.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
    sql = f"""
    WITH window_days AS (
      SELECT DISTINCT (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d
      FROM transactions t
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
        {store_clause}
    ), hours AS (
      SELECT generate_series(0,23) AS hour
    ), hourly AS (
      SELECT 
        EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS hour,
        SUM(t.total) AS sales_at_hour
      FROM transactions t
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
        {store_clause}
      GROUP BY 1
    )
    SELECT 
      h.hour,
      TO_CHAR(make_time(h.hour,0,0),'HH12:MI AM') AS hour_label,
      (COALESCE(hourly.sales_at_hour,0) / NULLIF((SELECT COUNT(*) FROM window_days),0))::numeric AS avg_sales_per_hour
    FROM hours h
    LEFT JOIN hourly ON hourly.hour = h.hour
    ORDER BY h.hour;
    """
    params: List[Any] = [str(current_start), str(current_end), str(current_start), str(current_end)]
    if store_ids:
        # window_days store filter, hourly store filter
        params.extend([store_ids, store_ids])
    df = execute_query_for_dashboard(sql, params=params)
    if df is None or df.empty:
        return pd.DataFrame({"hour": list(range(24)), "hour_label": [f"{((h%12) or 12)}:00 {'AM' if h<12 else 'PM'}" for h in range(24)], "avg_sales_per_hour": [0]*24})
    return df

def get_sales_by_category_pie(time_filter="7D", store_ids: Optional[List[int]] = None):
    """SQL query to get category sales for pie chart, with optional store filter.
    Uses intelligent date ranges.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    params = [str(current_start), str(current_end)]
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT 
        p.category, 
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
    {store_clause}
    GROUP BY p.category
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC;
    """
    return execute_query_for_dashboard(sql, params=params)

def get_inventory_by_category_pie(store_ids: Optional[List[int]] = None):
    """SQL query to get inventory value by category (quantity * unit_price), with optional store filter."""
    params = []
    store_clause = ""
    if store_ids:
        store_clause = "AND i.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT
        COALESCE(p.category, 'Unknown') as category,
        SUM(i.quantity_on_hand * p.unit_price) as total_inventory_value
    FROM inventory i
    JOIN products p ON i.product_id = p.id
    WHERE i.quantity_on_hand > 0
    AND p.unit_price > 0
    {store_clause}
    GROUP BY p.category
    HAVING SUM(i.quantity_on_hand * p.unit_price) > 0
    ORDER BY total_inventory_value DESC;
    """
    df = execute_query_for_dashboard(sql, params=params if params else None)
    return df

@st.cache_data(ttl=300)
def get_top_sellers_analysis(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Top products with momentum indicators, with optional store filter."""
    interval = get_time_filter_interval(time_filter)
    
    params_current = [interval]
    params_previous = [f'2 * {interval}', interval]
    
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params_current.append(store_ids)
        params_previous.append(store_ids)

    sql = f"""
    WITH current_period AS (
        SELECT 
            p.name as product_name,
            SUM(ti.item_total) as total_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
        {store_clause}
        GROUP BY p.name
    ),
    previous_period AS (
        SELECT 
            p.name as product_name,
            SUM(ti.item_total) as prev_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
        AND t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s
        {store_clause}
        GROUP BY p.name
    )
    SELECT
        cp.product_name,
        cp.total_revenue,
        COALESCE(pp.prev_revenue, 0) as prev_revenue,
        (cp.total_revenue - COALESCE(pp.prev_revenue, 0)) as revenue_change
    FROM current_period cp
    LEFT JOIN previous_period pp ON cp.product_name = pp.product_name
    ORDER BY cp.total_revenue DESC
    LIMIT 10;
    """
    # Combine params for a single query execution
    # This requires re-writing the query to not use CTEs that depend on different params
    # For simplicity and since the function is cached, separate executions are acceptable here.
    # A more optimized version would use a single query with CASE statements.
    
    # Let's create a single query for better performance
    single_sql = f"""
    SELECT
        p.name as product_name,
        SUM(CASE WHEN (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s THEN ti.item_total ELSE 0 END) as total_revenue,
        SUM(CASE WHEN t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s THEN ti.item_total ELSE 0 END) as prev_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
    {store_clause}
    GROUP BY p.name
    ORDER BY total_revenue DESC
    LIMIT 10;
    """
    params = [interval, f'2 * {interval}', interval, f'2 * {interval}']
    if store_ids:
        params.append(store_ids)
        
    df = execute_query_for_dashboard(single_sql, params=params)
    if not df.empty:
        df['revenue_change'] = df['total_revenue'] - df['prev_revenue']
    return df


@st.cache_data(ttl=300)
def get_store_performance_heatmap(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Store performance data for heatmap, with optional store filter."""
    interval = get_time_filter_interval(time_filter)
    params = [interval]
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT 
        s.name as store_name,
        DATE_TRUNC('day', t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        SUM(t.total) as daily_sales
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s
    {store_clause}
    GROUP BY s.name, date
    ORDER BY s.name, date;
    """
    return execute_query_for_dashboard(sql, params=params)

@st.cache_data(ttl=300)
def get_fad_stays_analysis(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Products gaining/losing momentum."""
    df = get_top_sellers_analysis(time_filter, store_ids)
    if not df.empty:
        df['momentum'] = (df['total_revenue'] - df['prev_revenue']) / df['prev_revenue'].replace(0, np.nan)
        df_gainers = df.sort_values(by='revenue_change', ascending=False).head(5)
        df_losers = df.sort_values(by='revenue_change', ascending=True).head(5)
        return df_gainers, df_losers
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def get_business_highlights(time_filter="7D", store_ids: Optional[List[int]] = None):
    """AI-generated business insights for the period."""
    metrics = get_filtered_metrics(time_filter, store_ids)
    top_sellers = get_top_sellers_analysis(time_filter, store_ids)

    if not metrics or top_sellers.empty:
        return "Not enough data to generate highlights."

    client = get_claude_client()
    if not client:
        return "AI client not configured. Cannot generate highlights."

    prompt = f"""
    Based on the following data for the last {time_filter}, generate 3-4 key business highlights and actionable insights. Be concise and use bullet points.

    - Total Sales: ₱{metrics.get('sales', 0):,.0f}
    - Total Profit: ₱{metrics.get('profit', 0):,.0f}
    - Total Transactions: {metrics.get('transactions', 0):,}
    - Top Selling Product: {top_sellers['product_name'].iloc[0]} with ₱{top_sellers['total_revenue'].iloc[0]:,.0f} in sales.
    - Top Gaining Product: {top_sellers.sort_values('revenue_change', ascending=False).iloc[0]['product_name']}

    Generate the highlights now.
    """
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Could not generate AI highlights: {e}"

# --- NEW DASHBOARD DATA FUNCTIONS END ---


def create_calendar_heatmap(df_cal, date_col, value_col):
    """Create calendar heatmap visualization"""
    if df_cal is None or df_cal.empty:
        return None
    df_cal = df_cal.copy()
    df_cal[date_col] = pd.to_datetime(df_cal[date_col])
    df_cal = df_cal.sort_values(date_col)
    
    # Create week-based calendar
    df_cal['week'] = df_cal[date_col].dt.isocalendar().week
    df_cal['day_of_week'] = df_cal[date_col].dt.dayofweek
    df_cal['day_num'] = df_cal[date_col].dt.day
    
    min_week = df_cal['week'].min()
    df_cal['week_normalized'] = df_cal['week'] - min_week
    
    unique_weeks = sorted(df_cal['week_normalized'].unique())
    if not unique_weeks or len(df_cal) < 7:
        return None  # Return None if insufficient data
    
    matrix = []
    annotations = []
    
    # Build matrix from Monday (0) to Sunday (6)
    for day in range(7):  # 0 = Monday, 6 = Sunday
        row = []
        for week in unique_weeks:
            day_data = df_cal[(df_cal['week_normalized'] == week) & (df_cal['day_of_week'] == day)]
            if not day_data.empty:
                value = day_data.iloc[0][value_col]
                day_num = day_data.iloc[0]['day_num']
                row.append(value)
                text_color = 'black' if value <= max(df_cal[value_col]) * 0.5 else 'white'
                annotations.append(dict(
                    x=week, y=day,  # y-coordinate matches day index (0 = Monday, 6 = Sunday)
                    text=str(day_num),
                    showarrow=False,
                    font=dict(color=text_color, size=12, family='Arial Black'),
                    xref='x', yref='y'
                ))
            else:
                row.append(0)  # No data for this day
        matrix.append(row)
    
    # Create heatmap with UI-matched color scheme
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=[
            [0, '#f0f4f8'],    # Very light gray-blue for no data
            [0.2, '#d9e6f2'],  # Light blue for low sales
            [0.4, '#a3c9e0'],  # Medium blue for medium sales
            [0.6, '#6baed6'],  # Darker blue for high sales
            [0.8, '#3182bd'],  # Deep blue for higher sales
            [1, '#1b4d7e']     # Dark blue for maximum
        ],
        showscale=True,
        colorbar=dict(
            title="Sales (PHP)",
            tickmode="linear",
            tick0=0,
            dtick=max(df_cal[value_col]) / 5 if df_cal[value_col].max() > 0 else 1,
            tickformat=",.0f"
        ),
        hovertemplate='<b>%{text}</b><br>Sales: ₱%{z:,.0f}<extra></extra>',
        text=[[f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]}" for week in range(len(unique_weeks))] for day in range(7)]
    ))
    
    # Add day number annotations
    for ann in annotations:
        fig.add_annotation(ann)
    
    # Update layout with Monday at top, Sunday at bottom
    fig.update_layout(
        title=f'📅 Daily Sales Calendar ({df_cal[date_col].min().strftime("%b %d")} - {df_cal[date_col].max().strftime("%b %d, %Y")})',
        xaxis=dict(
            title="",
            tickvals=list(range(len(unique_weeks))),
            ticktext=[f"Week {i+1}" for i in range(len(unique_weeks))],
            side='top'
        ),
        yaxis=dict(
            title="",
            tickvals=list(range(7)),  # 0 = Monday, 6 = Sunday
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],  # Monday at top
            autorange=True  # Automatically adjust while respecting ticktext order
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        title_x=0.5,
        font=dict(size=11)
    )
    
    return fig

# CSS Styling
# CSS now loaded from supabot.ui.styles.css

# Initialize Session State
def init_session_state():
    # Use centralized session state initialization
    settings.initialize_session_state()
    
    # Add app-specific session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "training_system" not in st.session_state:
        st.session_state.training_system = get_training_system()
    if "schema_info" not in st.session_state:
        st.session_state.schema_info = None
    if "dashboard_time_filter" not in st.session_state:
        st.session_state.dashboard_time_filter = "7D"
    if "dashboard_store_filter" not in st.session_state:
        st.session_state.dashboard_store_filter = [
            "Rockwell", "Greenhills", "Magnolia", "North Edsa", "Fairview"
        ]
    # Add custom date session state
    if "custom_start_date" not in st.session_state:
        st.session_state.custom_start_date = date.today() - timedelta(days=7)
    if "custom_end_date" not in st.session_state:
        st.session_state.custom_end_date = date.today() - timedelta(days=1)

# --- MODIFICATION START: REPLACED render_dashboard() FUNCTION ---

def get_dashboard_top_sellers(time_filter="7D", store_filter_ids=None):
    """Simple, working top sellers query using intelligent date ranges"""
    
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    params = [str(current_start), str(current_end)]
    store_clause = ""
    if store_filter_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_filter_ids)
    
    sql = f"""
    SELECT 
        p.name as product_name,
        SUM(ti.item_total) as total_revenue,
        SUM(ti.quantity) as total_quantity,
        COUNT(DISTINCT t.ref_id) as transaction_count
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND COALESCE(t.is_cancelled, false) = false
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
    {store_clause}
    GROUP BY p.name, p.id
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    
    return execute_query_for_dashboard(sql, params=params)

def get_top_products_with_change(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Top 10 products in current range with previous-period comparison and % change.
    Uses intelligent date ranges and ANY(%s) for multi-store binding.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']

    # Get previous period dates for comparison
    prev_dates = get_previous_period_dates(current_start, current_end, time_filter)

    # Build store clause and params in the exact placeholder order matching the SQL
    store_clause = ""
    params: List[Any] = []
    
    # Parameters in SQL order:
    # 1. current_start (line 11)
    # 2. current_end (line 12)
    # 3. store_ids for current_period (line 13) - if present
    # 4. prev_start (line 24)
    # 5. prev_end (line 25)  
    # 6. store_ids for previous_period (line 26) - if present
    
    params.extend([str(current_start), str(current_end)])
    
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)  # For current_period
    
    params.extend([str(prev_dates['start_date']), str(prev_dates['end_date'])])
    
    if store_ids:
        params.append(store_ids)  # For previous_period

    sql = f"""
    WITH current_period AS (
        SELECT 
            p.id as product_id,
            p.name as product_name,
            COALESCE(SUM(ti.item_total), 0) AS curr_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' 
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY p.id, p.name
    ),
    previous_period AS (
        SELECT 
            p.id as product_id,
            COALESCE(SUM(ti.item_total), 0) AS prev_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' 
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY p.id
    )
    SELECT 
        c.product_name,
        c.curr_revenue AS total_revenue,
        COALESCE(p.prev_revenue, 0) AS prev_revenue,
        CASE 
            WHEN COALESCE(p.prev_revenue, 0) = 0 THEN NULL
            ELSE ((c.curr_revenue - p.prev_revenue) / p.prev_revenue) * 100.0
        END AS pct_change
    FROM current_period c
    LEFT JOIN previous_period p ON p.product_id = c.product_id
    WHERE c.curr_revenue > 0
    ORDER BY total_revenue DESC
    LIMIT 10;
    """

    df = execute_query_for_dashboard(sql, params=params)
    return df if df is not None else pd.DataFrame(columns=["product_name","total_revenue","prev_revenue","pct_change"])

def get_categories_with_change(time_filter="7D", store_ids: Optional[List[int]] = None):
    """All categories ranked by revenue with previous-period comparison and % change.
    Uses intelligent date ranges and ANY(%s) for multi-store binding.
    """
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']

    # Get previous period dates for comparison
    prev_dates = get_previous_period_dates(current_start, current_end, time_filter)

    # Build store clause and params in the exact placeholder order matching the SQL
    store_clause = ""
    params: List[Any] = []
    
    # Parameters in SQL order:
    # 1. current_start, 2. current_end, 3. store_ids (current), 4. prev_start, 5. prev_end, 6. store_ids (previous)
    params.extend([str(current_start), str(current_end)])
    
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)  # For current_period
    
    params.extend([str(prev_dates['start_date']), str(prev_dates['end_date'])])
    
    if store_ids:
        params.append(store_ids)  # For previous_period

    sql = f"""
    WITH current_period AS (
        SELECT 
            COALESCE(p.category, 'Uncategorized') AS category,
            COALESCE(SUM(ti.item_total), 0) AS curr_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' 
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY COALESCE(p.category, 'Uncategorized')
    ),
    previous_period AS (
        SELECT 
            COALESCE(p.category, 'Uncategorized') AS category,
            COALESCE(SUM(ti.item_total), 0) AS prev_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' 
          AND COALESCE(t.is_cancelled, false) = false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
          {store_clause}
        GROUP BY COALESCE(p.category, 'Uncategorized')
    )
    SELECT 
        c.category,
        c.curr_revenue AS total_revenue,
        COALESCE(p.prev_revenue, 0) AS prev_revenue,
        CASE 
            WHEN COALESCE(p.prev_revenue, 0) = 0 THEN NULL
            ELSE ((c.curr_revenue - p.prev_revenue) / p.prev_revenue) * 100.0
        END AS pct_change
    FROM current_period c
    LEFT JOIN previous_period p ON c.category = p.category
    WHERE c.curr_revenue > 0
    ORDER BY total_revenue DESC;
    """

    df = execute_query_for_dashboard(sql, params=params)
    return df if df is not None else pd.DataFrame(columns=["category","total_revenue","prev_revenue","pct_change"])

@st.cache_data(ttl=300)
def get_dashboard_metrics_simple(time_filter="7D", store_filter_ids=None):
    """WORKING version - Simple metrics calculation using direct queries"""
    try:
        days_map = {"1D": 1, "7D": 7, "1M": 30, "6M": 180, "1Y": 365}
        days = days_map.get(time_filter, 7)
        
        conn = create_db_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        
        # Build store filter
        store_clause = ""
        params = [days]
        if store_filter_ids and len(store_filter_ids) > 0:
            placeholders = ','.join(['%s'] * len(store_filter_ids))
            store_clause = f"AND t.store_id IN ({placeholders})"
            params.extend(store_filter_ids)
        
        # Current period query (working version)
        current_sql = f"""
        SELECT 
            COALESCE(SUM(t.total), 0) as current_sales,
            COALESCE(COUNT(DISTINCT t.ref_id), 0) as current_transactions,
            COALESCE(AVG(t.total), 0) as avg_transaction_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s DAY
        {store_clause}
        """
        
        cursor.execute(current_sql, params)
        current_result = cursor.fetchone()
        current_sales, current_transactions, avg_value = current_result
        
        # Previous period query
        prev_params = [days * 2, days]
        if store_filter_ids and len(store_filter_ids) > 0:
            prev_params.extend(store_filter_ids)
            
        prev_sql = f"""
        SELECT 
            COALESCE(SUM(t.total), 0) as prev_sales,
            COALESCE(COUNT(DISTINCT t.ref_id), 0) as prev_transactions
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s DAY
        AND (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL %s DAY
        {store_clause}
        """
        
        cursor.execute(prev_sql, prev_params)
        prev_result = cursor.fetchone()
        prev_sales, prev_transactions = prev_result
        
        conn.close()
        
        # Calculate profit (30% margin estimate)
        current_profit = current_sales * 0.3 if current_sales else 0
        prev_profit = prev_sales * 0.3 if prev_sales else 0
        
        return {
            'current_sales': float(current_sales) if current_sales else 0,
            'current_profit': float(current_profit),
            'current_transactions': int(current_transactions) if current_transactions else 0,
            'avg_transaction_value': float(avg_value) if avg_value else 0,
            'prev_sales': float(prev_sales) if prev_sales else 0,
            'prev_profit': float(prev_profit),
            'prev_transactions': int(prev_transactions) if prev_transactions else 0
        }
        
    except Exception as e:
        print(f"Error in get_dashboard_metrics_simple: {e}")
        return {}

@st.cache_data(ttl=300)
def get_sales_trend_analysis():
    """Get daily sales trend for last 7 days"""
    sql = """
    WITH date_series AS (
        SELECT generate_series(
            (NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL '7 days',
            (NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL '1 day',
            INTERVAL '1 day'
        )::date as date
    ),
    daily_sales AS (
        SELECT 
            DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
            COALESCE(SUM(t.total), 0) as daily_sales
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ((NOW() AT TIME ZONE 'Asia/Manila')::date - INTERVAL '7 days')
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <  (NOW() AT TIME ZONE 'Asia/Manila')::date
        GROUP BY 1
    )
    SELECT 
        ds.date,
        COALESCE(ds2.daily_sales, 0) as daily_sales
    FROM date_series ds
    LEFT JOIN daily_sales ds2 ON ds.date = ds2.date
    ORDER BY ds.date
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_inventory_by_category():
    """Get inventory value by category using unit prices"""
    sql = """
    SELECT 
        COALESCE(p.category, 'Uncategorized') as category,
        SUM(i.quantity_on_hand * p.unit_price) as inventory_value,
        COUNT(DISTINCT p.id) as product_count
    FROM inventory i
    JOIN products p ON p.id = i.product_id
    WHERE i.quantity_on_hand > 0
    AND p.unit_price > 0
    GROUP BY p.category
    HAVING SUM(i.quantity_on_hand * p.unit_price) > 0
    ORDER BY inventory_value DESC
    """
    return execute_query_for_dashboard(sql)


def get_dashboard_metrics(time_filter="7D", store_filter_ids=None, custom_start=None, custom_end=None):
    """Calculate metrics with proper period-over-period comparisons using intelligent date ranges"""
    
    # Get intelligent date ranges
    if time_filter == "Custom" and custom_start and custom_end:
        current_start = custom_start
        current_end = custom_end
    else:
        date_range = get_intelligent_date_range(time_filter)
        current_start = date_range['start_date']
        current_end = date_range['end_date']
    
    # Get previous period dates for comparison
    prev_dates = get_previous_period_dates(current_start, current_end, time_filter)
    
    # Build store filter clause
    store_clause = ""
    
    # Build parameters in the EXACT order they appear in SQL
    params = []
    
    # Parameters in SQL order:
    # 1. current_start (line 3)
    # 2. current_end (line 4)
    # 3. store_filter_ids (line 13) - if present
    # 4. prev_start (line 18)
    # 5. prev_end (line 19)
    # 6. store_filter_ids (line 24) - if present
    
    params.append(str(current_start))
    params.append(str(current_end))
    
    if store_filter_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_filter_ids)  # For current_period CTE
    
    params.append(str(prev_dates['start_date']))
    params.append(str(prev_dates['end_date']))
    
    if store_filter_ids:
        params.append(store_filter_ids)  # For previous_period CTE
    
    # SQL with both current and previous period calculations using intelligent date ranges
    sql = f"""
    WITH current_period AS (
        SELECT 
            COALESCE(SUM(t.total), 0) as sales,
            COALESCE(SUM(t.total) * 0.3, 0) as profit,
            COALESCE(COUNT(DISTINCT t.ref_id), 0) as transactions,
            COALESCE(AVG(t.total), 0) as avg_transaction_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
        {store_clause}
    ),
    previous_period AS (
        SELECT 
            COALESCE(SUM(t.total), 0) as prev_sales,
            COALESCE(SUM(t.total) * 0.3, 0) as prev_profit,
            COALESCE(COUNT(DISTINCT t.ref_id), 0) as prev_transactions,
            COALESCE(AVG(t.total), 0) as prev_avg_transaction_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
        {store_clause}
    )
    SELECT 
        cp.sales as current_sales,
        cp.profit as current_profit,
        cp.transactions as current_transactions,
        cp.avg_transaction_value,
        pp.prev_sales,
        pp.prev_profit,
        pp.prev_transactions,
        pp.prev_avg_transaction_value
    FROM current_period cp
    CROSS JOIN previous_period pp
    """
    
    try:
        # Debug: Show the SQL and parameters being executed (only if debug mode is enabled)
        if st.session_state.get('debug_mode', False):
            st.info(f"🔍 Debug SQL: {sql}")
            st.info(f"🔍 Debug Params: {params}")
            st.info(f"🔍 Store filter IDs type: {type(store_filter_ids)}")
            st.info(f"🔍 Store filter IDs value: {store_filter_ids}")
            st.info(f"🔍 Debug Store Filter IDs: {store_filter_ids}")
            st.info(f"🔍 Debug Store Clause: {store_clause}")
        
        result = execute_query_for_dashboard(sql, params=params)
        
        # CRITICAL: Add extensive debugging for comparison calculations
        if result is not None and not result.empty:
            row = result.iloc[0]
            current_sales = float(row['current_sales'])
            prev_sales = float(row['prev_sales'])
            current_transactions = int(row['current_transactions'])
            prev_transactions = int(row['prev_transactions'])
            
            # Calculate percentage changes
            sales_change_pct = ((current_sales - prev_sales) / max(prev_sales, 1)) * 100 if prev_sales > 0 else 100.0 if current_sales > 0 else 0.0
            transactions_change_pct = ((current_transactions - prev_transactions) / max(prev_transactions, 1)) * 100 if prev_transactions > 0 else 100.0 if current_transactions > 0 else 0.0
            
            # CRITICAL: Debug the actual values and calculations
            print(f"🔍 METRICS DEBUG for {time_filter}:")
            print(f"   Current Sales: ₱{current_sales:,.0f}")
            print(f"   Previous Sales: ₱{prev_sales:,.0f}")
            print(f"   Sales Change: ₱{current_sales - prev_sales:,.0f}")
            print(f"   Sales Change %: {sales_change_pct:.1f}%")
            print(f"   Current Transactions: {current_transactions:,}")
            print(f"   Previous Transactions: {prev_transactions:,}")
            print(f"   Transactions Change: {current_transactions - prev_transactions:,}")
            print(f"   Transactions Change %: {transactions_change_pct:.1f}%")
            print(f"   Is sales up? {current_sales > prev_sales}")
            print(f"   Is percentage positive? {sales_change_pct > 0}")
            
            if st.session_state.get('debug_mode', False):
                st.info(f"🔍 Debug Result: {result.to_dict('records')}")
                st.info(f"🔍 Sales: ₱{current_sales:,.0f} vs ₱{prev_sales:,.0f} = {sales_change_pct:+.1f}%")
                st.info(f"🔍 Transactions: {current_transactions:,} vs {prev_transactions:,} = {transactions_change_pct:+.1f}%")
            
            return {
                'current_sales': current_sales,
                'current_profit': float(row['current_profit']),
                'current_transactions': current_transactions,
                'avg_transaction_value': float(row['avg_transaction_value']),
                'prev_sales': prev_sales,
                'prev_profit': float(row['prev_profit']),
                'prev_transactions': prev_transactions,
                'prev_avg_transaction_value': float(row['prev_avg_transaction_value'])
            }
        else:
            if st.session_state.get('debug_mode', False):
                st.warning("🔍 Debug: Query returned no results")
            return {
                'current_sales': 0, 'current_profit': 0, 'current_transactions': 0,
                'avg_transaction_value': 0, 'prev_sales': 0, 'prev_profit': 0, 
                'prev_transactions': 0, 'prev_avg_transaction_value': 0
            }
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"🔍 Debug KPI Error: {e}")
            st.error(f"🔍 Debug SQL: {sql}")
            st.error(f"🔍 Debug Params: {params}")
        return {}

def get_dashboard_highlights(metrics, top_sellers_df, time_filter):
    """Generate business highlights from available data"""
    if not metrics or top_sellers_df.empty:
        return "📊 **Current Status:** Data is being processed. Please check back in a moment for detailed insights."
    
    sales = metrics.get('current_sales', 0)
    transactions = metrics.get('current_transactions', 0)
    
    if sales == 0:
        return "📊 **Current Status:** No sales data available for the selected period and filters."
    
    top_product = top_sellers_df.iloc[0]['product_name'] if not top_sellers_df.empty else "N/A"
    top_revenue = top_sellers_df.iloc[0]['total_revenue'] if not top_sellers_df.empty else 0
    
    highlights = f"""**💰 Period Performance:** Generated ₱{sales:,.0f} in sales across {transactions:,} transactions.

**🏆 Top Performer:** {top_product} leads with ₱{top_revenue:,.0f} in revenue.

**📈 Business Focus:** {"Strong performance period" if sales > 100000 else "Growth opportunity period"} - {"continue current strategies" if sales > 100000 else "consider promotional activities"}."""
    
    return highlights

def get_dashboard_fad_stays(time_filter="7D", store_filter_ids=None):
    """Get trending products (simplified version)"""
    top_sellers = get_dashboard_top_sellers(time_filter, store_filter_ids)
    
    if top_sellers.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Split into gainers and losers based on revenue
    gainers = top_sellers.head(3).copy()
    gainers['revenue_change'] = gainers['total_revenue'] * 0.15  # Mock positive trend
    
    losers = top_sellers.tail(3).copy()
    losers['revenue_change'] = -(losers['total_revenue'] * 0.05)  # Mock negative trend
    
    return gainers, losers

def generate_dashboard_ai_insights(metrics, sales_cat_df, top_sellers_df, time_filter, store_filter_ids=None):
    """Generate AI-powered insights based on current dashboard data"""
    try:
        # Get Claude client
        client = get_claude_client()
        if not client:
            return "🤖 AI insights are not available. Please check your Claude API configuration in Settings."
        
        # Prepare data summary for AI analysis
        current_sales = metrics.get('current_sales', 0)
        prev_sales = metrics.get('prev_sales', 0)
        current_profit = metrics.get('current_profit', 0)
        prev_profit = metrics.get('prev_profit', 0)
        current_transactions = metrics.get('current_transactions', 0)
        prev_transactions = metrics.get('prev_transactions', 0)
        avg_transaction = metrics.get('avg_transaction_value', 0)
        
        # Calculate percentage changes
        sales_change = ((current_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
        profit_change = ((current_profit - prev_profit) / prev_profit * 100) if prev_profit > 0 else 0
        transaction_change = ((current_transactions - prev_transactions) / prev_transactions * 100) if prev_transactions > 0 else 0
        
        # Get top categories
        top_categories = []
        if not sales_cat_df.empty:
            top_categories = sales_cat_df.head(3)['category'].tolist()
        
        # Get top products
        top_products = []
        if not top_sellers_df.empty:
            top_products = top_sellers_df.head(3)['product_name'].tolist()
        
        # Store filter context
        store_context = f" across {len(store_filter_ids)} selected stores" if store_filter_ids else " across all stores"
        
        # Create prompt for Claude
        prompt = f"""
Analyze this business dashboard data and provide 3-4 key insights with actionable recommendations:

**Period:** {time_filter}{store_context}

**Key Metrics:**
- Current Sales: ₱{current_sales:,.0f} (Change: {sales_change:+.1f}%)
- Current Profit: ₱{current_profit:,.0f} (Change: {profit_change:+.1f}%)
- Transactions: {current_transactions:,} (Change: {transaction_change:+.1f}%)
- Avg Transaction Value: ₱{avg_transaction:,.0f}

**Top Categories:** {', '.join(top_categories) if top_categories else 'No data'}
**Top Products:** {', '.join(top_products) if top_products else 'No data'}

Provide insights in this format:
- **Performance Analysis:** [Brief assessment of overall performance]
- **Key Opportunity:** [Main opportunity or concern to address]
- **Recommended Action:** [Specific actionable recommendation]
- **Focus Area:** [What to monitor or improve next]

Keep it concise, business-focused, and actionable.
"""
        
        # Get AI response
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"🤖 **AI Insights Temporarily Unavailable**\n\nError: {str(e)}\n\nPlease try again later or check your API configuration."

def get_category_insights(sales_cat_df, time_filter="7D"):
    """Generate AI-powered category performance insights"""
    if sales_cat_df.empty:
        return "No category data available for insights."
    
    client = get_claude_client()
    if not client:
        # Fallback to simple insights
        top_cat = sales_cat_df.iloc[0]['category']
        top_revenue = sales_cat_df.iloc[0]['total_revenue']
        total_revenue = sales_cat_df['total_revenue'].sum()
        percentage = (top_revenue / total_revenue) * 100
        return f"**🏆 Top Category:** {top_cat} dominates with {percentage:.1f}% of total sales (₱{top_revenue:,.0f})"
    
    top_3_categories = sales_cat_df.head(3)
    category_summary = ""
    for _, row in top_3_categories.iterrows():
        category_summary += f"- {row['category']}: ₱{row['total_revenue']:,.0f}\n"
    
    prompt = f"""Analyze these category sales data for the last {time_filter} and provide 2-3 actionable business insights:

{category_summary}

Focus on:
1. Which categories are performing best/worst
2. Specific recommendations for improvement
3. Market opportunities or concerns

Be concise and business-focused. Use bullet points."""
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except:
        top_cat = sales_cat_df.iloc[0]['category']
        return f"**🏆 Top Category:** {top_cat} leads sales this period with strong performance."

def get_category_color_map():
    """Complete color mapping for all product categories with contrasting colors"""
    return {
        'tradsnax': '#95a5a6',           # Light gray-blue
        'aji mix': '#3498db',            # Your primary blue
        'indi': '#16a085',               # Teal (matches your theme)
        'ccp': '#e74c3c',                # Red accent
        'toys': '#f39c12',               # Orange accent
        'gummies': '#9b59b6',            # Purple accent
        'candy': '#e67e22',              # Dark orange
        'snacks': '#2ecc71',             # Green accent
        'beverages': '#1abc9c',          # Turquoise
        'chips': '#8e44ad',              # Dark purple
        'sweets': '#ff6b9d',             # Pink accent
        'crackers': '#d35400',           # Dark red-orange
        'nuts': '#27ae60',               # Dark green
        'biscuits': '#34495e',           # Dark blue-gray
        'chocolates': '#c0392b',         # Dark red
        'uncategorized': '#7f8c8d'       # Neutral gray
    }

def apply_category_colors(data, category_col):
    """Apply colors with case-insensitive matching for Streamlit/Plotly"""
    if data is None or data.empty:
        return []
        
    color_map = get_category_color_map()
    colors = []
    
    for category in data[category_col]:
        # Convert to lowercase and remove extra spaces for matching
        cat_key = str(category).lower().strip()
        color = color_map.get(cat_key, '#7f8c8d')  # Default to dark gray
        colors.append(color)
    
    return colors

 # Global fixed mapping and helpers
FIXED_CATEGORY_COLOR_MAP = {
    'n/a': '#6D8299',          # Cool Gray
    'aji mix': '#00C9FF',      # Neon Cyan
    'bev': '#2D98DA',          # Sky Blue
    'ccp': '#FF6B6B',          # Bright Coral
    'choco': '#E84118',        # Deep Red
    'indi': '#92FE9D',         # Mint Green
    'mint': '#38ADA9',         # Aqua Teal
    'nuts': '#FF9F43',         # Sunset Orange
    'oceana': '#1E90FF',       # Vibrant Blue
    'per gram': '#C65DFF',     # Electric Purple
    'seasonal': '#B33771',     # Magenta Pink
    'toys': '#FFD93D',         # Vivid Yellow
    'tradsnax': '#009432'      # Emerald Green
}

ALIAS_TO_CANONICAL = {
    # beverages -> bev
    'beverage': 'bev', 'beverages': 'bev', 'drink': 'bev', 'drinks': 'bev', 'soda': 'bev', 'soft drinks': 'bev', 'soft drink': 'bev',
    # chocolates -> choco
    'chocolate': 'choco', 'chocolates': 'choco', 'chocs': 'choco', 'choco': 'choco',
    # mint
    'mint': 'mint', 'mints': 'mint',
    # nuts
    'nut': 'nuts', 'nuts': 'nuts', 'trail mix': 'nuts',
    # toys
    'toy': 'toys', 'toys': 'toys',
    # tradsnax variations
    'trad snax': 'tradsnax', 'trad-snax': 'tradsnax', 'tradsnacks': 'tradsnax', 'trad snacks': 'tradsnax',
    # aji mix variations
    'aji-mix': 'aji mix', 'aji  mix': 'aji mix',
    # seasonal stays seasonal
    'seasonal': 'seasonal',
    # per gram
    'per gram': 'per gram', 'per-gram': 'per gram', 'pergram': 'per gram',
    # oceana
    'oceana': 'oceana',
    # n/a / unknowns bucket to n/a
    'n/a': 'n/a', 'na': 'n/a', 'n a': 'n/a', 'none': 'n/a', 'unknown': 'n/a', 'others': 'n/a', 'other': 'n/a', 'misc': 'n/a'
}

def normalize_category_label(label: str) -> str:
    return ' '.join(str(label).split()).strip().lower()

def canonicalize_category_label(raw_label: str) -> str:
    norm = normalize_category_label(raw_label)
    canonical = ALIAS_TO_CANONICAL.get(norm, norm)
    return canonical if canonical in FIXED_CATEGORY_COLOR_MAP else 'n/a'

def get_fixed_category_color_map():
    return FIXED_CATEGORY_COLOR_MAP

def get_pie_chart_colors(data, category_column):
    """Return a color map dict for Plotly: {actual_label: hex_color}."""
    color_map = {}
    seen = set()
    for raw in data[category_column].astype(str):
        if raw in seen:
            continue
        seen.add(raw)
        canonical = canonicalize_category_label(raw)
        color_map[raw] = FIXED_CATEGORY_COLOR_MAP[canonical]
    return color_map

def _normalize_store_name(name: str) -> str:
    """Normalize a store name for comparison: trim, lowercase, collapse whitespace."""
    if name is None:
        return ""
    # Convert to string, strip ends, lowercase, and collapse multiple spaces/tabs
    return " ".join(str(name).strip().lower().split())

def resolve_store_ids(store_df, selected_stores, debug=False):
    """
    Resolve a list of selected store display names to store IDs robustly.

    - Case-insensitive match
    - Ignores extra internal/leading/trailing whitespace
    - Falls back to partial substring match when no exact normalized match

    Returns: (store_ids or None, unmatched_names)
    """
    try:
        import pandas as pd  # ensure available if this module is imported elsewhere first
    except Exception:
        pd = None

    if store_df is None or getattr(store_df, "empty", True):
        return None, list(selected_stores or [])

    # Build normalized name column without mutating original object for callers
    df = store_df.copy()
    df['__name_norm'] = df['name'].astype(str).map(_normalize_store_name)

    # Map of normalized name -> list of IDs (in case of duplicates)
    exact_map = df.groupby('__name_norm')['id'].apply(list).to_dict()

    store_ids = []
    unmatched = []

    for raw_name in selected_stores or []:
        norm = _normalize_store_name(raw_name)

        # 1) Exact normalized match
        ids = exact_map.get(norm)
        if ids:
            store_ids.extend(ids)
            continue

        # 2) Partial contains match on normalized names
        partial_df = df[df['__name_norm'].str.contains(norm, na=False)] if norm else df.iloc[0:0]
        if not partial_df.empty:
            unique_norms = partial_df['__name_norm'].unique()
            if len(unique_norms) == 1:
                store_ids.extend(partial_df['id'].tolist())
            else:
                # Ambiguous partial match
                if debug:
                    import streamlit as st
                    st.warning(f"Store '{raw_name}' matched multiple stores: {list(unique_norms)}. Please refine selection.")
                unmatched.append(raw_name)
        else:
            unmatched.append(raw_name)

    # Deduplicate while preserving order
    if store_ids:
        seen = set()
        deduped = []
        for _id in store_ids:
            if _id not in seen:
                seen.add(_id)
                deduped.append(_id)
        store_ids = deduped
    else:
        store_ids = None

    return store_ids, unmatched

def render_dashboard():
    """Renders the BI dashboard with tabbed navigation separating data visuals and AI hub."""
    # Import mobile dashboard components
    try:
        from supabot.ui.components.mobile_dashboard import MobileDashboard
        mobile_available = True
    except ImportError:
        mobile_available = False
        st.warning("Mobile components not available. Using desktop layout only.")
    
    # Check if mobile components are available and render responsive dashboard
    if mobile_available:
        render_responsive_dashboard()
    else:
        render_legacy_dashboard()

def render_responsive_dashboard():
    """Renders the responsive dashboard with mobile optimization."""
    from supabot.ui.components.mobile_dashboard import MobileDashboard
    
    # Get dashboard data
    dashboard_data = get_dashboard_data()
    if not dashboard_data:
        st.error("Failed to load dashboard data")
        return
    
    metrics, sales_df, sales_cat_df, inv_cat_df, top_change_df, cat_change_df, time_filter, selected_stores = dashboard_data
    
    # Render responsive dashboard
    MobileDashboard.render_responsive_dashboard(
        metrics=metrics,
        sales_df=sales_df,
        sales_cat_df=sales_cat_df,
        inv_cat_df=inv_cat_df,
        top_change_df=top_change_df,
        cat_change_df=cat_change_df,
        time_filter=time_filter,
        selected_stores=selected_stores
    )

def get_dashboard_data():
    """Get all dashboard data for responsive rendering."""
    # --- 1. Time and Store Selectors ---
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
    with filter_col1:
        time_options = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
        time_index = time_options.index(st.session_state.dashboard_time_filter) if st.session_state.dashboard_time_filter in time_options else 1
        st.session_state.dashboard_time_filter = st.radio(
            "Select Time Period:", options=time_options, index=time_index,
            horizontal=True, key="time_filter_selector"
        )
        
        # Custom date range (only show if Custom is selected)
        if st.session_state.dashboard_time_filter == "Custom":
            st.write("**Custom Date Range:**")
            
            custom_start = st.date_input(
                "From Date", 
                value=st.session_state.custom_start_date,
                key="custom_date_start_widget"
            )
            
            custom_end = st.date_input(
                "To Date", 
                value=st.session_state.custom_end_date,
                key="custom_date_end_widget"
            )
            
            # Update session state
            st.session_state.custom_start_date = custom_start
            st.session_state.custom_end_date = custom_end
            
            # Validate dates
            if custom_start > custom_end:
                st.error("Start date cannot be after end date!")
                return None
    
    with filter_col2:
        # Middle column - can be used for additional filters or left empty
        pass

    with filter_col3:
        store_df = get_store_list()
        store_list = store_df['name'].tolist()
        all_stores_option = "All Stores"
        
        st.session_state.dashboard_store_filter = st.multiselect(
            "Select Store(s):",
            options=[all_stores_option] + store_list,
            default=st.session_state.dashboard_store_filter
        )

    # --- Process Filters ---
    time_filter = st.session_state.dashboard_time_filter
    selected_stores = st.session_state.dashboard_store_filter

    # Process store filter with robust resolution
    store_filter_ids = None
    if selected_stores and "All Stores" not in selected_stores and not store_df.empty:
        debug_mode = st.session_state.get('debug_mode', False)
        store_filter_ids, unmatched_names = resolve_store_ids(store_df, selected_stores, debug=debug_mode)
        # Surface warnings for unmatched selections
        for nm in unmatched_names:
            st.warning(f"Store '{nm}' not found or ambiguous in database after normalization")
        if store_filter_ids is None:
            st.warning("No valid stores found in selection")
    
    # Get all dashboard data
    try:
        with st.spinner("Loading dashboard data..."):
            # Get metrics
            metrics = get_dashboard_metrics(time_filter, store_filter_ids)
            
            # Get sales trend data
            sales_df = get_sales_trend_data(time_filter, store_filter_ids)
            
            # Get category data
            sales_cat_df = get_sales_by_category_data(time_filter, store_filter_ids)
            inv_cat_df = get_inventory_by_category_data(store_filter_ids)
            
            # Get top products and categories with change data
            top_change_df = get_top_products_with_change(time_filter, store_filter_ids)
            cat_change_df = get_categories_with_change(time_filter, store_filter_ids)
            
            return (metrics, sales_df, sales_cat_df, inv_cat_df, top_change_df, cat_change_df, time_filter, selected_stores)
    
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        return None

def get_sales_trend_data(time_filter: str, store_filter_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Get sales trend data for charts."""
    try:
        date_range = get_intelligent_date_range(time_filter)
        start_date = date_range['start_date']
        end_date = date_range['end_date']
        
        # Build SQL query
        sql = """
        SELECT 
            DATE(t.transaction_date) as date,
            SUM(t.total) as total_revenue
        FROM transactions t
        WHERE t.transaction_date BETWEEN %s AND %s
        AND LOWER(t.transaction_type) = 'sale'
        """
        
        params = [start_date, end_date]
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
        GROUP BY DATE(t.transaction_date)
        ORDER BY date
        """
        
        result = execute_query_for_dashboard(sql, params)
        return result if result is not None else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting sales trend data: {str(e)}")
        return pd.DataFrame()

def get_sales_by_category_data(time_filter: str, store_filter_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Get sales by category data."""
    try:
        date_range = get_intelligent_date_range(time_filter)
        start_date = date_range['start_date']
        end_date = date_range['end_date']
        
        sql = """
        SELECT 
            p.category,
            SUM(t.total) as total_revenue
        FROM transactions t
        JOIN products p ON t.product_id = p.id
        WHERE t.transaction_date BETWEEN %s AND %s
        AND LOWER(t.transaction_type) = 'sale'
        """
        
        params = [start_date, end_date]
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
        GROUP BY p.category
        ORDER BY total_revenue DESC
        """
        
        result = execute_query_for_dashboard(sql, params)
        return result if result is not None else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting sales by category data: {str(e)}")
        return pd.DataFrame()

def get_inventory_by_category_data(store_filter_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Get inventory by category data."""
    try:
        sql = """
        SELECT 
            p.category,
            SUM(i.quantity * p.price) as total_inventory_value
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        WHERE i.quantity > 0
        """
        
        params = []
        
        if store_filter_ids:
            sql += " AND i.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
        GROUP BY p.category
        ORDER BY total_inventory_value DESC
        """
        
        result = execute_query_for_dashboard(sql, params)
        return result if result is not None else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting inventory by category data: {str(e)}")
        return pd.DataFrame()

def get_top_products_with_change(time_filter: str, store_filter_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Get top products with percentage change data."""
    try:
        date_range = get_intelligent_date_range(time_filter)
        start_date = date_range['start_date']
        end_date = date_range['end_date']
        
        # Get previous period dates
        prev_dates = get_previous_period_dates(start_date, end_date, time_filter)
        prev_start = prev_dates['start_date']
        prev_end = prev_dates['end_date']
        
        sql = """
        WITH current_period AS (
            SELECT 
                p.product_name,
                SUM(t.total) as total_revenue
            FROM transactions t
            JOIN products p ON t.product_id = p.id
            WHERE t.transaction_date BETWEEN %s AND %s
            AND LOWER(t.transaction_type) = 'sale'
        """
        
        params = [start_date, end_date]
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
            GROUP BY p.product_name
        ),
        previous_period AS (
            SELECT 
                p.product_name,
                SUM(t.total) as total_revenue
            FROM transactions t
            JOIN products p ON t.product_id = p.id
            WHERE t.transaction_date BETWEEN %s AND %s
            AND LOWER(t.transaction_type) = 'sale'
        """
        
        params.extend([prev_start, prev_end])
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
            GROUP BY p.product_name
        )
        SELECT 
            COALESCE(cp.product_name, pp.product_name) as product_name,
            COALESCE(cp.total_revenue, 0) as total_revenue,
            COALESCE(pp.total_revenue, 0) as prev_revenue,
            CASE 
                WHEN COALESCE(pp.total_revenue, 0) = 0 THEN NULL
                ELSE ((COALESCE(cp.total_revenue, 0) - COALESCE(pp.total_revenue, 0)) / COALESCE(pp.total_revenue, 0)) * 100
            END as pct_change
        FROM current_period cp
        FULL OUTER JOIN previous_period pp ON cp.product_name = pp.product_name
        ORDER BY COALESCE(cp.total_revenue, 0) DESC
        LIMIT 10
        """
        
        result = execute_query_for_dashboard(sql, params)
        return result if result is not None else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting top products with change data: {str(e)}")
        return pd.DataFrame()

def get_categories_with_change(time_filter: str, store_filter_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Get categories with percentage change data."""
    try:
        date_range = get_intelligent_date_range(time_filter)
        start_date = date_range['start_date']
        end_date = date_range['end_date']
        
        # Get previous period dates
        prev_dates = get_previous_period_dates(start_date, end_date, time_filter)
        prev_start = prev_dates['start_date']
        prev_end = prev_dates['end_date']
        
        sql = """
        WITH current_period AS (
            SELECT 
                p.category,
                SUM(t.total) as total_revenue
            FROM transactions t
            JOIN products p ON t.product_id = p.id
            WHERE t.transaction_date BETWEEN %s AND %s
            AND LOWER(t.transaction_type) = 'sale'
        """
        
        params = [start_date, end_date]
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
            GROUP BY p.category
        ),
        previous_period AS (
            SELECT 
                p.category,
                SUM(t.total) as total_revenue
            FROM transactions t
            JOIN products p ON t.product_id = p.id
            WHERE t.transaction_date BETWEEN %s AND %s
            AND LOWER(t.transaction_type) = 'sale'
        """
        
        params.extend([prev_start, prev_end])
        
        if store_filter_ids:
            sql += " AND t.store_id = ANY(%s)"
            params.append(store_filter_ids)
        
        sql += """
            GROUP BY p.category
        )
        SELECT 
            COALESCE(cp.category, pp.category) as category,
            COALESCE(cp.total_revenue, 0) as total_revenue,
            COALESCE(pp.total_revenue, 0) as prev_revenue,
            CASE 
                WHEN COALESCE(pp.total_revenue, 0) = 0 THEN NULL
                ELSE ((COALESCE(cp.total_revenue, 0) - COALESCE(pp.total_revenue, 0)) / COALESCE(pp.total_revenue, 0)) * 100
            END as pct_change
        FROM current_period cp
        FULL OUTER JOIN previous_period pp ON cp.category = pp.category
        ORDER BY COALESCE(cp.total_revenue, 0) DESC
        """
        
        result = execute_query_for_dashboard(sql, params)
        return result if result is not None else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error getting categories with change data: {str(e)}")
        return pd.DataFrame()

def render_legacy_dashboard():
    """Renders the legacy desktop-only dashboard."""
    st.markdown('<div class="main-header"><h1>📊 SupaBot Ultimate BI Dashboard</h1><p>Real-time Business Intelligence powered by AI</p></div>', unsafe_allow_html=True)

    dashboard_tab = st.container()

    with dashboard_tab:
        # --- 1. Time and Store Selectors ---
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
        with filter_col1:
            time_options = ["1D", "7D", "1M", "6M", "1Y", "Custom"]
            time_index = time_options.index(st.session_state.dashboard_time_filter) if st.session_state.dashboard_time_filter in time_options else 1
            st.session_state.dashboard_time_filter = st.radio(
                "Select Time Period:", options=time_options, index=time_index,
                horizontal=True, key="time_filter_selector"
            )
            
            # Custom date range (only show if Custom is selected)
            if st.session_state.dashboard_time_filter == "Custom":
                st.write("**Custom Date Range:**")
                
                custom_start = st.date_input(
                    "From Date", 
                    value=st.session_state.custom_start_date,
                    key="custom_date_start_widget"
                )
                
                custom_end = st.date_input(
                    "To Date", 
                    value=st.session_state.custom_end_date,
                    key="custom_date_end_widget"
                )
                
                # Update session state
                st.session_state.custom_start_date = custom_start
                st.session_state.custom_end_date = custom_end
                
                # Validate dates
                if custom_start > custom_end:
                    st.error("Start date cannot be after end date!")
                    return
        
        with filter_col2:
            # Middle column - can be used for additional filters or left empty
            pass

        with filter_col3:
            store_df = get_store_list()
            store_list = store_df['name'].tolist()
            all_stores_option = "All Stores"
            
            st.session_state.dashboard_store_filter = st.multiselect(
                "Select Store(s):",
                options=[all_stores_option] + store_list,
                default=st.session_state.dashboard_store_filter
            )

        # --- Process Filters ---
        time_filter = st.session_state.dashboard_time_filter
        selected_stores = st.session_state.dashboard_store_filter

        # Process store filter with robust resolution
        store_filter_ids = None
        if selected_stores and "All Stores" not in selected_stores and not store_df.empty:
            debug_mode = st.session_state.get('debug_mode', False)
            store_filter_ids, unmatched_names = resolve_store_ids(store_df, selected_stores, debug=debug_mode)
            # Surface warnings for unmatched selections
            for nm in unmatched_names:
                st.warning(f"Store '{nm}' not found or ambiguous in database after normalization")
            if store_filter_ids is None:
                st.warning("No valid stores found in selection")
        
        # Debug: Show what we're working with (only in debug mode)
        if st.session_state.get('debug_mode', False):
            st.info(f"🔍 Debug: Selected stores: {selected_stores}")
            st.info(f"🔍 Debug: Store filter IDs: {store_filter_ids}")
            st.info(f"🔍 Debug: Store dataframe: {store_df.to_dict('records') if not store_df.empty else 'Empty'}")
            
            # Test database connection with a simple query
            try:
                test_sql = "SELECT COUNT(*) as total_transactions FROM transactions WHERE LOWER(transaction_type) = 'sale'"
                test_result = execute_query_for_dashboard(test_sql)
                if test_result is not None and not test_result.empty:
                    st.success(f"🔍 Database Test: Found {test_result.iloc[0]['total_transactions']} total sales transactions")
                else:
                    st.warning("🔍 Database Test: No results from simple query")
            except Exception as test_e:
                st.error(f"🔍 Database Test Error: {test_e}")
            
            # Test specific store and time period data
            if store_filter_ids and time_filter != "Custom":
                try:
                    date_range = get_intelligent_date_range(time_filter)
                    start_date = date_range['start_date']
                    end_date = date_range['end_date']
                    
                    test_store_sql = f"""
                    SELECT 
                        COUNT(*) as transaction_count,
                        COUNT(DISTINCT t.store_id) as store_count,
                        COALESCE(SUM(t.total), 0) as total_sales
                    FROM transactions t
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND COALESCE(t.is_cancelled, false) = false
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s
                    AND t.store_id = ANY(%s)
                    """
                    
                    test_store_result = execute_query_for_dashboard(test_store_sql, params=[str(start_date), str(end_date), store_filter_ids])
                    if test_store_result is not None and not test_store_result.empty:
                        row = test_store_result.iloc[0]
                        st.info(f"🔍 Store Test: Found {row['transaction_count']} transactions, {row['store_count']} stores, ₱{row['total_sales']:,.0f} sales for {time_filter} period")
                    else:
                        st.warning("🔍 Store Test: No results for specific stores and time period")
                except Exception as store_test_e:
                    st.error(f"🔍 Store Test Error: {store_test_e}")
        
        # Get custom dates if applicable
        custom_start = st.session_state.custom_start_date if time_filter == "Custom" else None
        custom_end = st.session_state.custom_end_date if time_filter == "Custom" else None
        
        # Clear insights when filters change to ensure fresh insights for new filter combination
        current_filter_key = f"{time_filter}_{store_filter_ids or 'all'}_{custom_start or 'none'}_{custom_end or 'none'}"
        if "last_filter_key" not in st.session_state:
            st.session_state.last_filter_key = current_filter_key
        elif st.session_state.last_filter_key != current_filter_key:
            # Filters changed, clear all insights
            st.session_state.ai_insights = None
            st.session_state.category_insights = None
            st.session_state.business_highlights = None
            st.session_state.last_filter_key = current_filter_key

        # --- Data Fetching with Working Functions ---
        with st.spinner(f"Loading dashboard data for the last {time_filter}..."):
            # Debug information (only show if debug mode is enabled)
            debug_mode = st.session_state.get('debug_mode', False)
            
            # Get metrics with proper error handling
            try:
                metrics = get_dashboard_metrics(time_filter, store_filter_ids, custom_start, custom_end)
                
                if not metrics or all(v == 0 for v in [metrics.get('current_sales', 0), metrics.get('current_transactions', 0)]):
                    st.warning("No data found for selected filters")
                    return
                else:
                    # Data loaded successfully (no message displayed)
                    total_sales = metrics.get('current_sales', 0)
                    total_transactions = metrics.get('current_transactions', 0)
                    
            except Exception as e:
                st.error(f"Error loading metrics: {e}")
                return
            
            try:
                sales_cat_df = get_sales_by_category_pie(time_filter, store_filter_ids)
                inv_cat_df = get_inventory_by_category_pie(store_filter_ids)
                top_change_df = get_top_products_with_change(time_filter, store_filter_ids)
                cat_change_df = get_categories_with_change(time_filter, store_filter_ids)
                daily_trend_df = get_daily_trend(days={"1D":1, "7D":7, "1M":30, "6M":180, "1Y":365}.get(time_filter, 7), store_ids=store_filter_ids)
                
            except Exception as e:
                st.error(f"❌ Error loading dashboard data: {e}")
                # Provide fallback empty data
                sales_cat_df = pd.DataFrame()
                inv_cat_df = pd.DataFrame()
                top_change_df = pd.DataFrame()
                cat_change_df = pd.DataFrame()
                daily_trend_df = pd.DataFrame()
        # End of with st.spinner block

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- 2. KPI Hero Section (Fixed Percentages) ---
        st.subheader("🚀 Key Performance Indicators")

        def format_percentage_change(current, previous):
            """Calculate and format percentage change"""
            if previous is None or previous == 0:
                if current > 0:
                    return "New data ↗"
                return "No data"
            
            if current is None:
                current = 0
            
            # CRITICAL FIX: Use previous instead of abs(previous) for correct percentage calculation
            change = ((current - previous) / previous) * 100
            
            # Debug the calculation
            print(f"🔍 PERCENTAGE DEBUG:")
            print(f"   Current: {current}")
            print(f"   Previous: {previous}")
            print(f"   Change: {current - previous}")
            print(f"   Percentage: {change:.1f}%")
            print(f"   Is positive? {change > 0}")
            
            if abs(change) > 999:
                return ">999% ↗" if change > 0 else ">999% ↘"
            
            if abs(change) < 0.1:
                return "→ 0.0%"
            
            arrow = "↗" if change > 0 else "↘"
            return f"{change:+.1f}% {arrow}"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            sales = metrics.get('current_sales', 0)
            prev_sales = metrics.get('prev_sales', 0)
            delta = format_percentage_change(sales, prev_sales)
            st.metric("Total Sales", f"₱{sales:,.0f}", delta)

        with kpi2:
            profit = metrics.get('current_profit', 0)
            prev_profit = metrics.get('prev_profit', 0)
            delta = format_percentage_change(profit, prev_profit)
            st.metric("Total Profit", f"₱{profit:,.0f}", delta)

        with kpi3:
            transactions = metrics.get('current_transactions', 0)
            prev_transactions = metrics.get('prev_transactions', 0)
            delta = format_percentage_change(transactions, prev_transactions)
            st.metric("Transactions", f"{transactions:,}", delta)

        with kpi4:
            avg_value = metrics.get('avg_transaction_value', 0)
            prev_avg_value = metrics.get('prev_avg_transaction_value', 0)
            delta = format_percentage_change(avg_value, prev_avg_value)
            st.metric("Avg Transaction Value", f"₱{avg_value:,.0f}", delta)

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- Debug/Raw Data Section for KPIs ---
        with st.expander("🔍 Raw KPI Data (Debug)", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Period Data")
                st.write(f"**Period:** {time_filter}")
                st.write(f"**Total Sales:** ₱{metrics.get('current_sales', 0):,.2f}")
                st.write(f"**Total Profit:** ₱{metrics.get('current_profit', 0):,.2f}")
                st.write(f"**Transactions:** {metrics.get('current_transactions', 0):,}")
                st.write(f"**Avg Transaction:** ₱{metrics.get('avg_transaction_value', 0):,.2f}")
                
                # Add date range info
                date_range = get_intelligent_date_range(time_filter)
                current_start_date = date_range['start_date']
                current_end_date = date_range['end_date']
                st.write(f"**Date Range:** {current_start_date} to {current_end_date}")
                
            with col2:
                st.subheader("Previous Period Data")
                st.write(f"**Total Sales:** ₱{metrics.get('prev_sales', 0):,.2f}")
                st.write(f"**Total Profit:** ₱{metrics.get('prev_profit', 0):,.2f}")
                st.write(f"**Transactions:** {metrics.get('prev_transactions', 0):,}")
                st.write(f"**Avg Transaction:** ₱{metrics.get('prev_avg_transaction_value', 0):,.2f}")
                
                # Add date range info
                prev_dates = get_previous_period_dates(current_start_date, current_end_date, time_filter)
                previous_start_date = prev_dates['start_date']
                previous_end_date = prev_dates['end_date']
                st.write(f"**Date Range:** {previous_start_date} to {previous_end_date}")
            
            # Calculation Details
            st.subheader("Calculation Details")
            current_sales = metrics.get('current_sales', 0)
            previous_sales = metrics.get('prev_sales', 0)
            current_profit = metrics.get('current_profit', 0)
            previous_profit = metrics.get('prev_profit', 0)
            current_transactions = metrics.get('current_transactions', 0)
            previous_transactions = metrics.get('prev_transactions', 0)
            
            st.write(f"**Sales Change:** ₱{current_sales - previous_sales:,.2f}")
            sales_growth = ((current_sales - previous_sales) / max(previous_sales, 1)) * 100 if previous_sales > 0 else 100.0 if current_sales > 0 else 0.0
            st.write(f"**Sales Change %:** {sales_growth:.1f}%")
            st.write(f"**Profit Change:** ₱{current_profit - previous_profit:,.2f}")
            profit_growth = ((current_profit - previous_profit) / max(previous_profit, 1)) * 100 if previous_profit > 0 else 100.0 if current_profit > 0 else 0.0
            st.write(f"**Profit Change %:** {profit_growth:.1f}%")
            st.write(f"**Transaction Change:** {current_transactions - previous_transactions:,}")
            trans_growth = ((current_transactions - previous_transactions) / max(previous_transactions, 1)) * 100 if previous_transactions > 0 else 100.0 if current_transactions > 0 else 0.0
            st.write(f"**Transaction Change %:** {trans_growth:.1f}%")
            
            # SQL Queries Display
            st.subheader("SQL Queries Used")
            with st.expander("View SQL Queries"):
                # Get the SQL queries from the get_dashboard_metrics function
                current_period_sql = f"""
                WITH current_period AS (
                    SELECT 
                        COALESCE(SUM(t.total), 0) as sales,
                        COALESCE(SUM(t.total) * 0.3, 0) as profit,
                        COALESCE(COUNT(DISTINCT t.ref_id), 0) as transactions,
                        COALESCE(AVG(t.total), 0) as avg_transaction_value
                    FROM transactions t
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND COALESCE(t.is_cancelled, false) = false
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= '{current_start_date}'
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= '{current_end_date}'
                    {f"AND t.store_id = ANY({store_filter_ids})" if store_filter_ids else ""}
                )
                """
                
                previous_period_sql = f"""
                WITH previous_period AS (
                    SELECT 
                        COALESCE(SUM(t.total), 0) as prev_sales,
                        COALESCE(SUM(t.total) * 0.3, 0) as prev_profit,
                        COALESCE(COUNT(DISTINCT t.ref_id), 0) as prev_transactions,
                        COALESCE(AVG(t.total), 0) as prev_avg_transaction_value
                    FROM transactions t
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND COALESCE(t.is_cancelled, false) = false
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= '{previous_start_date}'
                    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= '{previous_end_date}'
                    {f"AND t.store_id = ANY({store_filter_ids})" if store_filter_ids else ""}
                )
                """
                
                st.code(f"""
Current Period Query:
{current_period_sql}

Previous Period Query: 
{previous_period_sql}
                """, language="sql")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- 3. Main Grid Layout (AI removed from this tab) ---
        left_col, center_col = st.columns([1, 1], gap="large")

        # --- LEFT COLUMN - PRODUCT & CATEGORY ANALYTICS ---
        with left_col:
            pie_col1, pie_col2 = st.columns(2)
            
            with pie_col1:
                with st.container(border=True):
                    st.markdown("##### 💰 Sales by Category")
                    if not sales_cat_df.empty:
                        df_plot = sales_cat_df.head(10).copy()
                        df_plot['category_canonical'] = df_plot['category'].astype(str).apply(canonicalize_category_label)
                        df_plot = df_plot.groupby('category_canonical', as_index=False)['total_revenue'].sum()
                        fig = px.pie(
                            df_plot,
                            values='total_revenue',
                            names='category_canonical',
                            color='category_canonical',
                            hole=0.4,
                            color_discrete_map=get_fixed_category_color_map()
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No sales data available for selected period/stores.")
            
            with pie_col2:
                with st.container(border=True):
                    st.markdown("##### 📦 Inventory by Category")
                    if not inv_cat_df.empty:
                        df_plot = inv_cat_df.copy()
                        df_plot['category_canonical'] = df_plot['category'].astype(str).apply(canonicalize_category_label)
                        df_plot = df_plot.groupby('category_canonical', as_index=False)['total_inventory_value'].sum()
                        fig = px.pie(
                            df_plot,
                            values='total_inventory_value',
                            names='category_canonical',
                            color='category_canonical',
                            hole=0.4,
                            color_discrete_map=get_fixed_category_color_map()
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No inventory data available for selected stores.")

            # Top 10 Products with % change vs previous period
            with st.container(border=True):
                st.markdown("##### 🏆 Top 10 Products (with % change)")
                if not top_change_df.empty:
                    df_disp = top_change_df.copy()
                    # Build display columns
                    df_disp.rename(columns={'product_name': 'Product', 'total_revenue': 'Sales'}, inplace=True)
                    def _pct_cell(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return "New"
                        arrow = '▲' if x >= 0 else '▼'
                        return f"{arrow} {abs(x):.1f}%"
                    df_disp['Δ %'] = df_disp['pct_change'].apply(_pct_cell)
                    show_df = df_disp[['Product', 'Sales', 'Δ %']].copy()
                    # Styling
                    def style_change(col):
                        styles = []
                        for val in col:
                            if isinstance(val, str) and val.startswith('▲'):
                                styles.append('color: #00c853; font-weight: 600')
                            elif isinstance(val, str) and val.startswith('▼'):
                                styles.append('color: #ff5252; font-weight: 600')
                            else:
                                styles.append('color: #aaaaaa')
                        return styles
                    styled = (show_df.style
                        .format({'Sales': '₱{:,.0f}'})
                        .apply(style_change, subset=['Δ %']))
                    # Render Styler via st.write to avoid KeyError from st.dataframe on Styler objects
                    st.write(styled)
                else:
                    st.info("No product data available for selected period/stores.")

            # Categories Ranked, stacked under Top Products in the same (left) column
            with st.container(border=True):
                st.markdown("##### 🗂️ Categories Ranked (with % change)")
                if not cat_change_df.empty:
                    dfc = cat_change_df.copy()
                    dfc.rename(columns={'category': 'Category', 'total_revenue': 'Sales'}, inplace=True)
                    def _pct_cell2(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return "New"
                        arrow = '▲' if x >= 0 else '▼'
                        return f"{arrow} {abs(x):.1f}%"
                    dfc['Δ %'] = dfc['pct_change'].apply(_pct_cell2)
                    show_c = dfc[['Category', 'Sales', 'Δ %']].copy()
                    def style_change2(col):
                        styles = []
                        for val in col:
                            if isinstance(val, str) and val.startswith('▲'):
                                styles.append('color: #00c853; font-weight: 600')
                            elif isinstance(val, str) and val.startswith('▼'):
                                styles.append('color: #ff5252; font-weight: 600')
                            else:
                                styles.append('color: #aaaaaa')
                        return styles
                    styled_c = (show_c.style
                        .format({'Sales': '₱{:,.0f}'})
                        .apply(style_change2, subset=['Δ %']))
                    # Render Styler via st.write to avoid KeyError from st.dataframe on Styler objects
                    st.write(styled_c)
                else:
                    st.info("No category data available for selected period/stores.")

        # --- CENTER COLUMN - STORE & SALES ANALYTICS ---
        with center_col:
            with st.container(border=True):
                st.markdown("##### 🏪 Store Performance")
                store_performance = get_store_performance_with_comparison(time_filter, store_filter_ids if store_filter_ids else None)
                if not store_performance.empty:
                    # Fixed store colors to match Chart View
                    store_color_map = {
                        'Rockwell': '#E74C3C',
                        'Greenhills': '#2ECC71',
                        'Magnolia': '#F1C40F',
                        'North Edsa': '#3498DB',
                        'Fairview': '#9B59B6'
                    }
                    
                    # Create the bar chart
                    fig = px.bar(
                        store_performance.head(5),
                        x='store_name', y='total_sales',
                        title='Top Stores by Sales',
                        color='store_name',
                        color_discrete_map=store_color_map
                    )
                    
                    # Add percentage change annotations on top of each bar BEFORE displaying
                    for idx, (_, store) in enumerate(store_performance.head(5).iterrows()):
                        current_sales = store['total_sales']
                        pct_change = store.get('pct_change')
                        
                        # Format percentage change with arrow
                        if pct_change is not None:
                            if pct_change > 0:
                                annotation_text = f"↗ +{pct_change:.1f}%"
                                annotation_color = "#2ECC71"  # Green for positive
                            elif pct_change < 0:
                                annotation_text = f"↘ {pct_change:.1f}%"
                                annotation_color = "#E74C3C"  # Red for negative
                            else:
                                annotation_text = "→ 0.0%"
                                annotation_color = "#95A5A6"  # Gray for no change
                        else:
                            annotation_text = "New ↗"
                            annotation_color = "#F39C12"  # Orange for new data
                        
                        # Add annotation on top of each bar
                        fig.add_annotation(
                            x=store['store_name'],
                            y=current_sales,
                            text=annotation_text,
                            showarrow=False,
                            font=dict(color=annotation_color, size=11, family="Arial Black"),
                            yshift=15,  # Position above the bar
                            bgcolor="rgba(0,0,0,0)",  # Transparent background
                            bordercolor="rgba(0,0,0,0)",  # No border
                            borderwidth=0
                        )
                    
                    # Update layout after adding annotations
                    fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), template="plotly_dark", 
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    fig.update_yaxes(tickprefix='₱', separatethousands=True)
                    fig.update_xaxes(title_text="")  # Remove x-axis label
                    
                    # Display the chart with annotations
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No store performance data available.")

            with st.container(border=True):
                st.markdown("##### 📈 Sales Trend Analysis")
                if not daily_trend_df.empty:
                    fig = px.area(daily_trend_df, x='date', y='daily_sales', 
                                 title=f"Sales Trend for Last {time_filter}")
                    fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), 
                                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                                    plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sales trend data for this period.")
            
            # New Section: Average Sales Per Hour (across selected window)
            with st.container(border=True):
                st.markdown("##### 🕒 Average Sales Per Hour")
                try:
                    df = get_avg_sales_per_hour(time_filter, store_filter_ids if store_filter_ids else None)
                    if df is not None and len(df) > 0:
                        # Display only hours with sales
                        df_plot = df[df["avg_sales_per_hour"] > 0].copy()
                        if df_plot.empty:
                            st.info("No hourly sales were recorded in this period.")
                        else:
                            fig = go.Figure()
                            fig.add_bar(name="Avg per Hour", x=df_plot["hour_label"], y=df_plot["avg_sales_per_hour"], marker_color="#2E86DE")
                            fig.update_layout(template="plotly_dark", height=380,
                                              margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(tickangle=-45),
                                              showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            fig.update_yaxes(tickprefix='₱', separatethousands=True)
                            fig.update_traces(hovertemplate='%{x}: ₱%{y:,.0f}<extra></extra>')
                            st.plotly_chart(fig, use_container_width=True)
                            # Caption uses full df to reflect daily total across all hours
                            st.caption(
                                f"Average daily total (sum of hourly avgs): ₱{float(df['avg_sales_per_hour'].sum()):,.0f}"
                            )
                    else:
                        st.info("No data for the selected period/stores.")
                except Exception:
                    st.info("No data for the selected period/stores.")

            # Center column keeps Store Performance and Sales Trend only

        # (Removed bottom Sales Trend Analysis and Inventory by Category sections)
        # --- AI Intelligence Section (Fullscreen) ---
        st.markdown("---")
        st.markdown("### 🧠 AI Business Intelligence")
        st.markdown("Generate comprehensive 4-week business intelligence reports powered by AI.")
        
        # AI Intelligence generation
        if st.button("🔍 Generate AI Intelligence Report", type="primary", use_container_width=True, key="dashboard_ai_generate"):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Fetch data
                status_text.text("📊 Fetching weekly sales data...")
                progress_bar.progress(20)
                overall_weekly = get_overall_weekly_data()
                
                status_text.text("🏪 Analyzing store performance...")
                progress_bar.progress(35)
                stores_weekly = get_stores_weekly_data()
                
                status_text.text("📦 Processing product data...")
                progress_bar.progress(50)
                products_weekly = get_products_weekly_data()
                
                status_text.text("📊 Computing category insights...")
                progress_bar.progress(65)
                category_weekly = get_category_weekly_data()
                
                status_text.text("⏰ Analyzing time patterns...")
                progress_bar.progress(75)
                time_patterns = get_time_patterns_data()
                
                status_text.text("🤖 Generating AI insights...")
                progress_bar.progress(85)
                
                if overall_weekly:
                    summary = generate_ai_intelligence_summary()
                    progress_bar.progress(100)
                    status_text.text("✅ Report generated successfully!")
                    st.session_state['dashboard_intelligence_summary'] = summary
                    
                    # Clear progress indicators after 2 seconds
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Unable to fetch data for analysis.")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error generating report: {e}")
        
        # Display results
        if 'dashboard_intelligence_summary' in st.session_state:
            st.markdown("---")
            st.markdown("### 📋 AI Business Intelligence Report")
            st.markdown(st.session_state['dashboard_intelligence_summary'])
            
            # Download option
            st.download_button(
                label="📥 Download Report",
                data=st.session_state['dashboard_intelligence_summary'],
                file_name=f"dashboard_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                key="dashboard_download"
            )
        else:
            st.info("👆 Click 'Generate AI Intelligence Report' to create your comprehensive 4-week business analysis.")
            
            # Show what the report includes
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📈 Performance Analysis:**
                - Store performance rankings & movers
                - Product winners & losers
                - Category momentum analysis
                """)
            
            with col2:
                st.markdown("""
                **🎯 Actionable Insights:**
                - Time pattern analysis (best/worst days & hours)
                - Per-week spotlight with standout performers
                - Business risks & opportunities identification
                - Strategic action plan with impact estimates
                """)
            
        
        

# --- MODIFICATION END ---

# Enhanced Chat Page with Assistant
def render_chat():
    st.markdown('<div class="main-header"><h1>🧠 SupaBot AI Assistant</h1><p>Ask ANYTHING about your data - I learn from your feedback!</p></div>', unsafe_allow_html=True)

    if st.session_state.schema_info is None:
        with st.spinner("🔍 Learning about your database..."):
            st.session_state.schema_info = get_database_schema()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">💭 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            if message.get("interpretation"): 
                st.markdown(f'<div class="ai-message">{message["interpretation"]}</div>', unsafe_allow_html=True)
            
            if message.get("sql"):
                with st.expander("🔍 SQL Query & Training", expanded=False):
                    st.code(message["sql"], language="sql")
                    
                    st.markdown("**Was this SQL correct?**")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button("✅ Correct", key=f"correct_{i}"):
                            explanation = st.text_input(
                                "Why was this correct?",
                                placeholder="e.g., Perfect grouping for hourly totals",
                                key=f"correct_explanation_{i}"
                            )
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                message["sql"], 
                                "correct", 
                                explanation
                            ):
                                st.success("✅ Saved as correct example!")
                            else:
                                st.error("❌ Failed to save")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Wrong", key=f"wrong_{i}"):
                            st.session_state[f"show_correction_{i}"] = True
                            st.rerun()
                    
                    # Show correction interface
                    if st.session_state.get(f"show_correction_{i}", False):
                        st.markdown("**Provide the correct SQL:**")
                        corrected_sql = st.text_area(
                            "Correct SQL:", 
                            value=message["sql"], 
                            height=100,
                            key=f"corrected_sql_{i}"
                        )
                        explanation = st.text_input(
                            "What was wrong?",
                            placeholder="e.g., Should group by store_id for per-store breakdown",
                            key=f"correction_explanation_{i}"
                        )
                        
                        if st.button("💾 Save Correction", key=f"save_correction_{i}"):
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                corrected_sql, 
                                "corrected", 
                                explanation
                            ):
                                st.success("✅ Correction saved!")
                            else:
                                st.error("❌ Failed to save correction")
                            st.session_state[f"show_correction_{i}"] = False
                            st.rerun()
            
            if message.get("results") is not None:
                results = message["results"]
                if isinstance(results, pd.DataFrame) and not results.empty:
                    # Apply dynamic formatting to the dataframe
                    column_config = get_column_config(results)
                    with st.expander(f"📊 View Data ({len(results)} rows)", expanded=False): 
                        st.dataframe(results, column_config=column_config, use_container_width=True, hide_index=True)
                    if message.get("chart"): 
                        st.plotly_chart(message["chart"], use_container_width=True)
            elif message.get("error"): 
                st.error(message["error"])

    if not st.session_state.messages:
        st.markdown("### 💡 Example Questions You Can Ask:")
        c1, c2 = st.columns(2)
        c1.markdown("**🎯 Enhanced Chart Examples:**")
        c1.markdown("- **Pie Chart**: 'Sales distribution by category'")
        c1.markdown("- **Treemap**: 'Product revenue hierarchy'")
        c1.markdown("- **Scatter Plot**: 'Revenue vs quantity relationship'")
        c1.markdown("- **Line Chart**: 'Sales trend over time'")
        
        c2.markdown("**📊 Business Questions:**")
        c2.markdown("- **Performance**: 'Top 10 products by revenue'")
        c2.markdown("- **Time Analysis**: 'Sales per hour total of all stores'")
        c2.markdown("- **Inventory**: 'Which products are almost out of stock?'")
        c2.markdown("- **Correlation**: 'Is there a relationship between price and sales?'")
        
        # Show training system status
        if len(st.session_state.training_system.training_data) > 0:
            st.info(f"🎓 Training System Active: {len(st.session_state.training_system.training_data)} examples learned")

    # Updated chat input with training system
    if prompt := st.chat_input("Ask me anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Thinking with training data..."):
            # Pass training system to SQL generation
            sql = generate_smart_sql(prompt, st.session_state.schema_info, st.session_state.training_system)
            if sql:
                with st.spinner("📊 Analyzing your data..."):
                    results = execute_query_for_assistant(sql)
                if results is not None:
                    with st.spinner("💡 Generating insights & smart visualization..."):
                        interpretation = interpret_results(prompt, results, sql)
                        chart = create_smart_visualization(results, prompt)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "question": prompt, 
                        "sql": sql, 
                        "results": results, 
                        "interpretation": interpretation, 
                        "chart": chart, 
                        "error": None
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "error": "I couldn't process that question. The query failed."
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "error": "I couldn't generate a query for that question. Try being more specific."
                })
        st.rerun()

# MODIFICATION START: Patched Chart View data fetching and rendering logic
@st.cache_data(ttl=300)
def get_filter_options():
    """Fetches all filter options (stores, categories, products) in one go."""
    stores_list = ["Rockwell", "Greenhills", "Magnolia", "North Edsa", "Fairview"]
    
    categories_sql = "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
    products_sql = "SELECT name FROM products ORDER BY name"
    
    categories = execute_query_for_dashboard(categories_sql)
    products = execute_query_for_dashboard(products_sql)
    
    return {
        "stores": stores_list,
        "categories": categories['category'].tolist() if categories is not None else [],
        "products": products['name'].tolist() if products is not None else []
    }

@st.cache_data(ttl=300)
def get_products_by_categories(categories: Optional[List[str]] = None):
    """Fetches product names, optionally filtered by categories."""
    if not categories:
        # If no categories are selected, return all products.
        return get_filter_options()["products"]

    sql = "SELECT name FROM products WHERE category = ANY(%s) ORDER BY name"
    params = (categories,)
        
    products_df = execute_query_for_dashboard(sql, params=params)
    return products_df['name'].tolist() if products_df is not None else []

def get_chart_view_data(time_range, metric_type, granularity, filters, store_filters):
    """
    Fetch aggregated data for the Chart View, plotting each selected item as a separate series.
    """
    if not filters:
        return pd.DataFrame()

    params = []
    
    # Time Range Filter - Updated to use intelligent date ranges
    time_filter_map = {"1d": "1D", "7d": "7D", "1m": "1M", "3m": "6M", "6m": "6M", "1y": "1Y"}
    time_filter = time_filter_map.get(time_range, "7D")
    
    # Get intelligent date ranges
    date_range = get_intelligent_date_range(time_filter)
    current_start = date_range['start_date']
    current_end = date_range['end_date']
    
    # If grouping by week, widen the window to capture full weeks so recent buckets aren't truncated
    if granularity == "Week":
        # Calculate days difference and add buffer for full weeks
        days_diff = (current_end - current_start).days
        widen_days = {1: 7, 7: 14, 30: 35, 90: 98, 180: 196, 365: 371}
        buffer_days = widen_days.get(days_diff, days_diff)
        # Adjust start date to include buffer
        from datetime import timedelta
        current_start = current_start - timedelta(days=buffer_days - days_diff)
    
    # Build time condition: for day/week/month granularities, use whole days
    time_condition = ""
    if granularity in ["Day", "Week", "Month"]:
        # Use intelligent date ranges
        time_condition = (
            "AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s\n"
            "AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s"
        )
        params.extend([str(current_start), str(current_end)])
    else:
        # For hour/minute granularity, use the same date range
        time_condition = (
            "AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s\n"
            "AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= %s"
        )
        params.extend([str(current_start), str(current_end)])

    # Time Granularity for GROUP BY
    granularity_map = {"Minute": "minute", "Hour": "hour", "Day": "day", "Week": "week", "Month": "month"}
    sql_granularity = granularity_map.get(granularity, "day")
    time_agg = f"DATE_TRUNC('{sql_granularity}', t.transaction_time AT TIME ZONE 'Asia/Manila')"

    # Initialize SQL parts
    store_filter_sql = ""
    metric_filter_sql = ""
    series_name_sql = ""
    base_name_sql = ""
    group_by_sql = ""
    metric_calculation_sql = ""
    category_select_sql = ""  # only populated for Products metric
    from_sql = ""  # conditional FROM/JOINs based on metric type
    having_sql = ""  # conditional HAVING based on metric column

    if metric_type == "Stores":
        # Transaction-level revenue: no item join
        series_name_sql = "s.name"
        base_name_sql = "s.name"
        metric_calculation_sql = "SUM(t.total) AS metric_value"
        group_by_sql = "GROUP BY 1, 2, s.name"
        metric_filter_sql = "AND s.name = ANY(%s)"
        params.append(filters)
        from_sql = (
            "FROM transactions t\n"
            "JOIN stores s ON t.store_id = s.id"
        )
        having_sql = "HAVING SUM(t.total) > 0"
    elif metric_type == "Avg Transaction Value":
        # ATV derived from transactions: no item join
        metric_calculation_sql = "SUM(t.total) / NULLIF(COUNT(DISTINCT t.ref_id),0) AS metric_value"
        series_name_sql = "s.name"
        base_name_sql = "s.name"
        group_by_sql = "GROUP BY 1, 2, s.name"
        metric_filter_sql = "AND s.name = ANY(%s)"
        params.append(filters)
        from_sql = (
            "FROM transactions t\n"
            "JOIN stores s ON t.store_id = s.id"
        )
        having_sql = "HAVING SUM(t.total) > 0"
    else: # Product Categories or Products (requires item-level granularity)
        if store_filters:
            store_filter_sql = "AND s.name = ANY(%s)"
            params.append(store_filters)
        
        if metric_type == "Product Categories":
            series_name_sql = "p.category || ' - ' || s.name"
            base_name_sql = "p.category"
            group_by_sql = "GROUP BY 1, 2, s.name"
            metric_filter_sql = "AND p.category = ANY(%s)"
            params.append(filters)
            metric_calculation_sql = "SUM(ti.item_total) AS metric_value"
            having_sql = "HAVING SUM(ti.item_total) > 0"
        elif metric_type == "Products":
            series_name_sql = "p.name || ' - ' || s.name"
            base_name_sql = "p.name"
            # Include category to support consistent per-category coloring for single-store views
            group_by_sql = "GROUP BY 1, 2, s.name, p.category"
            metric_filter_sql = "AND p.name = ANY(%s)"
            params.append(filters)
            category_select_sql = "p.category AS product_category,"
            metric_calculation_sql = "SUM(ti.item_total) AS metric_value"
            having_sql = "HAVING SUM(ti.item_total) > 0"
        else:
            return pd.DataFrame()

        # FROM/JOINs for item-level metrics
        from_sql = (
            "FROM transaction_items ti\n"
            "JOIN transactions t ON ti.transaction_ref_id = t.ref_id\n"
            "JOIN products p ON ti.product_id = p.id\n"
            "JOIN stores s ON t.store_id = s.id"
        )

    sql = f"""
    SELECT
        {time_agg} AS date,
        {base_name_sql} AS base_name,
        {series_name_sql} AS series_name,
        s.name AS store_name,
        {category_select_sql}
        {metric_calculation_sql}
    {from_sql}
    WHERE LOWER(t.transaction_type) = 'sale'
    AND COALESCE(t.is_cancelled, false) = false
    {time_condition}
    {store_filter_sql}
    {metric_filter_sql}
    {group_by_sql}
    {having_sql} -- Ensure we only keep groups with positive value
    ORDER BY 1, 3
    """
    
    # Optional debug: set st.session_state.debug_chart_sql = True in Settings to inspect
    if st.session_state.get('debug_chart_sql'):
        with st.expander("Debug: Chart SQL", expanded=False):
            st.code(sql)
            st.write({"params": params})
    
    df = execute_query_for_dashboard(sql, tuple(params))
    
    # Rename metric column for consistency in the calling function
    if df is not None and 'metric_value' in df.columns:
        df.rename(columns={'metric_value': 'total_revenue'}, inplace=True) # Keep 'total_revenue' for downstream compatibility
        
    return df

def render_chart_view():
    """Render the enhanced Chart View page with multi-select, live search, and comparison."""
    st.markdown('<div class="main-header"><h1>📈 Chart View</h1><p>Deep dive analytics with interactive visualizations</p></div>', unsafe_allow_html=True)

    # --- Session State Initialization ---
    if 'cv_time' not in st.session_state: st.session_state.cv_time = "7d"
    if 'cv_metric_type' not in st.session_state: st.session_state.cv_metric_type = "Stores"
    if 'cv_granularity' not in st.session_state: st.session_state.cv_granularity = "Day"
    if 'comparison_sets' not in st.session_state: st.session_state.comparison_sets = [{}] 

    # Fetch all possible filter options once
    filter_options = get_filter_options()

    # --- Time Period Selector ---
    st.markdown("### ⏱️ Time Period")
    time_ranges = ["1d", "7d", "1m", "3m", "6m", "1y"]
    current_time_index = time_ranges.index(st.session_state.cv_time) if st.session_state.cv_time in time_ranges else 1
    st.session_state.cv_time = st.radio("", time_ranges, index=current_time_index, horizontal=True, key="time_range_selector")
    
    # --- Main Controls ---
    st.markdown("### 🎛️ Analytics Controls")
    with st.expander("Debug", expanded=False):
        st.checkbox("Show Chart SQL", key="debug_chart_sql")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Analyze by", ["Stores", "Product Categories", "Products", "Avg Transaction Value"], key="cv_metric_type")
    with c2:
        # Do not set a default index when using a session-state-backed key to avoid Streamlit warning
        st.selectbox("Time Granularity", ["Minute", "Hour", "Day", "Week", "Month"], key="cv_granularity")

    # --- Render Filter Sets ---
    all_data_frames = []
    
    for i in range(len(st.session_state.comparison_sets)):
        is_primary = (i == 0)
        label = "Primary Selection" if is_primary else f"Comparison Set {i}"
        
        with st.container(border=True):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{label}**")
            with cols[1]:
                if not is_primary:
                    if st.button(f"❌ Remove", key=f"remove_{i}"):
                        st.session_state.comparison_sets.pop(i)
                        st.rerun()

            current_filters = st.session_state.comparison_sets[i]

            # Store selector (always visible unless analyzing by store-based metrics)
            selected_stores = []
            if st.session_state.cv_metric_type in ["Product Categories", "Products"]:
                selected_stores = st.multiselect(
                    "Select Store(s)", options=filter_options["stores"],
                    default=current_filters.get("stores", []), placeholder="All Stores", key=f"stores_{i}"
                )
                st.session_state.comparison_sets[i]['stores'] = selected_stores
            
            # Metric-specific filters
            metric_filters = []
            if st.session_state.cv_metric_type in ["Stores", "Avg Transaction Value"]:
                label = "Select Store(s) to Plot" if st.session_state.cv_metric_type == "Stores" else "Select Store(s) for Avg. Transaction Value"
                selected = st.multiselect(label, filter_options["stores"], default=current_filters.get("filters", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['filters'] = selected
                if selected: metric_filters.extend(selected)
            
            elif st.session_state.cv_metric_type == "Product Categories":
                category_options = ["All"] + filter_options["categories"]
                selected_cats = st.multiselect("Select Product Category(s)", category_options, default=current_filters.get("filters_with_all", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['filters_with_all'] = selected_cats

                if "All" in selected_cats:
                    final_selected_cats = filter_options["categories"]
                else:
                    final_selected_cats = selected_cats
                
                st.session_state.comparison_sets[i]['filters'] = final_selected_cats
                if final_selected_cats: metric_filters.extend(final_selected_cats)

            elif st.session_state.cv_metric_type == "Products":
                # Category filter for products
                selected_prod_categories = st.multiselect(
                    "Filter by Product Category", options=filter_options["categories"],
                    default=current_filters.get("prod_categories", []), key=f"prod_cat_filter_{i}"
                )
                st.session_state.comparison_sets[i]['prod_categories'] = selected_prod_categories
                
                available_products = get_products_by_categories(selected_prod_categories)
                
                # Add "All Products" option to the product selector
                product_options = ["All Products"] + available_products
                selected_products_input = st.multiselect("Select Product(s)", product_options, default=current_filters.get("product_selection", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['product_selection'] = selected_products_input
                
                # Process the selection
                if "All Products" in selected_products_input:
                    final_selected_products = available_products
                else:
                    final_selected_products = selected_products_input
                
                st.session_state.comparison_sets[i]['filters'] = final_selected_products
                if final_selected_products: metric_filters.extend(final_selected_products)

            # Sanitize metric filters: expand any 'All' selections to actual lists
            if st.session_state.cv_metric_type == "Product Categories":
                if (not metric_filters) or (set(metric_filters) == {"All"}) or ("All" in metric_filters):
                    metric_filters = [c for c in filter_options.get("categories", []) if c]
            elif st.session_state.cv_metric_type == "Products":
                if (not metric_filters) or ("All Products" in metric_filters):
                    # If user selected All Products or none, fetch available products for chosen categories
                    selected_prod_categories = st.session_state.comparison_sets[i].get('prod_categories', [])
                    metric_filters = get_products_by_categories(selected_prod_categories)

            # Fetch data for this specific set
            if metric_filters:
                with st.spinner(f"Loading data for {label}..."):
                    data_subset = get_chart_view_data(
                        st.session_state.cv_time, st.session_state.cv_metric_type,
                        st.session_state.cv_granularity, metric_filters, selected_stores
                    )
                    if not data_subset.empty:
                        data_subset['set_index'] = i
                        all_data_frames.append(data_subset)

    if st.button("🆚 Add Comparison"):
        st.session_state.comparison_sets.append({})
        st.rerun()

    # --- Data Fetching and Visualization ---
    if not all_data_frames:
        st.info("Please select at least one item in a comparison set to view the chart.")
        return

    data = pd.concat(all_data_frames)

    if data.empty:
        st.info("No data available for the selected filters.")
        return

    # Backfill weekly gaps with zeros so single-store weekly selections always render
    if st.session_state.cv_granularity == "Week":
        try:
            data['date'] = pd.to_datetime(data['date'])
            overall_start = data['date'].min()
            overall_end = data['date'].max()
            # Align to week starts (Mon) for range
            start_week = (overall_start - pd.to_timedelta(overall_start.weekday(), unit='D')).normalize()
            end_week = (overall_end - pd.to_timedelta(overall_end.weekday(), unit='D')).normalize()
            week_index = pd.date_range(start=start_week, end=end_week, freq='W-MON')

            filled_frames = []
            for series, g in data.groupby('series_name'):
                g = g.set_index('date').sort_index()
                g = g.reindex(week_index)
                g['series_name'] = series
                # forward fill static columns
                g['base_name'] = g['base_name'].ffill().bfill()
                g['store_name'] = g['store_name'].ffill().bfill()
                # fill metric to zero where missing
                if 'total_revenue' in g.columns:
                    g['total_revenue'] = g['total_revenue'].fillna(0)
                # restore index as date
                g = g.reset_index().rename(columns={'index': 'date'})
                filled_frames.append(g)
            data = pd.concat(filled_frames, ignore_index=True)
        except Exception as _:
            pass

    # --- Visualization with advanced styling ---
    y_axis_title = "💰 Revenue (PHP)"
    chart_title = "Revenue Comparison"
    if st.session_state.cv_metric_type == "Avg Transaction Value":
        y_axis_title = "💳 Avg Transaction Value (PHP)"
        chart_title = "Avg Transaction Value Comparison"

    total_visible_metric = data['total_revenue'].sum()
    st.metric(f"Total Value (Visible in Chart)", f"₱{total_visible_metric:,.0f}")

    fig = go.Figure()

    # --- Conditional Coloring Logic ---
    primary_set = st.session_state.comparison_sets[0]
    stores_in_primary_set = []
    if st.session_state.cv_metric_type in ["Product Categories", "Products"]:
        stores_in_primary_set = primary_set.get("stores", [])
    elif st.session_state.cv_metric_type in ["Stores", "Avg Transaction Value"]:
        stores_in_primary_set = primary_set.get("filters", [])
    
    num_stores_selected = len(stores_in_primary_set) if stores_in_primary_set else len(filter_options["stores"])
    # Only when viewing Product Categories with a single store do we color by category
    color_by_category = (
        st.session_state.cv_metric_type == "Product Categories" and num_stores_selected == 1
    )

    # Define color maps
    store_color_map = {'Rockwell': '#E74C3C', 'Greenhills': '#2ECC71', 'Magnolia': '#F1C40F', 'North Edsa': '#3498DB', 'Fairview': '#9B59B6'}
    store_fill_color_map = {'Rockwell': 'rgba(231, 76, 60, 0.15)', 'Greenhills': 'rgba(46, 204, 113, 0.15)', 'Magnolia': 'rgba(241, 196, 15, 0.15)', 'North Edsa': 'rgba(52, 152, 219, 0.15)', 'Fairview': 'rgba(155, 89, 182, 0.15)'}
    # Fixed category color scheme (matches dashboard pies)
    category_color_map = {
        'n/a': '#6D8299',
        'aji mix': '#00C9FF',
        'bev': '#2D98DA',
        'ccp': '#FF6B6B',
        'choco': '#E84118',
        'indi': '#92FE9D',
        'mint': '#38ADA9',
        'nuts': '#FF9F43',
        'oceana': '#1E90FF',
        'per gram': '#C65DFF',
        'seasonal': '#B33771',
        'toys': '#FFD93D',
        'tradsnax': '#009432',
    }
    def _cat_color(key):
        k = (str(key).strip().lower() if key is not None else 'n/a')
        return category_color_map.get(k, '#6D8299')

    style_palette = [{'dash': 'solid', 'width': 2.5}, {'dash': 'dash', 'width': 2.0}, {'dash': 'dot', 'width': 2.0}, {'dash': 'dashdot', 'width': 2.0}]
    entity_style_map = {}
    
    for series_name in sorted(data['series_name'].unique()):
        series_df = data[data['series_name'] == series_name]
        base_name = series_df['base_name'].iloc[0]
        set_index = series_df['set_index'].iloc[0]

        if color_by_category:
            # Use fixed category palette; base_name holds the category in this mode
            color = _cat_color(base_name)
        else:
            store_name = series_df['store_name'].iloc[0]
            color = store_color_map.get(store_name, '#FFFFFF')

        try:
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)'
        except:
            fillcolor = 'rgba(255, 255, 255, 0.1)'

        style_key = (set_index, base_name)
        if style_key not in entity_style_map:
            style_idx = len([k for k in entity_style_map if k[0] == set_index])
            entity_style_map[style_key] = style_palette[style_idx % len(style_palette)]
        style = entity_style_map[style_key]
        # Force solid lines when using category colors (single-store Product Categories)
        if color_by_category:
            style = {'dash': 'solid', 'width': style['width']}

        fig.add_trace(go.Scatter(
            x=series_df['date'], y=series_df['total_revenue'], name=series_name + "_glow",
            line=dict(color=color, width=style['width'] * 2.5, dash=style['dash'], shape='spline'),
            opacity=0.2, mode='lines', showlegend=False, hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=series_df['date'], y=series_df['total_revenue'], name=series_name,
            line=dict(color=color, width=style['width'], dash=style['dash'], shape='spline'),
            fill='tozeroy', fillcolor=fillcolor, mode='lines'
        ))

    fig.update_layout(
        title_text=chart_title, template="plotly_dark", plot_bgcolor='#131722', paper_bgcolor='#131722',
        height=500, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        xaxis=dict(title_text='<b>Date</b>', title_font=dict(size=14), tickfont=dict(color='#B0B0B0'), gridcolor='rgba(255, 255, 255, 0.1)', showgrid=True, zeroline=False),
        yaxis=dict(title_text=f'<b>{y_axis_title}</b>', title_font=dict(size=14), tickfont=dict(color='#B0B0B0'), gridcolor='rgba(255, 255, 255, 0.1)', tickprefix="₱", tickformat=",.0f", hoverformat=",.0f", showgrid=True, zeroline=False),
        hoverlabel=dict(bgcolor="#2A2E39", font_size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Detailed Data"):
        st.dataframe(data, use_container_width=True, hide_index=True)

class DailyInsightsGenerator:
    """Automated daily business intelligence"""
    def __init__(self, db_connection_func, claude_client_func):
        self.get_db_connection = db_connection_func
        self.get_claude_client = claude_client_func

    @st.cache_data(ttl=3600)
    def generate_daily_summary(_self):
        """Generate AI-powered daily business summary"""
        # Get key metrics for yesterday
        sql = """
        SELECT 
            COUNT(DISTINCT t.ref_id) as transactions_yesterday,
            SUM(t.total) as sales_yesterday,
            COUNT(DISTINCT t.store_id) as active_stores,
            AVG(t.total) as avg_transaction_value
        FROM transactions t
        WHERE DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = DATE((NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '1 day')
        AND LOWER(t.transaction_type) = 'sale'
        AND COALESCE(t.is_cancelled, false) = false
        """
        conn = _self.get_db_connection()
        if not conn:
            return {"summary": "Unable to generate insights - database connection failed"}
        try:
            df = pd.read_sql(sql, conn)
            if df.empty or df.iloc[0]['transactions_yesterday'] == 0:
                return {"summary": "No sales data available for yesterday to generate insights."}
            
            metrics = df.iloc[0]
            
            # Use Claude to generate insights
            client = _self.get_claude_client()
            if client:
                prompt = f"""Generate a brief daily business summary based on these metrics for yesterday:
                - Transactions: {metrics['transactions_yesterday']}
                - Sales: ₱{metrics['sales_yesterday']:,.0f}
                - Active Stores: {metrics['active_stores']}
                - Avg Transaction: ₱{metrics['avg_transaction_value']:,.0f}
                Provide 3-4 bullet points with insights and recommendations."""
                try:
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    summary = response.content[0].text
                except:
                    summary = "AI insights temporarily unavailable"
            else:
                summary = "AI insights not configured"
            
            return {
                "summary": summary,
                "transactions": int(metrics['transactions_yesterday']),
                "sales": float(metrics['sales_yesterday']),
                "active_stores": int(metrics['active_stores']),
                "avg_transaction": float(metrics['avg_transaction_value'])
            }
        except Exception as e:
            return {"summary": f"Error generating insights: {e}"}
        finally:
            if conn:
                conn.close()

class SmartAlertManager:
    """Proactive alert system"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func

    def _execute_query(self, sql):
        """Internal query executor"""
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        try:
            return pd.read_sql(sql, conn)
        except:
            return pd.DataFrame()
        finally:
            if conn: conn.close()

    @st.cache_data(ttl=900)  # 15 minutes
    def get_active_alerts(_self):
        """Get current active alerts"""
        alerts = []
        
        # Critical stockout alerts
        sql_stockout = """
        SELECT p.name as product_name, s.name as store_name, i.quantity_on_hand
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        JOIN stores s ON i.store_id = s.id
        WHERE i.quantity_on_hand = 0
        LIMIT 10
        """
        df_stockout = _self._execute_query(sql_stockout)
        for _, row in df_stockout.iterrows():
            alerts.append({
                "type": "CRITICAL",
                "icon": "🚨",
                "message": f"STOCKOUT: {row['product_name']} at {row['store_name']}",
                "action": "Immediate restock required"
            })
            
        # Low inventory warnings
        sql_low_stock = """
        SELECT p.name as product_name, s.name as store_name,
               i.quantity_on_hand, i.warning_stock
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        JOIN stores s ON i.store_id = s.id
        WHERE i.quantity_on_hand > 0 
        AND i.quantity_on_hand <= COALESCE(i.warning_stock, 5)
        ORDER BY i.quantity_on_hand ASC
        LIMIT 15
        """
        df_low_stock = _self._execute_query(sql_low_stock)
        for _, row in df_low_stock.iterrows():
            alerts.append({
                "type": "WARNING",
                "icon": "⚠️",
                "message": f"LOW STOCK: {row['product_name']} at {row['store_name']} ({int(row['quantity_on_hand'])} left)",
                "action": "Consider restocking soon"
            })
            
        return alerts[:20]  # Limit to 20 most critical

# --- NEW AI HUB V2 CLASSES ---
class PredictiveForecastingEngine:
    """Advanced demand forecasting with multiple algorithms"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=7200)
    def forecast_demand_trends(_self, days_ahead=30, confidence_threshold=0.75):
        # This is a simplified SQL-based approach to forecasting, using linear regression for trends
        # and moving averages. A full implementation would require ML libraries.
        sql = """
        WITH weekly_sales AS (
            SELECT
                p.id as product_id,
                t.store_id,
                DATE_TRUNC('week', t.transaction_time AT TIME ZONE 'Asia/Manila') as week,
                SUM(ti.quantity) as weekly_qty
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '180 days'
            AND LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
            GROUP BY p.id, t.store_id, week
        ),
        trend_analysis AS (
            SELECT
                product_id,
                store_id,
                REGR_SLOPE(weekly_qty, EXTRACT(EPOCH FROM week) / (7*24*3600)) 
                    OVER (PARTITION BY product_id, store_id ORDER BY week ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as trend_slope,
                AVG(weekly_qty) 
                    OVER (PARTITION BY product_id, store_id ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as moving_avg_4week,
                week
            FROM weekly_sales
        ),
        latest_trends AS (
            SELECT DISTINCT ON (product_id, store_id)
                product_id, store_id, trend_slope, moving_avg_4week
            FROM trend_analysis
            ORDER BY product_id, store_id, week DESC
        )
        SELECT
            p.name as product_name,
            s.name as store_name,
            p.category,
            COALESCE(lt.moving_avg_4week, 0) as current_weekly_avg,
            GREATEST(0, COALESCE(lt.moving_avg_4week, 0) + (COALESCE(lt.trend_slope, 0) * 4)) as forecasted_weekly_avg,
            CASE
                WHEN COALESCE(lt.trend_slope, 0) > 0.1 THEN 'UP'
                WHEN COALESCE(lt.trend_slope, 0) < -0.1 THEN 'DOWN'
                ELSE 'STABLE'
            END as trend_direction,
            LEAST(1.0, ABS(COALESCE(lt.trend_slope, 0)) / 2.0) * 100 as trend_confidence
        FROM latest_trends lt
        JOIN products p ON lt.product_id = p.id
        JOIN stores s ON lt.store_id = s.id
        WHERE lt.moving_avg_4week > 0
        ORDER BY trend_confidence DESC, forecasted_weekly_avg DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def identify_seasonal_products(_self):
        # Simplified seasonal identification using coefficient of variation on monthly sales
        # Updated to use intelligent date ranges (1Y = current year to date)
        sql = """
        WITH monthly_sales AS (
            SELECT
                p.name as product_name,
                p.category,
                EXTRACT(MONTH FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as month,
                SUM(ti.quantity) as total_quantity
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= DATE_TRUNC('year', CURRENT_DATE)
            AND LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
            GROUP BY p.name, p.category, month
        ),
        seasonality_stats AS (
            SELECT
                product_name,
                category,
                STDDEV(total_quantity) / AVG(total_quantity) as coeff_variation,
                SUM(total_quantity) as total_sales_volume
            FROM monthly_sales
            GROUP BY product_name, category
            HAVING AVG(total_quantity) > 10
        )
        SELECT
            product_name,
            category,
            total_sales_volume,
            coeff_variation as seasonal_strength,
            'Check peak months for planning' as recommendation
        FROM seasonality_stats
        WHERE coeff_variation > 0.5
        ORDER BY seasonal_strength DESC
        LIMIT 50;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=1800)
    def analyze_product_lifecycle(_self):
        # Simplified lifecycle analysis based on sales trend and volume
        # Updated to use intelligent date ranges (1M = current month to date)
        sql = """
        WITH product_trends AS (
             SELECT
                p.name as product_name,
                MIN(t.transaction_time) as first_sale,
                MAX(t.transaction_time) as last_sale,
                SUM(ti.quantity) as total_units_sold,
                SUM(CASE WHEN DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= DATE_TRUNC('month', CURRENT_DATE) THEN ti.quantity ELSE 0 END) as last_month_units
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
            GROUP BY p.name
        )
        SELECT
            product_name,
            total_units_sold,
            last_month_units,
            CASE
                WHEN DATE((NOW() AT TIME ZONE 'Asia/Manila')) - DATE(first_sale) < 90 AND last_month_units > 50 THEN 'Introduction/Growth'
                WHEN total_units_sold > 1000 AND last_month_units > (total_units_sold / 24) THEN 'Maturity'
                WHEN last_month_units < (total_units_sold / 50) AND DATE((NOW() AT TIME ZONE 'Asia/Manila')) - DATE(last_sale) > 60 THEN 'Decline'
                ELSE 'Stable'
            END as lifecycle_stage
        FROM product_trends
        ORDER BY total_units_sold DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

class CustomerIntelligenceEngine:
    """Deep customer behavior and pattern analysis"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_shopping_patterns(_self):
        # Updated to use intelligent date ranges (6M = current month to date)
        sql = """
        SELECT
            EXTRACT(ISODOW FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as day_of_week,
            EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
            COUNT(DISTINCT t.ref_id) as transaction_count,
            AVG(t.total) as avg_basket_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def perform_basket_analysis(_self, min_support=0.01):
        # Simplified basket analysis to find top co-purchased product pairs
        sql = """
        WITH item_pairs AS (
            SELECT
                a.product_id as product_a,
                b.product_id as product_b,
                COUNT(DISTINCT a.transaction_ref_id) as pair_frequency
            FROM transaction_items a
            JOIN transaction_items b ON a.transaction_ref_id = b.transaction_ref_id AND a.product_id < b.product_id
            GROUP BY 1, 2
        )
        SELECT
            p1.name as product_1,
            p2.name as product_2,
            ip.pair_frequency
        FROM item_pairs ip
        JOIN products p1 ON ip.product_a = p1.id
        JOIN products p2 ON ip.product_b = p2.id
        ORDER BY ip.pair_frequency DESC
        LIMIT 25;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=7200)
    def segment_customers(_self):
        # RFM Analysis for Customer Segmentation
        st.info("Note: RFM analysis requires a `customer_ref_id` in the `transactions` table. Using anonymous buckets if not available.")
        sql = """
        WITH customer_metrics AS (
            SELECT
                COALESCE(t.customer_ref_id, 'Anonymous_' || (ROW_NUMBER() OVER () % 100)::text) as customer_id,
                MAX(DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')) as last_purchase_date,
                COUNT(DISTINCT t.ref_id) as frequency,
                SUM(t.total) as monetary_value,
                DATE((NOW() AT TIME ZONE 'Asia/Manila')) - MAX(DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')) as recency_days
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '365 days'
            GROUP BY 1
        ),
        rfm_scores AS (
            SELECT *,
                NTILE(5) OVER (ORDER BY recency_days ASC) as recency_score,
                NTILE(5) OVER (ORDER BY frequency DESC) as frequency_score,
                NTILE(5) OVER (ORDER BY monetary_value DESC) as monetary_score
            FROM customer_metrics
        )
        SELECT
            customer_id,
            recency_score, frequency_score, monetary_score,
            CASE
                WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'VIP'
                WHEN recency_score >= 3 AND frequency_score >= 3 THEN 'Loyal'
                WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
                WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost'
                ELSE 'Regular'
            END as customer_segment,
            monetary_value as lifetime_value,
            frequency as purchase_frequency,
            recency_days
        FROM rfm_scores
        ORDER BY monetary_score DESC, frequency_score DESC, recency_score DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

# Missing classes for AI Intelligence Hub

class PredictiveForecastingEngine:
    """Predictive forecasting and trend analysis"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query
    
    def generate_forecast(self):
        """Generate sales forecast (placeholder)"""
        return {"forecast": "Advanced forecasting available in full version"}
    
    @st.cache_data(ttl=7200)
    def forecast_demand_trends(_self, days_ahead=30, confidence_threshold=0.75):
        """Forecast demand trends for products (placeholder implementation)"""
        try:
            # Simplified forecast - return sample data structure
            import pandas as pd
            
            # Create sample forecast data
            forecast_data = {
                'product_name': ['Sample Product A', 'Sample Product B', 'Sample Product C'],
                'current_demand': [100, 150, 80],
                'predicted_demand': [120, 140, 95],
                'confidence_score': [0.85, 0.78, 0.92],
                'trend': ['Increasing', 'Stable', 'Increasing']
            }
            
            return pd.DataFrame(forecast_data)
        except Exception as e:
            return pd.DataFrame()  # Return empty DataFrame on error

class AutomatedInsightEngine:
    """Automated insights generation"""
    def __init__(self, db_connection_func, claude_client_func):
        self.get_db_connection = db_connection_func
        self.get_claude_client = claude_client_func
    
    def generate_weekly_business_review(self):
        """Generate weekly business review"""
        return {
            "summary": "📊 **Weekly Business Review**\n\n• Sales performance tracking active\n• Customer behavior analysis in progress\n• Market trend monitoring enabled\n\n*Full AI insights available with complete analytics setup*",
            "metrics": {
                "current_week_sales": 100000,
                "previous_week_sales": 95000,
                "current_week_tx": 500,
                "previous_week_tx": 480
            }
        }

class MarketIntelligenceEngine:
    """Competitive and market trend analysis"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_price_elasticity(_self):
        # This is a placeholder as true price elasticity requires price change data.
        # This query shows sales volume at different price points.
        sql = """
        SELECT
            p.name as product_name,
            ti.unit_price,
            SUM(ti.quantity) as total_quantity_sold
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT ti.transaction_ref_id) > 10 -- Only for products with some sales history
        ORDER BY 1, 2
        LIMIT 200;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def detect_market_opportunities(_self):
        sql = """
        SELECT
            p.category,
            SUM(ti.item_total) as total_revenue,
            AVG(ti.item_total) as avg_item_value,
            COUNT(DISTINCT p.id) as num_products
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY 1
        ORDER BY 2 DESC;
        """
        return _self._execute_query(sql)

class AutomatedInsightEngine:
    """AI-powered automated business intelligence"""
    def __init__(self, db_connection_func, claude_client_func):
        self.get_db_connection = db_connection_func
        self.get_claude_client = claude_client_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def generate_weekly_business_review(_self):
        sql = """
        WITH weekly_metrics AS (
            SELECT
                SUM(CASE WHEN (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '7 days' THEN t.total ELSE 0 END) as current_week_sales,
                SUM(CASE WHEN (t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '14 days' AND (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '8 days' THEN t.total ELSE 0 END) as previous_week_sales,
                COUNT(DISTINCT CASE WHEN (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '7 days' THEN t.ref_id END) as current_week_tx,
                COUNT(DISTINCT CASE WHEN (t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '14 days' AND (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '8 days' THEN t.ref_id END) as previous_week_tx
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '14 days'
        )
        SELECT * FROM weekly_metrics;
        """
        df = _self._execute_query(sql)
        if df.empty: return {"summary": "Not enough data for a weekly review."}
        
        metrics = df.iloc[0]
        client = _self.get_claude_client()
        if not client: return {"summary": "AI client not configured."}

        prompt = f"Analyze these weekly metrics and provide a business review with key wins, concerns, and recommendations. Current week sales: ₱{metrics['current_week_sales']:,.0f}. Previous week sales: ₱{metrics['previous_week_sales']:,.0f}. Current week transactions: {metrics['current_week_tx']}. Previous week transactions: {metrics['previous_week_tx']}."
        try:
            response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=600, messages=[{"role": "user", "content": prompt}])
            summary = response.content[0].text
            return {"summary": summary, "metrics": metrics.to_dict()}
        except Exception as e:
            return {"summary": f"Failed to generate AI insights: {e}"}

    @st.cache_data(ttl=1800)
    def create_predictive_alerts(_self):
        # Using a simplified version of forecasting for alerts
        forecasting_engine = PredictiveForecastingEngine(create_db_connection)
        forecasts = forecasting_engine.forecast_demand_trends()
        alerts = []
        if not forecasts.empty:
            trending_up = forecasts[forecasts['trend_direction'] == 'UP'].head(3)
            for _, row in trending_up.iterrows():
                alerts.append({"type": "TRENDING_UP", "message": f"Trending Up: {row['product_name']} at {row['store_name']} shows strong upward trend."})
        return alerts

    @st.cache_data(ttl=7200)
    def generate_store_intelligence(_self, store_id):
        return {"summary": f"Store-specific intelligence for store ID {store_id} is under development."}

class PerformanceOptimizationEngine:
    """Performance tracking and optimization"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_store_performance(_self):
        sql = """
        SELECT
            s.name as store_name,
            SUM(t.total) as total_revenue,
            COUNT(DISTINCT t.ref_id) as total_transactions,
            SUM(t.total) / COUNT(DISTINCT t.ref_id) as avg_transaction_value,
            AVG(ti.quantity) as avg_items_per_tx
        FROM transactions t
        JOIN stores s ON t.store_id = s.id
        JOIN transaction_items ti ON t.ref_id = ti.transaction_ref_id
        WHERE (t.transaction_time AT TIME ZONE 'Asia/Manila') >= (NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '90 days'
        AND LOWER(t.transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
        GROUP BY 1
        ORDER BY 2 DESC;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=1800)
    def track_kpi_trends(_self):
        sql = """
        SELECT
            DATE_TRUNC('week', transaction_time AT TIME ZONE 'Asia/Manila') as week,
            SUM(total) as weekly_sales,
            COUNT(DISTINCT ref_id) as weekly_transactions,
            AVG(total) as weekly_avg_tx_value
        FROM transactions
        WHERE LOWER(transaction_type) = 'sale' AND COALESCE(t.is_cancelled, false) = false
        GROUP BY 1
        ORDER BY 1;
        """
        return _self._execute_query(sql)

def render_enhanced_analytics_tab(analytics_engine, forecasting_engine):
    """ENHANCED Analytics Tab with forecasting"""
    st.header("📊 Enhanced Analytics Suite")
    
    # EXISTING functionality
    st.subheader("🔍 Hidden Demand Detection")
    st.info("💡 Identifies products that would sell more if they were in stock")
    if st.button("Analyze Hidden Demand", key="hd_btn"):
        with st.spinner("Analyzing sales patterns and inventory levels..."):
            hidden_demand_df = analytics_engine.detect_hidden_demand()
            if not hidden_demand_df.empty:
                def color_score(val):
                    if val >= 70: return 'background-color: #ff4444; color: white'
                    elif val >= 40: return 'background-color: #ffaa00; color: white'
                    else: return ''
                st.dataframe(hidden_demand_df.style.applymap(color_score, subset=['hidden_demand_score']), use_container_width=True)
            else:
                st.info("No hidden demand patterns detected")
    
    st.markdown("---")
    st.subheader("⏰ Stockout Predictions")
    st.info("🎯 Predicts when products will run out of stock")
    if st.button("Predict Stockouts", key="so_btn"):
        with st.spinner("Calculating stockout predictions..."):
            stockout_df = analytics_engine.predict_stockouts()
            if not stockout_df.empty:
                def color_urgency(val):
                    if val == 'CRITICAL': return 'background-color: #ff0000; color: white'
                    elif val == 'HIGH': return 'background-color: #ff8800; color: white'
                    elif val == 'MEDIUM': return 'background-color: #ffff00; color: black'
                    else: return ''
                st.dataframe(stockout_df.style.applymap(color_urgency, subset=['urgency_level']), use_container_width=True)
            else:
                st.info("No stockout risks detected")

    # NEW ADDITIONS:
    st.markdown("---")
    st.subheader("🔮 Demand Forecasting")
    st.info("🎯 Predicts future demand trends with confidence scoring")
    if st.button("🔮 Generate Demand Forecasts", type="primary"):
        with st.spinner("Analyzing historical patterns and generating forecasts..."):
            forecast_df = forecasting_engine.forecast_demand_trends()
            if not forecast_df.empty:
                st.write(f"Generated forecasts for {len(forecast_df)} products:")
                def color_trend(val):
                    if val == 'UP': return 'background-color: #00ff00; color: black'
                    elif val == 'DOWN': return 'background-color: #ff8888; color: black'
                    else: return ''
                st.dataframe(forecast_df.style.applymap(color_trend, subset=['trend_direction']), use_container_width=True)
                trend_counts = forecast_df['trend_direction'].value_counts()
                fig = px.bar(x=trend_counts.index, y=trend_counts.values, title="Product Trend Direction Summary", color=trend_counts.index, color_discrete_map={'UP': '#00ff00', 'DOWN': '#ff4444', 'STABLE': '#ffaa00'})
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient historical data for forecasting.")

    st.markdown("---")
    st.subheader("🌿 Seasonal Intelligence")
    st.info("📅 Identifies seasonal products and upcoming seasonal changes")
    if st.button("🌿 Analyze Seasonal Patterns", type="primary"):
        with st.spinner("Identifying seasonal patterns..."):
            seasonal_df = forecasting_engine.identify_seasonal_products()
            if not seasonal_df.empty:
                st.dataframe(seasonal_df, use_container_width=True)
                if 'seasonal_strength' in seasonal_df.columns:
                    fig = px.scatter(seasonal_df, x='category', y='seasonal_strength', size='total_sales_volume', color='seasonal_strength', title="Product Seasonality Analysis")
                    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant seasonal patterns found.")

def render_predictive_intelligence_tab(forecasting_engine, market_engine):
    """NEW: Predictive Intelligence Tab"""
    st.header("🔮 Predictive Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Product Lifecycle Analysis")
        if st.button("Analyze Product Lifecycles"):
            with st.spinner("Analyzing product lifecycles..."):
                lifecycle_df = forecasting_engine.analyze_product_lifecycle()
                if not lifecycle_df.empty:
                    st.dataframe(lifecycle_df, use_container_width=True)
    with col2:
        st.subheader("💰 Price Optimization")
        if st.button("Analyze Price Elasticity"):
            with st.spinner("Calculating optimal pricing..."):
                pricing_df = market_engine.analyze_price_elasticity()
                if not pricing_df.empty:
                    st.dataframe(pricing_df, use_container_width=True)
    st.markdown("---")
    st.subheader("🎯 Market Opportunities")
    if st.button("🎯 Detect Market Opportunities", type="primary"):
        with st.spinner("Scanning for market opportunities..."):
            opportunities_df = market_engine.detect_market_opportunities()
            if not opportunities_df.empty:
                st.dataframe(opportunities_df, use_container_width=True)

def render_customer_intelligence_tab(customer_engine):
    """NEW: Customer Intelligence Tab"""
    st.header("🎯 Customer Intelligence")
    tab1, tab2, tab3 = st.tabs(["🛒 Shopping Patterns", "🛍️ Basket Analysis", "👥 Customer Segments"])
    with tab1:
        st.subheader("🛒 Shopping Pattern Analysis")
        if st.button("Analyze Shopping Patterns"):
            with st.spinner("Analyzing customer shopping patterns..."):
                patterns_df = customer_engine.analyze_shopping_patterns()
                if not patterns_df.empty:
                    if 'hour' in patterns_df.columns and 'day_of_week' in patterns_df.columns:
                        pivot_df = patterns_df.pivot_table(index='hour', columns='day_of_week', values='transaction_count', fill_value=0)
                        fig = px.imshow(pivot_df, title="Shopping Activity Heatmap (Transactions per Hour)", aspect="auto")
                        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(patterns_df, use_container_width=True)
    with tab2:
        st.subheader("🛍️ Market Basket Analysis")
        if st.button("Perform Basket Analysis"):
            with st.spinner("Analyzing product associations..."):
                basket_df = customer_engine.perform_basket_analysis()
                if not basket_df.empty:
                    st.dataframe(basket_df, use_container_width=True)
    with tab3:
        st.subheader("👥 Customer Segmentation")
        if st.button("Segment Customers"):
            with st.spinner("Segmenting customers..."):
                segments_df = customer_engine.segment_customers()
                if not segments_df.empty:
                    st.dataframe(segments_df, use_container_width=True)

def render_automated_insights_tab(insight_engine):
    """NEW: Automated Insights Tab"""
    st.header("💡 Automated Business Intelligence")
    st.subheader("📊 Weekly Business Review")
    if st.button("📊 Generate Weekly Review", type="primary"):
        with st.spinner("Generating comprehensive business review..."):
            review = insight_engine.generate_weekly_business_review()
            if review:
                st.markdown(review.get('summary', 'No review available'))
                if 'metrics' in review:
                    metrics = review['metrics']
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("Sales Growth", f"{(metrics.get('current_week_sales', 0) / metrics.get('previous_week_sales', 1) - 1) * 100:.1f}%")
                    with mcol2:
                        st.metric("Transaction Growth", f"{(metrics.get('current_week_tx', 0) / metrics.get('previous_week_tx', 1) - 1) * 100:.1f}%")

def render_smart_alerts_tab(alert_manager, insight_engine):
    """Smart Alerts Tab Content"""
    st.header("🚨 Smart Alerts")
    
    st.subheader("⚡ Real-time Alerts")
    with st.spinner("Fetching active alerts..."):
        alerts = alert_manager.get_active_alerts()
    if alerts:
        critical_alerts = [a for a in alerts if a['type'] == 'CRITICAL']
        warning_alerts = [a for a in alerts if a['type'] == 'WARNING']
        if critical_alerts:
            st.error(f"**🚨 {len(critical_alerts)} Critical Alerts**")
            for alert in critical_alerts: st.markdown(f"- **{alert['icon']} {alert['message']}**: {alert['action']}")
        if warning_alerts:
            st.warning(f"**⚠️ {len(warning_alerts)} Warning Alerts**")
            for alert in warning_alerts: st.markdown(f"- **{alert['icon']} {alert['message']}**: {alert['action']}")
    else:
        st.success("✅ No active real-time alerts.")
        
    st.markdown("---")
    st.subheader("🔮 Predictive Alerts")
    with st.spinner("Generating predictive alerts..."):
        predictive_alerts = insight_engine.create_predictive_alerts()
        if predictive_alerts:
            for alert in predictive_alerts:
                if alert['type'] == 'TRENDING_UP': st.success(f"📈 {alert['message']}")
                elif alert['type'] == 'SEASONAL_PREP': st.info(f"🌿 {alert['message']}")
                elif alert['type'] == 'OPPORTUNITY': st.warning(f"� {alert['message']}")
                else: st.info(f"💡 {alert['message']}")
        else:
            st.success("✅ No predictive alerts at this time.")

def render_ai_learning_tab():
    """AI Learning Tab Content"""
    st.header("🎓 AI Learning Center")
    if 'training_system' in st.session_state:
        training_data = st.session_state.training_system.training_data
        st.subheader("📊 Learning Progress")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Examples", len(training_data))
        with col2: st.metric("Correct Examples", len([ex for ex in training_data if ex.get("feedback") == "correct"]))
        with col3: st.metric("Corrected Examples", len([ex for ex in training_data if ex.get("feedback") == "corrected"]))
        if training_data:
            df_training = pd.DataFrame(training_data)
            df_training['timestamp'] = pd.to_datetime(df_training['timestamp'])
            df_training['date'] = df_training['timestamp'].dt.date
            daily_learning = df_training.groupby('date').size().reset_index(name='examples')
            if len(daily_learning) > 1:
                fig = px.line(daily_learning, x='date', y='examples', title='Daily Learning Progress', markers=True)
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("📝 Recent Training Examples")
        if training_data:
            for ex in reversed(training_data[-5:]):
                with st.expander(f"Q: {ex['question'][:50]}..."):
                    st.write(f"**Status:** {ex['feedback']}")
                    if ex.get('explanation'): st.write(f"**Note:** {ex['explanation']}")
                    st.write(f"**Date:** {ex['timestamp'][:10]}")

def render_settings():
    """Settings page for configuration and system status"""
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>System configuration and database management</p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔧 System Status", "🗄️ Database Info", "🎨 UI Settings"])
    
    with tab1:
        st.subheader("📊 System Status")
        
        # Database connection status
        try:
            conn = create_db_connection()
            if conn:
                st.success("✅ Database: Connected")
                
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transactions")
                tx_count = cursor.fetchone()[0]
                st.metric("Total Transactions", f"{tx_count:,}")
                
                cursor.execute("SELECT COUNT(*) FROM products")
                prod_count = cursor.fetchone()[0]
                st.metric("Total Products", f"{prod_count:,}")
                
                cursor.execute("SELECT COUNT(*) FROM stores")
                store_count = cursor.fetchone()[0]
                st.metric("Total Stores", f"{store_count:,}")
                
                conn.close()
            else:
                st.error("❌ Database: Connection Failed")
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
        
        # AI Status
        try:
            claude_client = get_claude_client()
            if claude_client:
                st.success("✅ AI Assistant: Available")
            else:
                st.error("❌ AI Assistant: API Key Missing")
        except Exception as e:
            st.error(f"❌ AI Assistant Error: {e}")
    
    with tab2:
        st.subheader("🗄️ Database Schema")
        
        if st.button("Load Database Schema"):
            schema_info = get_database_schema()
            if schema_info:
                for table_name, table_info in schema_info.items():
                    with st.expander(f"📋 Table: {table_name} ({table_info['row_count']:,} rows)"):
                        st.write("**Columns:**")
                        for col_name, col_type, nullable, default in table_info['columns']:
                            null_text = "NULL" if nullable == "YES" else "NOT NULL"
                            default_text = f", DEFAULT: {default}" if default else ""
                            st.code(f"{col_name}: {col_type} ({null_text}{default_text})")
            else:
                st.error("Failed to load database schema")
    
    with tab3:
        st.subheader("🎨 User Interface Settings")
        
        # Debug Mode Toggle
        st.subheader("🔍 Debug Settings")
        debug_mode = st.checkbox(
            "Enable Debug Mode", 
            value=st.session_state.get('debug_mode', False),
            help="Show detailed debug information on the dashboard including date ranges, store filters, and data availability"
        )
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.success("✅ Debug mode enabled - detailed information will be shown on the dashboard")
        else:
            st.info("Debug mode disabled - dashboard will show standard interface")
        
        st.markdown("---")
        
        # Clear cache
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared successfully")
        
        # Reset session state
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ['current_page']:  # Keep current page
                    del st.session_state[key]
            st.success("✅ Session reset successfully")
        
        st.markdown("---")
        st.subheader("📖 Application Info")
        st.write("**Version:** SupaBot v3.8")
        st.write("**Framework:** Streamlit + PostgreSQL + Claude AI")
        st.write("**Architecture:** Modular Production-Ready")

def render_ai_intelligence_hub():
    """New Intelligence page replacing old AI Hub: six sections, Manila TZ, cached SQL, Plotly dark."""
    st.markdown('<div class="main-header"><h1>🤖 AI Intelligence Hub</h1><p>Executive insights, demand signals, forecasts, and risk radar</p></div>', unsafe_allow_html=True)

    # Reuse global filters
    time_filter = st.session_state.get("dashboard_time_filter", "7D")
    selected_stores = st.session_state.get("dashboard_store_filter", [])
    store_df = get_store_list()
    store_filter_ids = None
    if selected_stores and "All Stores" not in selected_stores and not store_df.empty:
        ids = []
        for store_name in selected_stores:
            ids.extend(store_df[store_df['name'] == store_name]['id'].tolist())
        store_filter_ids = ids or None
    st.caption(f"Filters • Period: {time_filter} • Stores: {'All' if not store_filter_ids else len(store_filter_ids)}")

    # 1) Executive Briefing
    with st.container(border=True):
        st.markdown("### 🧠 Executive Briefing (Auto-Analyst)")
        briefing_md = generate_exec_briefing(time_filter, store_filter_ids)
        st.markdown(briefing_md)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📄 Copy as Markdown", data=briefing_md, file_name="executive_briefing.md")
        with c2:
            if st.button("🔊 Enhance with LLM"):
                st.markdown(enhance_briefing_with_llm(briefing_md))
        with c3:
            if hasattr(settings, 'get_n8n_webhook_url') and settings.get_n8n_webhook_url():
                if st.button("✉️ Email Briefing"):
                    send_briefing_email(briefing_md)
                    st.success("Queued email for delivery")

    # 2) Hidden Demand Detection
    with st.container(border=True):
        st.markdown("### 🔎 Hidden Demand Detection")
        st.caption("Find SKUs under-selling due to OOS/insufficient facing — last 30 full days")
        if st.button("Analyze Hidden Demand", key="hd_analyze"):
            hd_df = hidden_demand_detection(store_filter_ids)
            if hd_df is not None and not hd_df.empty:
                st.dataframe(hd_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV", data=hd_df.to_csv(index=False), file_name="hidden_demand.csv")
                top = hd_df.sort_values("est_impact_php", ascending=False).head(15)
                fig = px.bar(top, x="sku", y="est_impact_php", color="est_impact_php")
                fig.update_layout(template="plotly_dark", margin=dict(t=10,l=10,r=10,b=10), showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hidden demand detected.")

    # 3) Stockout Predictions
    with st.container(border=True):
        st.markdown("### 🧯 Stockout Predictions")
        st.caption("Predict when SKU×Store will stock out based on avg daily demand and inbound ≤7d")
        if st.button("Predict Stockouts", key="btn_predict_stockouts"):
            so_df = stockout_predictions(store_filter_ids)
            if so_df is not None and not so_df.empty:
                st.dataframe(so_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV", data=so_df.to_csv(index=False), file_name="stockout_predictions.csv")
            else:
                st.info("No imminent stockouts detected.")

    # 4) Demand Forecasting
    with st.container(border=True):
        st.markdown("### 📊 Demand Forecasting")
        st.caption("Forecast next 7/30/90 days; aggregate to warehouse/supplier where available")
        if st.button("Generate Demand Forecasts", key="btn_generate_forecasts"):
            fc = demand_forecasts(store_filter_ids)
            if isinstance(fc, dict) and fc.get('store_view') is not None and not fc['store_view'].empty:
                tabs = st.tabs(["Store", "Warehouse", "Supplier"])
                with tabs[0]:
                    st.dataframe(fc['store_view'], use_container_width=True, hide_index=True)
                    st.download_button("⬇️ CSV", data=fc['store_view'].to_csv(index=False), file_name="forecast_store.csv")
                if fc.get('warehouse_view') is not None:
                    with tabs[1]:
                        st.dataframe(fc['warehouse_view'], use_container_width=True, hide_index=True)
                        st.download_button("⬇️ CSV", data=fc['warehouse_view'].to_csv(index=False), file_name="forecast_warehouse.csv")
                if fc.get('supplier_view') is not None:
                    with tabs[2]:
                        st.dataframe(fc['supplier_view'], use_container_width=True, hide_index=True)
                        st.download_button("⬇️ CSV", data=fc['supplier_view'].to_csv(index=False), file_name="forecast_supplier.csv")
                top = fc['store_view'].nlargest(20, 'D+30')
                fig = px.bar(top, x="sku", y="D+30", color="D+30")
                fig.update_layout(template="plotly_dark", margin=dict(t=10,l=10,r=10,b=10), showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No forecast generated.")

    # 5) Seasonal Intelligence
    with st.container(border=True):
        st.markdown("### 🌿 Seasonal Intelligence")
        st.caption("Detect seasonal SKUs and upcoming peaks")
        if st.button("Analyze Seasonal Patterns", key="btn_analyze_seasonal"):
            seas = seasonal_intelligence(store_filter_ids)
            if seas is not None and not seas.empty:
                st.dataframe(seas, use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV", data=seas.to_csv(index=False), file_name="seasonal_intelligence.csv")
            else:
                st.info("No strong seasonality detected.")

    # 6) Inventory Risk Radar
    with st.container(border=True):
        st.markdown("### 🚦 Inventory Risk Radar")
        st.caption("Understock • Overstock • Imbalance")
        tabs = st.tabs(["Understock", "Overstock", "Imbalance"])
        risks = get_inventory_risk(time_filter, store_filter_ids)
        for i, key in enumerate(["understock","overstock","imbalance"]):
            with tabs[i]:
                df = risks.get(key)
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("⬇️ CSV", data=df.to_csv(index=False), file_name=f"inventory_risk_{key}.csv")
                else:
                    st.info("No items in this category.")

# New Intelligence Helpers (cached, parameterized, Manila TZ)

@st.cache_data(ttl=300)
def _fetch_daily_sales_units_365(store_ids: Optional[List[int]] = None) -> pd.DataFrame:
    # Updated to use intelligent date ranges (1Y = current year to date)
    store_clause = ""
    params: List[Any] = []
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)
    sql = f"""
    SELECT 
      DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') AS d,
      ti.product_id AS product_id,
      t.store_id AS store_id,
      SUM(ti.quantity)::numeric AS units,
      SUM(ti.item_total)::numeric AS revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
      AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= DATE_TRUNC('year', CURRENT_DATE)
      AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= CURRENT_DATE
      {store_clause}
    GROUP BY 1,2,3
    """
    return execute_query_for_dashboard(sql, params=params if params else None)

def generate_exec_briefing(time_filter: str, store_ids: Optional[List[int]]):
    try:
        # Determine date windows using intelligent date ranges
        if time_filter == "Custom" and st.session_state.get('custom_start_date') and st.session_state.get('custom_end_date'):
            start = st.session_state['custom_start_date']
            end = st.session_state['custom_end_date']
        else:
            # Get intelligent date ranges
            date_range = get_intelligent_date_range(time_filter)
            start = date_range['start_date']
            end = date_range['end_date']
        
        # Get previous period dates
        prev_dates = get_previous_period_dates(start, end, time_filter)
        prev_start = prev_dates['start_date']
        prev_end = prev_dates['end_date']

        store_clause = ""
        params: List[Any] = [start, end, prev_start, prev_end]
        if store_ids:
            store_clause = " AND t.store_id = ANY(%s)"
            params.append(store_ids)
        sql = f"""
        WITH cur AS (
          SELECT SUM(t.total)::numeric AS sales, COUNT(DISTINCT t.ref_id) AS tx
          FROM transactions t
          WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') <  %s
            {store_clause}
        ), prev AS (
          SELECT SUM(t.total)::numeric AS sales, COUNT(DISTINCT t.ref_id) AS tx
          FROM transactions t
          WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= %s
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') <  %s
            {store_clause}
        )
        SELECT cur.sales AS cur_sales, cur.tx AS cur_tx, prev.sales AS prev_sales, prev.tx AS prev_tx
        FROM cur, prev
        """
        df = execute_query_for_dashboard(sql, params=params)
        if df is None or df.empty:
            return "No material changes"
        row = df.iloc[0].fillna(0)
        cur_sales = float(row.get('cur_sales', 0))
        prev_sales = float(row.get('prev_sales', 0))
        cur_tx = int(row.get('cur_tx', 0))
        prev_tx = int(row.get('prev_tx', 0))
        sales_delta = ((cur_sales - prev_sales) / prev_sales * 100) if prev_sales else 0
        tx_delta = ((cur_tx - prev_tx) / prev_tx * 100) if prev_tx else 0
        lines = [
            f"- Total Sales: ₱{cur_sales:,.0f} ({sales_delta:+.1f}% vs prev)",
            f"- Transactions: {cur_tx:,} ({tx_delta:+.1f}% vs prev)",
            f"- Window: {start} → {end - timedelta(days=1)} (prev: {prev_start} → {prev_end - timedelta(days=1)})",
        ]
        return "\n".join(["**Executive Summary**", *lines])
    except Exception:
        return "No material changes"

def enhance_briefing_with_llm(md_text: str) -> str:
    client = get_claude_client()
    if not client:
        return md_text
    try:
        resp = client.messages.create(model="claude-3-haiku-20240307", max_tokens=600,
                                      messages=[{"role":"user","content":f"Improve this executive retail briefing, keep bullet points:\n\n{md_text}"}])
        return resp.content[0].text
    except Exception:
        return md_text

def send_briefing_email(md_text: str):
    # Stub for n8n/Gmail webhook
    pass

###############################################
# Weekly Performance Review (WPR)
###############################################

WPR_SYSTEM_PROMPT = (
    "You are a Senior Business Analyst for Aji Ichiban, tasked with generating comprehensive weekly business briefings.\n\n"
    "You will receive one JSON object named analysis_json that covers the LAST 4 FULL ISO WEEKS in Asia/Manila timezone.\n"
    "The data is specifically filtered to include ONLY these five stores: 'Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', and 'Fairview'.\n\n"
    "The JSON includes: meta (timezone, window, week_starts), overall_weekly totals, stores_weekly, products_weekly (with categories),\n"
    "daily totals, hourly patterns, category_weekly (with within-week share), and inventory_low.\n\n"
    "Your job: produce an executive-quality, narrative business briefing that reads like a professional analyst report.\n"
    "Focus on actionable insights and clear business implications. For all week-over-week (WoW) comparisons, compare the latest week (Week 4) against the prior week (Week 3).\n\n"
    "REQUIRED OUTPUT SECTIONS (in this exact order):\n\n"
    "## Executive Summary\n"
    "   - Total sales trend across the 4 weeks, comparing the final week's sales to the first week's (W4 vs W1). Call the trend: UP / DOWN / MIXED.\n"
    "   - Best week and worst week by total sales, with pesos and % vs its prior week.\n\n"
    "## Store Performance\n"
    "   - Top 3 stores by latest week's sales and Top 3 WoW improvers by absolute ₱ increase.\n"
    "   - Bottom 3 stores by WoW decline by absolute ₱ decrease.\n"
    "   - One line per highlighted store: ₱this_week (vs ₱last_week, ±X.X%). If a store had no sales in the prior week, label it as \"(New Activity)\" instead of showing a percentage.\n\n"
    "## Product Trends\n"
    "   - Top 5 products UP and Top 5 DOWN by absolute ₱ revenue change in the latest week vs. the prior week.\n"
    "   - Call out notable category shifts affecting these products. If a product is new to the top movers, mention it.\n\n"
    "## Category Momentum\n"
    "   - Which categories gained or lost the most market share in the latest week vs. the prior week (report change in percentage points, e.g., \"+1.5 pts\").\n"
    "   - Briefly explain what this momentum implies for sales focus.\n\n"
    "## Time Patterns\n"
    "   - Best weekday (average) and worst weekday (average) by sales over the 4-week period.\n"
    "   - Peak sales hour overall (0-23). Note if weekend vs. weekday peak hours differ materially.\n\n"
    "## Inventory Watch\n"
    "   - From inventory_low: list the 5 most urgent items. Urgency is ranked by the largest deficit (warning_stock - on_hand).\n"
    "   - Provide a clear action for each item: [Store: Product] is at X/Y. Action: Restock / Redistribute / Monitor.\n\n"
    "## Risks & Opportunities\n"
    "   - 2 critical risks (e.g., a top store is declining, a key category is losing share, a bestseller is near stockout).\n"
    "   - 2 high-potential opportunities (e.g., a product is surging, a store is showing strong momentum, a weekend time slot is outperforming).\n\n"
    "FORMATTING RULES:\n"
    "- Use markdown headers (##) for each section as shown above\n"
    "- Always show pesos like ₱1,234,567 (no decimals). Percentages with one decimal (±X.X%).\n"
    "- Base everything on the provided JSON only. If data is missing or too sparse for a section, state that explicitly.\n"
    "- Keep the narrative professional and actionable: \"what happened, why it matters, what to do next\".\n"
    "- Write in a business analyst tone - clear, concise, and focused on insights that drive decisions."
)

@st.cache_data(ttl=3600)
def fetch_weekly_performance_review_data(store_ids: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Execute the one-shot SQL to produce a single JSON snapshot for the last 4 full ISO weeks.
    Returns a Python dict or None.
    """
    has_store_ids = bool(store_ids)
    store_filter_clause = "AND t.store_id = ANY(%(store_ids)s)" if has_store_ids else ""
    sql = f"""
-- Optimized SQL for Weekly Performance Review
WITH tz AS (
  SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
),
bounds AS (
  SELECT
    DATE_TRUNC('week', today_mnl)::date AS this_week_start,
    (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
    DATE_TRUNC('week', today_mnl)::date AS window_end
  FROM tz
),
weeks AS (
  SELECT (window_end - INTERVAL '28 days')::date AS week_start FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '21 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '14 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '7 days')::date FROM bounds
),
sales_tx AS (
  SELECT
    t.ref_id,
    t.store_id,
    (t.transaction_time AT TIME ZONE 'Asia/Manila') AS ts_mnl,
    (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d_mnl,
    DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS wk_start,
    t.total::numeric AS total
  FROM transactions t, bounds b
  WHERE LOWER(t.transaction_type)='sale'
    AND COALESCE(t.is_cancelled,false)=false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
    {store_filter_clause} -- Placeholder for dynamic store filter
),
item_level_sales AS (
  SELECT
    st.wk_start,
    st.store_id,
    ti.product_id,
    p.name AS product_name,
    COALESCE(p.category, 'Uncategorized') AS category,
    ti.item_total::numeric AS revenue
  FROM transaction_items ti
  JOIN products p ON p.id = ti.product_id
  JOIN sales_tx st ON st.ref_id = ti.transaction_ref_id
),
overall_weekly AS (
  SELECT st.wk_start AS week_start, SUM(st.total)::numeric AS sales, COUNT(DISTINCT st.ref_id)::bigint AS tx_count
  FROM sales_tx st GROUP BY st.wk_start
),
stores_weekly AS (
  SELECT st.wk_start AS week_start, s.id AS store_id, s.name AS store_name, SUM(st.total)::numeric AS sales
  FROM sales_tx st JOIN stores s ON s.id = st.store_id
  GROUP BY st.wk_start, s.id, s.name
),
products_weekly AS (
  SELECT week_start, product_id, product_name, category, SUM(revenue)::numeric AS revenue
  FROM item_level_sales
  GROUP BY 1, 2, 3, 4
),
daily AS (
  SELECT st.d_mnl AS sales_date, st.wk_start AS week_start,
         EXTRACT(ISODOW FROM st.d_mnl)::int AS dow_iso, TRIM(TO_CHAR(st.d_mnl, 'Day')) AS dow_name,
         SUM(st.total)::numeric AS sales
  FROM sales_tx st GROUP BY 1, 2, 3, 4
),
hourly AS (
  SELECT h.hour,
         COALESCE(SUM(st.total), 0)::numeric AS sales_all,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) BETWEEN 1 AND 5), 0)::numeric AS sales_weekday,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) IN (6,7)), 0)::numeric AS sales_weekend
  FROM generate_series(0,23) AS h(hour)
  LEFT JOIN sales_tx st ON EXTRACT(HOUR FROM st.ts_mnl)::int = h.hour
  GROUP BY 1
),
category_weekly AS (
  SELECT
    week_start,
    category,
    revenue,
    (revenue / NULLIF(SUM(revenue) OVER (PARTITION BY week_start), 0))::numeric AS share
  FROM (
    SELECT week_start, category, SUM(revenue)::numeric as revenue
    FROM item_level_sales
    GROUP BY 1, 2
  ) AS cat_base
),
inventory_low AS (
  SELECT i.product_id, p.name AS product_name, i.store_id, s.name AS store_name,
         i.quantity_on_hand AS on_hand, i.warning_stock
  FROM inventory i JOIN products p ON p.id = i.product_id JOIN stores s ON s.id = i.store_id
  WHERE i.quantity_on_hand <= COALESCE(i.warning_stock,0)
  AND (NOT %(has_store_ids)s OR i.store_id = ANY(%(store_ids)s))
)
SELECT jsonb_build_object(
  'meta', jsonb_build_object(
    'timezone', 'Asia/Manila',
    'window_start', (SELECT window_start FROM bounds),
    'window_end', (SELECT window_end FROM bounds),
    'week_starts', (SELECT jsonb_agg(w.week_start ORDER BY w.week_start) FROM weeks w),
    'store_filter', CASE WHEN NOT %(has_store_ids)s THEN 'ALL'::jsonb ELSE to_jsonb(%(store_ids)s) END
  ),
  'overall_weekly', (SELECT COALESCE(jsonb_agg(ow ORDER BY ow.week_start),'[]'::jsonb) FROM overall_weekly ow),
  'stores_weekly', (SELECT COALESCE(jsonb_agg(sw ORDER BY sw.week_start, sw.store_name),'[]'::jsonb) FROM stores_weekly sw),
  'products_weekly', (SELECT COALESCE(jsonb_agg(pw ORDER BY pw.week_start, pw.revenue DESC),'[]'::jsonb) FROM products_weekly pw),
  'daily', (SELECT COALESCE(jsonb_agg(d ORDER BY d.sales_date),'[]'::jsonb) FROM daily d),
  'hourly', (SELECT COALESCE(jsonb_agg(h ORDER BY h.hour),'[]'::jsonb) FROM hourly h),
  'category_weekly', (SELECT COALESCE(jsonb_agg(cw ORDER BY cw.week_start, cw.revenue DESC),'[]'::jsonb) FROM category_weekly cw),
  'inventory_low', (SELECT COALESCE(jsonb_agg(il ORDER BY il.store_name, il.product_name),'[]'::jsonb) FROM inventory_low il)
) AS analysis_json;
"""
    params = {"store_ids": store_ids or [], "has_store_ids": has_store_ids}
    df = execute_query_for_dashboard(sql, params=params)
    if df is None or df.empty or 'analysis_json' not in df.columns:
        return None
    val = df.iloc[0]['analysis_json']
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    if isinstance(val, (dict, list)):
        return val  # already parsed by driver
    return None

def analyze_weekly_performance_with_ai(analysis: Dict[str, Any]) -> str:
    """Send the analysis JSON to Claude 3.5 Sonnet with the business analyst system prompt."""
    client = get_claude_client()
    if not client:
        return "Claude API key not configured. Cannot generate Weekly Performance Review."
    try:
        payload = json.dumps({"analysis_json": analysis})
        resp = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1800,
            system=WPR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"AI generation failed: {e}"

# Backwards-compatible aliases with names referenced in the spec
@st.cache_data(ttl=3600)
def get_weekly_review_json_corrected() -> Optional[Dict[str, Any]]:
    """Fetch Weekly Performance Review analysis_json for a fixed set of five stores.
    Uses the hardcoded optimized SQL (Asset 1) permanently filtered to the target stores.
    """
    sql = """
-- Hardcoded SQL for Weekly Performance Review
WITH tz AS (
  SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
),
bounds AS (
  SELECT 
    DATE_TRUNC('week', today_mnl)::date AS this_week_start,
    (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
    DATE_TRUNC('week', today_mnl)::date AS window_end
  FROM tz
),
weeks AS (
  SELECT (window_end - INTERVAL '28 days')::date AS week_start FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '21 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '14 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '7 days')::date FROM bounds
),
sales_tx AS (
  SELECT
    t.ref_id,
    t.store_id,
    (t.transaction_time AT TIME ZONE 'Asia/Manila') AS ts_mnl,
    (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d_mnl,
    DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS wk_start,
    t.total::numeric AS total
  FROM transactions t
  JOIN stores s ON t.store_id = s.id, bounds b
  WHERE
    LOWER(t.transaction_type)='sale'
    AND COALESCE(t.is_cancelled,false)=false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
    AND s.name IN ('Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview')
),
item_level_sales AS (
  SELECT
    st.wk_start,
    st.store_id,
    ti.product_id,
    p.name AS product_name,
    COALESCE(p.category, 'Uncategorized') AS category,
    ti.item_total::numeric AS revenue
  FROM transaction_items ti
  JOIN products p ON p.id = ti.product_id
  JOIN sales_tx st ON st.ref_id = ti.transaction_ref_id
),
overall_weekly AS (
  SELECT st.wk_start AS week_start, SUM(st.total)::numeric AS sales, COUNT(DISTINCT st.ref_id)::bigint AS tx_count
  FROM sales_tx st GROUP BY st.wk_start
),
stores_weekly AS (
  SELECT st.wk_start AS week_start, s.id AS store_id, s.name AS store_name, SUM(st.total)::numeric AS sales
  FROM sales_tx st JOIN stores s ON s.id = st.store_id
  GROUP BY st.wk_start, s.id, s.name
),
products_weekly AS (
  SELECT wk_start AS week_start, product_id, product_name, category, SUM(revenue)::numeric AS revenue
  FROM item_level_sales
  GROUP BY 1, 2, 3, 4
),
daily AS (
  SELECT st.d_mnl AS sales_date, st.wk_start AS week_start,
         EXTRACT(ISODOW FROM st.d_mnl)::int AS dow_iso, TRIM(TO_CHAR(st.d_mnl, 'Day')) AS dow_name,
         SUM(st.total)::numeric AS sales
  FROM sales_tx st GROUP BY 1, 2, 3, 4
),
hourly AS (
  SELECT h.hour,
         COALESCE(SUM(st.total), 0)::numeric AS sales_all,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) BETWEEN 1 AND 5), 0)::numeric AS sales_weekday,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) IN (6,7)), 0)::numeric AS sales_weekend
  FROM generate_series(0,23) AS h(hour)
  LEFT JOIN sales_tx st ON EXTRACT(HOUR FROM st.ts_mnl)::int = h.hour
  GROUP BY 1
),
category_weekly AS (
  SELECT
    wk_start AS week_start,
    category,
    revenue,
    (revenue / NULLIF(SUM(revenue) OVER (PARTITION BY wk_start), 0))::numeric AS share
  FROM (
    SELECT wk_start, category, SUM(revenue)::numeric as revenue
    FROM item_level_sales
    GROUP BY 1, 2
  ) AS cat_base
),
inventory_low AS (
  SELECT i.product_id, p.name AS product_name, i.store_id, s.name AS store_name,
         i.quantity_on_hand AS on_hand, i.warning_stock
  FROM inventory i JOIN products p ON p.id = i.product_id JOIN stores s ON s.id = i.store_id
  WHERE i.quantity_on_hand <= COALESCE(i.warning_stock,0)
  AND s.name IN ('Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview')
)
SELECT jsonb_build_object(
  'meta', jsonb_build_object(
    'timezone', 'Asia/Manila',
    'window_start', (SELECT window_start FROM bounds),
    'window_end', (SELECT window_end FROM bounds),
    'week_starts', (SELECT jsonb_agg(w.week_start ORDER BY w.week_start) FROM weeks w),
    'store_filter', to_jsonb(ARRAY['Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview'])
  ),
  'overall_weekly', (SELECT COALESCE(jsonb_agg(ow ORDER BY ow.week_start),'[]'::jsonb) FROM overall_weekly ow),
  'stores_weekly', (SELECT COALESCE(jsonb_agg(sw ORDER BY sw.week_start, sw.store_name),'[]'::jsonb) FROM stores_weekly sw),
  'products_weekly', (SELECT COALESCE(jsonb_agg(pw ORDER BY pw.week_start, pw.revenue DESC),'[]'::jsonb) FROM products_weekly pw),
  'daily', (SELECT COALESCE(jsonb_agg(d ORDER BY d.sales_date),'[]'::jsonb) FROM daily d),
  'hourly', (SELECT COALESCE(jsonb_agg(h ORDER BY h.hour),'[]'::jsonb) FROM hourly h),
  'category_weekly', (SELECT COALESCE(jsonb_agg(cw ORDER BY cw.week_start, cw.revenue DESC),'[]'::jsonb) FROM category_weekly cw),
  'inventory_low', (SELECT COALESCE(jsonb_agg(il ORDER BY il.store_name, il.product_name),'[]'::jsonb) FROM inventory_low il)
) AS analysis_json;
"""
    df = execute_query_for_dashboard(sql)
    if df is None or df.empty or 'analysis_json' not in df.columns:
        return None
    val = df.iloc[0]['analysis_json']
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    if isinstance(val, (dict, list)):
        return val
    return None

# Function moved to avoid duplication - see line 5700

def render_weekly_performance_review():
    """UI for Weekly Performance Review with store name -> ID mapping via get_store_list()."""
    st.markdown('<div class="main-header"><h1>📊 Weekly Performance Review</h1><p>Executive-quality briefing across the last 4 full weeks</p></div>', unsafe_allow_html=True)

    # Load store list using existing helper that returns both id and name
    try:
        store_df = get_store_list()  # expected columns: id, name
    except Exception:
        store_df = execute_query_for_dashboard("SELECT id, name FROM stores")

    names = store_df['name'].tolist() if store_df is not None and not store_df.empty else []

    # Multiselect of names only; empty selection => ALL stores
    selection = st.multiselect("Select Stores (optional)", options=names, default=[])
    selected_ids: Optional[List[int]] = None
    if selection and store_df is not None and not store_df.empty:
        selected_ids = store_df[store_df['name'].isin(selection)]['id'].astype(int).tolist()
    else:
        selected_ids = None  # treat as ALL stores

    if st.button("Generate Weekly Review", type="primary"):
        with st.spinner("Fetching weekly data snapshot..."):
            analysis = get_weekly_review_json(selected_ids)
        if not analysis:
            st.warning("No sales data was found for the selected stores in the last 4 full weeks.")
            return
        with st.spinner("Generating executive briefing with AI..."):
            report_md = get_ai_weekly_briefing(analysis)
        st.markdown(report_md)
        with st.expander("View raw analysis_json"):
            st.json(analysis)

def render_weekly_review():
    """Repurposed AI Intelligence Hub page showing the automated Weekly Performance Review.
    Fixed to five stores; no user-facing filters.
    """
    st.markdown(
        '<div class="main-header"><h1>🧠 AI Intelligence Hub</h1><p>Automated Weekly Performance Review</p></div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate Weekly Review", type="primary"):
        with st.spinner("Fetching weekly data snapshot..."):
            analysis = get_weekly_review_json()
        if not analysis:
            st.warning("No sales data was found for the selected stores in the last 4 full weeks.")
            return
        with st.spinner("Generating executive briefing with AI..."):
            report_md = get_ai_weekly_briefing(analysis)
        st.markdown(report_md)
        with st.expander("View raw analysis_json"):
            st.json(analysis)

@st.cache_data(ttl=1800)
def get_overall_weekly_data():
    """Get overall weekly sales totals (4 rows only)"""
    conn = create_db_connection()
    if not conn:
        return []
    
    try:
        sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            DATE_TRUNC('week', today_mnl)::date AS this_week_start,
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        ),
        sales_tx AS (
          SELECT
            DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS wk_start,
            t.total::numeric AS total
          FROM transactions t, bounds b
          WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
        )
        SELECT 
          wk_start AS week_start,
          SUM(total)::numeric AS sales,
          COUNT(*)::bigint AS tx_count
        FROM sales_tx
        GROUP BY wk_start
        ORDER BY wk_start
        """
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return [{"week_start": row[0].strftime('%Y-%m-%d'), "sales": float(row[1]), "tx_count": int(row[2])} for row in results]
        
    except Exception as e:
        print(f"Error in overall weekly: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=1800)
def get_stores_weekly_data():
    """Get top stores by week (limited to top 10 stores)"""
    conn = create_db_connection()
    if not conn:
        return []
    
    try:
        sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        ),
        store_totals AS (
          SELECT 
            s.name AS store_name,
            SUM(t.total)::numeric AS total_sales
          FROM transactions t
          JOIN stores s ON s.id = t.store_id, bounds b
          WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
          GROUP BY s.name
          ORDER BY total_sales DESC
          LIMIT 10
        )
        SELECT 
          DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS week_start,
          s.name AS store_name,
          SUM(t.total)::numeric AS sales
        FROM transactions t
        JOIN stores s ON s.id = t.store_id
        JOIN store_totals st ON st.store_name = s.name, bounds b
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
        GROUP BY week_start, s.name
        ORDER BY week_start, sales DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return [{"week_start": row[0].strftime('%Y-%m-%d'), "store_name": row[1], "sales": float(row[2])} for row in results]
        
    except Exception as e:
        print(f"Error in stores weekly: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=1800)
def get_products_weekly_data():
    """Get top products by week (limited to top 20 products)"""
    conn = create_db_connection()
    if not conn:
        return []
    
    try:
        sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        ),
        top_products AS (
          SELECT 
            p.name AS product_name,
            SUM(ti.item_total)::numeric AS total_revenue
          FROM transaction_items ti
          JOIN transactions t ON t.ref_id = ti.transaction_ref_id
          JOIN products p ON p.id = ti.product_id, bounds b
          WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
          GROUP BY p.name
          ORDER BY total_revenue DESC
          LIMIT 20
        )
        SELECT 
          DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS week_start,
          p.name AS product_name,
          COALESCE(p.category, 'Uncategorized') AS category,
          SUM(ti.item_total)::numeric AS revenue
        FROM transaction_items ti
        JOIN transactions t ON t.ref_id = ti.transaction_ref_id
        JOIN products p ON p.id = ti.product_id
        JOIN top_products tp ON tp.product_name = p.name, bounds b
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
        GROUP BY week_start, p.name, p.category
        ORDER BY week_start, revenue DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return [{"week_start": row[0].strftime('%Y-%m-%d'), "product_name": row[1], "category": row[2], "revenue": float(row[3])} for row in results]
        
    except Exception as e:
        print(f"Error in products weekly: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=1800)
def get_category_weekly_data():
    """Get category performance by week"""
    conn = create_db_connection()
    if not conn:
        return []
    
    try:
        sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        ),
        cat_wk AS (
          SELECT 
            DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS week_start,
            COALESCE(p.category, 'Uncategorized') AS category,
            SUM(ti.item_total)::numeric AS revenue
          FROM transaction_items ti
          JOIN transactions t ON t.ref_id = ti.transaction_ref_id
          JOIN products p ON p.id = ti.product_id, bounds b
          WHERE LOWER(t.transaction_type) = 'sale'
            AND COALESCE(t.is_cancelled, false) = false
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
            AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
          GROUP BY week_start, p.category
        ),
        cat_tot AS (
          SELECT week_start, SUM(revenue)::numeric AS total_rev
          FROM cat_wk GROUP BY week_start
        )
        SELECT 
          c.week_start,
          c.category,
          c.revenue,
          (c.revenue / NULLIF(t.total_rev, 0))::numeric AS share
        FROM cat_wk c
        JOIN cat_tot t USING (week_start)
        ORDER BY c.week_start, c.revenue DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return [{"week_start": row[0].strftime('%Y-%m-%d'), "category": row[1], "revenue": float(row[2]), "share": float(row[3] or 0)} for row in results]
        
    except Exception as e:
        print(f"Error in category weekly: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=1800)
def get_time_patterns_data():
    """Get daily and hourly patterns (aggregated)"""
    conn = create_db_connection()
    if not conn:
        return {"daily": [], "hourly": []}
    
    try:
        # Daily patterns - Fixed query
        daily_sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        )
        SELECT 
          EXTRACT(ISODOW FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS dow_iso,
          CASE EXTRACT(ISODOW FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int
            WHEN 1 THEN 'Monday'
            WHEN 2 THEN 'Tuesday' 
            WHEN 3 THEN 'Wednesday'
            WHEN 4 THEN 'Thursday'
            WHEN 5 THEN 'Friday'
            WHEN 6 THEN 'Saturday'
            WHEN 7 THEN 'Sunday'
          END AS dow_name,
          SUM(t.total)::numeric AS total_sales,
          COUNT(DISTINCT (t.transaction_time AT TIME ZONE 'Asia/Manila')::date) AS days_count,
          (SUM(t.total) / COUNT(DISTINCT (t.transaction_time AT TIME ZONE 'Asia/Manila')::date))::numeric AS avg_daily_sales
        FROM transactions t, bounds b
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
        GROUP BY dow_iso, dow_name
        HAVING COUNT(*) > 0
        ORDER BY avg_daily_sales DESC
        """
        
        # Hourly patterns - Enhanced query with time formatting
        hourly_sql = """
        WITH tz AS (
          SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
        ),
        bounds AS (
          SELECT 
            (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
            DATE_TRUNC('week', today_mnl)::date AS window_end
          FROM tz
        )
        SELECT 
          EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int AS hour,
          CASE 
            WHEN EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int = 0 THEN '12:00 AM'
            WHEN EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int < 12 THEN 
              EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int || ':00 AM'
            WHEN EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int = 12 THEN '12:00 PM'
            ELSE 
              (EXTRACT(HOUR FROM (t.transaction_time AT TIME ZONE 'Asia/Manila'))::int - 12) || ':00 PM'
          END AS hour_formatted,
          SUM(t.total)::numeric AS total_sales,
          SUM(CASE WHEN EXTRACT(ISODOW FROM (t.transaction_time AT TIME ZONE 'Asia/Manila')) BETWEEN 1 AND 5 THEN t.total ELSE 0 END)::numeric AS weekday_sales,
          SUM(CASE WHEN EXTRACT(ISODOW FROM (t.transaction_time AT TIME ZONE 'Asia/Manila')) IN (6, 7) THEN t.total ELSE 0 END)::numeric AS weekend_sales
        FROM transactions t, bounds b
        WHERE LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
          AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
        GROUP BY hour, hour_formatted
        HAVING SUM(t.total) > 0
        ORDER BY total_sales DESC
        """
        
        cursor = conn.cursor()
        
        # Get daily data
        cursor.execute(daily_sql)
        daily_results = cursor.fetchall()
        daily_data = [{"dow_iso": row[0], "dow_name": row[1], "total_sales": float(row[2]), "days_count": row[3], "avg_daily_sales": float(row[4])} for row in daily_results]
        
        # Get hourly data
        cursor.execute(hourly_sql)
        hourly_results = cursor.fetchall()
        hourly_data = [{"hour": row[0], "hour_formatted": row[1], "total_sales": float(row[2]), "weekday_sales": float(row[3]), "weekend_sales": float(row[4])} for row in hourly_results]
        
        return {"daily": daily_data, "hourly": hourly_data}
        
    except Exception as e:
        print(f"Error in time patterns: {e}")
        return {"daily": [], "hourly": []}
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=1800)
def get_inventory_alerts():
    """Get low inventory items (top 5 only)"""
    conn = create_db_connection()
    if not conn:
        return []
    
    try:
        sql = """
        SELECT 
          s.name AS store_name,
          p.name AS product_name,
          i.quantity_on_hand AS on_hand,
          COALESCE(i.warning_stock, 0) AS warning_stock
        FROM inventory i
        JOIN products p ON p.id = i.product_id
        JOIN stores s ON s.id = i.store_id
        WHERE i.quantity_on_hand <= COALESCE(i.warning_stock, 0)
        ORDER BY (COALESCE(i.warning_stock, 0) - i.quantity_on_hand) DESC
        LIMIT 5
        """
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return [{"store_name": row[0], "product_name": row[1], "on_hand": row[2], "warning_stock": row[3]} for row in results]
        
    except Exception as e:
        print(f"Error in inventory alerts: {e}")
        return []
    finally:
        if conn:
            conn.close()

def generate_ai_intelligence_summary():
    """Generate AI-powered business intelligence summary using split data queries."""
    client = get_openai_client()
    if not client:
        return "❌ OpenAI API not configured. Please check your API key in Settings."
    
    # Fetch data from split queries
    overall_weekly = get_overall_weekly_data()
    stores_weekly = get_stores_weekly_data()
    products_weekly = get_products_weekly_data()
    category_weekly = get_category_weekly_data()
    time_patterns = get_time_patterns_data()
    inventory_alerts = get_inventory_alerts()
    
    if not overall_weekly:
        return "❌ No data available for analysis."
    
    # Prepare compact data for AI
    analysis_data = {
        "overall_weekly": overall_weekly,
        "stores_weekly": stores_weekly,
        "products_weekly": products_weekly,
        "category_weekly": category_weekly,
        "daily_patterns": time_patterns["daily"],
        "hourly_patterns": time_patterns["hourly"],
        "inventory_alerts": inventory_alerts
    }

    prompt = f"""You are an expert business intelligence analyst. Create a concise 4-week business briefing.

DATA CONTEXT:
- Dataset covers last 4 full ISO weeks in Asia/Manila timezone
- All figures are in Philippine Pesos (₱)
- Week-over-Week (WoW) refers to Week 4 vs Week 3 unless specified
- Sales-only transactions included

CRITICAL FORMATTING RULES:
- ALWAYS show both actual values when comparing periods: "₱500,000 (Week 4) vs ₱450,000 (Week 3): +11.1%"
- ALWAYS specify what periods are being compared: "Week 4 vs Week 3" or "Week 2 vs Week 1"
- Format: "Current Value vs Previous Value: +/-X.X% change"
- Use ₱1,234,567 format for peso amounts
- Use +/-X.X% format for percentages (one decimal)
- Every claim must be backed by numbers from the data
- If data insufficient for a section, write "Insufficient data for <section>"

REQUIRED SECTIONS:

1) 🏬 Store Performance
- Top 3 stores by sales in Week 4 with exact amounts and share of Week 4 total
- Top 3 WoW improvers with format: "Store: ₱X (Week 4) vs ₱Y (Week 3): +X.X%"
- Bottom 3 WoW decliners with format: "Store: ₱X (Week 4) vs ₱Y (Week 3): -X.X%"

2) 🛒 Product & Category Insights
- Top 5 products UP by revenue with format: "Product: ₱X (Week 4) vs ₱Y (Week 3): +X.X%"
- Top 5 products DOWN by revenue with format: "Product: ₱X (Week 4) vs ₱Y (Week 3): -X.X%"
- Category momentum with format: "Category: X.X% share (Week 4) vs Y.Y% share (Week 3): +/-Z.Z percentage points"

3) ⏰ Time Patterns
- Best and worst weekday with format: "Best: [Day] ₱X average daily sales | Worst: [Day] ₱Y average daily sales"
- Top 5 weekday hours with format: "1. 3:00 PM: ₱X | 2. 7:00 PM: ₱Y | 3. 12:00 PM: ₱Z" etc.
- Top 5 weekend hours with format: "1. 2:00 PM: ₱X | 2. 8:00 PM: ₱Y | 3. 1:00 PM: ₱Z" etc.

4) 📆 Per-Week Spotlight
CRITICAL: Week 1 compares forward to Week 2 (since no previous week exists). Other weeks compare to previous week.
- "Week 1: ₱X vs Week 2's ₱Y: +/-X.X% (forward comparison)"
- "Week 2: ₱X vs Week 1's ₱Y: +/-X.X%"
- "Week 3: ₱X vs Week 2's ₱Y: +/-X.X%"
- "Week 4: ₱X vs Week 3's ₱Y: +/-X.X%"

For each week, include:
- What went UP that week vs comparison week (stores, products, categories)
- What went DOWN that week vs comparison week (stores, products, categories)
- Key standouts with actual numbers

5) ⚠️ Risks & 🚀 Opportunities
- 2-3 critical risks with exact numbers and comparisons (NO INVENTORY ITEMS)
- Focus on: declining stores, declining products, market share losses
- 2-3 high-leverage opportunities with exact numbers and comparisons
- Focus on: growing stores, growing products, category expansions
- Format: "Risk/Opportunity: ₱X vs ₱Y represents Z.Z% change"

6) ✅ Action Plan
- 3-4 decisive actions with estimated impact when supportable (NO INVENTORY ACTIONS, NO INVENTORY MENTIONS)
- Focus ONLY on: marketing strategies, store promotions, product campaigns, category optimization, customer experience
- Format: "Action: [Marketing/Promotion strategy] — est. impact ₱X based on data showing ₱Y vs ₱Z trends"
- NEVER mention inventory, stock levels, replenishing, or restocking

DO NOT INCLUDE:
- Any inventory-related risks, opportunities, or actions
- Any inventory alerts or stock level mentions
- Any references to "increase inventory" or "replenish stock"

EXAMPLE:
Week 1: ₱1,379,438 vs Week 2's ₱1,463,829: -5.7%
Action: Launch Sunday marketing campaign — est. impact ₱361,014

DATA TO ANALYZE:
{json.dumps(analysis_data, indent=1)}

Generate the analysis now with BOTH VALUES in every comparison:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert business intelligence analyst. ALWAYS show both actual values when making comparisons, never just percentages. Format: '₱X (Period A) vs ₱Y (Period B): +/-Z.Z%'. Every comparison must include the raw numbers from both periods being compared."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Failed to generate intelligence summary: {e}"

@st.cache_data(ttl=300)
def hidden_demand_detection(store_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    store_clause = ""
    params: List[Any] = []
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)
    sql = f"""
    WITH window AS (
      SELECT (DATE(NOW() AT TIME ZONE 'Asia/Manila') - INTERVAL '30 days') AS start_d,
             (DATE(NOW() AT TIME ZONE 'Asia/Manila')) AS end_d
    ), sales_daily AS (
      SELECT DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') AS d,
             t.store_id, ti.product_id,
             SUM(ti.quantity)::numeric AS units,
             SUM(ti.item_total)::numeric AS revenue
      FROM transaction_items ti
      JOIN transactions t ON ti.transaction_ref_id = t.ref_id, window w
      WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= w.start_d
        AND (t.transaction_time AT TIME ZONE 'Asia/Manila') <  w.end_d
        {store_clause}
      GROUP BY 1,2,3
    ), inv_daily AS (
      SELECT i.store_id, i.product_id,
             DATE(i.updated_at AT TIME ZONE 'Asia/Manila') AS d,
             MAX(i.quantity_on_hand)::numeric AS on_hand
      FROM inventory i, window w
      WHERE (i.updated_at AT TIME ZONE 'Asia/Manila') >= w.start_d
        AND (i.updated_at AT TIME ZONE 'Asia/Manila') <  w.end_d
      GROUP BY 1,2,3
    ), joined AS (
      SELECT COALESCE(sd.d, id.d) AS d,
             COALESCE(sd.store_id, id.store_id) AS store_id,
             COALESCE(sd.product_id, id.product_id) AS product_id,
             COALESCE(sd.units,0) AS units,
             COALESCE(sd.revenue,0) AS revenue,
             COALESCE(id.on_hand,0) AS on_hand
      FROM sales_daily sd
      FULL OUTER JOIN inv_daily id
        ON sd.d = id.d AND sd.store_id = id.store_id AND sd.product_id = id.product_id
    ), baseline AS (
      SELECT store_id, product_id,
             PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY units) FILTER (WHERE on_hand > 0) AS units_median
      FROM joined
      GROUP BY 1,2
    )
    SELECT 
      j.product_id AS product_id,
      p.name AS sku,
      s.name AS store,
      SUM(GREATEST(b.units_median - j.units, 0))::numeric AS suppressed_30d,
      COUNT(*) FILTER (WHERE j.on_hand = 0) AS oos_days,
      (SUM(GREATEST(b.units_median - j.units, 0)) / NULLIF(b.units_median*30,0))::numeric AS score,
      COALESCE(SUM(j.revenue)/NULLIF(SUM(j.units),0),0) * SUM(GREATEST(b.units_median - j.units, 0)) AS est_impact_php
    FROM joined j
    JOIN baseline b ON b.store_id = j.store_id AND b.product_id = j.product_id
    LEFT JOIN products p ON p.id = j.product_id
    LEFT JOIN stores s ON s.id = j.store_id
    GROUP BY 1,2,3, b.units_median
    HAVING b.units_median > 0
    ORDER BY est_impact_php DESC
    LIMIT 200
    """
    df = execute_query_for_dashboard(sql, params=params if params else None)
    if df is None:
        return None
    df['score'] = df['score'].clip(lower=0, upper=1)
    return df.rename(columns={"suppressed_30d":"Suppressed Units (30d)", "score":"Score", "oos_days":"OOS Days", "est_impact_php":"est_impact_php"})

@st.cache_data(ttl=300)
def stockout_predictions(store_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    # Updated to use intelligent date ranges (7D = current week to date)
    store_clause = ""
    params: List[Any] = []
    if store_ids:
        store_clause = "AND dv.store_id = ANY(%s)"
        params.append(store_ids)
    sql = f"""
    WITH dv AS (
      SELECT i.store_id, i.product_id,
             GREATEST(AVG(u.units), 0.0)::numeric AS avg_daily
      FROM (
        SELECT DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') AS d,
               t.store_id, ti.product_id, SUM(ti.quantity)::numeric AS units
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        WHERE LOWER(t.transaction_type)='sale' AND COALESCE(t.is_cancelled,false)=false
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= DATE_TRUNC('week', CURRENT_DATE)
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') <= CURRENT_DATE
        GROUP BY 1,2,3
      ) u
      RIGHT JOIN inventory i ON i.store_id = u.store_id AND i.product_id = u.product_id
      GROUP BY 1,2
    ), inbound AS (
      SELECT po.store_id, poi.product_id,
             SUM(poi.quantity) FILTER (WHERE po.eta <= (NOW() AT TIME ZONE 'Asia/Manila') + INTERVAL '7 days')::numeric AS inbound_7d
      FROM purchase_orders po
      JOIN purchase_order_items poi ON poi.po_id = po.id
      GROUP BY 1,2
    )
    SELECT p.name AS sku, s.name AS store,
           COALESCE(i.quantity_on_hand,0)::numeric AS on_hand,
           COALESCE(inb.inbound_7d,0)::numeric AS inbound_7d,
           COALESCE(dv.avg_daily,0)::numeric AS avg_daily,
           (COALESCE(i.quantity_on_hand,0)+COALESCE(inb.inbound_7d,0))/NULLIF(COALESCE(dv.avg_daily,0),0) AS days_cover,
           (DATE(NOW() AT TIME ZONE 'Asia/Manila') + ((COALESCE(i.quantity_on_hand,0)+COALESCE(inb.inbound_7d,0))/NULLIF(COALESCE(dv.avg_daily,0),0))::int) AS predicted_stockout_date
    FROM inventory i
    LEFT JOIN dv ON dv.store_id = i.store_id AND dv.product_id = i.product_id
    LEFT JOIN inbound inb ON inb.store_id = i.store_id AND inb.product_id = i.product_id
    LEFT JOIN products p ON p.id = i.product_id
    LEFT JOIN stores s ON s.id = i.store_id
    WHERE 1=1 {store_clause}
    ORDER BY predicted_stockout_date NULLS LAST
    LIMIT 200
    """
    return execute_query_for_dashboard(sql, params=params if params else None)

@st.cache_data(ttl=300)
def demand_forecasts(store_ids: Optional[List[int]] = None) -> Optional[Dict[str, pd.DataFrame]]:
    df = _fetch_daily_sales_units_365(store_ids)
    if df is None or df.empty:
        return None
    df['weekday'] = pd.to_datetime(df['d']).dt.weekday
    grp = df.groupby(['product_id','store_id'])
    ma28 = grp['units'].transform(lambda s: s.rolling(28, min_periods=7).mean())
    df['ma28'] = ma28.fillna(grp['units'].transform(lambda s: s.rolling(14, min_periods=7).mean())).fillna(0)
    base = df.groupby(['product_id','store_id'])['ma28'].mean().rename('base')
    wk = df.groupby(['product_id','store_id','weekday'])['units'].mean().rename('wk_mean').to_frame().join(base, on=['product_id','store_id'])
    wk['factor'] = (wk['wk_mean'] / wk['base']).replace([np.inf,-np.inf],1).fillna(1)
    wk = wk['factor'].reset_index()
    start = pd.Timestamp(date.today())
    horizon = [7,30,90]
    out_rows = []
    for (pid, sid), g in df.groupby(['product_id','store_id']):
        base_val = g['ma28'].iloc[-1] if len(g)>0 else 0
        vals = {}
        for h in horizon:
            fut = [start + pd.Timedelta(days=i) for i in range(1, h+1)]
            pred = 0
            for d_ in fut:
                wd = d_.weekday()
                f = wk[(wk['product_id']==pid)&(wk['store_id']==sid)&(wk['weekday']==wd)]['factor']
                f = float(f.iloc[0]) if len(f)>0 else 1.0
                pred += base_val * f
            vals[h] = pred
        out_rows.append({'product_id':pid,'store_id':sid,'D+7':vals[7],'D+30':vals[30],'D+90':vals[90]})
    out = pd.DataFrame(out_rows)
    names = execute_query_for_dashboard("SELECT id as product_id, name as sku FROM products")
    stores = execute_query_for_dashboard("SELECT id as store_id, name as store FROM stores")
    store_view = out.merge(names, on='product_id', how='left').merge(stores, on='store_id', how='left')
    store_view = store_view[['sku','store','D+7','D+30','D+90']].fillna(0)
    warehouse_view = None
    supplier_view = None
    return {'store_view': store_view, 'warehouse_view': warehouse_view, 'supplier_view': supplier_view}

@st.cache_data(ttl=300)
def seasonal_intelligence(store_ids: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    df = _fetch_daily_sales_units_365(store_ids)
    if df is None or df.empty:
        return None
    df['week'] = pd.to_datetime(df['d']).dt.isocalendar().week.astype(int)
    agg = df.groupby(['product_id','store_id','week'])['units'].sum().reset_index()
    total_var = agg.groupby(['product_id','store_id'])['units'].var().rename('totvar')
    seas = agg.groupby(['product_id','store_id','week'])['units'].mean().rename('wkmean').reset_index().join(total_var, on=['product_id','store_id'])
    strength = seas.groupby(['product_id','store_id'])['wkmean'].var() / seas.groupby(['product_id','store_id'])['totvar'].first()
    strength = strength.replace([np.inf,-np.inf],0).fillna(0).reset_index().rename(columns={0:'seasonal_strength'})
    top_week = agg.loc[agg.groupby(['product_id','store_id'])['units'].idxmax()][['product_id','store_id','week']]
    names = execute_query_for_dashboard("SELECT id as product_id, name as sku FROM products")
    stores = execute_query_for_dashboard("SELECT id as store_id, name as store FROM stores")
    out = strength.merge(top_week, on=['product_id','store_id']).merge(names, on='product_id', how='left').merge(stores, on='store_id', how='left')
    out['next_peak_window'] = out['week'].apply(lambda w: f"Week {int(w)}")
    out['action'] = np.where(out['seasonal_strength']>0.5, 'Increase facing/stock before peak', 'Monitor')
    return out[['sku','store','week','seasonal_strength','next_peak_window','action']].rename(columns={'week':'Peak Weeks'})

@st.cache_data(ttl=300)
def get_inventory_risk(time_filter: str, store_ids: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
    dv = stockout_predictions(store_ids)
    if dv is None or dv.empty:
        return {"understock": pd.DataFrame(), "overstock": pd.DataFrame(), "imbalance": pd.DataFrame()}
    df = dv.copy()
    df['days_cover'] = (df['on_hand'] + df.get('inbound_7d', 0)) / df['avg_daily'].replace(0, np.nan)
    df['d14_forecast'] = df['avg_daily'] * 14
    under = df[(df['on_hand'] + df.get('inbound_7d',0)) < df['d14_forecast']].copy()
    under['severity'] = ((under['d14_forecast'] - (under['on_hand'] + under.get('inbound_7d',0))) / under['d14_forecast']).clip(lower=0) * 100
    under = under.rename(columns={'sku':'SKU','store':'Store','on_hand':'On-Hand'})
    under['Suggested Action'] = 'Expedite PO / Replenish'
    under = under[['SKU','Store','On-Hand','days_cover','d14_forecast','severity','Suggested Action']].rename(columns={'days_cover':'Days Cover','d14_forecast':'D+14 Forecast','severity':'Severity'})
    under = under.sort_values('Severity', ascending=False)
    over = df[(df['days_cover'] > 90) | (df['on_hand'] > df['avg_daily'] * 60)].copy()
    over = over.rename(columns={'sku':'SKU','store':'Store','on_hand':'On-Hand'})
    over['Suggested Action'] = 'Mark down / Transfer out'
    over = over[['SKU','Store','On-Hand','days_cover','d14_forecast','avg_daily','Suggested Action']].rename(columns={'days_cover':'Days Cover','d14_forecast':'D+14 Forecast','avg_daily':'Avg Daily'})
    over = over.sort_values('Days Cover', ascending=False)
    tmp = df[['sku','store','on_hand','avg_daily']].copy()
    tmp['cover'] = tmp['on_hand'] / tmp['avg_daily'].replace(0,np.nan)
    under_sku = tmp[tmp['cover'] < 7][['sku','store']].rename(columns={'store':'under_store'})
    over_sku = tmp[tmp['cover'] > 60][['sku','store']].rename(columns={'store':'over_store'})
    imb = under_sku.merge(over_sku, on='sku', how='inner')
    imb = imb.rename(columns={'sku':'SKU'})
    return {"understock": under, "overstock": over, "imbalance": imb}

# --- AI INTELLIGENCE HUB END ---

def get_weekly_analysis_data():
    """Alias for get_weekly_review_json_corrected to provide weekly analysis data."""
    return get_weekly_review_json_corrected()

def get_weekly_review_json():
    """Run the hardcoded Weekly Performance Review SQL and return the JSON object (dict).
    Uses a fixed set of five stores and returns a Python dict parsed from the jsonb result.
    """
    sql = """
-- Hardcoded SQL for Weekly Performance Review
WITH tz AS (
  SELECT (now() AT TIME ZONE 'Asia/Manila')::date AS today_mnl
),
bounds AS (
  SELECT
    DATE_TRUNC('week', today_mnl)::date AS this_week_start,
    (DATE_TRUNC('week', today_mnl)::date - INTERVAL '28 days') AS window_start,
    DATE_TRUNC('week', today_mnl)::date AS window_end
  FROM tz
),
weeks AS (
  SELECT (window_end - INTERVAL '28 days')::date AS week_start FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '21 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '14 days')::date FROM bounds
  UNION ALL SELECT (window_end - INTERVAL '7 days')::date FROM bounds
),
sales_tx AS (
  SELECT
    t.ref_id,
    t.store_id,
    (t.transaction_time AT TIME ZONE 'Asia/Manila') AS ts_mnl,
    (t.transaction_time AT TIME ZONE 'Asia/Manila')::date AS d_mnl,
    DATE_TRUNC('week', (t.transaction_time AT TIME ZONE 'Asia/Manila'))::date AS wk_start,
    t.total::numeric AS total
  FROM transactions t
  JOIN stores s ON t.store_id = s.id, bounds b
  WHERE
    LOWER(t.transaction_type)='sale'
    AND COALESCE(t.is_cancelled,false)=false
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') >= b.window_start
    AND (t.transaction_time AT TIME ZONE 'Asia/Manila') < b.window_end
    AND s.name IN ('Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview')
),
item_level_sales AS (
  SELECT
    st.wk_start,
    st.store_id,
    ti.product_id,
    p.name AS product_name,
    COALESCE(p.category, 'Uncategorized') AS category,
    ti.item_total::numeric AS revenue
  FROM transaction_items ti
  JOIN products p ON p.id = ti.product_id
  JOIN sales_tx st ON st.ref_id = ti.transaction_ref_id
),
overall_weekly AS (
  SELECT st.wk_start AS week_start, SUM(st.total)::numeric AS sales, COUNT(DISTINCT st.ref_id)::bigint AS tx_count
  FROM sales_tx st GROUP BY st.wk_start
),
stores_weekly AS (
  SELECT st.wk_start AS week_start, s.id AS store_id, s.name AS store_name, SUM(st.total)::numeric AS sales
  FROM sales_tx st JOIN stores s ON s.id = st.store_id
  GROUP BY st.wk_start, s.id, s.name
),
products_weekly AS (
  SELECT wk_start AS week_start, product_id, product_name, category, SUM(revenue)::numeric AS revenue
  FROM item_level_sales
  GROUP BY 1, 2, 3, 4
),
daily AS (
  SELECT st.d_mnl AS sales_date, st.wk_start AS week_start,
         EXTRACT(ISODOW FROM st.d_mnl)::int AS dow_iso, TRIM(TO_CHAR(st.d_mnl, 'Day')) AS dow_name,
         SUM(st.total)::numeric AS sales
  FROM sales_tx st GROUP BY 1, 2, 3, 4
),
hourly AS (
  SELECT h.hour,
         COALESCE(SUM(st.total), 0)::numeric AS sales_all,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) BETWEEN 1 AND 5), 0)::numeric AS sales_weekday,
         COALESCE(SUM(st.total) FILTER (WHERE EXTRACT(ISODOW FROM st.ts_mnl) IN (6,7)), 0)::numeric AS sales_weekend
  FROM generate_series(0,23) AS h(hour)
  LEFT JOIN sales_tx st ON EXTRACT(HOUR FROM st.ts_mnl)::int = h.hour
  GROUP BY 1
),
category_weekly AS (
  SELECT
    week_start,
    category,
    revenue,
    (revenue / NULLIF(SUM(revenue) OVER (PARTITION BY week_start), 0))::numeric AS share
  FROM (
    SELECT week_start, category, SUM(revenue)::numeric as revenue
    FROM item_level_sales
    GROUP BY 1, 2
  ) AS cat_base
),
inventory_low AS (
  SELECT i.product_id, p.name AS product_name, i.store_id, s.name AS store_name,
         i.quantity_on_hand AS on_hand, i.warning_stock
  FROM inventory i JOIN products p ON p.id = i.product_id JOIN stores s ON s.id = i.store_id
  WHERE i.quantity_on_hand <= COALESCE(i.warning_stock,0)
  AND s.name IN ('Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview')
)
SELECT jsonb_build_object(
  'meta', jsonb_build_object(
    'timezone', 'Asia/Manila',
    'window_start', (SELECT window_start FROM bounds),
    'window_end', (SELECT window_end FROM bounds),
    'week_starts', (SELECT jsonb_agg(w.week_start ORDER BY w.week_start) FROM weeks w),
    'store_filter', to_jsonb(ARRAY['Rockwell', 'Greenhills', 'Magnolia', 'North Edsa', 'Fairview'])
  ),
  'overall_weekly', (SELECT COALESCE(jsonb_agg(ow ORDER BY ow.week_start),'[]'::jsonb) FROM overall_weekly ow),
  'stores_weekly', (SELECT COALESCE(jsonb_agg(sw ORDER BY sw.week_start, sw.store_name),'[]'::jsonb) FROM stores_weekly sw),
  'products_weekly', (SELECT COALESCE(jsonb_agg(pw ORDER BY pw.week_start, pw.revenue DESC),'[]'::jsonb) FROM products_weekly pw),
  'daily', (SELECT COALESCE(jsonb_agg(d ORDER BY d.sales_date),'[]'::jsonb) FROM daily d),
  'hourly', (SELECT COALESCE(jsonb_agg(h ORDER BY h.hour),'[]'::jsonb) FROM hourly h),
  'category_weekly', (SELECT COALESCE(jsonb_agg(cw ORDER BY cw.week_start, cw.revenue DESC),'[]'::jsonb) FROM category_weekly cw),
  'inventory_low', (SELECT COALESCE(jsonb_agg(il ORDER BY il.store_name, il.product_name),'[]'::jsonb) FROM inventory_low il)
) AS analysis_json;
"""

    # Use assistant executor to surface SQL errors in the UI if the query fails
    df = execute_query_for_assistant(sql)
    if df is None or df.empty:
        return None
    value = df.iloc[0]["analysis_json"]
    # Convert to dict if needed
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


# === NEW SEQUENTIAL BRIEFING SYSTEM ===

import time

def _make_api_call_with_retry(client, system_prompt: str, relevant_data: dict, section_name: str, model_type: str = "claude", max_retries: int = 3) -> str:
    """
    Make API call with exponential backoff retry logic for rate limit handling.
    Supports both Claude and OpenAI models.
    """
    for attempt in range(max_retries):
        try:
            if model_type == "openai":
                # For OpenAI, limit data size to prevent context length issues
                if len(json.dumps(relevant_data)) > 8000:  # Conservative limit
                    # Truncate data for OpenAI to prevent context length errors
                    truncated_data = {}
                    for key, value in relevant_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Keep only first 10 items for lists to reduce size
                            truncated_data[key] = value[:10]
                        else:
                            truncated_data[key] = value
                    relevant_data = truncated_data
                
                # OpenAI API call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Relevant data: {json.dumps(relevant_data)}"}
                    ],
                    max_tokens=600,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            else:
                # Claude API call (default)
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=600,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"Relevant data: {json.dumps(relevant_data)}"}
                    ]
                )
                return response.content[0].text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    time.sleep(wait_time)
                    continue
                else:
                    return f"## {section_name}\n\n*Rate limit exceeded after {max_retries} attempts. Please try again later.*"
            elif "context_length_exceeded" in error_msg.lower() or "maximum context length" in error_msg.lower():
                # Try to fallback to Claude if OpenAI hits context limits
                try:
                    claude_client = get_claude_client()
                    if claude_client:
                        # Retry with Claude
                        claude_response = claude_client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=600,
                            temperature=0.2,
                            system=system_prompt,
                            messages=[
                                {"role": "user", "content": f"Relevant data: {json.dumps(relevant_data)}"}
                            ]
                        )
                        return f"## {section_name}\n\n{claude_response.content[0].text.strip()}\n\n*Note: Generated with Claude due to OpenAI context limits*"
                    else:
                        return f"## {section_name}\n\n*OpenAI context length exceeded. Data too large for GPT-3.5. Try Claude instead.*"
                except Exception as claude_error:
                    return f"## {section_name}\n\n*OpenAI context length exceeded. Data too large for GPT-3.5. Try Claude instead.*"
            else:
                return f"## {section_name}\n\n*Error: {error_msg}*"
    
    return f"## {section_name}\n\n*Failed to generate after {max_retries} attempts.*"

def generate_full_briefing(analysis_json: dict, progress_callback=None, delay_between_calls: int = 3, model_choice: str = "claude") -> str:
    """
    Orchestrator function that generates the weekly briefing through sequential API calls.
    Each section is generated separately to handle API rate limits gracefully.
    
    Args:
        analysis_json: The analysis data from the database
        progress_callback: Optional callback for progress updates
        delay_between_calls: Seconds to wait between API calls (default: 3)
        model_choice: AI model to use - "claude" or "openai" (default: "claude")
    """
    if model_choice == "openai":
        client = get_openai_client()
        model_type = "openai"
        if not client:
            return "OpenAI client not configured. Set OPENAI_API_KEY in secrets."
    else:
        client = get_claude_client()
        model_type = "claude"
        if not client:
            return "Claude client not configured. Set ANTHROPIC_API_KEY in secrets."

    sections = [
        {"name": "Executive Summary", "func": generate_executive_summary},
        {"name": "Store Performance", "func": generate_store_performance}, 
        {"name": "Product Trends", "func": generate_product_trends},
        {"name": "Category Momentum", "func": generate_category_momentum},
        {"name": "Time Patterns", "func": generate_time_patterns},
        {"name": "Inventory Watch", "func": generate_inventory_watch},
        {"name": "Risks & Opportunities", "func": generate_risks_opportunities}
    ]
    
    generated_sections = []
    previous_context = ""
    
    for i, section in enumerate(sections):
        if progress_callback:
            progress_callback(f"Generating {section['name']}...", (i + 1) / len(sections))
        
        # Add delay between API calls to respect rate limits
        if i > 0:  # Don't delay before the first call
            if progress_callback:
                progress_callback(f"Waiting {delay_between_calls}s before next section...", (i + 0.5) / len(sections))
            time.sleep(delay_between_calls)
        
        try:
            section_content = section["func"](analysis_json, previous_context, client, model_type)
            if section_content:
                generated_sections.append(section_content)
                # Update context with this section for next iterations
                previous_context += f"\n\n{section_content}"
            else:
                generated_sections.append(f"## {section['name']}\n\n*Section generation failed - please try again.*")
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                # Handle rate limit errors specifically
                generated_sections.append(f"## {section['name']}\n\n*Rate limit reached - please wait a moment and try again.*")
                if progress_callback:
                    progress_callback(f"Rate limit hit - waiting 10s before continuing...", (i + 1) / len(sections))
                time.sleep(10)  # Wait 10 seconds before continuing
            else:
                generated_sections.append(f"## {section['name']}\n\n*Error generating section: {error_msg}*")
    
    # Assemble final report
    final_report = "\n\n---\n\n".join(generated_sections)
    return final_report


def generate_executive_summary(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Executive Summary section."""
    # Extract only essential data to reduce token count
    overall_weekly = analysis_json.get("overall_weekly", [])
    if overall_weekly:
        # Keep only key metrics: week_start, revenue, transactions
        relevant_data = {
            "overall_weekly": [
                {
                    "week_start": week.get("week_start"),
                    "revenue": week.get("revenue"),
                    "transactions": week.get("transactions")
                }
                for week in overall_weekly
            ]
        }
    else:
        relevant_data = {"overall_weekly": []}
    
    system_prompt = """You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Executive Summary" section of a weekly business briefing.

**Instructions for this section:**
- Analyze the provided `overall_weekly` data covering the last 4 full ISO weeks
- Compare the final week's sales to the first week's (W4 vs W1). Call the trend: UP / DOWN / MIXED
- Identify the best week and worst week by total sales, with pesos and % vs its prior week
- Keep the analysis concise and executive-focused
- Always show pesos like ₱1,234,567 (no decimals). Percentages with one decimal (±X.X%)

Generate ONLY the markdown for the "Executive Summary" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Executive Summary", model_type)


def generate_store_performance(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Store Performance section."""
    relevant_data = {
        "stores_weekly": analysis_json.get("stores_weekly", []),
        "meta": analysis_json.get("meta", {})
    }
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Store Performance" section of a weekly business briefing.

**Context from Previous Sections:**
{previous_context}

**Instructions for this section:**
- Analyze the provided `stores_weekly` data
- Identify the Top 3 stores by the latest week's sales
- Identify the Top 3 week-over-week improvers by absolute ₱ increase
- Identify the Bottom 3 stores by WoW decline by absolute ₱ decrease
- Format each highlighted store on a single line: ₱this_week (vs ₱last_week, ±X.X%)
- If a store had no sales in the prior week, label it as "(New Activity)" instead of showing a percentage
- Keep the analysis concise and focused only on store performance

Generate ONLY the markdown for the "Store Performance" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Store Performance", model_type)


def generate_product_trends(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Product Trends section."""
    relevant_data = {
        "products_weekly": analysis_json.get("products_weekly", [])
    }
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Product Trends" section of a weekly business briefing.

**Context from Previous Sections:**
{previous_context}

**Instructions for this section:**
- Analyze the provided `products_weekly` data
- Identify Top 5 products UP by absolute ₱ revenue change in the latest week vs. the prior week
- Identify Top 5 products DOWN by absolute ₱ revenue change in the latest week vs. the prior week
- Call out notable category shifts affecting these products
- If a product is new to the top movers, mention it
- Keep the analysis concise and focused only on product trends

Generate ONLY the markdown for the "Product Trends" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Product Trends", model_type)


def generate_category_momentum(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Category Momentum section."""
    relevant_data = {
        "category_weekly": analysis_json.get("category_weekly", [])
    }
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Category Momentum" section of a weekly business briefing.

**Context from Previous Sections:**
{previous_context}

**Instructions for this section:**
- Analyze the provided `category_weekly` data
- Identify which categories gained or lost the most market share in the latest week vs. the prior week
- Report change in percentage points (e.g., "+1.5 pts")
- Briefly explain what this momentum implies for sales focus
- Keep the analysis concise and focused only on category momentum

Generate ONLY the markdown for the "Category Momentum" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Category Momentum", model_type)


def generate_time_patterns(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Time Patterns section."""
    relevant_data = {
        "daily": analysis_json.get("daily", []),
        "hourly": analysis_json.get("hourly", [])
    }
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Time Patterns" section of a weekly business briefing.

**Context from Previous Sections:**
{previous_context}

**Instructions for this section:**
- Analyze the provided `daily` and `hourly` data
- Identify the best weekday (average) and worst weekday (average) by sales over the 4-week period
- Identify the peak sales hour overall (0-23)
- Note if weekend vs. weekday peak hours differ materially
- Keep the analysis concise and focused only on time patterns

Generate ONLY the markdown for the "Time Patterns" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Time Patterns", model_type)


def generate_inventory_watch(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Inventory Watch section."""
    relevant_data = {
        "inventory_low": analysis_json.get("inventory_low", [])
    }
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Inventory Watch" section of a weekly business briefing.

**Context from Previous Sections:**
{previous_context}

**Instructions for this section:**
- Analyze the provided `inventory_low` data
- List the 5 most urgent items. Urgency is ranked by the largest deficit (warning_stock - on_hand)
- Provide a clear action for each item: [Store: Product] is at X/Y. Action: Restock / Redistribute / Monitor
- Keep the analysis concise and focused only on inventory concerns

Generate ONLY the markdown for the "Inventory Watch" section."""

    return _make_api_call_with_retry(client, system_prompt, relevant_data, "Inventory Watch", model_type)


def generate_risks_opportunities(analysis_json: dict, previous_context: str, client, model_type: str = "claude") -> str:
    """Generate Risks & Opportunities section - synthesis of all previous sections."""
    
    system_prompt = f"""You are the Business Analyst and Strategy AI for Aji Ichiban.

**Your Current Task:** Write the "Risks & Opportunities" section of a weekly business briefing.

**Context from All Previous Sections:**
{previous_context}

**Instructions for this section:**
- Synthesize insights from all previous sections to identify strategic implications
- Identify 2 critical risks (e.g., a top store is declining, a key category is losing share, a bestseller is near stockout)
- Identify 2 high-potential opportunities (e.g., a product is surging, a store is showing strong momentum, a weekend time slot is outperforming)
- Base your analysis on the patterns and insights from the previous sections
- Keep the analysis strategic and actionable

Generate ONLY the markdown for the "Risks & Opportunities" section."""

    return _make_api_call_with_retry(client, system_prompt, {"synthesis": "Based on previous sections"}, "Risks & Opportunities", model_type)


# Legacy function for backward compatibility
def get_ai_weekly_briefing(analysis_json: dict) -> str:
    """Legacy function - now uses the new sequential approach."""
    return generate_full_briefing(analysis_json)


def render_ai_intelligence_hub():
    """Render the AI Intelligence Hub page - placeholder for future features."""
    st.markdown('<div class="main-header"><h1>🧠 AI Intelligence Hub</h1><p>Advanced AI Intelligence Features - Coming Soon</p></div>', unsafe_allow_html=True)
    
    st.info("🚧 **AI Intelligence Hub is currently under development.**")
    
    st.markdown("### 📋 Planned Features:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔮 Advanced Analytics:**
        - Predictive modeling
        - Customer segmentation
        - Demand forecasting
        - Anomaly detection
        """)
    
    with col2:
        st.markdown("""
        **🎯 Strategic Insights:**
        - Market trend analysis
        - Competitive intelligence
        - Growth opportunity identification
        - Risk assessment
        """)
    
    st.markdown("### 💡 Current AI Intelligence")
    st.info("The AI Intelligence functionality has been moved to the **Dashboard** page for better integration with your workflow.")
def render_smart_reports():
    """Deprecated placeholder to keep references; replaced by Product Sales Report."""
    st.info("Smart Reports has been replaced by Product Sales Report. Use the sidebar to open it.")

def render_settings():
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Manage your dashboard</p></div>', unsafe_allow_html=True)
    
    # Configuration Status
    st.subheader("🔧 Configuration Status")
    
    # Check database connection
    db_conn = create_db_connection()
    if db_conn:
        st.success("✅ Database connection successful")
        db_conn.close()
    else:
        st.error("❌ Database connection failed")
        st.info("Add your database credentials to .streamlit/secrets.toml:")
        st.code("""
[postgres]
host = "your-database-host"
database = "your-database-name"
user = "your-database-user"
password = "your-database-password"
port = "5432"

# OR use individual keys:
# host = "your-database-host"
# database = "your-database-name"
# user = "your-database-user"
# password = "your-database-password"
# port = "5432"
        """)
    
    # Check API keys
    claude_client = get_claude_client()
    if claude_client:
        st.success("✅ Claude API key configured")
    else:
        st.error("❌ Claude API key missing")
        st.info("Add your Anthropic API key to .streamlit/secrets.toml:")
        st.code("""
[anthropic]
api_key = "your-anthropic-api-key"

# OR use direct key:
# CLAUDE_API_KEY = "your-anthropic-api-key"
        """)
    
    # Add OpenAI status check
    openai_client = get_openai_client()
    if openai_client:
        st.success("✅ OpenAI API key configured (for Intelligence Hub)")
    else:
        st.error("❌ OpenAI API key missing (Intelligence Hub won't work)")
        st.info("Add your OpenAI API key to .streamlit/secrets.toml:")
        st.code("""
[openai]
api_key = "your-openai-api-key"

# OR use direct key:
# OPENAI_API_KEY = "your-openai-api-key"
        """)
    
    st.subheader("🎓 Training System")
    training_count = len(st.session_state.training_system.training_data)
    st.metric("Training Examples", training_count)
    if training_count > 0:
        correct_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "correct"])
        corrected_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "corrected"])
        st.write(f"✅ Correct: {correct_count}")
        st.write(f"🔧 Corrected: {corrected_count}")
        
        with st.expander("📋 View Training Data"):
            for example in st.session_state.training_system.training_data[-5:]:
                st.write(f"**Q:** {example['question']}")
                st.write(f"**Status:** {example['feedback']}")
                if example.get('explanation'):
                    st.write(f"**Note:** {example['explanation']}")
                st.write("---")
    
    st.subheader("🛠️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Training Data"):
            st.session_state.training_system.training_data = []
            st.session_state.training_system.save_training_data()
            st.success("Training data cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Cache"):
            st.cache_data.clear()
            st.success("Cache refreshed!")
            st.rerun()

# --- PRODUCT SALES REPORT (Polars) ---

@st.cache_data(ttl=600)
def get_store_list_polars() -> pl.DataFrame:
    """Fetch stores using Polars."""
    conn = create_db_connection()
    if not conn:
        return pl.DataFrame()
    try:
        return pl.read_database("SELECT id, name FROM stores ORDER BY name;", conn)
    except Exception as e:
        st.error(f"Failed to load stores: {e}")
        return pl.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass

@st.cache_data(ttl=600)
def get_product_sales_report_data(primary_store_id: int, comparison_store_id: int, start_date: str, end_date: str) -> pl.DataFrame:
    """Generate product sales report using Polars with parameterized SQL."""
    conn = create_db_connection()
    if not conn:
        return pl.DataFrame()
    sql = """
    WITH sales_data AS (
        SELECT 
            p.name AS product_name,
            p.sku, p.id AS product_id, p.barcode, p.category,
            SUM(ti.quantity) AS quantity_sold,
            SUM(ti.item_total) AS total_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE t.store_id = %s
          AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN %s AND %s
          AND LOWER(t.transaction_type) = 'sale'
          AND COALESCE(t.is_cancelled, false) = false
        GROUP BY p.name, p.sku, p.id, p.barcode, p.category
    ),
    inventory_primary AS (
        SELECT product_id, quantity_on_hand AS primary_inventory FROM inventory WHERE store_id = %s
    ),
    inventory_comparison AS (
        SELECT product_id, quantity_on_hand AS comparison_inventory FROM inventory WHERE store_id = %s
    )
    SELECT sd.*, COALESCE(ip.primary_inventory, 0) AS primary_store_inventory,
           COALESCE(ic.comparison_inventory, 0) AS comparison_store_inventory
    FROM sales_data sd
    LEFT JOIN inventory_primary ip ON sd.product_id = ip.product_id
    LEFT JOIN inventory_comparison ic ON sd.product_id = ic.product_id
    ORDER BY sd.category, sd.quantity_sold DESC;
    """
    try:
        return pl.read_database(
            sql,
            conn,
            execute_options={"parameters": [primary_store_id, start_date, end_date, primary_store_id, comparison_store_id]},
        )
    except Exception as e:
        st.error(f"Database query failed: {e}")
        return pl.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass

def process_report_data_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Light processing and sorting using Polars."""
    if df.is_empty():
        return df
    return (
        df.with_columns([
            pl.col("total_revenue").fill_null(0),
            pl.col("quantity_sold").fill_null(0),
            pl.col("category").fill_null("Uncategorized"),
        ]).sort(["category", "quantity_sold"], descending=[False, True])
    )

@st.cache_data
def to_csv_polars(df: pl.DataFrame) -> bytes:
    """Return CSV bytes for download from a Polars DataFrame without revenue."""
    # Keep only selected columns and exclude revenue
    preferred_cols = [
        "product_name",  # Product
        "sku",           # SKU
        "product_id",    # Product ID
        "quantity_sold", # Quantity
        "primary_store_inventory",      # Primary Stock
        "comparison_store_inventory",   # Comparison Stock
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    safe_df = df.select(cols) if cols else df
    return safe_df.to_pandas().to_csv(index=False).encode("utf-8")

def render_product_sales_report():
    """Render the Product Sales Report (Polars)."""
    st.markdown('<div class="main-header"><h1>📊 Product Sales Report</h1><p>Advanced sales analytics powered by Polars</p></div>', unsafe_allow_html=True)

    stores_df = get_store_list_polars()
    if stores_df.is_empty():
        st.error("Could not fetch store list. Check database connection.")
        return

    store_options = {row["name"]: row["id"] for row in stores_df.iter_rows(named=True)}
    store_names = list(store_options.keys())

    st.markdown("### 🎛️ Report Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        primary_store = st.selectbox("Primary Store", store_names, index=0)
    with col2:
        comparison_stores = [s for s in store_names if s != primary_store]
        comparison_store = st.selectbox("Comparison Store", comparison_stores, index=0) if comparison_stores else primary_store
    with col3:
        start_date = st.date_input("Start Date", date.today() - timedelta(days=7))
    with col4:
        end_date = st.date_input("End Date", date.today())

    if st.button("🚀 Generate Polars Report", type="primary", use_container_width=True):
        if start_date > end_date:
            st.error("Start date cannot be after end date")
            return

        primary_id = store_options[primary_store]
        comparison_id = store_options[comparison_store]

        with st.spinner("⚡ Processing with Polars..."):
            report_df = get_product_sales_report_data(primary_id, comparison_id, str(start_date), str(end_date))
            if report_df.is_empty():
                st.warning("No sales data found for the selected criteria")
                return

            processed_df = process_report_data_polars(report_df)
            st.session_state["polars_report_df"] = processed_df
            st.session_state["polars_report_params"] = {
                "primary_store": primary_store,
                "comparison_store": comparison_store,
                "start_date": start_date,
                "end_date": end_date,
            }
        st.success(f"✅ Report generated with {len(processed_df)} products using Polars!")

    if "polars_report_df" in st.session_state:
        df = st.session_state["polars_report_df"]
        params = st.session_state["polars_report_params"]

        st.markdown("---")
        st.subheader("📈 Product Sales Analysis")

        head1, head2 = st.columns([3, 1])
        with head1:
            st.markdown(f"**Primary:** `{params['primary_store']}` | **Comparison:** `{params['comparison_store']}` | **Period:** `{params['start_date']}` to `{params['end_date']}`")
        with head2:
            st.download_button(
                "📥 Download CSV",
                to_csv_polars(df),
                f"{params['primary_store']}_Sales_{params['start_date']}.csv",
                "text/csv",
                use_container_width=True,
            )

        display_df = df.to_pandas()
        display_df = display_df.rename(columns={
            "product_id": "Product ID",
            "product_name": "Product",
            "sku": "SKU",
            "quantity_sold": "Qty Sold",
            # no Revenue column
            "primary_store_inventory": f"{params['primary_store']} Stock",
            "comparison_store_inventory": f"{params['comparison_store']} Stock",
        })

        for category in sorted(display_df["category"].unique()):
            category_df = display_df[display_df["category"] == category]
            with st.expander(f"📦 {category} ({len(category_df)} products)", expanded=True):
                column_config = {
                    "Product": st.column_config.TextColumn("Product Name", width="large"),
                    "SKU": st.column_config.TextColumn("SKU", width="small"),
                    "Product ID": st.column_config.TextColumn("Product ID", width="small"),
                    "Qty Sold": st.column_config.NumberColumn("Quantity", format="%d"),
                    f"{params['primary_store']} Stock": st.column_config.NumberColumn("Primary Stock", format="%d"),
                    f"{params['comparison_store']} Stock": st.column_config.NumberColumn("Comparison Stock", format="%d"),
                }
                st.dataframe(
                    category_df[[
                        "Product",
                        "SKU",
                        "Product ID",
                        "Qty Sold",
                        f"{params['primary_store']} Stock",
                        f"{params['comparison_store']} Stock",
                    ]],
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True,
                )

def main():
    try:
        load_css()
        init_session_state()
        
        # Show pool status in sidebar for debugging
        show_pool_status()
        monitor_connection_health()
        
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            # Add logo or branding here if you want
            st.markdown("---")
            
            pages = ["📊 Dashboard", "📈 Product Sales Report", "📈 Chart View", "🧠 AI Assistant", "⚙️ Settings"]
            
            for page in pages:
                page_name = page.split(" ", 1)[1]
                # Highlight current page
                if st.session_state.current_page == page_name:
                    st.markdown(f"**→ {page}**")
                else:
                    if st.button(page, key=f"nav_{page_name}", use_container_width=True):
                        st.session_state.current_page = page_name
                        st.rerun()
            
            st.markdown("---")
            
            # Add quick stats in sidebar
            st.markdown("### 📊 Quick Stats")
            latest_data = get_latest_metrics()
            if latest_data is not None and not latest_data.empty:
                latest_sales = latest_data.iloc[0]['latest_sales']
                st.metric("Today's Sales", f"₱{latest_sales:,.0f}")
            
            # Add timestamp
            st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.8rem;'>Last updated: {datetime.now().strftime('%I:%M %p')}</p>", unsafe_allow_html=True)
        
        # Page Routing
        if st.session_state.current_page == "Dashboard":
            render_dashboard()
        elif st.session_state.current_page == "Product Sales Report":
            render_product_sales_report()
        elif st.session_state.current_page == "Chart View":
            render_chart_view()
        elif st.session_state.current_page == "AI Assistant":
            render_chat()

        elif st.session_state.current_page == "Settings":
            render_settings()
        
        st.markdown("<hr><div style='text-align:center;color:#666;'><p>🧠 Enhanced SupaBot with Smart Visualizations | Powered by Claude Sonnet 3.5</p></div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")

if __name__ == "__main__":
    main()
