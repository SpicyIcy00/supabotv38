"""
Claude AI client for SQL generation and query assistance.
"""

import streamlit as st
import anthropic
from typing import Optional
from supabot.config.settings import settings


class ClaudeClient:
    """Claude AI client for SQL generation and assistance."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self) -> Optional[anthropic.Anthropic]:
        """Get or create the Claude client."""
        if self._client is None:
            api_key = settings.get_anthropic_api_key()
            if api_key:
                try:
                    self._client = anthropic.Anthropic(api_key=api_key)
                except Exception as e:
                    st.error(f"Failed to initialize Claude client: {e}")
                    return None
        return self._client
    
    def generate_sql(self, question: str, schema_info: Optional[dict] = None, 
                    training_context: str = "") -> Optional[str]:
        """Generate SQL query from natural language question."""
        if not self.client:
            return None
            
        try:
            # Build comprehensive prompt
            schema_context = ""
            if schema_info:
                schema_context = self._format_schema_context(schema_info)
            
            prompt = f"""
You are an expert PostgreSQL developer for a retail BI system. Generate ONLY the SQL query for this question:

QUESTION: {question}

DATABASE SCHEMA:
{schema_context}

TRAINING CONTEXT:
{training_context}

IMPORTANT RULES:
1. Use ONLY tables/columns that exist in the schema
2. Always use timezone 'Asia/Manila' for timestamp operations  
3. Filter sales with cancellations excluded using: LOWER(transaction_type) = 'sale' AND COALESCE(is_cancelled, false) = false
4. Return ONLY the SQL query, no explanations
5. Use proper JOINs between tables based on relationships
6. For aggregations, use appropriate GROUP BY clauses

Generate the SQL query:
"""

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            sql_query = response.content[0].text.strip()
            
            # Clean up the response to extract just the SQL
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            return sql_query
            
        except Exception as e:
            st.error(f"SQL generation failed: {e}")
            return None
    
    def interpret_results(self, question: str, results_df, sql_query: str) -> str:
        """Interpret query results and provide insights."""
        if not self.client:
            return "AI interpretation unavailable"
            
        try:
            # Prepare results summary
            results_summary = ""
            if results_df is not None and not results_df.empty:
                results_summary = f"Results: {len(results_df)} rows returned\n"
                results_summary += f"Columns: {', '.join(results_df.columns)}\n"
                if len(results_df) <= 5:
                    results_summary += f"Sample data:\n{results_df.to_string()}"
                else:
                    results_summary += f"First 3 rows:\n{results_df.head(3).to_string()}"
            else:
                results_summary = "No results returned"
            
            prompt = f"""
Analyze these SQL query results and provide business insights:

ORIGINAL QUESTION: {question}
SQL QUERY: {sql_query}
{results_summary}

Provide a concise business interpretation focusing on:
1. Key findings and trends
2. Business implications  
3. Actionable recommendations

Keep response under 200 words and focus on business value.
"""

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"Interpretation failed: {e}"
    
    def _format_schema_context(self, schema_info: dict) -> str:
        """Format database schema information for the prompt."""
        context = ""
        for table_name, table_info in schema_info.items():
            context += f"\nTable: {table_name}\n"
            context += f"Rows: {table_info['row_count']}\n"
            context += "Columns:\n"
            for col_name, col_type, nullable, default in table_info['columns']:
                context += f"  - {col_name} ({col_type})\n"
        return context


# Global singleton instance
_claude_client = None

def get_claude_client() -> ClaudeClient:
    """Get the global ClaudeClient instance."""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client

# Legacy compatibility function
def get_claude_client_legacy():
    """Legacy compatibility function."""
    return get_claude_client().client

