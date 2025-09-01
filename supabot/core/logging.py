"""
Logging configuration for SupaBot BI Dashboard.
Provides structured logging with performance monitoring.
"""

import logging
import time
import functools
from typing import Any, Callable, Optional
from datetime import datetime
import streamlit as st


class SupaBotLogger:
    """Custom logger for SupaBot with performance monitoring."""
    
    def __init__(self, name: str = "supabot"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration."""
        if not self.logger.handlers:
            # Create console handler
            handler = logging.StreamHandler()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} | {context}" if context else message
        self.logger.info(full_message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} | {context}" if context else message
        self.logger.error(full_message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} | {context}" if context else message
        self.logger.warning(full_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} | {context}" if context else message
        self.logger.debug(full_message)


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        function_name = func.__name__
        
        logger = get_logger()
        logger.info(f"Starting {function_name}", 
                   args_count=len(args), kwargs_count=len(kwargs))
        
        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            
            # Log performance metrics
            logger.info(f"Completed {function_name}", 
                       duration_ms=round(duration, 2))
            
            # Store in session state for dashboard monitoring
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            
            st.session_state.performance_metrics[function_name] = {
                'duration_ms': round(duration, 2),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            logger.error(f"Failed {function_name}", 
                        duration_ms=round(duration, 2), 
                        error=str(e))
            
            # Store error metrics
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            
            st.session_state.performance_metrics[function_name] = {
                'duration_ms': round(duration, 2),
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            
            raise
    
    return wrapper


def log_query_performance(sql_preview: str, params_count: int = 0):
    """Decorator for database query performance logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger = get_logger()
            
            logger.info("Executing database query", 
                       function=func.__name__,
                       sql_preview=sql_preview[:100] + "..." if len(sql_preview) > 100 else sql_preview,
                       params_count=params_count)
            
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start_time) * 1000
                
                # Determine result size
                result_size = 0
                if hasattr(result, '__len__'):
                    result_size = len(result)
                elif hasattr(result, 'shape'):
                    result_size = result.shape[0]
                
                logger.info("Query completed successfully",
                           function=func.__name__,
                           duration_ms=round(duration, 2),
                           result_rows=result_size)
                
                return result
                
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                logger.error("Query failed",
                            function=func.__name__,
                            duration_ms=round(duration, 2),
                            error=str(e))
                raise
        
        return wrapper
    return decorator


def log_user_action(action_type: str, details: Optional[dict] = None):
    """Log user actions for analytics."""
    logger = get_logger()
    
    context = {
        'action_type': action_type,
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    if details:
        context.update(details)
    
    logger.info("User action logged", **context)
    
    # Store in session state for analytics
    if 'user_actions' not in st.session_state:
        st.session_state.user_actions = []
    
    st.session_state.user_actions.append(context)


def get_performance_summary() -> dict:
    """Get performance metrics summary for monitoring."""
    metrics = st.session_state.get('performance_metrics', {})
    
    summary = {
        'total_functions': len(metrics),
        'successful_calls': sum(1 for m in metrics.values() if m.get('status') == 'success'),
        'failed_calls': sum(1 for m in metrics.values() if m.get('status') == 'error'),
        'avg_duration_ms': 0,
        'slowest_function': None,
        'fastest_function': None
    }
    
    if metrics:
        durations = [m['duration_ms'] for m in metrics.values() if 'duration_ms' in m]
        if durations:
            summary['avg_duration_ms'] = round(sum(durations) / len(durations), 2)
            
            # Find slowest and fastest
            slowest_time = max(durations)
            fastest_time = min(durations)
            
            for func_name, data in metrics.items():
                if data.get('duration_ms') == slowest_time:
                    summary['slowest_function'] = {'name': func_name, 'duration_ms': slowest_time}
                if data.get('duration_ms') == fastest_time:
                    summary['fastest_function'] = {'name': func_name, 'duration_ms': fastest_time}
    
    return summary


# Global logger instance
_logger = None

def get_logger() -> SupaBotLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = SupaBotLogger()
    return _logger


# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message."""
    get_logger().info(message, **kwargs)

def log_error(message: str, **kwargs):
    """Log error message."""
    get_logger().error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message."""
    get_logger().warning(message, **kwargs)
