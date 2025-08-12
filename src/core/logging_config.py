"""
Enterprise Logging System
=========================

Structured logging with correlation IDs, performance metrics,
and distributed tracing capabilities.
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps
import threading
from enum import Enum

# Thread-local storage for correlation context
_context = threading.local()


class LogLevel(Enum):
    """Log levels with numeric values"""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with correlation tracking.
    """
    
    def __init__(self, include_trace: bool = True):
        super().__init__()
        self.include_trace = include_trace
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(_context, 'correlation_id', None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add user context if available
        user_context = getattr(_context, 'user_context', None)
        if user_context:
            log_entry["user"] = user_context
        
        # Add request context if available
        request_context = getattr(_context, 'request_context', None)
        if request_context:
            log_entry["request"] = request_context
        
        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_entry["duration_ms"] = record.duration
        
        # Add business context if available
        if hasattr(record, 'business_context'):
            log_entry["business"] = record.business_context
        
        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'duration', 'business_context']:
                try:
                    # Ensure the value is JSON serializable
                    json.dumps(value)
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if self.include_trace else None
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class PerformanceLogger:
    """
    Performance monitoring and logging utilities.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @contextmanager
    def timer(self, operation: str, threshold_ms: float = 1000.0):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed
            threshold_ms: Log warning if operation exceeds this threshold
        """
        start_time = time.perf_counter()
        correlation_id = str(uuid.uuid4())
        
        try:
            with correlation_context(correlation_id):
                self.logger.info(
                    f"Starting operation: {operation}",
                    extra={"operation": operation, "phase": "start"}
                )
                yield
                
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Operation failed: {operation}",
                extra={
                    "operation": operation,
                    "phase": "error",
                    "duration": duration_ms,
                    "error": str(exc)
                },
                exc_info=True
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log with appropriate level based on duration
            if duration_ms > threshold_ms:
                log_level = logging.WARNING
                message = f"Slow operation completed: {operation}"
            else:
                log_level = logging.INFO
                message = f"Operation completed: {operation}"
            
            self.logger.log(
                log_level,
                message,
                extra={
                    "operation": operation,
                    "phase": "complete",
                    "duration": duration_ms,
                    "threshold_exceeded": duration_ms > threshold_ms
                }
            )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str = "count",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            context: Additional context
        """
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "metric": {
                    "name": metric_name,
                    "value": value,
                    "unit": unit,
                    "context": context or {}
                }
            }
        )


@contextmanager
def correlation_context(correlation_id: str):
    """
    Context manager for correlation ID tracking across operations.
    
    Args:
        correlation_id: Unique identifier for request correlation
    """
    old_id = getattr(_context, 'correlation_id', None)
    _context.correlation_id = correlation_id
    try:
        yield correlation_id
    finally:
        _context.correlation_id = old_id


@contextmanager
def user_context(user_id: str, session_id: Optional[str] = None):
    """
    Context manager for user context tracking.
    
    Args:
        user_id: User identifier
        session_id: Optional session identifier
    """
    old_context = getattr(_context, 'user_context', None)
    _context.user_context = {
        "user_id": user_id,
        "session_id": session_id
    }
    try:
        yield
    finally:
        _context.user_context = old_context


@contextmanager
def request_context(endpoint: str, method: str = "GET", request_id: Optional[str] = None):
    """
    Context manager for request context tracking.
    
    Args:
        endpoint: API endpoint or operation name
        method: HTTP method or operation type
        request_id: Optional request identifier
    """
    old_context = getattr(_context, 'request_context', None)
    _context.request_context = {
        "endpoint": endpoint,
        "method": method,
        "request_id": request_id or str(uuid.uuid4())
    }
    try:
        yield
    finally:
        _context.request_context = old_context


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
    console_output: bool = True,
    json_format: bool = True
) -> logging.Logger:
    """
    Configure enterprise-grade logging system.
    
    Args:
        log_level: Minimum log level to capture
        log_dir: Directory for log files
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        
    Returns:
        Configured root logger
    """
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "quantum_consensus.log"),
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler (errors only)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "quantum_consensus_errors.log"),
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            # Use simpler formatting for console
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = formatter
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Performance logger
    perf_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "quantum_consensus_performance.log"),
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    perf_handler.setFormatter(formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger("performance")
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("mlx").setLevel(logging.ERROR)
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            "config": {
                "log_level": log_level,
                "log_dir": log_dir,
                "json_format": json_format,
                "console_output": console_output
            }
        }
    )
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with performance tracking capabilities.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger with performance tracking
    """
    logger = logging.getLogger(name)
    logger.performance = PerformanceLogger(logger)
    return logger


def log_function_call(
    include_args: bool = False,
    include_result: bool = False,
    level: int = logging.DEBUG
):
    """
    Decorator to log function calls with performance tracking.
    
    Args:
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        level: Log level for the entries
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            log_data = {"function": func_name, "phase": "enter"}
            if include_args:
                log_data["args"] = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }
            
            logger.log(level, f"Entering function: {func_name}", extra=log_data)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log function exit
                log_data = {
                    "function": func_name,
                    "phase": "exit",
                    "duration": duration_ms,
                    "success": True
                }
                if include_result:
                    log_data["result"] = str(result)
                
                logger.log(level, f"Exiting function: {func_name}", extra=log_data)
                return result
                
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log function error
                logger.error(
                    f"Function error: {func_name}",
                    extra={
                        "function": func_name,
                        "phase": "error",
                        "duration": duration_ms,
                        "error": str(exc)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Business event logging
class BusinessEventLogger:
    """
    Logger for business events and user actions.
    """
    
    def __init__(self):
        self.logger = get_logger("business_events")
    
    def log_user_action(
        self,
        action: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log user action for audit trail"""
        self.logger.info(
            f"User action: {action}",
            extra={
                "event_type": "user_action",
                "action": action,
                "user_id": user_id,
                "details": details or {}
            }
        )
    
    def log_trade_event(
        self,
        event_type: str,
        symbol: str,
        details: Dict[str, Any]
    ):
        """Log trading-related events"""
        self.logger.info(
            f"Trade event: {event_type}",
            extra={
                "event_type": "trade",
                "trade_event": event_type,
                "symbol": symbol,
                "details": details
            }
        )
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log system events"""
        self.logger.info(
            f"System event: {event_type}",
            extra={
                "event_type": "system",
                "system_event": event_type,
                "component": component,
                "details": details or {}
            }
        )


# Global business event logger
business_events = BusinessEventLogger()