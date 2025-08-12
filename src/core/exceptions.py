"""
Enterprise Exception Handling System
===================================

Comprehensive exception hierarchy with context preservation,
error tracking, and automated recovery mechanisms.
"""

import sys
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting"""
    CRITICAL = "critical"    # System failure, immediate attention
    HIGH = "high"           # Major functionality impacted
    MEDIUM = "medium"       # Minor functionality impacted  
    LOW = "low"            # Non-impacting issues
    INFO = "info"          # Informational events


class ErrorCategory(Enum):
    """Error categories for classification and handling"""
    NETWORK = "network"           # Network connectivity issues
    DATA = "data"                # Data validation/processing errors
    AI_INFERENCE = "ai_inference" # AI/ML model errors
    AUTHENTICATION = "auth"       # Authentication/authorization
    CONFIGURATION = "config"      # Configuration errors
    EXTERNAL_API = "external_api" # Third-party API issues
    SYSTEM = "system"            # System-level errors
    USER_INPUT = "user_input"    # User input validation
    BUSINESS_LOGIC = "business"  # Business rule violations


class BaseRiverException(Exception):
    """
    Base exception class for River Trading System.
    
    Provides structured error handling with:
    - Unique error tracking IDs
    - Severity classification
    - Context preservation
    - Automated logging
    - Recovery hints
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        user_message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.user_message = user_message or "An error occurred. Please try again."
        self.cause = cause
        self.stack_trace = traceback.format_exc()
        
        # Automatically log the error
        self._log_error()
    
    def _log_error(self):
        """Log error with structured information"""
        log_data = {
            "error_id": self.error_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": str(self),
            "context": self.context,
            "recovery_hint": self.recovery_hint,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace
        }
        
        if self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            logger.error(f"River Error [{self.error_id}]: {self}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"River Warning [{self.error_id}]: {self}", extra=log_data)
        else:
            logger.info(f"River Info [{self.error_id}]: {self}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": str(self),
            "context": self.context,
            "recovery_hint": self.recovery_hint,
            "user_message": self.user_message,
            "stack_trace": self.stack_trace,
            "cause": str(self.cause) if self.cause else None
        }


class NetworkException(BaseRiverException):
    """Network connectivity and communication errors"""
    
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        context = kwargs.get('context', {})
        context['endpoint'] = endpoint
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.NETWORK
        kwargs['recovery_hint'] = "Check network connectivity and endpoint availability"
        super().__init__(message, **kwargs)


class DataValidationException(BaseRiverException):
    """Data validation and processing errors"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({'field': field, 'value': str(value) if value else None})
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.DATA
        kwargs['recovery_hint'] = "Verify data format and content"
        super().__init__(message, **kwargs)


class AIInferenceException(BaseRiverException):
    """AI model inference and processing errors"""
    
    def __init__(self, message: str, model: str = None, input_tokens: int = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({'model': model, 'input_tokens': input_tokens})
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.AI_INFERENCE
        kwargs['recovery_hint'] = "Check model availability and input format"
        super().__init__(message, **kwargs)


class ExternalAPIException(BaseRiverException):
    """External API communication errors"""
    
    def __init__(self, message: str, api: str = None, status_code: int = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({'api': api, 'status_code': status_code})
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.EXTERNAL_API
        kwargs['recovery_hint'] = "Check API availability and credentials"
        super().__init__(message, **kwargs)


class ConfigurationException(BaseRiverException):
    """Configuration and settings errors"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        context = kwargs.get('context', {})
        context['config_key'] = config_key
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.CONFIGURATION
        kwargs['severity'] = ErrorSeverity.HIGH
        kwargs['recovery_hint'] = "Check configuration file and environment variables"
        super().__init__(message, **kwargs)


class BusinessLogicException(BaseRiverException):
    """Business rule and logic violations"""
    
    def __init__(self, message: str, rule: str = None, **kwargs):
        context = kwargs.get('context', {})
        context['rule'] = rule
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.BUSINESS_LOGIC
        kwargs['user_message'] = message  # Business errors are user-facing
        super().__init__(message, **kwargs)


class RateLimitExceededException(BaseRiverException):
    """Rate limit exceeded errors"""
    
    def __init__(self, message: str, limit: int = None, window: str = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({'limit': limit, 'window': window})
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.SYSTEM
        kwargs['severity'] = ErrorSeverity.MEDIUM
        kwargs['recovery_hint'] = "Wait for rate limit window to reset"
        kwargs['user_message'] = "Rate limit exceeded. Please try again later."
        super().__init__(message, **kwargs)


class SystemResourceException(BaseRiverException):
    """System resource exhaustion errors"""
    
    def __init__(self, message: str, resource: str = None, usage: float = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({'resource': resource, 'usage': usage})
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.SYSTEM
        kwargs['severity'] = ErrorSeverity.HIGH
        kwargs['recovery_hint'] = "Check system resources and scaling"
        super().__init__(message, **kwargs)


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides:
    - Exception capture and processing
    - Automatic retry mechanisms
    - Error aggregation and reporting
    - Recovery strategy execution
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[BaseRiverException] = []
        self.max_recent_errors = 100
    
    def handle_exception(
        self,
        exc: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> BaseRiverException:
        """
        Handle any exception and convert to River exception.
        
        Args:
            exc: The exception to handle
            context: Additional context information
            severity: Error severity level
            
        Returns:
            BaseRiverException: Structured River exception
        """
        if isinstance(exc, BaseRiverException):
            river_exc = exc
        else:
            # Convert standard exceptions to River exceptions
            river_exc = self._convert_standard_exception(exc, context, severity)
        
        # Track error occurrence
        self._track_error(river_exc)
        
        # Add to recent errors for monitoring
        self.recent_errors.append(river_exc)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        return river_exc
    
    def _convert_standard_exception(
        self,
        exc: Exception,
        context: Optional[Dict[str, Any]],
        severity: ErrorSeverity
    ) -> BaseRiverException:
        """Convert standard Python exceptions to River exceptions"""
        
        exc_type = type(exc).__name__
        message = str(exc)
        
        # Map common exceptions to appropriate River exceptions
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return NetworkException(
                message=f"{exc_type}: {message}",
                severity=severity,
                context=context,
                cause=exc
            )
        elif isinstance(exc, ValueError):
            return DataValidationException(
                message=f"{exc_type}: {message}",
                severity=severity,
                context=context,
                cause=exc
            )
        elif isinstance(exc, (FileNotFoundError, PermissionError)):
            return SystemResourceException(
                message=f"{exc_type}: {message}",
                severity=severity,
                context=context,
                cause=exc
            )
        else:
            return BaseRiverException(
                message=f"{exc_type}: {message}",
                severity=severity,
                context=context,
                cause=exc
            )
    
    def _track_error(self, exc: BaseRiverException):
        """Track error occurrence for pattern analysis"""
        error_key = f"{exc.category.value}:{type(exc).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        return {
            "total_errors": len(self.recent_errors),
            "error_counts": self.error_counts,
            "recent_errors": [exc.to_dict() for exc in self.recent_errors[-10:]],
            "critical_errors": len([
                exc for exc in self.recent_errors
                if exc.severity == ErrorSeverity.CRITICAL
            ])
        }


# Global error handler instance
global_error_handler = ErrorHandler()


def safe_execute(
    func,
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Execute function with automatic error handling and retry logic.
    
    Args:
        func: Function to execute
        *args: Function arguments
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        context: Additional context for error handling
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or raises BaseRiverException
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exception = global_error_handler.handle_exception(
                exc,
                context={
                    **(context or {}),
                    'function': func.__name__,
                    'attempt': attempt + 1,
                    'max_retries': max_retries
                }
            )
            
            if attempt < max_retries:
                logger.warning(
                    f"Retrying {func.__name__} after error: {exc} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(
                    f"Max retries exceeded for {func.__name__}: {exc}"
                )
    
    raise last_exception


def error_boundary(
    fallback_value=None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_errors: bool = True
):
    """
    Decorator for error boundary pattern - catches and handles exceptions.
    
    Args:
        fallback_value: Value to return on error
        severity: Error severity level
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if log_errors:
                    global_error_handler.handle_exception(
                        exc,
                        context={'function': func.__name__},
                        severity=severity
                    )
                return fallback_value
        return wrapper
    return decorator