"""
Resilience Patterns and Circuit Breakers
========================================

Enterprise-grade resilience patterns including circuit breakers,
bulkheads, timeouts, and graceful degradation.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union, List
from enum import Enum
from functools import wraps
from collections import deque, defaultdict
import logging

from .exceptions import (
    BaseRiverException, NetworkException, SystemResourceException,
    ErrorSeverity, safe_execute
)
from .logging_config import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation with exponential backoff.
    
    Provides automatic fault detection and system protection by:
    - Monitoring failure rates
    - Automatically opening circuit on threshold breach
    - Gradual recovery testing
    - Metrics collection
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Union[type, tuple] = Exception,
        fallback_function: Optional[Callable] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.total_requests = 0
        
        self._lock = threading.RLock()
        
        logger.info(
            f"Circuit breaker initialized: {name}",
            extra={
                "circuit_breaker": {
                    "name": name,
                    "failure_threshold": failure_threshold,
                    "recovery_timeout": recovery_timeout
                }
            }
        )
    
    def __call__(self, func):
        """Decorator interface"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback value
            
        Raises:
            BaseRiverException: If circuit is open and no fallback
        """
        with self._lock:
            self.total_requests += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker half-open: {self.name}",
                        extra={"circuit_breaker": self.get_stats()}
                    )
                else:
                    return self._handle_open_circuit()
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as exc:
                self._on_failure(exc)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _handle_open_circuit(self) -> Any:
        """Handle request when circuit is open"""
        if self.fallback_function:
            logger.warning(
                f"Circuit breaker open, using fallback: {self.name}",
                extra={"circuit_breaker": self.get_stats()}
            )
            return self.fallback_function()
        else:
            raise NetworkException(
                f"Circuit breaker is open for {self.name}",
                severity=ErrorSeverity.HIGH,
                context={"circuit_breaker": self.get_stats()},
                recovery_hint="Wait for circuit breaker to recover or check service health"
            )
    
    def _on_success(self):
        """Handle successful execution"""
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info(
                f"Circuit breaker closed: {self.name}",
                extra={"circuit_breaker": self.get_stats()}
            )
    
    def _on_failure(self, exc: Exception):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker opened from half-open: {self.name}",
                extra={"circuit_breaker": self.get_stats(), "error": str(exc)}
            )
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(
                f"Circuit breaker opened: {self.name}",
                extra={"circuit_breaker": self.get_stats(), "error": str(exc)}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "failure_rate": self.failure_count / self.total_requests if self.total_requests > 0 else 0,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker manually reset: {self.name}")


class RateLimiter:
    """
    Token bucket rate limiter with burst capability.
    """
    
    def __init__(
        self,
        name: str,
        max_requests: int,
        time_window: int = 60,  # seconds
        burst_size: Optional[int] = None
    ):
        self.name = name
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_size = burst_size or max_requests
        
        self.tokens = self.burst_size
        self.last_refill = time.time()
        self._lock = threading.RLock()
        
        logger.info(
            f"Rate limiter initialized: {name}",
            extra={
                "rate_limiter": {
                    "name": name,
                    "max_requests": max_requests,
                    "time_window": time_window,
                    "burst_size": self.burst_size
                }
            }
        )
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limit exceeded
        """
        with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = int(elapsed * (self.max_requests / self.time_window))
            
            if tokens_to_add > 0:
                self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
                self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                logger.warning(
                    f"Rate limit exceeded: {self.name}",
                    extra={
                        "rate_limiter": {
                            "name": self.name,
                            "tokens_requested": tokens,
                            "tokens_available": self.tokens
                        }
                    }
                )
                return False
    
    def __call__(self, func):
        """Decorator interface"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.acquire():
                raise NetworkException(
                    f"Rate limit exceeded for {self.name}",
                    severity=ErrorSeverity.MEDIUM,
                    context={"rate_limiter": self.name},
                    recovery_hint="Reduce request frequency or wait before retrying"
                )
            return func(*args, **kwargs)
        return wrapper


class Timeout:
    """
    Timeout decorator with context manager support.
    """
    
    def __init__(self, seconds: Union[int, float], error_message: str = None):
        self.seconds = seconds
        self.error_message = error_message or f"Operation timed out after {seconds} seconds"
    
    def __call__(self, func):
        """Decorator interface"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_timeout(func, *args, **kwargs)
        return wrapper
    
    def __enter__(self):
        """Context manager entry"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(self.error_message)
        
        # Set up timeout signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.seconds))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            raise NetworkException(
                self.error_message,
                severity=ErrorSeverity.MEDIUM,
                context={"timeout_seconds": self.seconds},
                recovery_hint="Increase timeout or optimize operation performance"
            )
        finally:
            signal.signal(signal.SIGALRM, old_handler)


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation.
    
    Isolates resources by limiting concurrent operations
    to prevent cascading failures.
    """
    
    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_operations = 0
        self.total_operations = 0
        self.rejected_operations = 0
        self._lock = threading.Lock()
        
        logger.info(
            f"Bulkhead initialized: {name}",
            extra={
                "bulkhead": {
                    "name": name,
                    "max_concurrent": max_concurrent
                }
            }
        )
    
    def __call__(self, func):
        """Decorator interface"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with bulkhead protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            SystemResourceException: If bulkhead is full
        """
        with self._lock:
            self.total_operations += 1
        
        if not self.semaphore.acquire(blocking=False):
            with self._lock:
                self.rejected_operations += 1
            
            logger.warning(
                f"Bulkhead full: {self.name}",
                extra={"bulkhead": self.get_stats()}
            )
            
            raise SystemResourceException(
                f"Bulkhead is full for {self.name}",
                resource="concurrency_limit",
                usage=self.max_concurrent,
                severity=ErrorSeverity.MEDIUM,
                context={"bulkhead": self.get_stats()},
                recovery_hint="Wait for concurrent operations to complete or increase bulkhead size"
            )
        
        try:
            with self._lock:
                self.active_operations += 1
            
            return func(*args, **kwargs)
            
        finally:
            with self._lock:
                self.active_operations -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        with self._lock:
            return {
                "name": self.name,
                "max_concurrent": self.max_concurrent,
                "active_operations": self.active_operations,
                "total_operations": self.total_operations,
                "rejected_operations": self.rejected_operations,
                "utilization": self.active_operations / self.max_concurrent
            }


class RetryPolicy:
    """
    Configurable retry policy with different backoff strategies.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions
    
    def __call__(self, func):
        """Decorator interface"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry policy"""
        import random
        
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except self.exceptions as exc:
                last_exception = exc
                
                if attempt < self.max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if self.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_attempts} for {func.__name__}",
                        extra={
                            "retry": {
                                "attempt": attempt + 1,
                                "max_attempts": self.max_attempts,
                                "delay": delay,
                                "error": str(exc)
                            }
                        }
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Max retry attempts exceeded for {func.__name__}",
                        extra={
                            "retry": {
                                "max_attempts": self.max_attempts,
                                "final_error": str(exc)
                            }
                        }
                    )
        
        raise last_exception


class HealthCheck:
    """
    Health check system for service monitoring.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 5.0,
        critical: bool = True
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
            timeout: Check timeout in seconds
            critical: Whether failure should mark service as unhealthy
        """
        with self._lock:
            self.checks[name] = {
                "func": check_func,
                "timeout": timeout,
                "critical": critical
            }
            
        logger.info(
            f"Health check registered: {name}",
            extra={
                "health_check": {
                    "service": self.name,
                    "check": name,
                    "critical": critical,
                    "timeout": timeout
                }
            }
        )
    
    def run_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Health check results with overall status
        """
        results = {}
        overall_healthy = True
        
        with self._lock:
            for check_name, check_config in self.checks.items():
                try:
                    with Timeout(check_config["timeout"]):
                        start_time = time.time()
                        healthy = check_config["func"]()
                        duration = time.time() - start_time
                        
                        results[check_name] = {
                            "healthy": healthy,
                            "duration": duration,
                            "critical": check_config["critical"],
                            "error": None
                        }
                        
                        if not healthy and check_config["critical"]:
                            overall_healthy = False
                            
                except Exception as exc:
                    results[check_name] = {
                        "healthy": False,
                        "duration": 0,
                        "critical": check_config["critical"],
                        "error": str(exc)
                    }
                    
                    if check_config["critical"]:
                        overall_healthy = False
                    
                    logger.error(
                        f"Health check failed: {check_name}",
                        extra={
                            "health_check": {
                                "service": self.name,
                                "check": check_name,
                                "error": str(exc)
                            }
                        },
                        exc_info=True
                    )
        
        self.results = results
        
        summary = {
            "service": self.name,
            "healthy": overall_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
        
        logger.info(
            f"Health check completed: {self.name}",
            extra={"health_check_summary": summary}
        )
        
        return summary


# Global resilience components registry
class ResilienceRegistry:
    """Registry for managing resilience components"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker"""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, **kwargs)
            return self.circuit_breakers[name]
    
    def get_rate_limiter(self, name: str, **kwargs) -> RateLimiter:
        """Get or create rate limiter"""
        with self._lock:
            if name not in self.rate_limiters:
                self.rate_limiters[name] = RateLimiter(name, **kwargs)
            return self.rate_limiters[name]
    
    def get_bulkhead(self, name: str, **kwargs) -> Bulkhead:
        """Get or create bulkhead"""
        with self._lock:
            if name not in self.bulkheads:
                self.bulkheads[name] = Bulkhead(name, **kwargs)
            return self.bulkheads[name]
    
    def get_health_check(self, name: str) -> HealthCheck:
        """Get or create health check"""
        with self._lock:
            if name not in self.health_checks:
                self.health_checks[name] = HealthCheck(name)
            return self.health_checks[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all resilience components"""
        with self._lock:
            return {
                "circuit_breakers": {
                    name: cb.get_stats() 
                    for name, cb in self.circuit_breakers.items()
                },
                "rate_limiters": {
                    name: {
                        "name": rl.name,
                        "tokens": rl.tokens,
                        "max_requests": rl.max_requests
                    }
                    for name, rl in self.rate_limiters.items()
                },
                "bulkheads": {
                    name: bh.get_stats()
                    for name, bh in self.bulkheads.items()
                },
                "health_checks": {
                    name: hc.results
                    for name, hc in self.health_checks.items()
                }
            }


# Global registry instance
resilience_registry = ResilienceRegistry()