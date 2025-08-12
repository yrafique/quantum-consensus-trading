"""
API Middleware
=============

Custom middleware for request processing, security, and monitoring.
"""

import time
import uuid
from typing import Callable, Dict, Any
import asyncio

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import Counter, Histogram
import redis

from ..core.config import get_config
from ..core.exceptions import RateLimitExceededException

logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

RATE_LIMIT_EXCEEDED = Counter(
    'rate_limit_exceeded_total',
    'Rate limit exceeded events',
    ['endpoint', 'client_ip']
)


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """
    Add unique request ID to each request for tracing.
    
    Args:
        request: The incoming request
        call_next: The next middleware/route handler
        
    Returns:
        Response with request ID header
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """
    Add security headers to all responses.
    
    Args:
        request: The incoming request
        call_next: The next middleware/route handler
        
    Returns:
        Response with security headers
    """
    response = await call_next(request)
    
    # Add security headers
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    for header, value in security_headers.items():
        response.headers[header] = value
    
    return response


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Collect metrics for monitoring and observability.
    
    Args:
        request: The incoming request
        call_next: The next middleware/route handler
        
    Returns:
        Response with timing metrics collected
    """
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    # Extract endpoint pattern for metrics (avoid high cardinality)
    endpoint = _extract_endpoint_pattern(path)
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        
    except Exception as exc:
        # Handle exceptions and still record metrics
        status_code = 500
        logger.error(
            "Request failed with exception",
            method=method,
            path=path,
            error=str(exc),
            request_id=getattr(request.state, 'request_id', None),
            exc_info=True
        )
        raise
    
    finally:
        # Record metrics
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log request details
        logger.info(
            "Request processed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            request_id=getattr(request.state, 'request_id', None)
        )
    
    return response


async def rate_limiting_middleware(request: Request, call_next: Callable) -> Response:
    """
    Implement rate limiting based on client IP and endpoint.
    
    Args:
        request: The incoming request
        call_next: The next middleware/route handler
        
    Returns:
        Response or rate limit error
    """
    config = get_config()
    
    if not config.security.enable_rate_limiting:
        return await call_next(request)
    
    # Get client IP
    client_ip = _get_client_ip(request)
    endpoint = _extract_endpoint_pattern(request.url.path)
    
    # Create rate limit key
    rate_limit_key = f"rate_limit:{client_ip}:{endpoint}"
    
    try:
        # Check rate limit using Redis
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True
        )
        
        # Get current request count
        current_count = r.get(rate_limit_key)
        limit = config.api.rate_limit_per_minute
        
        if current_count is None:
            # First request in window
            r.setex(rate_limit_key, 60, 1)  # 60 seconds window
        else:
            current_count = int(current_count)
            if current_count >= limit:
                # Rate limit exceeded
                RATE_LIMIT_EXCEEDED.labels(
                    endpoint=endpoint,
                    client_ip=client_ip
                ).inc()
                
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    endpoint=endpoint,
                    current_count=current_count,
                    limit=limit,
                    request_id=getattr(request.state, 'request_id', None)
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "type": "RateLimitExceeded",
                            "message": f"Rate limit exceeded. Maximum {limit} requests per minute.",
                            "retry_after": 60,
                            "request_id": getattr(request.state, 'request_id', None)
                        }
                    },
                    headers={
                        "Retry-After": "60",
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + 60)
                    }
                )
            else:
                # Increment counter
                r.incr(rate_limit_key)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = max(0, limit - (int(current_count) if current_count else 0) - 1)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
        
    except redis.RedisError as exc:
        # If Redis is unavailable, allow request but log error
        logger.error(
            "Rate limiting failed due to Redis error",
            error=str(exc),
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=getattr(request.state, 'request_id', None)
        )
        return await call_next(request)
    
    except Exception as exc:
        # Unexpected error, allow request but log
        logger.error(
            "Rate limiting middleware error",
            error=str(exc),
            client_ip=client_ip,
            endpoint=endpoint,
            request_id=getattr(request.state, 'request_id', None),
            exc_info=True
        )
        return await call_next(request)


def _extract_endpoint_pattern(path: str) -> str:
    """
    Extract endpoint pattern from path for metrics (avoid high cardinality).
    
    Args:
        path: Request path
        
    Returns:
        Normalized endpoint pattern
    """
    # Simple pattern extraction - replace IDs with placeholders
    import re
    
    # Replace UUIDs
    path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
    
    # Replace numeric IDs
    path = re.sub(r'/\d+', '/{id}', path)
    
    # Replace common symbol patterns
    path = re.sub(r'/[A-Z]{1,5}', '/{symbol}', path)
    
    return path


def _get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: The incoming request
        
    Returns:
        Client IP address
    """
    # Check for forwarded headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to request client
    return request.client.host if request.client else "unknown"


class CircuitBreakerMiddleware:
    """
    Circuit breaker middleware for external service calls.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, Any] = {}
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        Apply circuit breaker pattern to requests.
        
        Args:
            request: The incoming request
            call_next: The next middleware/route handler
            
        Returns:
            Response or circuit breaker error
        """
        # Identify if this is an external service call
        path = request.url.path
        
        if self._is_external_service_call(path):
            service_name = self._extract_service_name(path)
            
            # Check circuit breaker state
            if self._is_circuit_open(service_name):
                logger.warning(
                    "Circuit breaker open for service",
                    service=service_name,
                    path=path,
                    request_id=getattr(request.state, 'request_id', None)
                )
                
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": {
                            "type": "ServiceUnavailable",
                            "message": f"Service {service_name} is temporarily unavailable",
                            "service": service_name,
                            "request_id": getattr(request.state, 'request_id', None)
                        }
                    }
                )
        
        return await call_next(request)
    
    def _is_external_service_call(self, path: str) -> bool:
        """Check if path represents external service call."""
        external_patterns = [
            "/api/v1/trading/execute",
            "/api/v1/analysis/external",
            "/api/v1/data/external"
        ]
        return any(pattern in path for pattern in external_patterns)
    
    def _extract_service_name(self, path: str) -> str:
        """Extract service name from path."""
        if "trading" in path:
            return "trading_service"
        elif "analysis" in path:
            return "analysis_service"
        elif "data" in path:
            return "data_service"
        return "unknown_service"
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for service."""
        # Simplified implementation - in production, use proper circuit breaker
        return False