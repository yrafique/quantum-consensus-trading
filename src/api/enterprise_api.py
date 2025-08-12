"""
Enterprise API Server
====================

Production-grade API server with comprehensive enterprise patterns:
- Circuit breakers and fault tolerance
- Rate limiting and throttling
- Comprehensive error handling
- Request/response logging with correlation IDs
- Performance monitoring and metrics
- Health checks and graceful shutdown
- API versioning and documentation
"""

import time
import uuid
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging

# Import enterprise core components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.exceptions import (
    BaseRiverException, NetworkException, DataValidationException,
    BusinessLogicException, ErrorSeverity, safe_execute
)
from src.core.logging_config import (
    get_logger, correlation_context, request_context, 
    business_events, setup_logging
)
from src.core.resilience import (
    CircuitBreaker, RateLimiter, Bulkhead, Timeout,
    resilience_registry
)
from src.core.monitoring import (
    metrics_collector, performance_monitor, alert_manager, get_system_health
)
from src.core.config import get_config, requires_config

# Import business components
from src.monitoring.enhanced_connection_monitor import EnterpriseConnectionMonitor

logger = get_logger(__name__)


class EnterpriseAPI:
    """
    Enterprise-grade API server with comprehensive monitoring and fault tolerance.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = config_override or get_config()
        self.app = Flask(__name__)
        self.connection_monitor: Optional[EnterpriseConnectionMonitor] = None
        
        # Setup Flask configuration
        self._configure_flask()
        
        # Initialize enterprise patterns
        self._setup_resilience_patterns()
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        logger.info(
            "Enterprise API server initialized",
            extra={
                "config": {
                    "host": self.config.api.host,
                    "port": self.config.api.port,
                    "workers": self.config.api.workers
                }
            }
        )
    
    def _configure_flask(self):
        """Configure Flask application with enterprise settings"""
        
        # CORS configuration
        if self.config.api.enable_cors:
            CORS(
                self.app,
                origins=self.config.api.cors_origins,
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"]
            )
        
        # Flask configuration
        self.app.config.update({
            'MAX_CONTENT_LENGTH': self.config.api.max_request_size,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': False
        })
        
        # Suppress Flask request logging (we handle it ourselves)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    def _setup_resilience_patterns(self):
        """Setup circuit breakers, rate limiters, and bulkheads"""
        
        # Global API circuit breaker
        self.api_circuit_breaker = resilience_registry.get_circuit_breaker(
            "api_server",
            failure_threshold=10,
            recovery_timeout=60
        )
        
        # Rate limiter for general API access
        self.api_rate_limiter = resilience_registry.get_rate_limiter(
            "api_general",
            max_requests=self.config.api.rate_limit_per_minute,
            time_window=60
        )
        
        # Bulkhead for concurrent request handling
        self.request_bulkhead = resilience_registry.get_bulkhead(
            "api_requests",
            max_concurrent=self.config.api.workers * 2
        )
        
        # Specific rate limiters for different endpoints
        self.analysis_rate_limiter = resilience_registry.get_rate_limiter(
            "api_analysis",
            max_requests=10,  # Limit analysis requests
            time_window=60
        )
        
        self.health_rate_limiter = resilience_registry.get_rate_limiter(
            "api_health",
            max_requests=60,  # Allow frequent health checks
            time_window=60
        )
    
    def _setup_middleware(self):
        """Setup middleware for logging, monitoring, and security"""
        
        @self.app.before_request
        def before_request():
            """Pre-request middleware"""
            
            # Generate correlation ID
            correlation_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())
            g.correlation_id = correlation_id
            g.start_time = time.perf_counter()
            
            # Setup request context for logging  
            g.request_context = request_context(
                endpoint=request.endpoint or request.path,
                method=request.method,
                request_id=correlation_id
            ).__enter__()
            
            # Setup correlation context
            g.correlation_context = correlation_context(correlation_id).__enter__()
            
            # Rate limiting check
            try:
                if not self.api_rate_limiter.acquire():
                    return self._error_response(
                        "Rate limit exceeded",
                        429,
                        {"retry_after": "60 seconds"}
                    )
            except Exception as exc:
                logger.error("Rate limiting error", exc_info=True)
            
            # Log request
            logger.info(
                f"API request: {request.method} {request.path}",
                extra={
                    "request": {
                        "method": request.method,
                        "path": request.path,
                        "remote_addr": request.remote_addr,
                        "user_agent": request.headers.get('User-Agent'),
                        "content_length": request.content_length
                    }
                }
            )
            
            # Increment request metrics
            metrics_collector.increment_counter(
                "api_requests_total",
                labels={
                    "method": request.method,
                    "endpoint": request.endpoint or "unknown"
                }
            )
        
        @self.app.after_request
        def after_request(response):
            """Post-request middleware"""
            
            try:
                # Calculate request duration
                duration_ms = (time.perf_counter() - g.start_time) * 1000
                
                # Record performance metrics
                performance_monitor.record_database_query(
                    duration_ms,
                    f"{request.method}_{request.endpoint or 'unknown'}"
                )
                
                # Record response metrics
                metrics_collector.record_histogram(
                    "api_request_duration_ms",
                    duration_ms,
                    labels={
                        "method": request.method,
                        "endpoint": request.endpoint or "unknown",
                        "status_code": str(response.status_code)
                    }
                )
                
                # Add correlation ID to response headers
                response.headers['X-Correlation-ID'] = g.correlation_id
                response.headers['X-Response-Time'] = f"{duration_ms:.2f}ms"
                
                # Log response
                logger.info(
                    f"API response: {response.status_code}",
                    extra={
                        "response": {
                            "status_code": response.status_code,
                            "duration_ms": duration_ms,
                            "content_length": response.content_length
                        }
                    }
                )
                
                # Business event logging for important actions
                if request.endpoint in ['analyze', 'opportunities']:
                    business_events.log_user_action(
                        f"api_{request.endpoint}",
                        request.remote_addr,  # Use IP as user ID for now
                        {
                            "endpoint": request.endpoint,
                            "method": request.method,
                            "status_code": response.status_code,
                            "duration_ms": duration_ms
                        }
                    )
                
            except Exception as exc:
                logger.error("Error in after_request middleware", exc_info=True)
            finally:
                # Cleanup contexts
                if hasattr(g, 'correlation_context'):
                    g.correlation_context.__exit__(None, None, None)
                if hasattr(g, 'request_context'):
                    g.request_context.__exit__(None, None, None)
            
            return response
    
    def _setup_error_handlers(self):
        """Setup comprehensive error handling"""
        
        @self.app.errorhandler(BaseRiverException)
        def handle_river_exception(exc: BaseRiverException):
            """Handle River system exceptions"""
            
            # Record error metrics
            metrics_collector.increment_counter(
                "api_errors_total",
                labels={
                    "error_type": type(exc).__name__,
                    "severity": exc.severity.value,
                    "category": exc.category.value
                }
            )
            
            # Determine HTTP status code based on exception type
            status_code = {
                'DataValidationException': 400,
                'BusinessLogicException': 400,
                'NetworkException': 502,
                'AIInferenceException': 503,
                'ConfigurationException': 500,
                'SystemResourceException': 503
            }.get(type(exc).__name__, 500)
            
            return self._error_response(
                exc.user_message,
                status_code,
                {
                    "error_id": exc.error_id,
                    "error_type": type(exc).__name__,
                    "severity": exc.severity.value,
                    "recovery_hint": exc.recovery_hint
                }
            )
        
        @self.app.errorhandler(404)
        def handle_not_found(error):
            """Handle 404 errors"""
            return self._error_response(
                "Endpoint not found",
                404,
                {"available_endpoints": self._get_available_endpoints()}
            )
        
        @self.app.errorhandler(405)
        def handle_method_not_allowed(error):
            """Handle method not allowed errors"""
            return self._error_response(
                "Method not allowed for this endpoint",
                405,
                {"allowed_methods": error.valid_methods if hasattr(error, 'valid_methods') else []}
            )
        
        @self.app.errorhandler(Exception)
        def handle_generic_exception(exc):
            """Handle unexpected exceptions"""
            
            # Log unexpected error
            logger.error(
                "Unexpected API error",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc()
                },
                exc_info=True
            )
            
            # Record error metrics
            metrics_collector.increment_counter(
                "api_errors_total",
                labels={
                    "error_type": "unexpected",
                    "severity": "high"
                }
            )
            
            return self._error_response(
                "Internal server error",
                500,
                {"error_id": str(uuid.uuid4())}
            )
    
    def _error_response(self, message: str, status_code: int, details: Dict[str, Any] = None):
        """Create standardized error response"""
        
        response_data = {
            "error": True,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": getattr(g, 'correlation_id', None)
        }
        
        if details:
            response_data.update(details)
        
        return jsonify(response_data), status_code
    
    def _success_response(self, data: Any = None, message: str = None, meta: Dict[str, Any] = None):
        """Create standardized success response"""
        
        response_data = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": getattr(g, 'correlation_id', None)
        }
        
        if message:
            response_data["message"] = message
        
        if data is not None:
            response_data["data"] = data
        
        if meta:
            response_data["meta"] = meta
        
        return jsonify(response_data)
    
    def _initialize_monitoring(self):
        """Initialize connection monitoring"""
        try:
            self.connection_monitor = EnterpriseConnectionMonitor()
            self.connection_monitor.start_monitoring()
            
            logger.info("Connection monitoring initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize connection monitoring",
                extra={"error": str(exc)},
                exc_info=True
            )
    
    def _get_available_endpoints(self) -> List[str]:
        """Get list of available API endpoints"""
        endpoints = []
        for rule in self.app.url_map.iter_rules():
            if rule.endpoint != 'static':
                endpoints.append({
                    "path": rule.rule,
                    "methods": list(rule.methods - {'HEAD', 'OPTIONS'})
                })
        return endpoints
    
    def _setup_routes(self):
        """Setup API routes with enterprise patterns"""
        
        @self.app.route('/health', methods=['GET'])
        @self.health_rate_limiter
        def health_check():
            """Comprehensive health check endpoint"""
            
            try:
                # Basic health info
                health_data = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": self.config.version,
                    "environment": self.config.environment.value
                }
                
                # Add connection monitoring data if available
                if self.connection_monitor:
                    health_data.update(self.connection_monitor.get_health_summary())
                
                # Add system metrics
                system_health = get_system_health()
                health_data["system_metrics"] = system_health
                
                # Determine overall status
                if self.connection_monitor:
                    overall_health = self.connection_monitor.get_overall_health()
                    if not overall_health["critical_components_healthy"]:
                        health_data["status"] = "degraded"
                        return jsonify(health_data), 503
                
                return jsonify(health_data), 200
                
            except Exception as exc:
                logger.error("Health check failed", exc_info=True)
                return jsonify({
                    "status": "unhealthy",
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat()
                }), 503
        
        @self.app.route('/status', methods=['GET'])
        @self.health_rate_limiter
        def detailed_status():
            """Detailed system status for monitoring"""
            
            try:
                status_data = {
                    "api_server": {
                        "status": "running",
                        "uptime_seconds": time.time() - g.start_time if hasattr(g, 'start_time') else 0,
                        "version": self.config.version
                    },
                    "resilience_patterns": resilience_registry.get_stats(),
                    "metrics_summary": {
                        name: metrics_collector.get_metric_summary(name)
                        for name in ["api_requests_total", "api_errors_total", "api_request_duration_ms"]
                        if metrics_collector.get_current_value(name) is not None
                    }
                }
                
                # Add connection monitoring if available
                if self.connection_monitor:
                    status_data["connection_monitoring"] = self.connection_monitor.get_health_summary()
                
                return self._success_response(status_data)
                
            except Exception as exc:
                raise NetworkException(
                    "Failed to get system status",
                    context={"error": str(exc)},
                    cause=exc
                )
        
        @self.app.route('/analyze', methods=['POST'])
        @self.analysis_rate_limiter
        @self.request_bulkhead
        def analyze_stock():
            """Stock analysis endpoint with comprehensive error handling"""
            
            try:
                # Validate request data
                if not request.is_json:
                    raise DataValidationException(
                        "Request must contain JSON data",
                        field="content_type",
                        value=request.content_type
                    )
                
                data = request.get_json()
                if not data or 'ticker' not in data:
                    raise DataValidationException(
                        "Ticker symbol is required",
                        field="ticker",
                        value=data.get('ticker') if data else None
                    )
                
                ticker = data['ticker'].upper().strip()
                if not ticker or len(ticker) > 10:
                    raise DataValidationException(
                        "Invalid ticker symbol",
                        field="ticker",
                        value=ticker
                    )
                
                # Check system readiness
                if self.connection_monitor and not self.connection_monitor.led_manager.is_system_ready():
                    raise BusinessLogicException(
                        "System is not ready for analysis. Please wait for all components to be online.",
                        rule="system_readiness_check"
                    )
                
                # Perform analysis with timeout and circuit breaker protection
                @self.api_circuit_breaker
                @safe_execute(max_retries=2, retry_delay=1.0)
                def perform_analysis():
                    # This would integrate with the actual ReAct agent
                    # For now, return a mock response
                    return {
                        "ticker": ticker,
                        "analysis": {
                            "recommendation": "HOLD",
                            "confidence_score": 0.75,
                            "reasoning": "Based on technical analysis and market conditions",
                            "price_target": 155.00,
                            "risk_level": "moderate"
                        },
                        "metadata": {
                            "analysis_time": datetime.utcnow().isoformat(),
                            "model_version": "1.0.0",
                            "data_sources": ["yahoo_finance", "technical_indicators"]
                        }
                    }
                
                with performance_monitor.track_request("analyze", "POST"):
                    result = perform_analysis()
                
                return self._success_response(
                    result,
                    f"Analysis completed for {ticker}",
                    {"processing_time_ms": (time.perf_counter() - g.start_time) * 1000}
                )
                
            except BaseRiverException:
                raise  # Re-raise River exceptions
            except Exception as exc:
                raise NetworkException(
                    f"Analysis failed for ticker {data.get('ticker', 'unknown')}",
                    context={"ticker": data.get('ticker'), "error": str(exc)},
                    cause=exc
                )
        
        @self.app.route('/opportunities', methods=['POST'])
        @self.analysis_rate_limiter
        @self.request_bulkhead
        def find_opportunities():
            """Market opportunities endpoint"""
            
            try:
                data = request.get_json() or {}
                
                # Validate request
                sectors = data.get('sectors', ['technology'])
                limit = min(data.get('limit', 10), 50)  # Cap at 50
                
                # Check system readiness
                if self.connection_monitor and not self.connection_monitor.led_manager.is_system_ready():
                    raise BusinessLogicException(
                        "System is not ready for opportunity hunting",
                        rule="system_readiness_check"
                    )
                
                @self.api_circuit_breaker
                def find_market_opportunities():
                    # Mock opportunity data
                    return {
                        "opportunities": [
                            {
                                "ticker": "AAPL",
                                "sector": "technology",
                                "opportunity_score": 0.85,
                                "reasoning": "Strong earnings growth with technical breakout",
                                "expected_return": 0.12,
                                "risk_score": 0.3
                            },
                            {
                                "ticker": "MSFT", 
                                "sector": "technology",
                                "opportunity_score": 0.78,
                                "reasoning": "Cloud growth momentum continues",
                                "expected_return": 0.10,
                                "risk_score": 0.25
                            }
                        ],
                        "scan_metadata": {
                            "sectors_scanned": sectors,
                            "total_stocks_analyzed": 500,
                            "scan_time": datetime.utcnow().isoformat()
                        }
                    }
                
                with performance_monitor.track_request("opportunities", "POST"):
                    result = find_market_opportunities()
                
                return self._success_response(
                    result,
                    f"Found {len(result['opportunities'])} opportunities",
                    {"sectors": sectors, "limit": limit}
                )
                
            except BaseRiverException:
                raise
            except Exception as exc:
                raise NetworkException(
                    "Opportunity hunting failed",
                    context={"error": str(exc)},
                    cause=exc
                )
        
        @self.app.route('/metrics', methods=['GET'])
        def prometheus_metrics():
            """Prometheus-compatible metrics endpoint"""
            
            try:
                metrics_data = metrics_collector.export_prometheus()
                
                from flask import Response
                return Response(metrics_data, mimetype='text/plain')
                
            except Exception as exc:
                logger.error("Failed to export metrics", exc_info=True)
                return self._error_response("Metrics export failed", 500)
        
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            """Get current system alerts"""
            
            try:
                alerts_data = {
                    "active_alerts": alert_manager.get_active_alerts(),
                    "alert_history": alert_manager.get_alert_history(limit=20)
                }
                
                return self._success_response(alerts_data)
                
            except Exception as exc:
                raise NetworkException(
                    "Failed to retrieve alerts",
                    context={"error": str(exc)},
                    cause=exc
                )
        
        # API documentation endpoint
        @self.app.route('/', methods=['GET'])
        def api_documentation():
            """API documentation endpoint"""
            
            return jsonify({
                "name": "River Trading System API",
                "version": self.config.version,
                "description": "Enterprise AI-powered trading platform API",
                "endpoints": self._get_available_endpoints(),
                "documentation": {
                    "health": "GET /health - System health check",
                    "status": "GET /status - Detailed system status",
                    "analyze": "POST /analyze - Analyze individual stocks",
                    "opportunities": "POST /opportunities - Find market opportunities",
                    "metrics": "GET /metrics - Prometheus metrics",
                    "alerts": "GET /alerts - System alerts"
                },
                "support": {
                    "correlation_id": "Include X-Correlation-ID header for request tracking",
                    "rate_limiting": f"Rate limit: {self.config.api.rate_limit_per_minute} requests per minute",
                    "error_handling": "All errors include correlation_id and recovery_hint"
                }
            })
    
    def run(self, debug: bool = False):
        """Run the API server"""
        
        try:
            logger.info(
                f"Starting Enterprise API server on {self.config.api.host}:{self.config.api.port}",
                extra={
                    "host": self.config.api.host,
                    "port": self.config.api.port,
                    "debug": debug,
                    "environment": self.config.environment.value
                }
            )
            
            self.app.run(
                host=self.config.api.host,
                port=self.config.api.port,
                debug=debug,
                threaded=True
            )
            
        except Exception as exc:
            logger.error(
                "Failed to start API server",
                extra={"error": str(exc)},
                exc_info=True
            )
            raise
    
    def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("Shutting down Enterprise API server")
        
        if self.connection_monitor:
            self.connection_monitor.stop_monitoring()
        
        # Stop metrics collection
        metrics_collector.stop_system_metrics_collection()
        alert_manager.stop_alert_evaluation()
        
        logger.info("Enterprise API server shutdown complete")


def create_app(config_override: Optional[Dict[str, Any]] = None) -> Flask:
    """Factory function to create Flask app"""
    
    api = EnterpriseAPI(config_override)
    return api.app


if __name__ == "__main__":
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_dir="logs",
        console_output=True,
        json_format=False  # Use simple format for development
    )
    
    # Create and run API
    api = EnterpriseAPI()
    
    try:
        api.run(debug=False)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        api.shutdown()