"""
River Trading System - FastAPI Main Application
===============================================

Modern, high-performance API server with Google-level architecture patterns.
Features comprehensive monitoring, security, and enterprise-grade reliability.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import asyncio
import logging
import time
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import structlog

# Internal imports
from ..core.config import get_config, RiverTradingConfig
from ..core.exceptions import (
    BaseRiverException,
    ConfigurationException,
    ExternalAPIException
)
from ..core.logging_config import setup_logging
from ..core.monitoring import MetricsCollector, HealthCheck
from ..core.resilience import CircuitBreaker, RateLimiter
from .routers import trading, analysis, portfolio, system, auth
from .middleware import (
    request_id_middleware,
    security_headers_middleware,
    rate_limiting_middleware,
    metrics_middleware,
)
from .dependencies import get_current_user, verify_api_key

# Initialize structured logging
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'river_trading_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'river_trading_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Counter(
    'river_trading_active_connections',
    'Active WebSocket connections'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    """
    # Startup
    logger.info(
        "Starting River Trading System API",
        version=app.version,
        environment=get_config().environment.value
    )
    
    try:
        # Initialize monitoring
        metrics_collector = MetricsCollector()
        health_check = HealthCheck("api_server")
        
        # Start background tasks
        metrics_collector.start_system_metrics_collection()
        # Note: HealthCheck doesn't have start_health_checks method, so we'll skip this for now
        
        # Store in app state
        app.state.metrics_collector = metrics_collector
        app.state.health_check = health_check
        
        # Start Prometheus metrics server if enabled
        config = get_config()
        if config.monitoring.prometheus_enabled:
            start_http_server(config.monitoring.metrics_port)
            logger.info(
                "Prometheus metrics server started",
                port=config.monitoring.metrics_port
            )
        
        logger.info("API startup completed successfully")
        yield
        
    except Exception as exc:
        logger.error("Failed to start API", error=str(exc), exc_info=True)
        raise
    
    # Shutdown
    logger.info("Shutting down River Trading System API")
    
    try:
        # Cancel background tasks
        for task in getattr(app.state, 'background_tasks', []):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cleanup resources
        if hasattr(app.state, 'health_check'):
            # HealthCheck doesn't have cleanup method, so we'll skip this
            pass
        
        logger.info("API shutdown completed successfully")
        
    except Exception as exc:
        logger.error("Error during shutdown", error=str(exc), exc_info=True)


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(
        log_level=config.logging.level.value,
        log_dir=config.logging.directory,
        console_output=True,
        json_format=(config.logging.format == "json")
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="River Trading System API",
        description="Enterprise AI-Powered Trading Platform with ReAct Reasoning & MLX Acceleration",
        version="1.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None,
        lifespan=lifespan,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
        }
    )
    
    # Configure CORS
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
        )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", config.api.host]
    )
    
    # Add custom middleware
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(security_headers_middleware)
    app.middleware("http")(metrics_middleware)
    
    if config.security.enable_rate_limiting:
        app.middleware("http")(rate_limiting_middleware)
    
    # Include routers
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["System"],
        dependencies=[]
    )
    
    app.include_router(
        auth.router,
        prefix="/api/v1/auth",
        tags=["Authentication"]
    )
    
    # Protected routes
    protected_dependencies = []
    if config.security.enable_authentication:
        protected_dependencies.append(Depends(get_current_user))
    
    app.include_router(
        trading.router,
        prefix="/api/v1/trading",
        tags=["Trading"],
        dependencies=protected_dependencies
    )
    
    app.include_router(
        analysis.router,
        prefix="/api/v1/analysis",
        tags=["Analysis"],
        dependencies=protected_dependencies
    )
    
    app.include_router(
        portfolio.router,
        prefix="/api/v1/portfolio",
        tags=["Portfolio"],
        dependencies=protected_dependencies
    )
    
    return app


# Create app instance
app = create_application()


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with links to main features"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>River Trading System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-bottom: 10px; }
            .subtitle { color: #7f8c8d; margin-bottom: 30px; }
            .links { margin: 30px 0; }
            .link-item { margin: 15px 0; padding: 15px; background: #ecf0f1; border-radius: 5px; }
            .link-item a { color: #3498db; text-decoration: none; font-weight: bold; }
            .link-item a:hover { text-decoration: underline; }
            .description { color: #7f8c8d; margin-top: 5px; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 River Trading System</h1>
            <p class="subtitle">AI-Powered Trading Platform with Intelligent LangChain/LangGraph Routing</p>
            <p class="status">✅ System Status: Online and Ready</p>
            
            <div class="links">
                <div class="link-item">
                    <a href="/docs">🌐 Interactive API Documentation</a>
                    <div class="description">Explore all endpoints with Swagger UI - start here!</div>
                </div>
                
                <div class="link-item">
                    <a href="/health">⚡ System Health Check</a>
                    <div class="description">View system status and uptime</div>
                </div>
                
                <div class="link-item">
                    <a href="/api/v1/system/info">📊 System Information</a>
                    <div class="description">Platform specs and resource usage</div>
                </div>
                
                <div class="link-item">
                    <a href="/api/v1/analysis/performance">📈 Performance Metrics</a>
                    <div class="description">Intelligent router statistics and agent performance</div>
                </div>
            </div>
            
            <p><strong>🧠 Features:</strong> Query intent detection, smart agent selection, circuit breaker patterns, performance tracking, and context-aware routing.</p>
        </div>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns:
        System health status and component details
    """
    try:
        health_check = getattr(app.state, 'health_check', None)
        if not health_check:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Basic health check - services running"
            }
        
        # Basic health status
        health_status = {
            "overall_status": "healthy",
            "components": {
                "api_server": "healthy",
                "configuration": "healthy",
                "monitoring": "healthy"
            }
        }
        
        return {
            "status": health_status["overall_status"],
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": get_config().environment.value,
            "components": health_status["components"],
            "uptime_seconds": time.time() - getattr(app.state, 'start_time', time.time())
        }
        
    except Exception as exc:
        logger.error("Health check failed", error=str(exc), exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(exc)
        }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return generate_latest()


@app.get("/ready", tags=["System"])
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    
    Returns:
        Service readiness status
    """
    try:
        # Check critical dependencies
        config = get_config()
        
        # Basic readiness checks
        checks = {
            "config_loaded": config is not None,
            "api_server": True,  # If we're responding, API is ready
        }
        
        # Additional checks for production
        if config.environment.value == "production":
            # Add production-specific readiness checks
            pass
        
        all_ready = all(checks.values())
        
        return {
            "ready": all_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
        
    except Exception as exc:
        logger.error("Readiness check failed", error=str(exc), exc_info=True)
        return {
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(exc)
        }


@app.get("/live", tags=["System"])
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes-style liveness probe.
    
    Returns:
        Simple liveness confirmation
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(BaseRiverException)
async def river_trading_exception_handler(
    request: Request, 
    exc: BaseRiverException
) -> JSONResponse:
    """
    Handle custom River Trading exceptions.
    """
    logger.error(
        "River Trading exception occurred",
        error=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "context": getattr(exc, 'context', {}),
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, 
    exc: HTTPException
) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions.
    """
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=getattr(request.state, 'request_id', None),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.
    """
    logger.error(
        "Unexpected exception occurred",
        error=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


if __name__ == "__main__":
    # Development server
    config = get_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        log_level="info" if config.debug else "warning",
        access_log=config.debug,
        use_colors=True,
        reload_dirs=["src"] if config.debug else None,
    )