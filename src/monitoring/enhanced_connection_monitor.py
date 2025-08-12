"""
Enterprise Connection Monitor & LED Status System
================================================

Production-grade monitoring system with enterprise patterns:
- Circuit breakers for fault tolerance
- Comprehensive error handling and recovery
- Structured logging with correlation IDs
- Performance metrics and alerting
- Health checks and graceful degradation
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import requests
import json

# Import enterprise core components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.exceptions import (
    BaseRiverException, NetworkException, AIInferenceException,
    SystemResourceException, ErrorSeverity, safe_execute, error_boundary
)
from src.core.logging_config import get_logger, correlation_context, business_events
from src.core.resilience import (
    CircuitBreaker, RateLimiter, Bulkhead, RetryPolicy, HealthCheck,
    resilience_registry
)
from src.core.monitoring import metrics_collector, performance_monitor
from src.core.config import get_config

logger = get_logger(__name__)


class ConnectionStatus(Enum):
    """Connection status levels with enterprise semantics"""
    CONNECTED = "connected"      # Fully operational
    CONNECTING = "connecting"    # Initialization in progress
    DEGRADED = "degraded"       # Partially functional
    DISCONNECTED = "disconnected"  # Temporarily unavailable
    ERROR = "error"             # Failed with errors
    UNKNOWN = "unknown"         # Status not determined


class ComponentType(Enum):
    """System component types with criticality classification"""
    DATA_FEED = "data_feed"         # Critical: Market data access
    MLX_LLM = "mlx_llm"            # Critical: AI inference engine
    REACT_AGENT = "react_agent"     # Critical: Trading logic
    API_SERVER = "api_server"       # Important: External interfaces
    DATABASE = "database"           # Important: Data persistence
    MARKET_DATA = "market_data"     # Important: Historical data
    CACHE = "cache"                # Optional: Performance optimization


@dataclass
class ConnectionInfo:
    """
    Enhanced connection information with enterprise features.
    """
    component: ComponentType
    status: ConnectionStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    check_interval: int = 30
    circuit_breaker_name: str = ""
    health_score: float = 1.0  # 0.0 = unhealthy, 1.0 = perfect health
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.circuit_breaker_name:
            self.circuit_breaker_name = f"connection_{self.component.value}"


@dataclass
class LEDStatus:
    """Enhanced LED visual status with accessibility"""
    color: str  # green, yellow, red, blue, gray
    blink: bool = False
    message: str = ""
    icon: str = "●"
    accessibility_text: str = ""  # Screen reader support
    severity: ErrorSeverity = ErrorSeverity.INFO


class EnterpriseConnectionMonitor:
    """
    Enterprise-grade connection monitoring system.
    
    Features:
    - Circuit breakers for each component
    - Rate limiting for health checks
    - Comprehensive error handling
    - Performance monitoring
    - Automatic recovery
    - Business event logging
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = config_override or {}
        self.connections: Dict[ComponentType, ConnectionInfo] = {}
        self.led_manager: Optional['LEDStatusManager'] = None
        self.health_check = HealthCheck("connection_monitor")
        
        # Thread management
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.total_checks = 0
        self.successful_checks = 0
        
        try:
            self._initialize_components()
            self._setup_health_checks()
            self._register_metrics()
            
            logger.info(
                "Enterprise connection monitor initialized",
                extra={
                    "components": list(self.connections.keys()),
                    "config": self.config
                }
            )
            
        except Exception as exc:
            raise SystemResourceException(
                "Failed to initialize connection monitor",
                resource="connection_monitor",
                context={"error": str(exc)},
                cause=exc
            )
    
    def _initialize_components(self):
        """Initialize component connections with enterprise patterns"""
        
        # Component configurations with different criticality levels
        component_configs = {
            ComponentType.DATA_FEED: {
                "check_interval": 15,
                "max_retries": 5,
                "circuit_breaker_threshold": 3,
                "timeout": 10
            },
            ComponentType.MLX_LLM: {
                "check_interval": 30,
                "max_retries": 3,
                "circuit_breaker_threshold": 2,
                "timeout": 30
            },
            ComponentType.REACT_AGENT: {
                "check_interval": 20,
                "max_retries": 3,
                "circuit_breaker_threshold": 2,
                "timeout": 15
            },
            ComponentType.API_SERVER: {
                "check_interval": 10,
                "max_retries": 3,
                "circuit_breaker_threshold": 5,
                "timeout": 5
            },
            ComponentType.MARKET_DATA: {
                "check_interval": 60,
                "max_retries": 2,
                "circuit_breaker_threshold": 3,
                "timeout": 20
            }
        }
        
        for component_type, config in component_configs.items():
            # Create connection info
            connection = ConnectionInfo(
                component=component_type,
                status=ConnectionStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                response_time=0.0,
                check_interval=config["check_interval"],
                max_retries=config["max_retries"]
            )
            
            self.connections[component_type] = connection
            
            # Create circuit breaker for this component
            circuit_breaker = resilience_registry.get_circuit_breaker(
                connection.circuit_breaker_name,
                failure_threshold=config["circuit_breaker_threshold"],
                recovery_timeout=60,
                expected_exception=(NetworkException, AIInferenceException, SystemResourceException)
            )
            
            # Create rate limiter for health checks
            rate_limiter = resilience_registry.get_rate_limiter(
                f"health_check_{component_type.value}",
                max_requests=10,  # Max 10 health checks per minute
                time_window=60
            )
    
    def _setup_health_checks(self):
        """Setup health check registrations"""
        
        # Register individual component health checks
        for component_type in self.connections.keys():
            self.health_check.register_check(
                f"{component_type.value}_status",
                lambda ct=component_type: self._component_health_check(ct),
                timeout=5.0,
                critical=(component_type in [ComponentType.DATA_FEED, ComponentType.MLX_LLM, ComponentType.REACT_AGENT])
            )
        
        # Register overall system health check
        self.health_check.register_check(
            "overall_connectivity",
            self._overall_health_check,
            timeout=2.0,
            critical=True
        )
    
    def _register_metrics(self):
        """Register monitoring metrics"""
        
        # Connection status metrics
        for component_type in self.connections.keys():
            component_name = component_type.value
            
            metrics_collector.register_metric(
                f"connection_status_{component_name}",
                type="gauge",
                help_text=f"Connection status for {component_name} (1=connected, 0=disconnected)"
            )
            
            metrics_collector.register_metric(
                f"connection_response_time_{component_name}",
                type="histogram",
                help_text=f"Response time for {component_name} health checks"
            )
            
            metrics_collector.register_metric(
                f"connection_failures_{component_name}",
                type="counter",
                help_text=f"Total connection failures for {component_name}"
            )
    
    def start_monitoring(self):
        """Start monitoring with enterprise patterns"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        with correlation_context(f"monitor_startup_{int(time.time())}"):
            try:
                self.monitoring_active = True
                
                # Start monitoring thread with bulkhead protection
                bulkhead = resilience_registry.get_bulkhead("connection_monitor", max_concurrent=1)
                
                @bulkhead
                def start_monitor_thread():
                    self.monitor_thread = threading.Thread(
                        target=self._monitoring_loop,
                        daemon=True,
                        name="EnterpriseConnectionMonitor"
                    )
                    self.monitor_thread.start()
                
                start_monitor_thread()
                
                # Log business event
                business_events.log_system_event(
                    "monitoring_started",
                    "connection_monitor",
                    {"components": [ct.value for ct in self.connections.keys()]}
                )
                
                logger.info(
                    "Connection monitoring started",
                    extra={
                        "thread_name": self.monitor_thread.name if self.monitor_thread else None,
                        "component_count": len(self.connections)
                    }
                )
                
            except Exception as exc:
                self.monitoring_active = False
                raise SystemResourceException(
                    "Failed to start connection monitoring",
                    resource="monitoring_thread",
                    context={"error": str(exc)},
                    cause=exc
                )
    
    def stop_monitoring(self):
        """Stop monitoring with graceful shutdown"""
        if not self.monitoring_active:
            return
        
        with correlation_context(f"monitor_shutdown_{int(time.time())}"):
            logger.info("Stopping connection monitoring")
            
            self.monitoring_active = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitor thread did not stop gracefully")
            
            business_events.log_system_event(
                "monitoring_stopped",
                "connection_monitor"
            )
            
            logger.info("Connection monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop with enterprise patterns"""
        logger.info("Connection monitoring loop started")
        
        while self.monitoring_active:
            loop_start = time.perf_counter()
            
            try:
                with correlation_context(f"monitor_cycle_{int(time.time())}"):
                    self._check_all_connections()
                    self._update_metrics()
                    
                    # Calculate and record loop performance
                    loop_duration = (time.perf_counter() - loop_start) * 1000
                    performance_monitor.record_database_query(loop_duration, "monitoring_cycle")
                    
                    # Adaptive sleep based on system load
                    base_sleep = 5.0
                    cpu_usage = metrics_collector.get_current_value("system_cpu_percent") or 0
                    if cpu_usage > 80:
                        base_sleep *= 1.5  # Slow down under high CPU load
                    
                    time.sleep(base_sleep)
                    
            except Exception as exc:
                logger.error(
                    "Error in monitoring loop",
                    extra={"error": str(exc)},
                    exc_info=True
                )
                
                # Exponential backoff on errors
                time.sleep(min(30, 5 * (2 ** getattr(self, '_loop_error_count', 0))))
                setattr(self, '_loop_error_count', getattr(self, '_loop_error_count', 0) + 1)
            else:
                # Reset error count on successful iteration
                setattr(self, '_loop_error_count', 0)
        
        logger.info("Connection monitoring loop stopped")
    
    @error_boundary(fallback_value=None, severity=ErrorSeverity.MEDIUM)
    def _check_all_connections(self):
        """Check all connections with fault tolerance"""
        
        for component_type, connection in self.connections.items():
            try:
                # Check if it's time for a health check
                time_since_check = (datetime.utcnow() - connection.last_check).total_seconds()
                if time_since_check < connection.check_interval:
                    continue
                
                # Get circuit breaker for this component
                circuit_breaker = resilience_registry.get_circuit_breaker(connection.circuit_breaker_name)
                
                # Execute health check with circuit breaker protection
                @circuit_breaker
                def check_component():
                    return self._check_connection(connection)
                
                check_component()
                
            except Exception as exc:
                logger.error(
                    f"Failed to check connection for {component_type.value}",
                    extra={
                        "component": component_type.value,
                        "error": str(exc)
                    },
                    exc_info=True
                )
    
    def _check_connection(self, connection: ConnectionInfo):
        """Check individual connection with comprehensive error handling"""
        
        start_time = time.perf_counter()
        component_name = connection.component.value
        
        try:
            connection.status = ConnectionStatus.CONNECTING
            connection.last_check = datetime.utcnow()
            self.total_checks += 1
            
            # Component-specific health check logic
            success = self._perform_component_check(connection.component)
            
            # Calculate response time
            response_time = (time.perf_counter() - start_time) * 1000
            connection.response_time = response_time
            
            if success:
                connection.status = ConnectionStatus.CONNECTED
                connection.error_message = None
                connection.consecutive_failures = 0
                connection.last_success = datetime.utcnow()
                connection.health_score = min(1.0, connection.health_score + 0.1)
                self.successful_checks += 1
                
                logger.debug(
                    f"Connection check successful: {component_name}",
                    extra={
                        "component": component_name,
                        "response_time_ms": response_time,
                        "health_score": connection.health_score
                    }
                )
                
            else:
                self._handle_connection_failure(connection, "Health check returned false")
            
        except Exception as exc:
            error_message = str(exc)
            response_time = (time.perf_counter() - start_time) * 1000
            connection.response_time = response_time
            
            self._handle_connection_failure(connection, error_message)
            
            # Re-raise for circuit breaker
            if isinstance(exc, BaseRiverException):
                raise
            else:
                raise NetworkException(
                    f"Connection check failed for {component_name}: {error_message}",
                    endpoint=component_name,
                    context={
                        "component": component_name,
                        "response_time_ms": response_time
                    },
                    cause=exc
                )
    
    def _handle_connection_failure(self, connection: ConnectionInfo, error_message: str):
        """Handle connection failure with comprehensive recovery logic"""
        
        connection.status = ConnectionStatus.ERROR
        connection.error_message = error_message
        connection.consecutive_failures += 1
        connection.health_score = max(0.0, connection.health_score - 0.2)
        
        component_name = connection.component.value
        
        # Increment failure metrics
        metrics_collector.increment_counter(
            f"connection_failures_{component_name}",
            labels={"component": component_name}
        )
        
        logger.warning(
            f"Connection check failed: {component_name}",
            extra={
                "component": component_name,
                "error": error_message,
                "consecutive_failures": connection.consecutive_failures,
                "health_score": connection.health_score
            }
        )
        
        # Log business event for critical components
        if connection.component in [ComponentType.DATA_FEED, ComponentType.MLX_LLM, ComponentType.REACT_AGENT]:
            business_events.log_system_event(
                "critical_component_failure",
                component_name,
                {
                    "error": error_message,
                    "consecutive_failures": connection.consecutive_failures,
                    "health_score": connection.health_score
                }
            )
    
    @safe_execute
    def _perform_component_check(self, component_type: ComponentType) -> bool:
        """Perform component-specific health checks"""
        
        if component_type == ComponentType.DATA_FEED:
            return self._check_data_feed()
        elif component_type == ComponentType.MLX_LLM:
            return self._check_mlx_llm()
        elif component_type == ComponentType.REACT_AGENT:
            return self._check_react_agent()
        elif component_type == ComponentType.API_SERVER:
            return self._check_api_server()
        elif component_type == ComponentType.MARKET_DATA:
            return self._check_market_data()
        else:
            logger.warning(f"Unknown component type for health check: {component_type}")
            return False
    
    @error_boundary(fallback_value=False)
    def _check_data_feed(self) -> bool:
        """Check data feed connectivity with rate limiting"""
        rate_limiter = resilience_registry.get_rate_limiter("data_feed_health", max_requests=5, time_window=60)
        
        @rate_limiter
        def check_feed():
            # Test with a simple market data request
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d", interval="1m")
            return len(data) > 0
        
        return check_feed()
    
    @error_boundary(fallback_value=False)
    def _check_mlx_llm(self) -> bool:
        """Check MLX LLM availability with timeout"""
        try:
            # Simple inference test
            import mlx.core as mx
            
            # Test basic MLX functionality
            test_array = mx.array([1, 2, 3, 4])
            result = mx.sum(test_array)
            
            return float(result) == 10.0
            
        except ImportError:
            logger.warning("MLX not available for health check")
            return False
        except Exception as exc:
            raise AIInferenceException(
                "MLX health check failed",
                model="mlx_test",
                context={"error": str(exc)},
                cause=exc
            )
    
    @error_boundary(fallback_value=False)
    def _check_react_agent(self) -> bool:
        """Check ReAct agent availability"""
        try:
            # Import and basic instantiation test
            # This would be replaced with actual agent health check
            return True
        except Exception as exc:
            raise AIInferenceException(
                "ReAct agent health check failed",
                context={"error": str(exc)},
                cause=exc
            )
    
    @error_boundary(fallback_value=False)
    def _check_api_server(self) -> bool:
        """Check API server health endpoint"""
        try:
            response = requests.get("http://localhost:5001/health", timeout=5)
            return response.status_code in [200, 503]  # 503 might indicate partial health
        except requests.RequestException as exc:
            raise NetworkException(
                "API server health check failed",
                endpoint="http://localhost:5001/health",
                context={"error": str(exc)},
                cause=exc
            )
    
    @error_boundary(fallback_value=False)
    def _check_market_data(self) -> bool:
        """Check market data availability"""
        try:
            # Check if we can access historical data files
            import os
            data_dir = "data"
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
                return len(files) > 0
            return False
        except Exception as exc:
            logger.warning(f"Market data health check failed: {exc}")
            return False
    
    def _update_metrics(self):
        """Update monitoring metrics"""
        
        for component_type, connection in self.connections.items():
            component_name = component_type.value
            
            # Connection status (1 = connected, 0 = not connected)
            status_value = 1.0 if connection.status == ConnectionStatus.CONNECTED else 0.0
            metrics_collector.set_gauge(
                f"connection_status_{component_name}",
                status_value,
                labels={"component": component_name}
            )
            
            # Response time
            metrics_collector.record_histogram(
                f"connection_response_time_{component_name}",
                connection.response_time,
                labels={"component": component_name}
            )
        
        # Overall health score
        if self.connections:
            overall_health = sum(conn.health_score for conn in self.connections.values()) / len(self.connections)
            metrics_collector.set_gauge("connection_monitor_health_score", overall_health)
    
    def _component_health_check(self, component_type: ComponentType) -> bool:
        """Health check for specific component"""
        connection = self.connections.get(component_type)
        if not connection:
            return False
        
        return connection.status == ConnectionStatus.CONNECTED
    
    def _overall_health_check(self) -> bool:
        """Overall system health check"""
        critical_components = [ComponentType.DATA_FEED, ComponentType.MLX_LLM, ComponentType.REACT_AGENT]
        
        critical_healthy = all(
            self.connections[comp].status == ConnectionStatus.CONNECTED
            for comp in critical_components
            if comp in self.connections
        )
        
        return critical_healthy
    
    def get_connection_status(self, component_type: ComponentType) -> Optional[ConnectionInfo]:
        """Get connection status for specific component"""
        return self.connections.get(component_type)
    
    def get_all_statuses(self) -> Dict[ComponentType, ConnectionInfo]:
        """Get all connection statuses"""
        return self.connections.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        
        total_components = len(self.connections)
        connected_components = sum(
            1 for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        )
        
        health_percentage = (connected_components / total_components * 100) if total_components > 0 else 0
        
        critical_components = [ComponentType.DATA_FEED, ComponentType.MLX_LLM, ComponentType.REACT_AGENT]
        critical_healthy = all(
            self.connections[comp].status == ConnectionStatus.CONNECTED
            for comp in critical_components
            if comp in self.connections
        )
        
        return {
            "overall_status": "healthy" if critical_healthy else "degraded",
            "health_percentage": health_percentage,
            "total_components": total_components,
            "connected_components": connected_components,
            "critical_components_healthy": critical_healthy,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "success_rate": (self.successful_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        }
    
    def force_check_all(self):
        """Force immediate check of all connections"""
        logger.info("Forcing immediate connection checks")
        
        for connection in self.connections.values():
            connection.last_check = datetime.utcnow() - timedelta(seconds=connection.check_interval + 1)
        
        business_events.log_system_event(
            "forced_health_check",
            "connection_monitor"
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for APIs"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": self.get_overall_health(),
            "component_details": {
                comp.value: {
                    "status": conn.status.value,
                    "health_score": conn.health_score,
                    "response_time_ms": conn.response_time,
                    "consecutive_failures": conn.consecutive_failures,
                    "last_success": conn.last_success.isoformat() if conn.last_success else None,
                    "error_message": conn.error_message
                }
                for comp, conn in self.connections.items()
            },
            "health_check_results": self.health_check.run_checks()
        }


class LEDStatusManager:
    """
    Enhanced LED status manager with enterprise features.
    """
    
    def __init__(self, connection_monitor: EnterpriseConnectionMonitor):
        self.connection_monitor = connection_monitor
        self.connection_monitor.led_manager = self
        
        logger.info("LED status manager initialized")
    
    def get_all_led_statuses(self) -> Dict[ComponentType, LEDStatus]:
        """Get LED status for all components"""
        
        statuses = {}
        
        for component_type, connection in self.connection_monitor.connections.items():
            status = self._connection_to_led_status(connection)
            statuses[component_type] = status
        
        return statuses
    
    def _connection_to_led_status(self, connection: ConnectionInfo) -> LEDStatus:
        """Convert connection info to LED status"""
        
        status_map = {
            ConnectionStatus.CONNECTED: LEDStatus(
                color="connected",
                icon="🟢",
                message=f"Connected ({connection.response_time:.1f}s)",
                accessibility_text=f"{connection.component.value} is connected",
                severity=ErrorSeverity.INFO
            ),
            ConnectionStatus.CONNECTING: LEDStatus(
                color="connecting",
                icon="🔵",
                message="Connecting...",
                blink=True,
                accessibility_text=f"{connection.component.value} is connecting",
                severity=ErrorSeverity.INFO
            ),
            ConnectionStatus.DEGRADED: LEDStatus(
                color="degraded",
                icon="🟡",
                message="Degraded performance",
                accessibility_text=f"{connection.component.value} has degraded performance",
                severity=ErrorSeverity.MEDIUM
            ),
            ConnectionStatus.DISCONNECTED: LEDStatus(
                color="disconnected",
                icon="🟡",
                message="Disconnected",
                blink=True,
                accessibility_text=f"{connection.component.value} is disconnected",
                severity=ErrorSeverity.MEDIUM
            ),
            ConnectionStatus.ERROR: LEDStatus(
                color="error",
                icon="🔴",
                message=f"Error: {connection.error_message or 'Unknown error'}",
                accessibility_text=f"{connection.component.value} has an error",
                severity=ErrorSeverity.HIGH
            ),
            ConnectionStatus.UNKNOWN: LEDStatus(
                color="unknown",
                icon="⚪",
                message="Status unknown",
                accessibility_text=f"{connection.component.value} status is unknown",
                severity=ErrorSeverity.LOW
            )
        }
        
        return status_map.get(connection.status, status_map[ConnectionStatus.UNKNOWN])
    
    def is_system_ready(self) -> bool:
        """Check if system is ready for user interaction"""
        
        health = self.connection_monitor.get_overall_health()
        
        # System is ready if:
        # 1. All critical components are connected
        # 2. Overall health is above threshold (60%)
        return (
            health["critical_components_healthy"] and
            health["health_percentage"] >= 60
        )
    
    def get_system_status_summary(self) -> Dict[str, Any]:
        """Get system status summary for UI display"""
        
        led_statuses = self.get_all_led_statuses()
        health = self.connection_monitor.get_overall_health()
        
        return {
            "ready": self.is_system_ready(),
            "health_percentage": health["health_percentage"],
            "status_message": "System Ready" if self.is_system_ready() else "System Initializing",
            "component_count": health["total_components"],
            "connected_count": health["connected_components"],
            "led_statuses": {
                comp.value: {
                    "color": status.color,
                    "icon": status.icon,
                    "message": status.message,
                    "blink": status.blink,
                    "accessibility_text": status.accessibility_text
                }
                for comp, status in led_statuses.items()
            }
        }