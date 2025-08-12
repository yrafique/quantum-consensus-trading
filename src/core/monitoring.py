"""
Enterprise Monitoring and Metrics System
========================================

Comprehensive monitoring with metrics collection, alerting,
and observability features.
"""

import time
import threading
import queue
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import psutil
import traceback

from .exceptions import BaseRiverException, ErrorSeverity, NetworkException
from .logging_config import get_logger, correlation_context
from .resilience import HealthCheck

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"         # Always increasing value
    GAUGE = "gauge"            # Point-in-time value
    HISTOGRAM = "histogram"    # Distribution of values
    TIMER = "timer"           # Duration measurements


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""


@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    condition: Callable[[Any], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 5
    enabled: bool = True
    last_fired: Optional[datetime] = None
    fire_count: int = 0


class MetricsCollector:
    """
    Thread-safe metrics collection and storage system.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.metric_definitions: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.system_metrics_interval = 30  # seconds
        self.system_metrics_thread = None
        
        # Start system metrics collection
        self.start_system_metrics_collection()
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        help_text: str = "",
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Register a metric definition.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            help_text: Description of the metric
            labels: Default labels for the metric
        """
        with self._lock:
            self.metric_definitions[name] = Metric(
                name=name,
                value=0,
                metric_type=metric_type,
                labels=labels or {},
                help_text=help_text
            )
            
        logger.debug(
            f"Metric registered: {name}",
            extra={"metric": {"name": name, "type": metric_type.value}}
        )
    
    def increment_counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        self._record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value"""
        self._record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a histogram value"""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def record_timer(
        self,
        name: str,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a timer duration"""
        self._record_metric(name, duration_ms, MetricType.TIMER, labels)
    
    def _record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """Internal metric recording"""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=datetime.utcnow()
            )
            
            # Store metric
            self.metrics[name].append(metric)
            
            # Update definition if not exists
            if name not in self.metric_definitions:
                self.metric_definitions[name] = metric
    
    def get_metric_values(
        self,
        name: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Metric]:
        """
        Get metric values with optional filtering.
        
        Args:
            name: Metric name
            since: Only return values since this timestamp
            limit: Maximum number of values to return
            
        Returns:
            List of metric values
        """
        with self._lock:
            if name not in self.metrics:
                return []
            
            values = list(self.metrics[name])
            
            # Filter by timestamp
            if since:
                values = [m for m in values if m.timestamp >= since]
            
            # Limit results
            if limit:
                values = values[-limit:]
            
            return values
    
    def get_current_value(self, name: str) -> Optional[Union[int, float]]:
        """Get the most recent value for a metric"""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            return self.metrics[name][-1].value
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get statistical summary of a metric"""
        values = self.get_metric_values(name)
        if not values:
            return {}
        
        numeric_values = [m.value for m in values]
        
        return {
            "count": len(values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "current": numeric_values[-1],
            "first_timestamp": values[0].timestamp.isoformat(),
            "last_timestamp": values[-1].timestamp.isoformat()
        }
    
    def start_system_metrics_collection(self):
        """Start collecting system metrics in background thread"""
        if self.system_metrics_thread and self.system_metrics_thread.is_alive():
            return
        
        self.system_metrics_enabled = True
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True,
            name="SystemMetricsCollector"
        )
        self.system_metrics_thread.start()
        
        logger.info("System metrics collection started")
    
    def stop_system_metrics_collection(self):
        """Stop collecting system metrics"""
        self.system_metrics_enabled = False
        if self.system_metrics_thread:
            self.system_metrics_thread.join(timeout=5)
        logger.info("System metrics collection stopped")
    
    def _collect_system_metrics(self):
        """Background thread for collecting system metrics"""
        # Register system metrics
        self.register_metric("system_cpu_percent", MetricType.GAUGE, "CPU utilization percentage")
        self.register_metric("system_memory_percent", MetricType.GAUGE, "Memory utilization percentage")
        self.register_metric("system_memory_bytes", MetricType.GAUGE, "Memory usage in bytes")
        self.register_metric("system_disk_percent", MetricType.GAUGE, "Disk utilization percentage")
        self.register_metric("system_network_bytes_sent", MetricType.COUNTER, "Network bytes sent")
        self.register_metric("system_network_bytes_recv", MetricType.COUNTER, "Network bytes received")
        
        while self.system_metrics_enabled:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system_cpu_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.set_gauge("system_memory_percent", memory.percent)
                self.set_gauge("system_memory_bytes", memory.used)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.set_gauge("system_disk_percent", disk_percent)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.set_gauge("system_network_bytes_sent", network.bytes_sent)
                self.set_gauge("system_network_bytes_recv", network.bytes_recv)
                
                time.sleep(self.system_metrics_interval)
                
            except Exception as exc:
                logger.error(
                    "Error collecting system metrics",
                    extra={"error": str(exc)},
                    exc_info=True
                )
                time.sleep(self.system_metrics_interval)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            for name, metric_def in self.metric_definitions.items():
                if name not in self.metrics or not self.metrics[name]:
                    continue
                
                # Add help text
                if metric_def.help_text:
                    lines.append(f"# HELP {name} {metric_def.help_text}")
                
                # Add type
                prom_type = {
                    MetricType.COUNTER: "counter",
                    MetricType.GAUGE: "gauge",
                    MetricType.HISTOGRAM: "histogram",
                    MetricType.TIMER: "histogram"
                }.get(metric_def.metric_type, "gauge")
                
                lines.append(f"# TYPE {name} {prom_type}")
                
                # Add current value
                current_metric = self.metrics[name][-1]
                label_str = ""
                if current_metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in current_metric.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{name}{label_str} {current_metric.value}")
        
        return "\n".join(lines)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics data"""
        with self._lock:
            result = {}
            for name in self.metrics:
                result[name] = {
                    "definition": self.metric_definitions.get(name, {}).__dict__ if name in self.metric_definitions else {},
                    "summary": self.get_metric_summary(name),
                    "current_value": self.get_current_value(name)
                }
            return result


class AlertManager:
    """
    Alert management and notification system.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        self._lock = threading.RLock()
        
        # Alert evaluation thread
        self.evaluation_enabled = True
        self.evaluation_interval = 30  # seconds
        self.evaluation_thread = None
        
        self._setup_default_alerts()
        self.start_alert_evaluation()
    
    def register_alert(
        self,
        name: str,
        condition: Callable[[MetricsCollector], bool],
        severity: AlertSeverity,
        message_template: str,
        cooldown_minutes: int = 5
    ):
        """
        Register an alert rule.
        
        Args:
            name: Alert name
            condition: Function that returns True if alert should fire
            severity: Alert severity level
            message_template: Message template for alert
            cooldown_minutes: Minimum minutes between alert firings
        """
        with self._lock:
            self.alerts[name] = Alert(
                name=name,
                condition=condition,
                severity=severity,
                message_template=message_template,
                cooldown_minutes=cooldown_minutes
            )
        
        logger.info(
            f"Alert registered: {name}",
            extra={"alert": {"name": name, "severity": severity.value}}
        )
    
    def add_notification_handler(self, handler: Callable[[str, AlertSeverity, str], None]):
        """
        Add notification handler for alerts.
        
        Args:
            handler: Function that handles alert notifications
                    Signature: handler(alert_name, severity, message)
        """
        self.notification_handlers.append(handler)
        logger.info(f"Notification handler added: {handler.__name__}")
    
    def _setup_default_alerts(self):
        """Setup default system alerts"""
        
        # High CPU usage alert
        self.register_alert(
            "high_cpu_usage",
            lambda mc: (mc.get_current_value("system_cpu_percent") or 0) > 80,
            AlertSeverity.WARNING,
            "High CPU usage detected: {cpu_percent:.1f}%"
        )
        
        # High memory usage alert
        self.register_alert(
            "high_memory_usage",
            lambda mc: (mc.get_current_value("system_memory_percent") or 0) > 85,
            AlertSeverity.CRITICAL,
            "High memory usage detected: {memory_percent:.1f}%"
        )
        
        # High error rate alert
        self.register_alert(
            "high_error_rate",
            self._check_error_rate,
            AlertSeverity.CRITICAL,
            "High error rate detected: {error_rate:.2f} errors/minute"
        )
    
    def _check_error_rate(self, metrics_collector: MetricsCollector) -> bool:
        """Check if error rate is too high"""
        # Get error count from last 5 minutes
        since = datetime.utcnow() - timedelta(minutes=5)
        error_metrics = metrics_collector.get_metric_values("errors_total", since=since)
        
        if len(error_metrics) < 2:
            return False
        
        # Calculate error rate per minute
        time_span = (error_metrics[-1].timestamp - error_metrics[0].timestamp).total_seconds() / 60
        if time_span <= 0:
            return False
        
        error_count = sum(m.value for m in error_metrics)
        error_rate = error_count / time_span
        
        return error_rate > 10  # More than 10 errors per minute
    
    def start_alert_evaluation(self):
        """Start alert evaluation in background thread"""
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            return
        
        self.evaluation_enabled = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluate_alerts,
            daemon=True,
            name="AlertEvaluator"
        )
        self.evaluation_thread.start()
        
        logger.info("Alert evaluation started")
    
    def stop_alert_evaluation(self):
        """Stop alert evaluation"""
        self.evaluation_enabled = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
        logger.info("Alert evaluation stopped")
    
    def _evaluate_alerts(self):
        """Background thread for evaluating alerts"""
        while self.evaluation_enabled:
            try:
                with self._lock:
                    for alert_name, alert in self.alerts.items():
                        if not alert.enabled:
                            continue
                        
                        # Check cooldown
                        if alert.last_fired:
                            cooldown_end = alert.last_fired + timedelta(minutes=alert.cooldown_minutes)
                            if datetime.utcnow() < cooldown_end:
                                continue
                        
                        # Evaluate condition
                        try:
                            should_fire = alert.condition(self.metrics_collector)
                        except Exception as exc:
                            logger.error(
                                f"Error evaluating alert condition: {alert_name}",
                                extra={"alert": alert_name, "error": str(exc)},
                                exc_info=True
                            )
                            continue
                        
                        if should_fire:
                            self._fire_alert(alert)
                
                time.sleep(self.evaluation_interval)
                
            except Exception as exc:
                logger.error(
                    "Error in alert evaluation loop",
                    extra={"error": str(exc)},
                    exc_info=True
                )
                time.sleep(self.evaluation_interval)
    
    def _fire_alert(self, alert: Alert):
        """Fire an alert and send notifications"""
        # Update alert state
        alert.last_fired = datetime.utcnow()
        alert.fire_count += 1
        
        # Format message with current metrics
        message_data = {}
        try:
            # Add common metric values to message data
            message_data.update({
                "cpu_percent": self.metrics_collector.get_current_value("system_cpu_percent") or 0,
                "memory_percent": self.metrics_collector.get_current_value("system_memory_percent") or 0,
            })
            
            message = alert.message_template.format(**message_data)
        except Exception:
            message = alert.message_template
        
        # Add to alert history
        alert_record = {
            "name": alert.name,
            "severity": alert.severity.value,
            "message": message,
            "timestamp": alert.last_fired.isoformat(),
            "fire_count": alert.fire_count
        }
        self.alert_history.append(alert_record)
        
        # Log alert
        log_level = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.INFO: logging.INFO
        }.get(alert.severity, logging.WARNING)
        
        logger.log(
            log_level,
            f"Alert fired: {alert.name}",
            extra={"alert": alert_record}
        )
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert.name, alert.severity, message)
            except Exception as exc:
                logger.error(
                    f"Error in notification handler: {handler.__name__}",
                    extra={"error": str(exc)},
                    exc_info=True
                )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts"""
        active = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        
        with self._lock:
            for alert in self.alerts.values():
                if alert.last_fired and alert.last_fired > cutoff_time:
                    active.append({
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "last_fired": alert.last_fired.isoformat(),
                        "fire_count": alert.fire_count
                    })
        
        return active
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        with self._lock:
            return list(self.alert_history)[-limit:]


class PerformanceMonitor:
    """
    Application performance monitoring with automatic instrumentation.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_requests: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Register performance metrics
        self._register_performance_metrics()
    
    def _register_performance_metrics(self):
        """Register performance-related metrics"""
        metrics = [
            ("request_duration_ms", MetricType.HISTOGRAM, "Request duration in milliseconds"),
            ("request_count", MetricType.COUNTER, "Total number of requests"),
            ("error_count", MetricType.COUNTER, "Total number of errors"),
            ("active_requests", MetricType.GAUGE, "Number of active requests"),
            ("database_query_duration_ms", MetricType.HISTOGRAM, "Database query duration"),
            ("cache_hit_rate", MetricType.GAUGE, "Cache hit rate percentage"),
            ("ai_inference_duration_ms", MetricType.HISTOGRAM, "AI inference duration"),
        ]
        
        for name, metric_type, help_text in metrics:
            self.metrics_collector.register_metric(name, metric_type, help_text)
    
    def track_request(self, endpoint: str, method: str = "GET"):
        """Context manager for tracking request performance"""
        return RequestTracker(self, endpoint, method)
    
    def record_database_query(self, duration_ms: float, query_type: str = "unknown"):
        """Record database query performance"""
        self.metrics_collector.record_histogram(
            "database_query_duration_ms",
            duration_ms,
            labels={"query_type": query_type}
        )
    
    def record_cache_hit(self, hit: bool, cache_type: str = "default"):
        """Record cache hit/miss"""
        hit_rate = 100.0 if hit else 0.0
        self.metrics_collector.set_gauge(
            "cache_hit_rate",
            hit_rate,
            labels={"cache_type": cache_type, "result": "hit" if hit else "miss"}
        )
    
    def record_ai_inference(self, duration_ms: float, model: str = "unknown"):
        """Record AI inference performance"""
        self.metrics_collector.record_histogram(
            "ai_inference_duration_ms",
            duration_ms,
            labels={"model": model}
        )


class RequestTracker:
    """Context manager for tracking individual request performance"""
    
    def __init__(self, performance_monitor: PerformanceMonitor, endpoint: str, method: str):
        self.performance_monitor = performance_monitor
        self.endpoint = endpoint
        self.method = method
        self.start_time = None
        self.request_id = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.request_id = f"{self.method}:{self.endpoint}"
        
        with self.performance_monitor._lock:
            self.performance_monitor.active_requests[self.request_id] = self.start_time
            active_count = len(self.performance_monitor.active_requests)
            self.performance_monitor.metrics_collector.set_gauge("active_requests", active_count)
        
        self.performance_monitor.metrics_collector.increment_counter(
            "request_count",
            labels={"endpoint": self.endpoint, "method": self.method}
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Record duration
        self.performance_monitor.metrics_collector.record_histogram(
            "request_duration_ms",
            duration_ms,
            labels={"endpoint": self.endpoint, "method": self.method}
        )
        
        # Record error if exception occurred
        if exc_type is not None:
            self.performance_monitor.metrics_collector.increment_counter(
                "error_count",
                labels={
                    "endpoint": self.endpoint,
                    "method": self.method,
                    "error_type": exc_type.__name__
                }
            )
        
        # Remove from active requests
        with self.performance_monitor._lock:
            self.performance_monitor.active_requests.pop(self.request_id, None)
            active_count = len(self.performance_monitor.active_requests)
            self.performance_monitor.metrics_collector.set_gauge("active_requests", active_count)


# Global monitoring components
metrics_collector = MetricsCollector()
alert_manager = AlertManager(metrics_collector)
performance_monitor = PerformanceMonitor(metrics_collector)


def console_notification_handler(alert_name: str, severity: AlertSeverity, message: str):
    """Default console notification handler"""
    severity_color = {
        AlertSeverity.CRITICAL: "\033[91m",  # Red
        AlertSeverity.WARNING: "\033[93m",   # Yellow
        AlertSeverity.INFO: "\033[94m"       # Blue
    }.get(severity, "\033[0m")
    
    print(f"{severity_color}[ALERT] {severity.value.upper()}: {alert_name}")
    print(f"  {message}\033[0m")


# Add default notification handler
alert_manager.add_notification_handler(console_notification_handler)


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health report"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_summary": {
            name: metrics_collector.get_metric_summary(name)
            for name in ["system_cpu_percent", "system_memory_percent", "request_count", "error_count"]
            if metrics_collector.get_current_value(name) is not None
        },
        "active_alerts": alert_manager.get_active_alerts(),
        "performance": {
            "active_requests": len(performance_monitor.active_requests),
            "avg_response_time": metrics_collector.get_metric_summary("request_duration_ms").get("avg", 0)
        }
    }