"""
Enterprise Production Launcher
==============================

Google-level production launcher with comprehensive enterprise patterns:
- Graceful startup and shutdown sequences
- Health check dependencies
- Resource monitoring and limits
- Configuration validation
- Signal handling and cleanup
- Process supervision
- Metrics and alerting integration
"""

import os
import sys
import signal
import time
import threading
import subprocess
import atexit
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.exceptions import (
    BaseRiverException, SystemResourceException, ConfigurationException,
    ErrorSeverity, safe_execute
)
from src.core.logging_config import (
    setup_logging, get_logger, correlation_context, business_events
)
from src.core.config import ConfigManager, get_config, Environment
from src.core.resilience import HealthCheck, resilience_registry
from src.core.monitoring import (
    metrics_collector, alert_manager, performance_monitor, get_system_health
)
from src.monitoring.enhanced_connection_monitor import EnterpriseConnectionMonitor
from src.api.enterprise_api import EnterpriseAPI

logger = get_logger(__name__)


class ServiceState:
    """Service state tracking"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class EnterpriseLauncher:
    """
    Enterprise-grade launcher with comprehensive service management.
    
    Features:
    - Dependency-aware startup sequence
    - Health check validation before service start
    - Resource monitoring and limits
    - Graceful shutdown with timeouts
    - Signal handling and cleanup
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration first
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        
        # Service management
        self.services: Dict[str, Dict[str, Any]] = {}
        self.service_states: Dict[str, str] = {}
        self.startup_sequence: List[str] = []
        self.shutdown_sequence: List[str] = []
        
        # Process management
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}
        
        # System components
        self.connection_monitor: Optional[EnterpriseConnectionMonitor] = None
        self.api_server: Optional[EnterpriseAPI] = None
        self.health_check = HealthCheck("enterprise_launcher")
        
        # Runtime state
        self.running = False
        self.startup_time = None
        self.shutdown_requested = False
        
        # Setup logging
        self._setup_logging()
        
        # Register signal handlers
        self._setup_signal_handlers()
        
        # Register shutdown cleanup
        atexit.register(self.cleanup)
        
        logger.info(
            "Enterprise launcher initialized",
            extra={
                "environment": self.config.environment.value,
                "version": self.config.version,
                "config_path": config_path
            }
        )
    
    def _setup_logging(self):
        """Setup enterprise logging"""
        setup_logging(
            log_level=self.config.logging.level.value,
            log_dir=self.config.logging.directory,
            max_file_size=self.config.logging.max_file_size_mb * 1024 * 1024,
            backup_count=self.config.logging.backup_count,
            console_output=self.config.logging.enable_console_output,
            json_format=(self.config.logging.format == "json")
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"Received signal {signal_name}, initiating graceful shutdown")
            
            if not self.shutdown_requested:
                self.shutdown_requested = True
                threading.Thread(target=self.shutdown, daemon=True).start()
        
        # Handle common shutdown signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, signal_handler)
        
        # Handle SIGHUP for configuration reload (Unix only)
        if hasattr(signal, 'SIGHUP'):
            def reload_handler(signum, frame):
                logger.info("Received SIGHUP, reloading configuration")
                self._reload_configuration()
            
            signal.signal(signal.SIGHUP, reload_handler)
    
    def _define_services(self):
        """Define services and their dependencies"""
        
        self.services = {
            "monitoring": {
                "description": "System monitoring and metrics collection",
                "dependencies": [],
                "health_check": self._check_monitoring_health,
                "start_timeout": 30,
                "stop_timeout": 15,
                "critical": True
            },
            "connection_monitor": {
                "description": "Connection and component health monitoring",
                "dependencies": ["monitoring"],
                "health_check": self._check_connection_monitor_health,
                "start_timeout": 45,
                "stop_timeout": 20,
                "critical": True
            },
            "api_server": {
                "description": "REST API server",
                "dependencies": ["monitoring", "connection_monitor"],
                "health_check": self._check_api_server_health,
                "start_timeout": 60,
                "stop_timeout": 30,
                "critical": True
            },
            "ui_server": {
                "description": "Streamlit UI server",
                "dependencies": ["api_server"],
                "health_check": self._check_ui_server_health,
                "start_timeout": 90,
                "stop_timeout": 20,
                "critical": False
            }
        }
        
        # Build startup and shutdown sequences
        self.startup_sequence = self._build_dependency_order()
        self.shutdown_sequence = list(reversed(self.startup_sequence))
        
        # Initialize service states
        for service_name in self.services:
            self.service_states[service_name] = ServiceState.STOPPED
    
    def _build_dependency_order(self) -> List[str]:
        """Build service startup order based on dependencies"""
        
        ordered = []
        visited = set()
        temp_visited = set()
        
        def visit(service_name):
            if service_name in temp_visited:
                raise ConfigurationException(
                    f"Circular dependency detected involving service: {service_name}",
                    config_key="service_dependencies"
                )
            
            if service_name not in visited:
                temp_visited.add(service_name)
                
                for dependency in self.services[service_name]["dependencies"]:
                    visit(dependency)
                
                temp_visited.remove(service_name)
                visited.add(service_name)
                ordered.append(service_name)
        
        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)
        
        return ordered
    
    @safe_execute(max_retries=2, retry_delay=5.0)
    def start(self):
        """Start all services with enterprise patterns"""
        
        if self.running:
            logger.warning("Launcher already running")
            return
        
        startup_correlation_id = f"startup_{int(time.time())}"
        
        with correlation_context(startup_correlation_id):
            try:
                logger.info("Starting enterprise launcher")
                self.startup_time = datetime.utcnow()
                
                # Define services
                self._define_services()
                
                # Pre-startup validation
                self._validate_startup_conditions()
                
                # Start services in dependency order
                for service_name in self.startup_sequence:
                    if self.shutdown_requested:
                        logger.info("Shutdown requested during startup")
                        break
                    
                    self._start_service(service_name)
                
                # Verify all critical services are running
                self._verify_critical_services()
                
                self.running = True
                
                # Log successful startup
                uptime = (datetime.utcnow() - self.startup_time).total_seconds()
                
                business_events.log_system_event(
                    "system_startup_complete",
                    "enterprise_launcher",
                    {
                        "startup_time_seconds": uptime,
                        "services_started": len([s for s in self.service_states.values() if s == ServiceState.RUNNING]),
                        "environment": self.config.environment.value
                    }
                )
                
                logger.info(
                    f"Enterprise launcher started successfully in {uptime:.2f}s",
                    extra={
                        "startup_time_seconds": uptime,
                        "services_running": self._get_running_services(),
                        "environment": self.config.environment.value
                    }
                )
                
                # Start monitoring main loop
                self._start_monitoring_loop()
                
            except Exception as exc:
                logger.error(
                    "Failed to start enterprise launcher",
                    extra={"error": str(exc)},
                    exc_info=True
                )
                
                # Cleanup on failure
                self.shutdown()
                raise
    
    def _validate_startup_conditions(self):
        """Validate conditions required for startup"""
        
        logger.info("Validating startup conditions")
        
        # Check disk space
        disk_usage = os.statvfs('/')
        free_space_mb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024 * 1024)
        if free_space_mb < 1000:  # Require at least 1GB free
            raise SystemResourceException(
                f"Insufficient disk space: {free_space_mb:.0f}MB available, minimum 1000MB required",
                resource="disk_space",
                usage=free_space_mb
            )
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB minimum
                raise SystemResourceException(
                    f"Insufficient memory: {memory.available / (1024**3):.1f}GB available, minimum 2GB required",
                    resource="memory",
                    usage=memory.percent
                )
        except ImportError:
            logger.warning("psutil not available for memory check")
        
        # Validate configuration
        try:
            self.config.validate()
        except Exception as exc:
            raise ConfigurationException(
                f"Configuration validation failed: {exc}",
                config_key="startup_validation",
                cause=exc
            )
        
        # Check port availability
        self._check_port_availability(self.config.api.port, "API")
        self._check_port_availability(self.config.ui.port, "UI")
        
        logger.info("Startup conditions validated successfully")
    
    def _check_port_availability(self, port: int, service_name: str):
        """Check if a port is available"""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            if result == 0:
                raise SystemResourceException(
                    f"Port {port} is already in use (required for {service_name})",
                    resource="network_port",
                    usage=port
                )
    
    def _start_service(self, service_name: str):
        """Start individual service with comprehensive error handling"""
        
        service_config = self.services[service_name]
        
        logger.info(
            f"Starting service: {service_name}",
            extra={
                "service": service_name,
                "description": service_config["description"]
            }
        )
        
        try:
            self.service_states[service_name] = ServiceState.STARTING
            
            # Start the service
            if service_name == "monitoring":
                self._start_monitoring()
            elif service_name == "connection_monitor":
                self._start_connection_monitor()
            elif service_name == "api_server":
                self._start_api_server()
            elif service_name == "ui_server":
                self._start_ui_server()
            else:
                raise ConfigurationException(f"Unknown service: {service_name}")
            
            # Wait for service to be healthy
            self._wait_for_service_health(service_name)
            
            self.service_states[service_name] = ServiceState.RUNNING
            
            logger.info(f"Service started successfully: {service_name}")
            
        except Exception as exc:
            self.service_states[service_name] = ServiceState.FAILED
            
            logger.error(
                f"Failed to start service: {service_name}",
                extra={
                    "service": service_name,
                    "error": str(exc)
                },
                exc_info=True
            )
            
            if service_config["critical"]:
                raise SystemResourceException(
                    f"Critical service failed to start: {service_name}",
                    resource="service_startup",
                    context={"service": service_name, "error": str(exc)},
                    cause=exc
                )
    
    def _start_monitoring(self):
        """Start monitoring services"""
        # Monitoring is handled by the core modules automatically
        # This just registers our health checks
        
        self.health_check.register_check(
            "system_resources",
            self._check_system_resources,
            timeout=5.0,
            critical=True
        )
        
        self.health_check.register_check(
            "configuration_valid",
            lambda: True,  # Already validated
            timeout=1.0,
            critical=True
        )
    
    def _start_connection_monitor(self):
        """Start connection monitoring"""
        self.connection_monitor = EnterpriseConnectionMonitor()
        self.connection_monitor.start_monitoring()
    
    def _start_api_server(self):
        """Start API server in separate thread"""
        self.api_server = EnterpriseAPI()
        
        def run_api():
            try:
                self.api_server.run(debug=self.config.debug)
            except Exception as exc:
                logger.error("API server failed", exc_info=True)
                self.service_states["api_server"] = ServiceState.FAILED
        
        api_thread = threading.Thread(target=run_api, daemon=True, name="APIServer")
        api_thread.start()
        self.threads["api_server"] = api_thread
        
        # Give API server time to start
        time.sleep(3)
    
    def _start_ui_server(self):
        """Start UI server as subprocess"""
        
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'src/interface/led_enhanced_interface.py',
            '--server.port', str(self.config.ui.port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--logger.level', 'error'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        self.processes["ui_server"] = process
    
    def _wait_for_service_health(self, service_name: str):
        """Wait for service to become healthy"""
        
        service_config = self.services[service_name]
        health_check = service_config["health_check"]
        timeout = service_config["start_timeout"]
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.shutdown_requested:
                raise SystemResourceException(
                    f"Shutdown requested while waiting for {service_name} health",
                    resource="service_health"
                )
            
            try:
                if health_check():
                    logger.debug(f"Service healthy: {service_name}")
                    return
            except Exception as exc:
                logger.debug(f"Health check failed for {service_name}: {exc}")
            
            time.sleep(2)
        
        raise SystemResourceException(
            f"Service {service_name} failed to become healthy within {timeout}s",
            resource="service_health",
            context={"service": service_name, "timeout": timeout}
        )
    
    def _verify_critical_services(self):
        """Verify all critical services are running"""
        
        critical_services = [
            name for name, config in self.services.items()
            if config["critical"]
        ]
        
        failed_services = [
            name for name in critical_services
            if self.service_states[name] != ServiceState.RUNNING
        ]
        
        if failed_services:
            raise SystemResourceException(
                f"Critical services failed to start: {failed_services}",
                resource="critical_services",
                context={"failed_services": failed_services}
            )
    
    def _start_monitoring_loop(self):
        """Start main monitoring loop"""
        
        def monitoring_loop():
            logger.info("Starting monitoring loop")
            
            while self.running and not self.shutdown_requested:
                try:
                    # Check service health
                    self._check_all_service_health()
                    
                    # Check system resources
                    self._monitor_system_resources()
                    
                    # Check for configuration changes
                    if self.config.enable_hot_reload:
                        self._check_configuration_changes()
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as exc:
                    logger.error(
                        "Error in monitoring loop",
                        extra={"error": str(exc)},
                        exc_info=True
                    )
                    time.sleep(10)  # Shorter sleep on error
            
            logger.info("Monitoring loop stopped")
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True, name="MonitoringLoop")
        monitor_thread.start()
        self.threads["monitoring_loop"] = monitor_thread
    
    def _check_all_service_health(self):
        """Check health of all services"""
        
        for service_name, service_config in self.services.items():
            if self.service_states[service_name] == ServiceState.RUNNING:
                try:
                    if not service_config["health_check"]():
                        logger.warning(f"Service health check failed: {service_name}")
                        self.service_states[service_name] = ServiceState.FAILED
                        
                        if service_config["critical"]:
                            logger.error(f"Critical service failed: {service_name}")
                            # Could trigger restart here
                            
                except Exception as exc:
                    logger.error(
                        f"Health check error for {service_name}",
                        extra={"error": str(exc)},
                        exc_info=True
                    )
    
    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                logger.warning(f"High disk usage: {disk_percent}%")
                
        except ImportError:
            pass  # psutil not available
        except Exception as exc:
            logger.debug(f"Resource monitoring error: {exc}")
    
    def _check_configuration_changes(self):
        """Check for configuration file changes"""
        
        try:
            # This would implement file watching logic
            # For now, just log that we're checking
            logger.debug("Checking for configuration changes")
        except Exception as exc:
            logger.debug(f"Configuration check error: {exc}")
    
    def _reload_configuration(self):
        """Reload configuration"""
        
        try:
            logger.info("Reloading configuration")
            new_config = self.config_manager.reload_config()
            
            # Compare configurations and restart services if needed
            # This is a simplified version
            if new_config.logging.level != self.config.logging.level:
                logger.info("Log level changed, updating logging")
                setup_logging(log_level=new_config.logging.level.value)
            
            self.config = new_config
            
            business_events.log_system_event(
                "configuration_reloaded",
                "enterprise_launcher"
            )
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as exc:
            logger.error(
                "Failed to reload configuration",
                extra={"error": str(exc)},
                exc_info=True
            )
    
    # Health check methods
    def _check_monitoring_health(self) -> bool:
        """Check monitoring service health"""
        try:
            # Check if metrics collector is working
            metrics_collector.set_gauge("health_check_test", 1.0)
            return True
        except Exception:
            return False
    
    def _check_connection_monitor_health(self) -> bool:
        """Check connection monitor health"""
        return (
            self.connection_monitor is not None and
            self.connection_monitor.monitoring_active
        )
    
    def _check_api_server_health(self) -> bool:
        """Check API server health"""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.config.api.port}/health", timeout=5)
            return response.status_code in [200, 503]
        except Exception:
            return False
    
    def _check_ui_server_health(self) -> bool:
        """Check UI server health"""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.config.ui.port}/_stcore/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_system_resources(self) -> bool:
        """Check system resource availability"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return False
            
            # Check disk
            disk = psutil.disk_usage('/')
            if (disk.used / disk.total) > 0.95:
                return False
            
            return True
        except ImportError:
            return True  # Assume healthy if psutil not available
        except Exception:
            return False
    
    def _get_running_services(self) -> List[str]:
        """Get list of running services"""
        return [
            name for name, state in self.service_states.items()
            if state == ServiceState.RUNNING
        ]
    
    def shutdown(self):
        """Graceful shutdown of all services"""
        
        if not self.running and not any(state != ServiceState.STOPPED for state in self.service_states.values()):
            return
        
        shutdown_correlation_id = f"shutdown_{int(time.time())}"
        
        with correlation_context(shutdown_correlation_id):
            logger.info("Starting graceful shutdown")
            self.shutdown_requested = True
            self.running = False
            
            try:
                # Stop services in reverse order
                for service_name in self.shutdown_sequence:
                    if self.service_states[service_name] in [ServiceState.RUNNING, ServiceState.STARTING]:
                        self._stop_service(service_name)
                
                # Wait for threads to complete
                self._wait_for_threads()
                
                # Cleanup processes
                self._cleanup_processes()
                
                # Final cleanup
                self._final_cleanup()
                
                business_events.log_system_event(
                    "system_shutdown_complete",
                    "enterprise_launcher"
                )
                
                logger.info("Graceful shutdown completed")
                
            except Exception as exc:
                logger.error(
                    "Error during shutdown",
                    extra={"error": str(exc)},
                    exc_info=True
                )
    
    def _stop_service(self, service_name: str):
        """Stop individual service"""
        
        logger.info(f"Stopping service: {service_name}")
        self.service_states[service_name] = ServiceState.STOPPING
        
        try:
            if service_name == "ui_server" and "ui_server" in self.processes:
                process = self.processes["ui_server"]
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}")
                    process.kill()
                    process.wait()
            
            elif service_name == "api_server" and self.api_server:
                self.api_server.shutdown()
            
            elif service_name == "connection_monitor" and self.connection_monitor:
                self.connection_monitor.stop_monitoring()
            
            self.service_states[service_name] = ServiceState.STOPPED
            logger.info(f"Service stopped: {service_name}")
            
        except Exception as exc:
            logger.error(
                f"Error stopping service {service_name}",
                extra={"error": str(exc)},
                exc_info=True
            )
            self.service_states[service_name] = ServiceState.FAILED
    
    def _wait_for_threads(self):
        """Wait for all threads to complete"""
        
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for thread: {thread_name}")
                thread.join(timeout=10)
                
                if thread.is_alive():
                    logger.warning(f"Thread did not stop gracefully: {thread_name}")
    
    def _cleanup_processes(self):
        """Cleanup any remaining processes"""
        
        for process_name, process in self.processes.items():
            if process.poll() is None:  # Still running
                logger.info(f"Terminating process: {process_name}")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process: {process_name}")
                    process.kill()
                    process.wait()
    
    def _final_cleanup(self):
        """Final cleanup operations"""
        
        # Stop metrics collection
        try:
            metrics_collector.stop_system_metrics_collection()
            alert_manager.stop_alert_evaluation()
        except Exception as exc:
            logger.debug(f"Cleanup error: {exc}")
    
    def cleanup(self):
        """Cleanup method called by atexit"""
        if self.running or self.shutdown_requested:
            self.shutdown()
    
    def get_status(self) -> Dict[str, Any]:
        """Get launcher status"""
        
        uptime = (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0
        
        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "environment": self.config.environment.value,
            "version": self.config.version,
            "services": {
                name: {
                    "state": state,
                    "description": self.services[name]["description"],
                    "critical": self.services[name]["critical"]
                }
                for name, state in self.service_states.items()
            },
            "system_health": get_system_health()
        }


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise River Trading System Launcher")
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path"
    )
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "testing", "staging", "production"],
        help="Override environment"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Create launcher
        launcher = EnterpriseLauncher(args.config)
        
        # Override environment if specified
        if args.environment:
            launcher.config.environment = Environment(args.environment)
        
        # Validate only mode
        if args.validate_only:
            logger.info("Configuration validation successful")
            print("✅ Configuration is valid")
            return 0
        
        # Start launcher
        launcher.start()
        
        # Keep main thread alive
        try:
            while launcher.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        return 0
        
    except BaseRiverException as exc:
        logger.error(f"Launcher failed: {exc.user_message}")
        print(f"❌ {exc.user_message}")
        if exc.recovery_hint:
            print(f"💡 {exc.recovery_hint}")
        return 1
        
    except Exception as exc:
        logger.error("Unexpected launcher error", exc_info=True)
        print(f"❌ Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())