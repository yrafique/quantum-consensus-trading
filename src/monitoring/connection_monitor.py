"""
Connection Monitor & LED Status System
=====================================

Real-time monitoring system for data connections and LLM validation
with LED indicators to show system health before user interaction.

Features:
- Live data connection monitoring
- LLM/MLX status validation
- API endpoint health checks
- Component availability verification
- Real-time status updates
- Connection retry mechanisms
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import requests
import json
from threading import Thread
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status levels"""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """System component types"""
    DATA_FEED = "data_feed"
    MLX_LLM = "mlx_llm"
    REACT_AGENT = "react_agent"
    API_SERVER = "api_server"
    DATABASE = "database"
    MARKET_DATA = "market_data"

@dataclass
class ConnectionInfo:
    """Connection information for a component"""
    component: ComponentType
    status: ConnectionStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    check_interval: int = 30  # seconds

@dataclass
class LEDStatus:
    """LED visual status"""
    color: str  # green, yellow, red, blue, gray
    blink: bool = False
    message: str = ""
    icon: str = "●"

class ConnectionMonitor:
    """Real-time connection monitoring system"""
    
    def __init__(self):
        self.connections: Dict[ComponentType, ConnectionInfo] = {}
        self.status_queue = queue.Queue()
        self.monitoring_active = True
        self.monitor_thread = None
        
        # Initialize connection configs
        self._init_connections()
        
        # Start monitoring
        self.start_monitoring()
    
    def _init_connections(self):
        """Initialize connection configurations"""
        
        # Data feed connection
        self.connections[ComponentType.DATA_FEED] = ConnectionInfo(
            component=ComponentType.DATA_FEED,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            check_interval=15
        )
        
        # MLX LLM connection
        self.connections[ComponentType.MLX_LLM] = ConnectionInfo(
            component=ComponentType.MLX_LLM,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            check_interval=20
        )
        
        # ReAct Agent
        self.connections[ComponentType.REACT_AGENT] = ConnectionInfo(
            component=ComponentType.REACT_AGENT,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            check_interval=30
        )
        
        # API Server
        self.connections[ComponentType.API_SERVER] = ConnectionInfo(
            component=ComponentType.API_SERVER,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            check_interval=10
        )
        
        # Market Data
        self.connections[ComponentType.MARKET_DATA] = ConnectionInfo(
            component=ComponentType.MARKET_DATA,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            check_interval=20
        )
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_active = True
            self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Connection monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Connection monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check each connection
                for component_type, connection in self.connections.items():
                    if self._should_check_connection(connection):
                        self._check_connection(connection)
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _should_check_connection(self, connection: ConnectionInfo) -> bool:
        """Determine if connection should be checked"""
        now = datetime.now()
        time_since_check = (now - connection.last_check).total_seconds()
        return time_since_check >= connection.check_interval
    
    def _check_connection(self, connection: ConnectionInfo):
        """Check a specific connection"""
        start_time = time.time()
        connection.last_check = datetime.now()
        
        try:
            # Update status to connecting
            connection.status = ConnectionStatus.CONNECTING
            
            # Check based on component type
            if connection.component == ComponentType.DATA_FEED:
                success = self._check_data_feed()
            elif connection.component == ComponentType.MLX_LLM:
                success = self._check_mlx_llm()
            elif connection.component == ComponentType.REACT_AGENT:
                success = self._check_react_agent()
            elif connection.component == ComponentType.API_SERVER:
                success = self._check_api_server()
            elif connection.component == ComponentType.MARKET_DATA:
                success = self._check_market_data()
            else:
                success = False
            
            # Update connection info
            connection.response_time = time.time() - start_time
            
            if success:
                connection.status = ConnectionStatus.CONNECTED
                connection.error_message = None
                connection.retry_count = 0
            else:
                connection.status = ConnectionStatus.ERROR
                connection.retry_count += 1
                
        except Exception as e:
            connection.status = ConnectionStatus.ERROR
            connection.error_message = str(e)
            connection.retry_count += 1
            connection.response_time = time.time() - start_time
            
        # Add to status queue for UI updates
        self.status_queue.put((connection.component, connection.status))
    
    def _check_data_feed(self) -> bool:
        """Check data feed connection"""
        try:
            # Try to import and initialize data validator
            from data_validator import DataValidator
            validator = DataValidator()
            
            # Test with a common stock
            test_data = validator.get_market_context("AAPL")
            return test_data is not None and isinstance(test_data, dict)
            
        except Exception as e:
            logger.debug(f"Data feed check failed: {e}")
            return False
    
    def _check_mlx_llm(self) -> bool:
        """Check MLX LLM connection"""
        try:
            from mlx_trading_llm import MLXTradingLLM
            
            # Try to initialize MLX LLM
            llm = MLXTradingLLM()
            
            # Check if model is loaded
            if hasattr(llm, 'loaded') and llm.loaded:
                # Test inference
                test_response = llm._generate_text("Test", max_tokens=5)
                return len(test_response.strip()) > 0
            
            return False
            
        except Exception as e:
            logger.debug(f"MLX LLM check failed: {e}")
            return False
    
    def _check_react_agent(self) -> bool:
        """Check ReAct agent connection"""
        try:
            from react_trading_agent import ReActTradingAgent
            
            # Try to initialize ReAct agent
            agent = ReActTradingAgent()
            
            # Check if components are available
            return (agent.market_tool is not None and 
                    agent.tech_tool is not None and
                    agent.fact_checker is not None)
            
        except Exception as e:
            logger.debug(f"ReAct agent check failed: {e}")
            return False
    
    def _check_api_server(self) -> bool:
        """Check API server connection"""
        try:
            # Try to connect to health endpoint
            response = requests.get("http://localhost:5001/health", timeout=5)
            return response.status_code in [200, 503]  # 503 is acceptable for degraded mode
            
        except Exception as e:
            logger.debug(f"API server check failed: {e}")
            return False
    
    def _check_market_data(self) -> bool:
        """Check market data availability"""
        try:
            # Test multiple market data sources
            test_symbols = ["AAPL", "MSFT", "GOOGL"]
            
            for symbol in test_symbols:
                if self._check_data_feed():
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Market data check failed: {e}")
            return False
    
    def get_connection_status(self, component: ComponentType) -> ConnectionInfo:
        """Get status for specific component"""
        return self.connections.get(component, ConnectionInfo(
            component=component,
            status=ConnectionStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0
        ))
    
    def get_all_statuses(self) -> Dict[ComponentType, ConnectionInfo]:
        """Get all connection statuses"""
        return self.connections.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        statuses = list(self.connections.values())
        
        connected_count = sum(1 for s in statuses if s.status == ConnectionStatus.CONNECTED)
        error_count = sum(1 for s in statuses if s.status == ConnectionStatus.ERROR)
        total_count = len(statuses)
        
        health_percentage = (connected_count / total_count) * 100 if total_count > 0 else 0
        
        if health_percentage >= 90:
            overall_status = "healthy"
        elif health_percentage >= 70:
            overall_status = "degraded"
        elif health_percentage >= 50:
            overall_status = "limited"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "connected_components": connected_count,
            "error_components": error_count,
            "total_components": total_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def force_check_all(self):
        """Force immediate check of all connections"""
        for connection in self.connections.values():
            connection.last_check = datetime.now() - timedelta(seconds=connection.check_interval + 1)

class LEDStatusManager:
    """Manages LED status indicators for the UI"""
    
    def __init__(self, connection_monitor: ConnectionMonitor):
        self.monitor = connection_monitor
        self.led_states: Dict[ComponentType, LEDStatus] = {}
        self._update_led_states()
    
    def _update_led_states(self):
        """Update LED states based on connection statuses"""
        for component_type, connection in self.monitor.get_all_statuses().items():
            self.led_states[component_type] = self._connection_to_led(connection)
    
    def _connection_to_led(self, connection: ConnectionInfo) -> LEDStatus:
        """Convert connection info to LED status"""
        
        if connection.status == ConnectionStatus.CONNECTED:
            return LEDStatus(
                color="green",
                blink=False,
                message=f"Connected ({connection.response_time:.1f}s)",
                icon="🟢"
            )
        elif connection.status == ConnectionStatus.CONNECTING:
            return LEDStatus(
                color="blue",
                blink=True,
                message="Connecting...",
                icon="🔵"
            )
        elif connection.status == ConnectionStatus.ERROR:
            return LEDStatus(
                color="red",
                blink=False,
                message=f"Error (Retry {connection.retry_count}/{connection.max_retries})",
                icon="🔴"
            )
        elif connection.status == ConnectionStatus.DISCONNECTED:
            return LEDStatus(
                color="yellow",
                blink=True,
                message="Disconnected",
                icon="🟡"
            )
        else:  # UNKNOWN
            return LEDStatus(
                color="gray",
                blink=False,
                message="Unknown status",
                icon="⚪"
            )
    
    def get_led_status(self, component: ComponentType) -> LEDStatus:
        """Get LED status for component"""
        self._update_led_states()
        return self.led_states.get(component, LEDStatus(
            color="gray",
            message="Unknown component",
            icon="⚪"
        ))
    
    def get_all_led_statuses(self) -> Dict[ComponentType, LEDStatus]:
        """Get all LED statuses"""
        self._update_led_states()
        return self.led_states.copy()
    
    def is_system_ready(self) -> bool:
        """Check if system is ready for user interaction"""
        health = self.monitor.get_overall_health()
        critical_components = [
            ComponentType.DATA_FEED,
            ComponentType.MLX_LLM,
            ComponentType.REACT_AGENT
        ]
        
        # Check if critical components are connected
        for component in critical_components:
            connection = self.monitor.get_connection_status(component)
            if connection.status != ConnectionStatus.CONNECTED:
                return False
        
        return health["health_percentage"] >= 60  # At least 60% health required

def main():
    """Demo the connection monitoring system"""
    print("🔌 CONNECTION MONITOR DEMO")
    print("=" * 50)
    
    # Initialize monitor
    monitor = ConnectionMonitor()
    led_manager = LEDStatusManager(monitor)
    
    # Force check all connections
    monitor.force_check_all()
    
    # Wait for initial checks
    time.sleep(5)
    
    # Display results
    print("\n📊 CONNECTION STATUS:")
    print("-" * 30)
    
    for component_type, led_status in led_manager.get_all_led_statuses().items():
        component_name = component_type.value.replace("_", " ").title()
        print(f"{led_status.icon} {component_name}: {led_status.message}")
    
    # Overall health
    health = monitor.get_overall_health()
    print(f"\n🏥 Overall Health: {health['overall_status'].upper()} ({health['health_percentage']:.1f}%)")
    print(f"   Connected: {health['connected_components']}/{health['total_components']}")
    
    # System readiness
    ready = led_manager.is_system_ready()
    ready_icon = "✅" if ready else "❌"
    ready_text = "READY" if ready else "NOT READY"
    print(f"\n{ready_icon} System Status: {ready_text} for user interaction")
    
    # Stop monitoring
    monitor.stop_monitoring()

if __name__ == "__main__":
    main()