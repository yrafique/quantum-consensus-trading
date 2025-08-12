"""
LED-Enhanced River Interface
===========================

Enhanced River Trading Interface with comprehensive LED status system
showing real-time data connection and LLM validation status before
allowing user interaction.

Features:
- Real-time LED status indicators
- Connection monitoring dashboard
- System readiness validation
- Initialization guards
- Auto-retry mechanisms
- Diagnostic information
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
import json

# Import connection monitoring
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.monitoring.connection_monitor import ConnectionMonitor, LEDStatusManager, ComponentType, ConnectionStatus
    MONITOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Connection monitor not available: {e}")
    MONITOR_AVAILABLE = False

# Enhanced interface not needed anymore - using built-in interface
ENHANCED_INTERFACE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LEDEnhancedInterface:
    """Enhanced interface with LED status monitoring"""
    
    def __init__(self):
        # Initialize connection monitor
        if MONITOR_AVAILABLE:
            if 'connection_monitor' not in st.session_state:
                st.session_state.connection_monitor = ConnectionMonitor()
                st.session_state.led_manager = LEDStatusManager(st.session_state.connection_monitor)
            
            self.monitor = st.session_state.connection_monitor
            self.led_manager = st.session_state.led_manager
        else:
            self.monitor = None
            self.led_manager = None
        
        # Initialize main interface
        if ENHANCED_INTERFACE_AVAILABLE:
            self.main_interface = EnhancedRiverInterface()
        else:
            self.main_interface = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        
        if 'last_health_check' not in st.session_state:
            st.session_state.last_health_check = datetime.now()
        
        if 'show_diagnostics' not in st.session_state:
            st.session_state.show_diagnostics = False
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def render(self):
        """Main render method with LED status validation"""
        try:
            self._setup_page()
            
            # Always show LED status at the top
            self._render_led_status_bar()
            
            # Check system readiness
            if self._is_system_ready():
                st.session_state.system_ready = True
                self._render_main_interface()
            else:
                st.session_state.system_ready = False
                self._render_initialization_screen()
                
        except Exception as e:
            logger.error(f"Render error: {e}")
            st.error("Application error occurred. Please refresh the page.")
    
    def _setup_page(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="River Flow Trading - System Monitor",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Minimal CSS for any remaining styling needs
        minimal_css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        
        st.markdown(minimal_css, unsafe_allow_html=True)
    
    def _render_led_status_bar(self):
        """Render the LED status bar at the top"""
        if not self.led_manager:
            st.warning("⚠️ System monitoring not available")
            return
        
        # Get current statuses
        led_statuses = self.led_manager.get_all_led_statuses()
        health = self.monitor.get_overall_health()
        
        # System status header with container
        status_container = st.container()
        with status_container:
            system_ready = self._is_system_ready()
            status_icon = "✅" if system_ready else "⚠️"
            status_text = "SYSTEM READY" if system_ready else "INITIALIZING"
            
            if system_ready:
                st.success(f"{status_icon} **{status_text}**")
            else:
                st.warning(f"{status_icon} **{status_text}**")
            
            # Health progress bar
            health_percentage = health["health_percentage"]
            st.progress(health_percentage / 100)
            st.caption(f"System Health: {health_percentage:.0f}% ({health['connected_components']}/{health['total_components']} components)")
        
        # Component status grid
        st.markdown("**📊 Component Status:**")
        
        # Create columns for component status display
        num_components = len(led_statuses)
        cols = st.columns(min(num_components, 5))  # Max 5 columns
        
        for idx, (component_type, led_status) in enumerate(led_statuses.items()):
            with cols[idx % len(cols)]:
                component_name = component_type.value.replace("_", " ").title()
                
                # Choose status display based on color
                if led_status.color == "connected":
                    st.success(f"{led_status.icon} {component_name}")
                    st.caption(led_status.message)
                elif led_status.color == "connecting":
                    st.info(f"{led_status.icon} {component_name}")
                    st.caption(led_status.message)
                elif led_status.color == "error":
                    st.error(f"{led_status.icon} {component_name}")
                    st.caption(led_status.message)
                elif led_status.color == "disconnected":
                    st.warning(f"{led_status.icon} {component_name}")
                    st.caption(led_status.message)
                else:
                    st.info(f"{led_status.icon} {component_name}")
                    st.caption(led_status.message)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Status", key="refresh_status"):
                self.monitor.force_check_all()
                st.rerun()
        
        with col2:
            if st.button("🔧 Show Diagnostics", key="toggle_diagnostics"):
                st.session_state.show_diagnostics = not st.session_state.show_diagnostics
                st.rerun()
        
        with col3:
            auto_refresh = st.checkbox("Auto Refresh", 
                                     value=st.session_state.auto_refresh,
                                     key="auto_refresh_checkbox")
            st.session_state.auto_refresh = auto_refresh
        
        with col4:
            if st.button("⚡ Force Retry All", key="force_retry"):
                self._force_retry_all_connections()
                st.rerun()
        
        # Show diagnostics if requested
        if st.session_state.show_diagnostics:
            self._render_diagnostic_panel()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(2)
            st.rerun()
    
    def _render_diagnostic_panel(self):
        """Render detailed diagnostic information"""
        st.markdown("### 🔧 System Diagnostics")
        
        if not self.monitor:
            st.error("Diagnostic information not available")
            return
        
        # Get detailed connection info
        connections = self.monitor.get_all_statuses()
        
        # Create diagnostic cards for each component
        for component_type, connection in connections.items():
            component_name = component_type.value.replace("_", " ").title()
            
            with st.expander(f"{component_name} Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Status indicator
                    if connection.status == ConnectionStatus.CONNECTED:
                        st.success(f"✅ Status: {connection.status.value.title()}")
                    elif connection.status == ConnectionStatus.CONNECTING:
                        st.info(f"🔄 Status: {connection.status.value.title()}")
                    elif connection.status == ConnectionStatus.ERROR:
                        st.error(f"❌ Status: {connection.status.value.title()}")
                    else:
                        st.warning(f"⚠️ Status: {connection.status.value.title()}")
                    
                    st.metric("Response Time", f"{connection.response_time:.3f}s")
                    
                with col2:
                    st.caption(f"Last Check: {connection.last_check.strftime('%H:%M:%S')}")
                    st.caption(f"Retry: {connection.retry_count}/{connection.max_retries}")
                    
                if connection.error_message:
                    st.error(f"Error: {connection.error_message}")
                
                st.divider()
        
        # Export diagnostics
        if st.button("💾 Export Diagnostics", key="export_diagnostics"):
            diagnostic_data = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": self.monitor.get_overall_health(),
                "component_details": {
                    comp.value: {
                        "status": conn.status.value,
                        "response_time": conn.response_time,
                        "error_message": conn.error_message,
                        "retry_count": conn.retry_count,
                        "last_check": conn.last_check.isoformat()
                    }
                    for comp, conn in connections.items()
                }
            }
            
            st.download_button(
                label="Download Diagnostic Report",
                data=json.dumps(diagnostic_data, indent=2),
                file_name=f"system_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _render_initialization_screen(self):
        """Render initialization screen when system is not ready"""
        
        # Create a nice header
        st.markdown("# 🌊 River Flow Trading System")
        st.markdown("### 🔧 System Initialization in Progress")
        st.info("Please wait while we verify all system components...")
        
        st.divider()
        
        # Show what's being checked
        if self.monitor:
            connections = self.monitor.get_all_statuses()
            
            st.markdown("### 📋 Component Status:")
            
            for component_type, connection in connections.items():
                component_name = component_type.value.replace("_", " ").title()
                
                if connection.status == ConnectionStatus.CONNECTED:
                    st.success(f"✅ {component_name}: Ready")
                elif connection.status == ConnectionStatus.CONNECTING:
                    st.info(f"🔄 {component_name}: Connecting...")
                elif connection.status == ConnectionStatus.ERROR:
                    st.error(f"❌ {component_name}: Error - {connection.error_message or 'Connection failed'}")
                else:
                    st.warning(f"⚠️ {component_name}: {connection.status.value.title()}")
        
        # Initialization progress
        progress_container = st.container()
        
        with progress_container:
            if self.monitor:
                health = self.monitor.get_overall_health()
                progress_value = health["health_percentage"] / 100
                
                st.progress(progress_value)
                st.markdown(f"**Initialization Progress: {health['health_percentage']:.0f}%**")
                
                if health["health_percentage"] < 60:
                    st.warning("⚠️ Some components are not responding. The system will retry automatically.")
                    
                    # Show retry button
                    if st.button("🔄 Retry Failed Components", key="retry_failed"):
                        self._retry_failed_connections()
                        st.rerun()
            else:
                st.error("❌ System monitor not available. Please check installation.")
        
        # Requirements information
        with st.expander("📋 System Requirements"):
            st.markdown("""
            **Required Components:**
            - 🔗 Data Feed Connection (Market data access)
            - 🧠 MLX LLM (Local AI processing)
            - 🤖 ReAct Agent (Advanced reasoning)
            
            **Optional Components:**
            - 🌐 API Server (REST endpoints)
            - 💾 Database (Historical data)
            
            **Minimum Health:** 60% (Critical components must be online)
            """)
        
        # Auto-retry countdown
        if st.session_state.auto_refresh:
            countdown_placeholder = st.empty()
            for i in range(5, 0, -1):
                countdown_placeholder.info(f"🔄 Auto-retry in {i} seconds...")
                time.sleep(1)
            countdown_placeholder.empty()
            st.rerun()
    
    def _render_main_interface(self):
        """Render main interface when system is ready"""
        st.markdown("---")  # Clear separator
        st.success("✅ System Ready - All critical components online")
        
        if self.main_interface:
            st.info("🔧 Loading enhanced ReAct trading interface...")
            # Render the enhanced interface
            try:
                # Remove the setup_page call to avoid duplicate page config
                if hasattr(self.main_interface, '_render_react_status'):
                    self.main_interface._render_react_status()
                
                if hasattr(self.main_interface, '_render_mode_selector'):
                    self.main_interface._render_mode_selector()
                
                if hasattr(self.main_interface, '_render_current_flow'):
                    self.main_interface._render_current_flow()
                
            except Exception as e:
                st.error(f"Enhanced interface error: {e}")
                logger.error(f"Enhanced interface error: {e}")
                # Fall back to simple interface
                self._render_simple_interface()
        else:
            st.warning("Enhanced ReAct interface not available. Using simplified trading mode.")
            self._render_simple_interface()
    
    def _render_simple_interface(self):
        """Render simple fallback trading interface"""
        st.markdown("### 🌊 River Trading System - Basic Mode")
        
        # Basic trading interface
        tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🎯 Opportunities", "🧠 ReAct Agent"])
        
        with tab1:
            st.subheader("Single Stock Analysis")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                ticker = st.text_input("Enter stock symbol (e.g., AAPL, TSLA):", key="analysis_ticker")
            
            with col2:
                analyze_btn = st.button("🔍 Analyze", key="analyze_stock", type="primary")
            
            if analyze_btn and ticker:
                with st.spinner(f"Analyzing {ticker.upper()}..."):
                    st.success(f"✅ Analysis for {ticker.upper()} would be performed here")
                    st.info("💡 This would show ReAct reasoning, market data, and recommendations")
        
        with tab2:
            st.subheader("Market Opportunity Hunter")
            
            sector = st.selectbox("Select sector:", 
                                ["All Sectors", "Technology", "Healthcare", "Finance", "Energy"])
            
            if st.button("🎯 Find Opportunities", key="find_opps", type="primary"):
                with st.spinner("Hunting for opportunities..."):
                    st.success("✅ Opportunity hunting would be performed here")
                    st.info("💡 This would show cross-validated opportunities with confidence scores")
        
        with tab3:
            st.subheader("ReAct Agent Direct Chat")
            st.info("💬 Direct chat with the ReAct reasoning agent")
            
            question = st.text_area("Ask the ReAct agent anything about trading:", 
                                   placeholder="What are the key technical indicators for AAPL?")
            
            if st.button("🧠 Ask ReAct Agent", key="ask_react", type="primary"):
                if question:
                    with st.spinner("ReAct agent is thinking..."):
                        st.success("✅ ReAct agent response would appear here")
                        st.info("💡 This would show step-by-step reasoning (Observation → Thought → Action → Reflection)")
                else:
                    st.warning("Please enter a question first")
    
    def _is_system_ready(self) -> bool:
        """Check if system is ready for user interaction"""
        if not self.led_manager:
            return False
        
        return self.led_manager.is_system_ready()
    
    def _force_retry_all_connections(self):
        """Force retry of all connections"""
        if self.monitor:
            self.monitor.force_check_all()
            st.info("🔄 Forcing retry of all connections...")
    
    def _retry_failed_connections(self):
        """Retry only failed connections"""
        if self.monitor:
            connections = self.monitor.get_all_statuses()
            failed_count = 0
            
            for component_type, connection in connections.items():
                if connection.status in [ConnectionStatus.ERROR, ConnectionStatus.DISCONNECTED]:
                    # Reset retry count and force check
                    connection.retry_count = 0
                    connection.last_check = datetime.now() - timedelta(seconds=connection.check_interval + 1)
                    failed_count += 1
            
            if failed_count > 0:
                st.info(f"🔄 Retrying {failed_count} failed connections...")
            else:
                st.success("✅ No failed connections to retry")

def main():
    """Main application entry point"""
    try:
        interface = LEDEnhancedInterface()
        interface.render()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        st.error("Critical application error. Please refresh the page.")
        
        # Show error details in debug mode
        if st.button("Show Error Details"):
            st.exception(e)

if __name__ == "__main__":
    main()