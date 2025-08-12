"""
Streamlit WebSocket Client Integration
=====================================

Provides WebSocket integration for Streamlit frontend.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import streamlit as st

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class StreamlitWebSocketClient:
    """
    WebSocket client integration for Streamlit.
    
    Provides real-time data updates in Streamlit applications
    using session state and periodic refresh.
    """
    
    def __init__(self, websocket_url: str = "ws://localhost:8001/quotes"):
        self.websocket_url = websocket_url
        self.client_id = f"streamlit_{int(time.time())}"
        
        # Initialize session state
        if "websocket_data" not in st.session_state:
            st.session_state.websocket_data = {}
        
        if "websocket_subscriptions" not in st.session_state:
            st.session_state.websocket_subscriptions = set()
        
        if "websocket_connected" not in st.session_state:
            st.session_state.websocket_connected = False
        
        if "websocket_last_update" not in st.session_state:
            st.session_state.websocket_last_update = {}
    
    def subscribe_to_symbol(self, symbol: str):
        """
        Subscribe to real-time updates for a symbol.
        
        Args:
            symbol: Stock symbol to subscribe to
        """
        symbol_upper = symbol.upper()
        st.session_state.websocket_subscriptions.add(symbol_upper)
        
        # Store subscription preference
        if "websocket_subscriptions_persistent" not in st.session_state:
            st.session_state.websocket_subscriptions_persistent = set()
        st.session_state.websocket_subscriptions_persistent.add(symbol_upper)
    
    def unsubscribe_from_symbol(self, symbol: str):
        """
        Unsubscribe from real-time updates for a symbol.
        
        Args:
            symbol: Stock symbol to unsubscribe from
        """
        symbol_upper = symbol.upper()
        st.session_state.websocket_subscriptions.discard(symbol_upper)
        
        # Remove from persistent subscriptions
        if "websocket_subscriptions_persistent" in st.session_state:
            st.session_state.websocket_subscriptions_persistent.discard(symbol_upper)
        
        # Clear cached data
        if symbol_upper in st.session_state.websocket_data:
            del st.session_state.websocket_data[symbol_upper]
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest quote data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest quote data or None if not available
        """
        symbol_upper = symbol.upper()
        return st.session_state.websocket_data.get(symbol_upper)
    
    def is_data_fresh(self, symbol: str, max_age_seconds: int = 30) -> bool:
        """
        Check if quote data is fresh.
        
        Args:
            symbol: Stock symbol
            max_age_seconds: Maximum age in seconds
            
        Returns:
            True if data is fresh
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in st.session_state.websocket_last_update:
            return False
        
        last_update = st.session_state.websocket_last_update[symbol_upper]
        age = time.time() - last_update
        return age <= max_age_seconds
    
    def display_realtime_quote_card(self, symbol: str):
        """
        Display a real-time quote card in Streamlit.
        
        Args:
            symbol: Stock symbol to display
        """
        symbol_upper = symbol.upper()
        quote_data = self.get_latest_quote(symbol_upper)
        
        if quote_data:
            price = quote_data.get("price", 0)
            change = quote_data.get("change", 0)
            change_percent = quote_data.get("change_percent", 0)
            volume = quote_data.get("volume", 0)
            timestamp = quote_data.get("timestamp", "")
            market_status = quote_data.get("market_status", "unknown")
            
            # Color coding for price changes
            if change > 0:
                color = "green"
                change_text = f"+${change:.2f} (+{change_percent:.2f}%)"
            elif change < 0:
                color = "red" 
                change_text = f"-${abs(change):.2f} ({change_percent:.2f}%)"
            else:
                color = "gray"
                change_text = "No Change"
            
            # Market status indicator
            status_color = {
                "open": "green",
                "pre_market": "orange", 
                "after_hours": "orange",
                "closed": "red"
            }.get(market_status, "gray")
            
            # Create card with real-time data
            st.markdown(f"""
            <div style="
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background: rgba(26, 28, 33, 0.95);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0; color: white;">{symbol_upper}</h3>
                        <div style="color: {status_color}; font-size: 12px;">● {market_status.replace('_', ' ').title()}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 24px; font-weight: bold; color: white;">${price:.2f}</div>
                        <div style="color: {color};">{change_text}</div>
                    </div>
                </div>
                <div style="margin-top: 12px; display: flex; justify-content: space-between; font-size: 14px; color: rgba(255, 255, 255, 0.7);">
                    <span>Volume: {volume:,}</span>
                    <span>Updated: {datetime.fromisoformat(timestamp.replace('Z', '')).strftime('%H:%M:%S') if timestamp else 'N/A'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Freshness indicator
            if self.is_data_fresh(symbol_upper):
                st.success("🟢 Live data")
            else:
                st.warning("🟡 Data may be stale")
                
        else:
            # No data available
            st.markdown(f"""
            <div style="
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background: rgba(26, 28, 33, 0.95);
                text-align: center;
            ">
                <h3 style="margin: 0; color: white;">{symbol_upper}</h3>
                <p style="color: rgba(255, 255, 255, 0.7);">Waiting for real-time data...</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_connection_status(self):
        """Display WebSocket connection status"""
        if st.session_state.websocket_connected:
            st.success("🔗 Connected to real-time data stream")
        else:
            st.error("❌ Real-time data stream disconnected")
        
        # Show subscribed symbols
        if st.session_state.websocket_subscriptions:
            symbols_list = ", ".join(sorted(st.session_state.websocket_subscriptions))
            st.info(f"📊 Subscribed to: {symbols_list}")
    
    def create_subscription_controls(self, available_symbols: list):
        """
        Create UI controls for managing subscriptions.
        
        Args:
            available_symbols: List of available symbols to subscribe to
        """
        st.subheader("Real-time Data Subscriptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add subscription
            new_symbol = st.selectbox(
                "Add Symbol",
                options=[""] + available_symbols,
                key="ws_add_symbol"
            )
            
            if st.button("Subscribe") and new_symbol:
                self.subscribe_to_symbol(new_symbol)
                st.success(f"Subscribed to {new_symbol}")
                st.rerun()
        
        with col2:
            # Remove subscription
            if st.session_state.websocket_subscriptions:
                remove_symbol = st.selectbox(
                    "Remove Symbol",
                    options=[""] + list(sorted(st.session_state.websocket_subscriptions)),
                    key="ws_remove_symbol"
                )
                
                if st.button("Unsubscribe") and remove_symbol:
                    self.unsubscribe_from_symbol(remove_symbol)
                    st.success(f"Unsubscribed from {remove_symbol}")
                    st.rerun()
        
        # Display current subscriptions
        if st.session_state.websocket_subscriptions:
            st.write("**Current Subscriptions:**")
            for symbol in sorted(st.session_state.websocket_subscriptions):
                self.display_realtime_quote_card(symbol)
    
    def simulate_websocket_update(self, symbol: str, quote_data: Dict[str, Any]):
        """
        Simulate a WebSocket update (for testing without actual WebSocket connection).
        
        Args:
            symbol: Stock symbol
            quote_data: Quote data to simulate
        """
        symbol_upper = symbol.upper()
        st.session_state.websocket_data[symbol_upper] = quote_data
        st.session_state.websocket_last_update[symbol_upper] = time.time()
        st.session_state.websocket_connected = True
    
    def auto_refresh_component(self, refresh_interval: int = 5):
        """
        Auto-refresh component for Streamlit.
        
        Args:
            refresh_interval: Refresh interval in seconds
        """
        # This would typically use st_autorefresh or similar component
        # For now, we'll use a placeholder that suggests manual refresh
        if st.button("🔄 Refresh Real-time Data"):
            st.rerun()
        
        # Show last refresh time
        if "last_manual_refresh" not in st.session_state:
            st.session_state.last_manual_refresh = time.time()
        
        if st.button("Auto-refresh: ON"):
            st.session_state.last_manual_refresh = time.time()
            st.rerun()


def create_websocket_javascript() -> str:
    """
    Generate JavaScript code for WebSocket connection in Streamlit.
    
    Returns:
        JavaScript code as string
    """
    return """
    <script>
    let ws = null;
    let clientId = 'streamlit_' + Date.now();
    let subscriptions = new Set();
    
    function connectWebSocket() {
        const wsUrl = 'ws://localhost:8001/quotes/' + clientId;
        ws = new WebSocket(wsUrl);
        
        ws.onopen = function(event) {
            console.log('WebSocket connected');
            // Re-subscribe to symbols
            subscriptions.forEach(symbol => {
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    data: { symbol: symbol }
                }));
            });
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'quote') {
                // Update Streamlit session state via custom component or callback
                console.log('Quote update:', message.data);
                
                // Store in sessionStorage for Streamlit to read
                sessionStorage.setItem(
                    'ws_quote_' + message.data.symbol, 
                    JSON.stringify(message.data)
                );
                
                // Trigger Streamlit update
                const event = new CustomEvent('websocket_quote_update', {
                    detail: message.data
                });
                document.dispatchEvent(event);
            }
        };
        
        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            // Attempt reconnection after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
    }
    
    function subscribeToSymbol(symbol) {
        subscriptions.add(symbol);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'subscribe',
                data: { symbol: symbol }
            }));
        }
    }
    
    function unsubscribeFromSymbol(symbol) {
        subscriptions.delete(symbol);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'unsubscribe',
                data: { symbol: symbol }
            }));
        }
    }
    
    // Initialize connection
    connectWebSocket();
    
    // Make functions globally available
    window.wsClient = {
        subscribe: subscribeToSymbol,
        unsubscribe: unsubscribeFromSymbol
    };
    </script>
    """