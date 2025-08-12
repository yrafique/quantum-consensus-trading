"""
Market Data Broadcaster
=======================

Coordinates real-time market data broadcasting across the application.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.logging_config import get_logger
from .websocket_manager import websocket_manager
from .quote_streamer import quote_streamer

logger = get_logger(__name__)


class MarketDataBroadcaster:
    """
    Central coordinator for real-time market data broadcasting.
    
    Features:
    - Manages WebSocket connections and quote streaming
    - Handles alerts and notifications
    - Coordinates market events
    - Provides unified interface for real-time data
    """
    
    def __init__(self):
        self._started = False
    
    async def start(self):
        """Start the market data broadcasting service"""
        if self._started:
            return
        
        try:
            # Start WebSocket manager
            await websocket_manager.start()
            
            # Start quote streamer
            await quote_streamer.start()
            
            self._started = True
            logger.info("Market data broadcaster started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start market data broadcaster: {e}")
            raise
    
    async def stop(self):
        """Stop the market data broadcasting service"""
        if not self._started:
            return
        
        try:
            # Stop quote streamer
            await quote_streamer.stop()
            
            # Stop WebSocket manager
            await websocket_manager.stop()
            
            self._started = False
            logger.info("Market data broadcaster stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping market data broadcaster: {e}")
    
    async def broadcast_quote_update(self, symbol: str, quote_data: Dict[str, Any]):
        """
        Manually broadcast a quote update.
        
        Args:
            symbol: Stock symbol
            quote_data: Quote information
        """
        await websocket_manager.broadcast_quote(symbol, quote_data)
    
    async def broadcast_market_alert(
        self, 
        alert_type: str, 
        message: str, 
        symbols: Optional[List[str]] = None,
        priority: str = "normal"
    ):
        """
        Broadcast market alert to relevant clients.
        
        Args:
            alert_type: Type of alert (price_alert, volume_spike, news, etc.)
            message: Alert message
            symbols: Related symbols (None = all clients)
            priority: Alert priority (low, normal, high, critical)
        """
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "symbols": symbols or [],
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine target clients based on symbol subscriptions
        if symbols:
            target_clients = []
            stats = websocket_manager.get_connection_stats()
            
            for symbol in symbols:
                symbol_upper = symbol.upper()
                if symbol_upper in stats.get("symbol_subscriber_counts", {}):
                    # Get clients subscribed to this symbol
                    # This would require extending websocket_manager to track client->symbol mapping
                    pass
            
            await websocket_manager.broadcast_alert(alert_data, target_clients)
        else:
            await websocket_manager.broadcast_alert(alert_data)
    
    async def broadcast_price_alert(
        self, 
        symbol: str, 
        current_price: float, 
        alert_price: float, 
        alert_type: str
    ):
        """
        Broadcast price alert when target price is reached.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            alert_price: Target price that was reached
            alert_type: Type of alert (above, below)
        """
        message = f"{symbol} has moved {'above' if alert_type == 'above' else 'below'} ${alert_price:.2f} (current: ${current_price:.2f})"
        
        await self.broadcast_market_alert(
            alert_type="price_alert",
            message=message,
            symbols=[symbol],
            priority="high"
        )
    
    async def broadcast_volume_spike(self, symbol: str, current_volume: int, average_volume: int):
        """
        Broadcast volume spike alert.
        
        Args:
            symbol: Stock symbol
            current_volume: Current volume
            average_volume: Average volume
        """
        volume_multiple = current_volume / average_volume if average_volume > 0 else 0
        message = f"{symbol} experiencing volume spike: {current_volume:,} ({volume_multiple:.1f}x average)"
        
        await self.broadcast_market_alert(
            alert_type="volume_spike",
            message=message,
            symbols=[symbol],
            priority="normal"
        )
    
    async def broadcast_market_status(self, status: str, message: Optional[str] = None):
        """
        Broadcast market status updates.
        
        Args:
            status: Market status (open, closed, pre_market, after_hours)
            message: Optional status message
        """
        alert_data = {
            "alert_type": "market_status",
            "status": status,
            "message": message or f"Market is now {status}",
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_alert(alert_data)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return websocket_manager.get_connection_stats()
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get quote streaming statistics"""
        return quote_streamer.get_status()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all broadcasting services"""
        return {
            "broadcaster_started": self._started,
            "websocket_connections": self.get_connection_stats(),
            "quote_streaming": self.get_streaming_stats(),
            "server_time": datetime.now().isoformat()
        }


# Global market data broadcaster instance
market_data_broadcaster = MarketDataBroadcaster()