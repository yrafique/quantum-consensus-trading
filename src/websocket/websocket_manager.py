"""
WebSocket Connection Manager
============================

Manages WebSocket connections, client subscriptions, and real-time broadcasting.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from contextlib import asynccontextmanager

from fastapi import WebSocket, WebSocketDisconnect
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe" 
    QUOTE = "quote"
    ALERT = "alert"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STATUS = "status"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format"""
    type: str
    timestamp: str
    data: Dict[str, Any]
    client_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


@dataclass
class ClientSubscription:
    """Client subscription tracking"""
    client_id: str
    websocket: WebSocket
    symbols: Set[str]
    connected_at: datetime
    last_heartbeat: datetime
    
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale"""
        return (datetime.now() - self.last_heartbeat).total_seconds() > timeout_seconds


class WebSocketManager:
    """
    Manages WebSocket connections and real-time data broadcasting.
    
    Features:
    - Client connection management
    - Symbol subscription tracking
    - Automatic heartbeat monitoring
    - Graceful connection cleanup
    - Broadcasting to subscribed clients
    """
    
    def __init__(self, heartbeat_interval: int = 30):
        self.connections: Dict[str, ClientSubscription] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> client_ids
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the WebSocket manager"""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            
        # Disconnect all clients
        await self._disconnect_all_clients()
        logger.info("WebSocket manager stopped")
    
    async def connect_client(self, websocket: WebSocket, client_id: str) -> bool:
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
            
        Returns:
            bool: True if connected successfully
        """
        try:
            await websocket.accept()
            
            async with self._lock:
                # Disconnect existing client with same ID
                if client_id in self.connections:
                    await self._disconnect_client_unsafe(client_id)
                
                # Add new client
                subscription = ClientSubscription(
                    client_id=client_id,
                    websocket=websocket,
                    symbols=set(),
                    connected_at=datetime.now(),
                    last_heartbeat=datetime.now()
                )
                self.connections[client_id] = subscription
            
            # Send welcome message
            welcome_msg = WebSocketMessage(
                type=MessageType.STATUS.value,
                timestamp=datetime.now().isoformat(),
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "server_time": datetime.now().isoformat()
                },
                client_id=client_id
            )
            await self._send_to_client(client_id, welcome_msg)
            
            logger.info(f"WebSocket client connected: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket client {client_id}: {e}")
            return False
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a WebSocket client"""
        async with self._lock:
            await self._disconnect_client_unsafe(client_id)
    
    async def _disconnect_client_unsafe(self, client_id: str):
        """Disconnect client without acquiring lock (internal use)"""
        if client_id not in self.connections:
            return
            
        subscription = self.connections[client_id]
        
        # Remove from symbol subscriptions
        for symbol in subscription.symbols:
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
        
        # Close WebSocket connection
        try:
            await subscription.websocket.close()
        except:
            pass  # Connection might already be closed
        
        # Remove from connections
        del self.connections[client_id]
        logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def _disconnect_all_clients(self):
        """Disconnect all clients"""
        client_ids = list(self.connections.keys())
        for client_id in client_ids:
            await self._disconnect_client_unsafe(client_id)
    
    async def subscribe_to_symbol(self, client_id: str, symbol: str) -> bool:
        """
        Subscribe client to symbol updates.
        
        Args:
            client_id: Client identifier
            symbol: Stock symbol to subscribe to
            
        Returns:
            bool: True if subscribed successfully
        """
        async with self._lock:
            if client_id not in self.connections:
                logger.warning(f"Cannot subscribe unknown client {client_id} to {symbol}")
                return False
            
            # Add to client's subscriptions
            self.connections[client_id].symbols.add(symbol.upper())
            
            # Add to symbol subscribers
            symbol_upper = symbol.upper()
            if symbol_upper not in self.symbol_subscribers:
                self.symbol_subscribers[symbol_upper] = set()
            self.symbol_subscribers[symbol_upper].add(client_id)
            
            logger.info(f"Client {client_id} subscribed to {symbol_upper}")
            return True
    
    async def unsubscribe_from_symbol(self, client_id: str, symbol: str) -> bool:
        """
        Unsubscribe client from symbol updates.
        
        Args:
            client_id: Client identifier  
            symbol: Stock symbol to unsubscribe from
            
        Returns:
            bool: True if unsubscribed successfully
        """
        async with self._lock:
            if client_id not in self.connections:
                return False
            
            symbol_upper = symbol.upper()
            
            # Remove from client's subscriptions
            self.connections[client_id].symbols.discard(symbol_upper)
            
            # Remove from symbol subscribers
            if symbol_upper in self.symbol_subscribers:
                self.symbol_subscribers[symbol_upper].discard(client_id)
                if not self.symbol_subscribers[symbol_upper]:
                    del self.symbol_subscribers[symbol_upper]
            
            logger.info(f"Client {client_id} unsubscribed from {symbol_upper}")
            return True
    
    async def broadcast_quote(self, symbol: str, quote_data: Dict[str, Any]):
        """
        Broadcast quote update to all subscribed clients.
        
        Args:
            symbol: Stock symbol
            quote_data: Quote information
        """
        symbol_upper = symbol.upper()
        
        async with self._lock:
            if symbol_upper not in self.symbol_subscribers:
                return  # No subscribers
            
            subscribers = self.symbol_subscribers[symbol_upper].copy()
        
        # Create message
        message = WebSocketMessage(
            type=MessageType.QUOTE.value,
            timestamp=datetime.now().isoformat(),
            data={
                "symbol": symbol_upper,
                "quote": quote_data
            }
        )
        
        # Send to all subscribers
        disconnected_clients = []
        for client_id in subscribers:
            if not await self._send_to_client(client_id, message):
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any], target_clients: Optional[List[str]] = None):
        """
        Broadcast alert to clients.
        
        Args:
            alert_data: Alert information
            target_clients: Specific clients to send to (None = all clients)
        """
        message = WebSocketMessage(
            type=MessageType.ALERT.value,
            timestamp=datetime.now().isoformat(),
            data=alert_data
        )
        
        if target_clients is None:
            async with self._lock:
                target_clients = list(self.connections.keys())
        
        # Send to target clients
        disconnected_clients = []
        for client_id in target_clients:
            if not await self._send_to_client(client_id, message):
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    async def _send_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """
        Send message to specific client.
        
        Args:
            client_id: Target client ID
            message: Message to send
            
        Returns:
            bool: True if sent successfully
        """
        if client_id not in self.connections:
            return False
        
        try:
            websocket = self.connections[client_id].websocket
            await websocket.send_text(message.to_json())
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to client {client_id}: {e}")
            return False
    
    async def handle_client_message(self, client_id: str, message_data: str):
        """
        Handle incoming message from client.
        
        Args:
            client_id: Client identifier
            message_data: Raw message data
        """
        try:
            data = json.loads(message_data)
            message_type = data.get("type")
            payload = data.get("data", {})
            
            if message_type == MessageType.SUBSCRIBE.value:
                symbol = payload.get("symbol")
                if symbol:
                    await self.subscribe_to_symbol(client_id, symbol)
                    
            elif message_type == MessageType.UNSUBSCRIBE.value:
                symbol = payload.get("symbol")
                if symbol:
                    await self.unsubscribe_from_symbol(client_id, symbol)
                    
            elif message_type == MessageType.HEARTBEAT.value:
                await self._update_client_heartbeat(client_id)
                
            else:
                logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")
            
            # Send error response
            error_msg = WebSocketMessage(
                type=MessageType.ERROR.value,
                timestamp=datetime.now().isoformat(),
                data={"error": "Invalid message format"},
                client_id=client_id
            )
            await self._send_to_client(client_id, error_msg)
    
    async def _update_client_heartbeat(self, client_id: str):
        """Update client's last heartbeat timestamp"""
        if client_id in self.connections:
            self.connections[client_id].last_heartbeat = datetime.now()
    
    async def _heartbeat_loop(self):
        """Background task for heartbeat monitoring"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check for stale connections
                stale_clients = []
                async with self._lock:
                    for client_id, subscription in self.connections.items():
                        if subscription.is_stale():
                            stale_clients.append(client_id)
                
                # Disconnect stale clients
                for client_id in stale_clients:
                    logger.info(f"Disconnecting stale client: {client_id}")
                    await self.disconnect_client(client_id)
                
                # Send heartbeat to active clients
                heartbeat_msg = WebSocketMessage(
                    type=MessageType.HEARTBEAT.value,
                    timestamp=datetime.now().isoformat(),
                    data={"server_time": datetime.now().isoformat()}
                )
                
                active_clients = []
                async with self._lock:
                    active_clients = list(self.connections.keys())
                
                for client_id in active_clients:
                    await self._send_to_client(client_id, heartbeat_msg)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.connections),
            "subscribed_symbols": len(self.symbol_subscribers),
            "symbol_subscriber_counts": {
                symbol: len(subscribers) 
                for symbol, subscribers in self.symbol_subscribers.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()