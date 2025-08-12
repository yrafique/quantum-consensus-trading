"""
WebSocket API Router
===================

FastAPI WebSocket endpoints for real-time market data streaming.
"""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse

from ...websocket.websocket_manager import websocket_manager, MessageType
from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/quotes/{client_id}")
async def websocket_quotes_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time quote streaming.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    # Validate client_id
    if not client_id or len(client_id) < 3:
        await websocket.close(code=1008, reason="Invalid client_id")
        return
    
    # Connect client
    connected = await websocket_manager.connect_client(websocket, client_id)
    if not connected:
        await websocket.close(code=1011, reason="Connection failed")
        return
    
    try:
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                await websocket_manager.handle_client_message(client_id, data)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected")
                break
                
            except Exception as e:
                logger.error(f"Error handling WebSocket message for {client_id}: {e}")
                # Send error response but continue connection
                continue
                
    finally:
        # Cleanup on disconnect
        await websocket_manager.disconnect_client(client_id)


@router.websocket("/quotes")  
async def websocket_quotes_endpoint_auto_id(websocket: WebSocket):
    """
    WebSocket endpoint with auto-generated client ID.
    """
    client_id = f"client_{uuid.uuid4().hex[:8]}"
    await websocket_quotes_endpoint(websocket, client_id)


@router.get("/status")
async def websocket_status():
    """Get WebSocket connection status and statistics."""
    stats = websocket_manager.get_connection_stats()
    
    return {
        "status": "active",
        "server_time": asyncio.get_event_loop().time(),
        "connections": stats
    }


@router.post("/broadcast/quote")
async def broadcast_quote_update(symbol: str, quote_data: dict):
    """
    Manually broadcast a quote update (for testing).
    
    Args:
        symbol: Stock symbol
        quote_data: Quote information
    """
    await websocket_manager.broadcast_quote(symbol, quote_data)
    return {"status": "broadcasted", "symbol": symbol}


@router.post("/broadcast/alert")
async def broadcast_alert(alert_data: dict, client_ids: Optional[list] = None):
    """
    Broadcast an alert to clients.
    
    Args:
        alert_data: Alert information
        client_ids: Target client IDs (None = all clients)
    """
    await websocket_manager.broadcast_alert(alert_data, client_ids)
    return {"status": "broadcasted", "targets": client_ids or "all"}


@router.get("/clients")
async def list_connected_clients():
    """List all connected WebSocket clients."""
    stats = websocket_manager.get_connection_stats()
    return {
        "total_clients": stats["total_connections"],
        "subscribed_symbols": stats["subscribed_symbols"], 
        "symbol_subscribers": stats["symbol_subscriber_counts"]
    }