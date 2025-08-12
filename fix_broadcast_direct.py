#!/usr/bin/env python3
"""
Direct Broadcast Fix
===================

Directly fixes the broadcast mechanism to ensure quotes reach connected WebSocket clients.
"""

import asyncio
import websockets
import json
import yfinance as yf
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def start_working_quote_service():
    """Start a working quote broadcasting service"""
    
    print("🔧 Starting Direct Quote Broadcasting Service")
    print("=" * 50)
    
    # Connected clients storage
    connected_clients = {}
    client_subscriptions = {}
    
    async def handle_client(websocket, path):
        """Handle individual WebSocket client connections"""
        client_id = path.split('/')[-1] if '/' in path else f"client_{len(connected_clients)}"
        
        # Add client to connected clients
        connected_clients[client_id] = websocket
        client_subscriptions[client_id] = set()
        
        print(f"🔗 Client connected: {client_id}")
        
        try:
            # Send connection status
            await websocket.send(json.dumps({
                "type": "status",
                "data": {"status": "connected", "client_id": client_id},
                "timestamp": datetime.now().isoformat()
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'subscribe':
                        symbol = data.get('data', {}).get('symbol', '').upper()
                        if symbol:
                            client_subscriptions[client_id].add(symbol)
                            print(f"📊 {client_id} subscribed to {symbol}")
                            
                            # Send immediate quote for this symbol
                            await send_quote_to_client(websocket, symbol)
                    
                    elif msg_type == 'unsubscribe':
                        symbol = data.get('data', {}).get('symbol', '').upper()
                        if symbol:
                            client_subscriptions[client_id].discard(symbol)
                            print(f"📉 {client_id} unsubscribed from {symbol}")
                
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON from {client_id}")
                except Exception as e:
                    print(f"❌ Error handling message from {client_id}: {e}")
        
        except websockets.exceptions.ConnectionClosedError:
            print(f"🔌 Client {client_id} disconnected normally")
        except Exception as e:
            print(f"❌ Client {client_id} error: {e}")
        finally:
            # Clean up client
            if client_id in connected_clients:
                del connected_clients[client_id]
            if client_id in client_subscriptions:
                del client_subscriptions[client_id]
            print(f"🧹 Cleaned up client {client_id}")
    
    async def send_quote_to_client(websocket, symbol):
        """Send a quote to a specific client"""
        try:
            # Fetch real market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if not hist.empty:
                current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
                previous_close = float(hist['Close'].iloc[-2] if len(hist) >= 2 else hist['Close'].iloc[-1])
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                volume = int(info.get('volume', hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0))
                
                quote_data = {
                    "type": "quote",
                    "data": {
                        "symbol": symbol,
                        "price": current_price,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": volume,
                        "timestamp": datetime.now().isoformat(),
                        "market_status": "open" if 9 <= datetime.now().hour < 16 else "closed"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(quote_data))
                return True
            else:
                print(f"⚠️ No data available for {symbol}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending quote for {symbol}: {e}")
            return False
    
    async def broadcast_quotes():
        """Broadcast quotes to all subscribed clients"""
        while True:
            try:
                if connected_clients:
                    print(f"📡 Broadcasting to {len(connected_clients)} clients")
                    
                    # Get all subscribed symbols
                    all_symbols = set()
                    for subscriptions in client_subscriptions.values():
                        all_symbols.update(subscriptions)
                    
                    if all_symbols:
                        print(f"📊 Updating quotes for: {', '.join(all_symbols)}")
                        
                        # Send quotes to each client for their subscribed symbols
                        for client_id, websocket in list(connected_clients.items()):
                            client_symbols = client_subscriptions.get(client_id, set())
                            
                            for symbol in client_symbols:
                                try:
                                    success = await send_quote_to_client(websocket, symbol)
                                    if success:
                                        print(f"   ✅ Sent {symbol} quote to {client_id}")
                                except Exception as e:
                                    print(f"   ❌ Failed to send {symbol} to {client_id}: {e}")
                    else:
                        print("⏳ No symbol subscriptions")
                else:
                    print("⏳ No connected clients")
                
                # Wait 10 seconds before next broadcast
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"❌ Broadcast error: {e}")
                await asyncio.sleep(5)
    
    # Start WebSocket server on a different port to avoid conflicts
    server_port = 8001
    print(f"🚀 Starting WebSocket server on port {server_port}")
    
    # Start the server
    server = await websockets.serve(handle_client, "localhost", server_port)
    print(f"✅ WebSocket server started on ws://localhost:{server_port}")
    
    # Start broadcasting task
    broadcast_task = asyncio.create_task(broadcast_quotes())
    
    print("🎉 Direct Quote Broadcasting Service is running!")
    print(f"🔗 Connect to: ws://localhost:{server_port}/quotes/your_client_id")
    print("📊 Service will run for 120 seconds...")
    print("\n💡 Test command:")
    print(f"python3 -c \"import asyncio, websockets, json; asyncio.run((lambda: websockets.connect('ws://localhost:{server_port}/quotes/test'))())\"")
    
    # Run for 2 minutes
    try:
        await asyncio.sleep(120)
    except KeyboardInterrupt:
        print("⏹️ Stopping service...")
    
    # Cleanup
    broadcast_task.cancel()
    server.close()
    await server.wait_closed()
    
    print("✅ Direct broadcasting service stopped")

if __name__ == "__main__":
    asyncio.run(start_working_quote_service())