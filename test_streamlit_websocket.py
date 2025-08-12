#!/usr/bin/env python3
"""
Streamlit WebSocket Connection Test
==================================

Test script to simulate Streamlit WebSocket integration and verify end-to-end functionality.
"""

import asyncio
import websockets
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def simulate_streamlit_connection():
    """Simulate how Streamlit would connect to WebSocket"""
    
    print("🔧 Simulating Streamlit WebSocket Connection")
    print("=" * 50)
    
    try:
        # Step 1: Start quote streaming services
        from src.websocket.websocket_manager import websocket_manager
        from src.websocket.quote_streamer import quote_streamer
        from src.websocket.market_data_broadcaster import market_data_broadcaster
        
        await market_data_broadcaster.start()
        print("✅ Services started")
        
        # Step 2: Connect as Streamlit client would
        client_id = "streamlit_user_123"
        uri = f"ws://localhost:8000/api/v1/ws/quotes/{client_id}"
        
        print(f"🔗 Connecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Step 3: Subscribe to watchlist symbols (like Streamlit would)
            watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            
            for symbol in watchlist:
                subscribe_msg = {
                    "type": "subscribe",
                    "data": {"symbol": symbol}
                }
                await websocket.send(json.dumps(subscribe_msg))
                print(f"📊 Subscribed to {symbol}")
            
            # Step 4: Manually trigger quote fetching and broadcasting
            print("🔄 Triggering quote broadcasts...")
            
            # Import yfinance here to fetch real data
            import yfinance as yf
            from datetime import datetime
            
            for symbol in watchlist:
                try:
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
                            "symbol": symbol,
                            "price": current_price,
                            "change": change,
                            "change_percent": change_percent,
                            "volume": volume,
                            "timestamp": datetime.now().isoformat(),
                            "market_status": "open" if 9 <= datetime.now().hour < 16 else "closed"
                        }
                        
                        # Broadcast the quote
                        await websocket_manager.broadcast_quote(symbol, quote_data)
                        print(f"   📈 Broadcast {symbol}: ${current_price:.2f}")
                        
                except Exception as e:
                    print(f"   ❌ Failed to broadcast {symbol}: {e}")
            
            # Step 5: Listen for incoming quotes
            print("👂 Listening for incoming quotes...")
            quotes_received = 0
            
            for i in range(10):  # Listen for 10 cycles
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    
                    if data.get('type') == 'quote':
                        quotes_received += 1
                        quote = data.get('data', {})
                        symbol = quote.get('symbol', 'Unknown')
                        price = quote.get('price', 0)
                        change_percent = quote.get('change_percent', 0)
                        market_status = quote.get('market_status', 'unknown')
                        
                        # This is what Streamlit would display
                        live_indicator = "🔴 LIVE" if market_status == "open" else "🟡 CACHED"
                        print(f"   {live_indicator} {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
                        
                    elif data.get('type') == 'status':
                        print(f"   ℹ️ Status: {data.get('data', {}).get('status', 'unknown')}")
                    else:
                        print(f"   📨 Message type: {data.get('type', 'unknown')}")
                        
                except asyncio.TimeoutError:
                    print(f"   ⏳ Cycle {i+1}/10 - waiting...")
                    continue
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    break
            
            print(f"✅ Simulation completed - received {quotes_received} quotes")
            
            # Step 6: Show what Streamlit connection status would display
            print("\n📊 STREAMLIT UI SIMULATION:")
            print("-" * 30)
            if quotes_received > 0:
                print("🔗 Connected to real-time data stream")
                print("📊 Subscribed to: AAPL, GOOGL, MSFT, AMZN, TSLA")
                print("🔴 LIVE data indicators active")
            else:
                print("❌ Real-time data stream disconnected")
                print("⚪ STATIC data fallback active")
    
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the simulation"""
    print("🚀 Streamlit WebSocket Integration Test")
    print("=" * 60)
    
    await simulate_streamlit_connection()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("If this test shows live quotes, the WebSocket system is working.")
    print("The issue is likely that Streamlit frontend isn't connecting properly.")
    print("\n🔧 TO FIX STREAMLIT:")
    print("1. Refresh your browser at http://localhost:8501")
    print("2. Go to AI Trading Agents page")
    print("3. Click 'Subscribe to Watchlist' button")
    print("4. Look for real-time indicators")

if __name__ == "__main__":
    asyncio.run(main())