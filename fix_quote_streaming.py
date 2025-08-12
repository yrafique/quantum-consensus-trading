#!/usr/bin/env python3
"""
Quote Streaming Fix
==================

Fixes the quote streaming issue by implementing active quote broadcasting.
"""

import asyncio
import json
import yfinance as yf
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def fix_and_test_streaming():
    """Fix quote streaming and test with live connection"""
    try:
        from src.websocket.websocket_manager import websocket_manager
        from src.websocket.quote_streamer import quote_streamer
        from src.websocket.market_data_broadcaster import market_data_broadcaster
        
        print("🔧 Starting Quote Streaming Fix...")
        
        # Step 1: Ensure services are running
        await market_data_broadcaster.start()
        print("✅ Market Data Broadcaster started")
        
        # Step 2: Subscribe to watchlist symbols in quote streamer
        watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        for symbol in watchlist:
            await quote_streamer.subscribe_symbol(symbol)
        print(f"✅ Subscribed to {len(watchlist)} symbols in quote streamer")
        
        # Step 3: Manually fetch and broadcast quotes for all symbols
        print("📊 Fetching and broadcasting quotes...")
        for symbol in watchlist:
            try:
                # Get real market data
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
                    previous_close = float(hist['Close'].iloc[-2] if len(hist) >= 2 else hist['Close'].iloc[-1])
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                    volume = int(info.get('volume', hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0))
                    
                    # Create quote data
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
                    print(f"   📈 {symbol}: ${current_price:.2f} ({change_percent:+.2f}%)")
                else:
                    print(f"   ⚠️ No data for {symbol}")
                    
            except Exception as e:
                print(f"   ❌ Failed to fetch {symbol}: {e}")
        
        print("✅ Quote broadcasting completed")
        
        # Step 4: Start continuous streaming loop
        print("🔄 Starting continuous quote streaming...")
        
        async def continuous_streaming():
            """Continuous quote streaming loop"""
            while True:
                try:
                    # Check if there are any connections
                    stats = websocket_manager.get_connection_stats()
                    active_connections = stats.get('active_connections', 0)
                    
                    if active_connections > 0:
                        print(f"📡 Broadcasting to {active_connections} active connections...")
                        
                        # Fetch and broadcast quotes for subscribed symbols
                        symbol_subscribers = stats.get('symbol_subscriber_counts', {})
                        for symbol in symbol_subscribers.keys():
                            await quote_streamer._fetch_and_broadcast_quote(symbol)
                    else:
                        print("⏳ No active connections, waiting...")
                    
                    # Wait 5 seconds before next update
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"❌ Streaming error: {e}")
                    await asyncio.sleep(5)
        
        # Start the continuous streaming task
        streaming_task = asyncio.create_task(continuous_streaming())
        
        print("🎉 Quote streaming is now active!")
        print("💡 Connect to WebSocket to see live quotes")
        print("🔗 WebSocket URL: ws://localhost:8000/api/v1/ws/quotes/your_client_id")
        print("📊 To test: refresh Streamlit page and click 'Subscribe to Watchlist'")
        print("\n⏰ Streaming will run for 60 seconds, then stop...")
        
        # Run for 60 seconds then stop
        await asyncio.sleep(60)
        
        # Cancel streaming task
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass
        
        print("✅ Streaming test completed")
        
        # Final status check
        final_stats = websocket_manager.get_connection_stats()
        print(f"📊 Final stats: {final_stats.get('active_connections', 0)} connections")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 QuantumConsensus Quote Streaming Fix")
    print("=" * 50)
    asyncio.run(fix_and_test_streaming())