#!/usr/bin/env python3
"""
Complete WebSocket Testing Suite
===============================

Comprehensive test script to verify WebSocket functionality and diagnose connection issues.
"""

import asyncio
import websockets
import json
import requests
import time
import yfinance as yf
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class WebSocketTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.streamlit_url = "http://localhost:8501"
    
    def test_1_server_health(self):
        """Test 1: Basic server health checks"""
        print("🔍 TEST 1: Server Health Checks")
        print("=" * 50)
        
        # Test FastAPI server
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            print(f"✅ FastAPI Server: {response.status_code} - {response.json().get('status', 'unknown')}")
        except Exception as e:
            print(f"❌ FastAPI Server: {e}")
            return False
        
        # Test Streamlit server
        try:
            response = requests.get(self.streamlit_url, timeout=5)
            print(f"✅ Streamlit Server: {response.status_code}")
        except Exception as e:
            print(f"❌ Streamlit Server: {e}")
        
        # Test WebSocket status endpoint
        try:
            response = requests.get(f"{self.base_url}/api/v1/ws/status", timeout=5)
            data = response.json()
            print(f"✅ WebSocket Status: {data.get('status', 'unknown')}")
            print(f"   - Active Connections: {data.get('connections', {}).get('total_connections', 0)}")
            print(f"   - Subscribed Symbols: {data.get('connections', {}).get('subscribed_symbols', 0)}")
        except Exception as e:
            print(f"❌ WebSocket Status: {e}")
            return False
        
        print()
        return True
    
    async def test_2_basic_websocket(self):
        """Test 2: Basic WebSocket connection"""
        print("🔍 TEST 2: Basic WebSocket Connection")
        print("=" * 50)
        
        try:
            uri = f"{self.ws_url}/api/v1/ws/quotes/test_client"
            async with websockets.connect(uri) as websocket:
                print("✅ WebSocket connection established")
                
                # Send test message
                test_msg = {"type": "test", "data": {"message": "hello"}}
                await websocket.send(json.dumps(test_msg))
                print("✅ Test message sent")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(f"✅ Response received: {data.get('type', 'unknown')}")
                    return True
                except asyncio.TimeoutError:
                    print("⚠️ No response within 5 seconds (acceptable)")
                    return True
                    
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def test_3_quote_subscription(self):
        """Test 3: Quote subscription flow"""
        print("🔍 TEST 3: Quote Subscription Flow")
        print("=" * 50)
        
        try:
            uri = f"{self.ws_url}/api/v1/ws/quotes/test_subscriber"
            async with websockets.connect(uri) as websocket:
                print("✅ Connected for subscription test")
                
                # Subscribe to AAPL
                subscribe_msg = {
                    "type": "subscribe",
                    "data": {"symbol": "AAPL"}
                }
                await websocket.send(json.dumps(subscribe_msg))
                print("✅ Subscription message sent for AAPL")
                
                # Listen for 10 seconds
                quotes_received = 0
                start_time = time.time()
                
                while (time.time() - start_time) < 10:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'quote':
                            quotes_received += 1
                            quote = data.get('data', {})
                            print(f"📊 Quote #{quotes_received}: {quote.get('symbol')} = ${quote.get('price', 0):.2f}")
                        elif data.get('type') == 'status':
                            print(f"ℹ️ Status: {data.get('data', {}).get('status')}")
                            
                    except asyncio.TimeoutError:
                        continue
                
                print(f"✅ Subscription test completed - {quotes_received} quotes received")
                return quotes_received > 0
                
        except Exception as e:
            print(f"❌ Subscription test failed: {e}")
            return False
    
    def test_4_market_data_fetch(self):
        """Test 4: Direct market data fetching"""
        print("🔍 TEST 4: Market Data Fetching")
        print("=" * 50)
        
        try:
            # Test direct Yahoo Finance access
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
                volume = int(info.get('volume', hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0))
                print(f"✅ AAPL Data: ${current_price:.2f}, Volume: {volume:,}")
                return True
            else:
                print("❌ No market data available")
                return False
                
        except Exception as e:
            print(f"❌ Market data fetch failed: {e}")
            return False
    
    async def test_5_manual_quote_broadcast(self):
        """Test 5: Manual quote broadcasting"""
        print("🔍 TEST 5: Manual Quote Broadcasting")
        print("=" * 50)
        
        try:
            # Import and use the internal services
            from src.websocket.websocket_manager import websocket_manager
            from src.websocket.quote_streamer import quote_streamer
            from src.websocket.market_data_broadcaster import market_data_broadcaster
            
            # Start services if not running
            if not market_data_broadcaster._started:
                await market_data_broadcaster.start()
                print("✅ Market data broadcaster started")
            
            # Subscribe to AAPL in the quote streamer
            await quote_streamer.subscribe_symbol("AAPL")
            print("✅ Subscribed to AAPL in quote streamer")
            
            # Get real market data and broadcast it
            ticker = yf.Ticker("AAPL")
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
                    "symbol": "AAPL",
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": volume,
                    "timestamp": datetime.now().isoformat(),
                    "market_status": "open"
                }
                
                # Broadcast the quote
                await websocket_manager.broadcast_quote("AAPL", quote_data)
                print(f"✅ Broadcast AAPL quote: ${current_price:.2f} ({change_percent:+.2f}%)")
                
                # Check connection stats
                stats = websocket_manager.get_connection_stats()
                print(f"📊 Active connections: {stats.get('active_connections', 0)}")
                print(f"📈 Symbol subscribers: {stats.get('symbol_subscriber_counts', {})}")
                
                return True
            else:
                print("❌ No market data to broadcast")
                return False
                
        except Exception as e:
            print(f"❌ Manual broadcast failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_6_concurrent_connections(self):
        """Test 6: Multiple concurrent connections"""
        print("🔍 TEST 6: Concurrent Connections Test")
        print("=" * 50)
        
        async def single_connection(client_id):
            try:
                uri = f"{self.ws_url}/api/v1/ws/quotes/{client_id}"
                async with websockets.connect(uri) as websocket:
                    # Subscribe to AAPL
                    subscribe_msg = {
                        "type": "subscribe",
                        "data": {"symbol": "AAPL"}
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    # Listen for 5 seconds
                    quotes_received = 0
                    start_time = time.time()
                    
                    while (time.time() - start_time) < 5:
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(response)
                            if data.get('type') == 'quote':
                                quotes_received += 1
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                    
                    return quotes_received
                    
            except Exception as e:
                print(f"❌ Client {client_id} failed: {e}")
                return 0
        
        # Test with 3 concurrent connections
        try:
            tasks = [single_connection(f"client_{i}") for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_connections = sum(1 for r in results if isinstance(r, int))
            total_quotes = sum(r for r in results if isinstance(r, int))
            
            print(f"✅ Successful connections: {successful_connections}/3")
            print(f"📊 Total quotes received: {total_quotes}")
            
            return successful_connections >= 2
            
        except Exception as e:
            print(f"❌ Concurrent connections test failed: {e}")
            return False
    
    def test_7_streamlit_integration(self):
        """Test 7: Streamlit WebSocket integration"""
        print("🔍 TEST 7: Streamlit Integration Test")
        print("=" * 50)
        
        try:
            # Test if Streamlit WebSocket client can be imported
            from src.websocket.streamlit_websocket_client import StreamlitWebSocketClient, create_websocket_javascript
            
            # Create a test client (without Streamlit session state)
            print("✅ StreamlitWebSocketClient imported successfully")
            
            # Test JavaScript generation
            js_code = create_websocket_javascript()
            if "WebSocket" in js_code and "localhost:8000" in js_code:
                print("✅ WebSocket JavaScript code generated")
            else:
                print("❌ WebSocket JavaScript code invalid")
                return False
            
            print("✅ Streamlit integration components working")
            return True
            
        except Exception as e:
            print(f"❌ Streamlit integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Run all tests"""
    print("🚀 QuantumConsensus WebSocket Testing Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tester = WebSocketTester()
    test_results = {}
    
    # Run all tests
    test_results["server_health"] = tester.test_1_server_health()
    test_results["basic_websocket"] = await tester.test_2_basic_websocket()
    test_results["quote_subscription"] = await tester.test_3_quote_subscription()
    test_results["market_data"] = tester.test_4_market_data_fetch()
    test_results["manual_broadcast"] = await tester.test_5_manual_quote_broadcast()
    test_results["concurrent_connections"] = await tester.test_6_concurrent_connections()
    test_results["streamlit_integration"] = tester.test_7_streamlit_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
        if result:
            passed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! WebSocket system is fully operational.")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ Most tests passed. Some minor issues may exist.")
    else:
        print("❌ Multiple test failures. WebSocket system needs attention.")
    
    # Diagnostic recommendations
    print("\n📝 DIAGNOSTIC RECOMMENDATIONS:")
    print("-" * 40)
    
    if not test_results["server_health"]:
        print("• Check if both FastAPI (port 8000) and Streamlit (port 8501) servers are running")
    
    if not test_results["basic_websocket"]:
        print("• WebSocket server is not accepting connections - check server logs")
    
    if not test_results["quote_subscription"]:
        print("• Quote streaming not working - check Market Data Broadcaster")
    
    if not test_results["market_data"]:
        print("• Market data fetching failed - check internet connection and Yahoo Finance access")
    
    if not test_results["manual_broadcast"]:
        print("• Internal broadcast system not working - check WebSocket Manager")
    
    if not test_results["streamlit_integration"]:
        print("• Streamlit integration has issues - check imports and session state")
    
    print("\n🔧 SUGGESTED FIXES:")
    print("-" * 30)
    print("1. Restart both servers: python3 quantum_start.py && python3 start_websocket_server.py")
    print("2. Refresh browser page at http://localhost:8501")
    print("3. Click 'Subscribe to Watchlist' button on AI Trading Agents page")
    print("4. Check network connectivity and firewall settings")

if __name__ == "__main__":
    asyncio.run(main())