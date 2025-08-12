#!/usr/bin/env python3
"""
WebSocket Management Utility
============================

Utility script for managing WebSocket services, testing connections, and monitoring performance.
"""

import asyncio
import websockets
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.logging_config import get_logger
from src.websocket.websocket_manager import websocket_manager
from src.websocket.quote_streamer import quote_streamer
from src.websocket.market_data_broadcaster import market_data_broadcaster

logger = get_logger(__name__)


class WebSocketTester:
    """Test WebSocket connections and functionality"""
    
    def __init__(self, websocket_url: str = "ws://localhost:8000/api/v1/ws/quotes"):
        self.websocket_url = websocket_url
        self.client_id = f"test_client_{int(time.time())}"
    
    async def test_connection(self) -> bool:
        """Test basic WebSocket connection"""
        try:
            logger.info(f"Testing connection to {self.websocket_url}/{self.client_id}")
            
            async with websockets.connect(f"{self.websocket_url}/{self.client_id}") as websocket:
                # Send a test message
                test_message = {
                    "type": "subscribe",
                    "data": {"symbol": "AAPL"}
                }
                await websocket.send(json.dumps(test_message))
                logger.info("✅ Sent test subscription message")
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    logger.info(f"✅ Received response: {response_data}")
                    return True
                    
                except asyncio.TimeoutError:
                    logger.warning("⚠️ No response received within 10 seconds")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    async def test_subscription_flow(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Test full subscription flow"""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]
        
        results = {
            "connection_success": False,
            "subscription_success": False,
            "quote_received": False,
            "symbols_tested": symbols,
            "quotes_received": [],
            "errors": []
        }
        
        try:
            logger.info(f"Testing subscription flow for symbols: {symbols}")
            
            async with websockets.connect(f"{self.websocket_url}/{self.client_id}") as websocket:
                results["connection_success"] = True
                logger.info("✅ WebSocket connected")
                
                # Subscribe to symbols
                for symbol in symbols:
                    subscribe_msg = {
                        "type": "subscribe",
                        "data": {"symbol": symbol}
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"📤 Subscribed to {symbol}")
                
                results["subscription_success"] = True
                
                # Listen for quotes
                timeout_duration = 30  # 30 seconds timeout
                start_time = time.time()
                
                while (time.time() - start_time) < timeout_duration:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_data = json.loads(response)
                        
                        if response_data.get("type") == "quote":
                            results["quote_received"] = True
                            results["quotes_received"].append(response_data)
                            logger.info(f"📊 Quote received: {response_data['data']['symbol']} = ${response_data['data']['price']:.2f}")
                        
                        # Stop after receiving quotes for all symbols
                        received_symbols = {q["data"]["symbol"] for q in results["quotes_received"]}
                        if len(received_symbols) >= len(symbols):
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        results["errors"].append(str(e))
                        logger.warning(f"⚠️ Error receiving quote: {e}")
                
                logger.info(f"✅ Test completed. Received {len(results['quotes_received'])} quotes")
                
        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Subscription flow test failed: {e}")
        
        return results
    
    async def stress_test(self, concurrent_clients: int = 10, duration_seconds: int = 60) -> Dict[str, Any]:
        """Stress test WebSocket server"""
        logger.info(f"Starting stress test: {concurrent_clients} clients for {duration_seconds}s")
        
        results = {
            "concurrent_clients": concurrent_clients,
            "duration_seconds": duration_seconds,
            "successful_connections": 0,
            "failed_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "errors": []
        }
        
        async def single_client_test(client_num: int):
            """Test function for a single client"""
            client_id = f"stress_test_client_{client_num}_{int(time.time())}"
            messages_sent = 0
            messages_received = 0
            
            try:
                async with websockets.connect(f"{self.websocket_url}/{client_id}") as websocket:
                    results["successful_connections"] += 1
                    
                    # Subscribe to a random symbol
                    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
                    symbol = symbols[client_num % len(symbols)]
                    
                    subscribe_msg = {
                        "type": "subscribe",
                        "data": {"symbol": symbol}
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    messages_sent += 1
                    
                    # Listen for messages during test duration
                    start_time = time.time()
                    while (time.time() - start_time) < duration_seconds:
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            messages_received += 1
                        except asyncio.TimeoutError:
                            continue
                            
            except Exception as e:
                results["failed_connections"] += 1
                results["errors"].append(f"Client {client_num}: {str(e)}")
            
            results["total_messages_sent"] += messages_sent
            results["total_messages_received"] += messages_received
        
        # Run concurrent clients
        tasks = [single_client_test(i) for i in range(concurrent_clients)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Stress test completed:")
        logger.info(f"   - Successful connections: {results['successful_connections']}")
        logger.info(f"   - Failed connections: {results['failed_connections']}")
        logger.info(f"   - Total messages sent: {results['total_messages_sent']}")
        logger.info(f"   - Total messages received: {results['total_messages_received']}")
        
        return results


async def start_services():
    """Start all WebSocket services"""
    logger.info("🚀 Starting WebSocket services...")
    await market_data_broadcaster.start()
    logger.info("✅ All services started successfully")


async def stop_services():
    """Stop all WebSocket services"""
    logger.info("🛑 Stopping WebSocket services...")
    await market_data_broadcaster.stop()
    logger.info("✅ All services stopped successfully")


async def get_status():
    """Get status of all WebSocket services"""
    logger.info("📊 WebSocket Services Status:")
    
    # Get comprehensive status
    status = market_data_broadcaster.get_comprehensive_status()
    
    print(f"Broadcaster Started: {status['broadcaster_started']}")
    print(f"Active Connections: {status['websocket_connections'].get('active_connections', 0)}")
    print(f"Subscribed Symbols: {list(status['websocket_connections'].get('symbol_subscriber_counts', {}).keys())}")
    print(f"Quote Streamer Running: {status['quote_streaming'].get('running', False)}")
    print(f"Server Time: {status['server_time']}")
    
    return status


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WebSocket Management Utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start WebSocket services")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop WebSocket services")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get service status")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test WebSocket functionality")
    test_parser.add_argument("--url", default="ws://localhost:8000/api/v1/ws/quotes", help="WebSocket URL")
    test_parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL"], help="Symbols to test")
    
    # Stress test command
    stress_parser = subparsers.add_parser("stress", help="Stress test WebSocket server")
    stress_parser.add_argument("--clients", type=int, default=10, help="Number of concurrent clients")
    stress_parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    stress_parser.add_argument("--url", default="ws://localhost:8000/api/v1/ws/quotes", help="WebSocket URL")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "start":
            await start_services()
            
        elif args.command == "stop":
            await stop_services()
            
        elif args.command == "status":
            await get_status()
            
        elif args.command == "test":
            tester = WebSocketTester(args.url)
            
            # Basic connection test
            logger.info("Running basic connection test...")
            connection_ok = await tester.test_connection()
            
            if connection_ok:
                # Full subscription flow test
                logger.info("Running subscription flow test...")
                results = await tester.test_subscription_flow(args.symbols)
                print(f"Test Results: {json.dumps(results, indent=2)}")
            else:
                logger.error("❌ Basic connection test failed. Skipping subscription test.")
                
        elif args.command == "stress":
            tester = WebSocketTester(args.url)
            logger.info("Running stress test...")
            results = await tester.stress_test(args.clients, args.duration)
            print(f"Stress Test Results: {json.dumps(results, indent=2)}")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())