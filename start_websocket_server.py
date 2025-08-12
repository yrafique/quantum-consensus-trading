#!/usr/bin/env python3
"""
WebSocket Server Startup Script
===============================

Starts the FastAPI server with WebSocket support for real-time market data.
"""

import asyncio
import uvicorn
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.logging_config import get_logger
from src.websocket.market_data_broadcaster import market_data_broadcaster

logger = get_logger(__name__)


async def start_websocket_services():
    """Start all WebSocket-related services"""
    try:
        logger.info("Starting WebSocket services...")
        await market_data_broadcaster.start()
        logger.info("✅ All WebSocket services started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start WebSocket services: {e}")
        raise


async def stop_websocket_services():
    """Stop all WebSocket-related services"""
    try:
        logger.info("Stopping WebSocket services...")
        await market_data_broadcaster.stop()
        logger.info("✅ All WebSocket services stopped successfully")
    except Exception as e:
        logger.error(f"❌ Error stopping WebSocket services: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start WebSocket server for real-time trading data")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"Starting QuantumConsensus WebSocket Server on {args.host}:{args.port}")
    
    # Configure uvicorn
    config = uvicorn.Config(
        app="src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
        access_log=True,
        use_colors=True,
        reload_dirs=[str(project_root / "src")] if args.reload else None,
    )
    
    server = uvicorn.Server(config)
    
    try:
        # Start the server
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()