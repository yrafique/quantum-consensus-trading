#!/usr/bin/env python3
"""
API Server Launcher
==================

Properly configured launcher for the River Trading API server.
Handles module imports and component loading correctly.
"""

import sys
import os

# Add current directory to Python path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set package context
__package__ = None

# Import and run the API
if __name__ == "__main__":
    from api_endpoints import app, logger
    
    logger.info("Starting River Trading API server...")
    logger.info(f"Python path: {sys.path[0]}")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)