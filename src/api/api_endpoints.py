#!/usr/bin/env python3
"""
API Endpoints for River Trading System
=====================================

REST API endpoints for external validation and integration.
Provides HTTP interfaces for all major trading system functionality.

Endpoints:
- GET /health - System health check
- POST /analyze - Stock analysis
- POST /opportunities - Hunt opportunities  
- POST /chat - AI conversation
- GET /metrics - Performance metrics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime
from typing import Dict, Any
import traceback

# Import trading components with proper module handling
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try relative imports first
    try:
        from .mlx_trading_llm import MLXTradingLLM
        from .opportunity_hunter import OpportunityHunter
        from .data_validator import DataValidator
        from .llm_reasoner import generate_recommendation
    except ImportError:
        # Fallback to direct imports
        from mlx_trading_llm import MLXTradingLLM
        from opportunity_hunter import OpportunityHunter
        from data_validator import DataValidator
        from llm_reasoner import generate_recommendation
    
    COMPONENTS_AVAILABLE = True
    logging.info("Trading components loaded successfully")
except ImportError as e:
    logging.warning(f"Trading components not available: {e}")
    COMPONENTS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components (lazy loaded)
_trading_components = None

def get_components():
    """Lazy load trading components."""
    global _trading_components
    if _trading_components is None and COMPONENTS_AVAILABLE:
        try:
            _trading_components = {
                'llm': MLXTradingLLM(),
                'hunter': OpportunityHunter(), 
                'validator': DataValidator()
            }
            logger.info("Trading components loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            _trading_components = {}
    return _trading_components or {}

@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint."""
    try:
        components = get_components()
        
        health_status = {
            "status": "healthy" if components else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "mlx_available": COMPONENTS_AVAILABLE,
                "llm_loaded": bool(components.get('llm')),
                "hunter_ready": bool(components.get('hunter')),
                "validator_ready": bool(components.get('validator'))
            }
        }
        
        if components.get('llm'):
            try:
                health_status["components"]["llm_model_loaded"] = components['llm'].loaded
            except:
                health_status["components"]["llm_model_loaded"] = False
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """Stock analysis endpoint."""
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({"error": "ticker required"}), 400
        
        ticker = data['ticker'].upper()
        debug_mode = data.get('debug', False)
        
        components = get_components()
        if not components.get('llm') or not components.get('validator'):
            return jsonify({"error": "Analysis components not available"}), 503
        
        # Get market data
        validator = components['validator']
        market_data = validator.get_market_context(ticker)
        
        if not market_data:
            return jsonify({"error": f"Could not fetch data for {ticker}"}), 404
        
        # Perform analysis
        llm = components['llm']
        analysis = llm.analyze_opportunity(ticker, market_data, debug_mode=debug_mode)
        
        if not analysis:
            return jsonify({"error": f"Analysis failed for {ticker}"}), 500
        
        response = {
            "ticker": ticker,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data if debug_mode else None
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Analysis endpoint failed: {e}")
        return jsonify({
            "error": "Analysis failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/opportunities', methods=['POST'])
def hunt_opportunities():
    """Opportunity hunting endpoint."""
    try:
        data = request.get_json() or {}
        max_opportunities = data.get('max_opportunities', 10)
        debug_mode = data.get('debug', False)
        
        components = get_components()
        if not components.get('hunter'):
            return jsonify({"error": "Opportunity hunter not available"}), 503
        
        hunter = components['hunter']
        opportunities = hunter.hunt_opportunities(max_opportunities, debug_mode)
        
        response = {
            "opportunities": opportunities if not debug_mode else opportunities.get("opportunities", []),
            "count": len(opportunities) if not debug_mode else len(opportunities.get("opportunities", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        if debug_mode and isinstance(opportunities, dict):
            response["screening_stats"] = opportunities.get("screening_stats")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Opportunities endpoint failed: {e}")
        return jsonify({
            "error": "Opportunity hunting failed", 
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """AI conversation endpoint."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "message required"}), 400
        
        message = data['message']
        components = get_components()
        
        if not components.get('llm'):
            return jsonify({
                "response": "AI chat temporarily unavailable. Please try again later.",
                "timestamp": datetime.now().isoformat()
            }), 200
        
        # Generate response
        llm = components['llm']
        prompt = f"""You are a professional trading advisor. User said: "{message}"
        
        Provide a helpful response that:
        1. Addresses their question directly
        2. Offers actionable insights
        3. Maintains professional tone
        4. Includes appropriate risk warnings
        
        Keep response concise (2-3 paragraphs max).
        """
        
        if hasattr(llm, '_generate_text') and llm.loaded:
            response_text = llm._generate_text(prompt, max_tokens=300)
        else:
            response_text = "AI system is initializing. Please try again in a moment."
        
        return jsonify({
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return jsonify({
            "error": "Chat failed",
            "details": str(e), 
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Performance metrics endpoint."""
    try:
        # Basic metrics - can be extended
        metrics = {
            "api_status": "operational",
            "components_loaded": bool(get_components()),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return jsonify({
            "error": "Metrics failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/validate', methods=['POST'])
def validate_system():
    """Comprehensive system validation endpoint."""
    try:
        validation_results = {
            "system_health": "unknown",
            "component_tests": {},
            "api_tests": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test components
        components = get_components()
        validation_results["component_tests"] = {
            "mlx_available": COMPONENTS_AVAILABLE,
            "llm_loaded": bool(components.get('llm')),
            "hunter_ready": bool(components.get('hunter')),
            "validator_ready": bool(components.get('validator'))
        }
        
        # Test basic functionality
        if components.get('validator'):
            try:
                # Test data validation with a known stock
                market_data = components['validator'].get_market_context('AAPL')
                validation_results["api_tests"]["data_fetch"] = bool(market_data)
            except:
                validation_results["api_tests"]["data_fetch"] = False
        
        if components.get('llm') and components['llm'].loaded:
            try:
                # Test MLX inference
                test_response = components['llm']._generate_text("Test", max_tokens=10)
                validation_results["api_tests"]["mlx_inference"] = bool(test_response)
            except:
                validation_results["api_tests"]["mlx_inference"] = False
        
        # Determine overall health
        all_tests = list(validation_results["component_tests"].values()) + list(validation_results["api_tests"].values())
        if all(all_tests):
            validation_results["system_health"] = "healthy"
        elif any(all_tests):
            validation_results["system_health"] = "degraded"
        else:
            validation_results["system_health"] = "error"
        
        status_code = 200 if validation_results["system_health"] in ["healthy", "degraded"] else 500
        return jsonify(validation_results), status_code
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return jsonify({
            "system_health": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Run the API server."""
    logger.info("Starting River Trading API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()