#!/usr/bin/env python3
"""
QuantumConsensus Platform Launcher
==================================

Simple launcher for the QuantumConsensus trading platform.
This script starts the Streamlit app with optimal settings.

Usage: python start_quantum_consensus.py
"""

import subprocess
import sys
import os

def main():
    """Launch QuantumConsensus platform"""
    print("🌊 Starting QuantumConsensus Trading Platform...")
    print("=" * 50)
    
    # Set environment for safety
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Launch command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'quantum_consensus_app.py',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    try:
        print("🚀 Launching QuantumConsensus...")
        print("🌐 Platform will be available at: http://localhost:8501")
        print("💡 Features:")
        print("  - Multi-Agent AI Trading Analysis")
        print("  - Real-time Stock Data & Charts")
        print("  - Advanced Technical Indicators")
        print("  - Kelly Criterion Position Sizing")
        print("  - Quantum Consensus Scoring")
        print("\n" + "=" * 50)
        
        # Start the application
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n🛑 QuantumConsensus platform stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting platform: {e}")
        print("💡 Try installing dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()