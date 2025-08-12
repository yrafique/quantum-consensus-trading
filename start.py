#!/usr/bin/env python3
"""
QuantumConsensus Default Startup Script
======================================

This is the main entry point for launching the QuantumConsensus trading platform.
It uses the safe startup mechanism to prevent memory issues and ensure stable operation.
"""

import subprocess
import sys
import os

def main():
    """Launch QuantumConsensus using the safe startup script."""
    # Use the safe quantum_start.py script
    safe_start_script = os.path.join(os.path.dirname(__file__), 'quantum_start.py')
    
    if not os.path.exists(safe_start_script):
        print("❌ Error: quantum_start.py not found!")
        print("Please ensure quantum_start.py is in the same directory.")
        sys.exit(1)
    
    # Launch using the safe startup
    try:
        subprocess.run([sys.executable, safe_start_script])
    except KeyboardInterrupt:
        print("\n👋 QuantumConsensus shutdown requested")
    except Exception as e:
        print(f"❌ Error launching QuantumConsensus: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Always use safe startup
    main()