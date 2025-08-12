#!/usr/bin/env python3
"""
Quantum Consensus Safe Startup Script
====================================

Ensures safe startup with memory monitoring and graceful AI model loading.
"""

import os
import sys
import subprocess
import psutil
import time

def check_system_resources():
    """Check if system has enough resources to run the app."""
    memory = psutil.virtual_memory()
    
    # Check minimum memory (4GB free)
    min_memory_gb = 4.0
    available_gb = memory.available / (1024 ** 3)
    
    print(f"💻 System Memory Check:")
    print(f"   Total: {memory.total / (1024 ** 3):.1f} GB")
    print(f"   Available: {available_gb:.1f} GB ({100 - memory.percent:.0f}% free)")
    print(f"   Required: {min_memory_gb} GB minimum")
    
    if available_gb < min_memory_gb:
        print(f"\n⚠️  WARNING: Low memory! Only {available_gb:.1f} GB available.")
        print(f"   The app may run with limited AI functionality.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    else:
        print(f"✅ Memory check passed")
    
    return True

def set_environment_variables():
    """Set environment variables for safe operation."""
    # Disable MLX auto-loading to control memory usage
    os.environ['MLX_LAZY_LOAD'] = '1'
    
    # Set memory limits
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Streamlit settings
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    
    print("✅ Environment variables set")

def launch_app():
    """Launch the Quantum Consensus app with monitoring."""
    print("\n🚀 Launching Quantum Consensus Trading System...")
    print("   Access the app at: http://localhost:8501\n")
    
    try:
        # Launch the app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "quantum_consensus_app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.runOnSave", "true",
            "--server.maxUploadSize", "200",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Quantum Consensus...")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    print("=" * 60)
    print("🟢 Quantum Consensus Trading System - Safe Startup")
    print("=" * 60)
    
    # Check system resources
    check_system_resources()
    
    # Set environment variables
    set_environment_variables()
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()