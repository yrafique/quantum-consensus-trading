#!/usr/bin/env python3
"""
River Trading System - Production Launcher
==========================================

Clean, professional launcher for the River Trading platform
with minimal logging and polished user experience.
"""

import os
import sys
import subprocess
import time
import requests
import logging
from datetime import datetime

# Suppress verbose logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('mlx').setLevel(logging.ERROR)

# Suppress MLX verbose output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['MLX_LOGGING_LEVEL'] = 'ERROR'

class RiverTradingLauncher:
    """Professional launcher for River Trading System"""
    
    def __init__(self):
        self.api_port = 5001
        self.ui_port = 8503
        self.api_process = None
        self.ui_process = None
    
    def print_header(self):
        """Print professional header"""
        print("\033[96m" + "=" * 60)
        print("🌊 RIVER TRADING SYSTEM")
        print("=" * 60 + "\033[0m")
        print("\033[92m✓ Enterprise AI Trading Platform")
        print("✓ ReAct-Enhanced Analysis Engine")
        print("✓ Real-Time LED Status Monitoring")
        print("✓ MLX-Accelerated Local Intelligence\033[0m\n")
    
    def check_dependencies(self):
        """Check system dependencies"""
        print("🔍 Checking system dependencies...")
        
        required_files = [
            'src/interface/led_enhanced_interface.py',
            'src/api/start_api.py',
            'src/monitoring/connection_monitor.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"\033[91m❌ Missing files: {', '.join(missing_files)}\033[0m")
            return False
        
        print("\033[92m✓ All dependencies found\033[0m")
        return True
    
    def start_api_server(self):
        """Start API server with clean output"""
        print("🚀 Starting API server...")
        
        # Kill any existing process on port
        try:
            subprocess.run(['lsof', '-ti:5001'], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['lsof', '-ti:5001', '|', 'xargs', 'kill', '-9'], 
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
        
        # Start API server with suppressed output
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        self.api_process = subprocess.Popen(
            [sys.executable, 'src/api/start_api.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for API to be ready
        for i in range(10):
            try:
                response = requests.get(f'http://localhost:{self.api_port}/health', timeout=2)
                if response.status_code in [200, 503]:
                    print(f"\033[92m✓ API server ready at http://localhost:{self.api_port}\033[0m")
                    return True
            except:
                if i == 0:
                    print("  ⏳ Waiting for API server startup...", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                time.sleep(1)
        
        print(f"\n\033[91m❌ API server failed to start\033[0m")
        return False
    
    def start_ui_server(self):
        """Start UI server with clean output"""
        print("\n🎨 Starting River Trading interface...")
        
        # Kill any existing streamlit processes
        try:
            subprocess.run(['pkill', '-f', 'streamlit'], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
        
        time.sleep(1)
        
        # Start Streamlit with suppressed output
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        self.ui_process = subprocess.Popen(
            [
                sys.executable, '-m', 'streamlit', 'run', 
                'src/interface/led_enhanced_interface.py',
                '--server.port', str(self.ui_port),
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false',
                '--logger.level', 'error'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for UI to be ready
        print("  ⏳ Initializing trading interface...", end="", flush=True)
        
        for i in range(15):
            try:
                response = requests.get(f'http://localhost:{self.ui_port}/_stcore/health', timeout=2)
                if response.status_code == 200:
                    print(f"\n\033[92m✓ Trading interface ready\033[0m")
                    return True
            except:
                print(".", end="", flush=True)
                time.sleep(1)
        
        print(f"\n\033[91m❌ Trading interface failed to start\033[0m")
        return False
    
    def print_access_info(self):
        """Print access information"""
        print("\n" + "\033[96m" + "=" * 60)
        print("🎯 RIVER TRADING SYSTEM - READY")
        print("=" * 60 + "\033[0m")
        print(f"\033[92m🌊 Trading Interface: \033[94mhttp://localhost:{self.ui_port}\033[0m")
        print(f"\033[92m🔗 API Endpoints:     \033[94mhttp://localhost:{self.api_port}\033[0m")
        print(f"\033[92m📊 Health Check:      \033[94mhttp://localhost:{self.api_port}/health\033[0m")
        
        print(f"\n\033[93m💡 FEATURES AVAILABLE:\033[0m")
        print("   🧠 ReAct AI Analysis with MLX acceleration")
        print("   🚨 Real-time LED status monitoring")
        print("   🎯 Advanced opportunity hunting")
        print("   🔍 Cross-validated recommendations")
        print("   📈 Live market data integration")
        
        print(f"\n\033[95m🎮 USAGE:\033[0m")
        print("   1. Open the trading interface in your browser")
        print("   2. Wait for LED status validation (all components green)")
        print("   3. Start analyzing stocks with AI-powered insights")
        
        print(f"\n\033[91m⚠️  Press Ctrl+C to stop all services\033[0m")
        print("=" * 60)
    
    def monitor_services(self):
        """Monitor running services"""
        try:
            while True:
                # Check if processes are still running
                if self.api_process and self.api_process.poll() is not None:
                    print(f"\n\033[91m❌ API server stopped unexpectedly\033[0m")
                    break
                
                if self.ui_process and self.ui_process.poll() is not None:
                    print(f"\n\033[91m❌ Trading interface stopped unexpectedly\033[0m")
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n\n\033[93m🛑 Shutting down River Trading System...\033[0m")
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown of all services"""
        print("   🔌 Stopping API server...")
        if self.api_process:
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
        
        print("   🎨 Stopping trading interface...")
        if self.ui_process:
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
        
        # Clean up any remaining processes
        try:
            subprocess.run(['pkill', '-f', 'streamlit'], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['lsof', '-ti:5001', '|', 'xargs', 'kill', '-9'], 
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
        
        print("\033[92m✓ River Trading System shutdown complete\033[0m")
        print("\033[96m🌊 Thank you for using River Trading System!\033[0m\n")
    
    def launch(self):
        """Launch the complete system"""
        self.print_header()
        
        if not self.check_dependencies():
            return False
        
        if not self.start_api_server():
            return False
        
        if not self.start_ui_server():
            self.cleanup()
            return False
        
        self.print_access_info()
        self.monitor_services()
        
        return True

def main():
    """Main launcher entry point"""
    launcher = RiverTradingLauncher()
    try:
        launcher.launch()
    except KeyboardInterrupt:
        launcher.cleanup()
    except Exception as e:
        print(f"\033[91m❌ Launcher error: {e}\033[0m")
        launcher.cleanup()

if __name__ == "__main__":
    main()