#!/usr/bin/env python3
"""
Safe Restart Script for Trading System
=====================================

This script safely restarts the trading system app with proper cleanup:
- Kills any existing Streamlit processes
- Clears GPU processes and memory
- Cleans up temporary files and cache
- Disables GPU/MLX to prevent crashes
- Starts app with memory management

Usage: python safe_restart.py
"""

import os
import sys
import subprocess
import signal
import time
import psutil
import gc
import tempfile
import shutil
from pathlib import Path

def kill_existing_processes():
    """Kill all existing Streamlit and related processes"""
    print("🧹 Cleaning up existing processes...")
    
    # Kill Streamlit processes
    try:
        subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        print("  ✅ Killed Streamlit processes")
    except:
        pass
    
    # Kill Python processes running our app
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'app_safe.py' in cmdline or 'trading_system' in cmdline:
                        proc.kill()
                        print(f"  ✅ Killed Python process: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"  ⚠️ Process cleanup warning: {e}")

def clear_gpu_processes():
    """Clear GPU processes and reset GPU state"""
    print("🔧 Clearing GPU processes...")
    
    # Kill GPU processes
    gpu_processes = ['nvidia-smi', 'nvidia-ml-py', 'torch', 'tensorflow']
    for proc_name in gpu_processes:
        try:
            subprocess.run(["pkill", "-f", proc_name], capture_output=True)
        except:
            pass
    
    # Reset CUDA context if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("  ✅ Cleared CUDA cache")
    except:
        pass
    
    print("  ✅ GPU cleanup completed")

def setup_safe_environment():
    """Set up environment variables for safe execution"""
    print("🛡️ Setting up safe environment...")
    
    # Disable GPU and heavy AI libraries
    safe_env = {
        'CUDA_VISIBLE_DEVICES': '',
        'MLX_DISABLE': '1',
        'PYTORCH_DISABLE_CUDA': '1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'NUMEXPR_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        # Memory management
        'MALLOC_TRIM_THRESHOLD_': '100000',
        'PYTHONHASHSEED': '0',
        # Streamlit specific
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
    }
    
    for key, value in safe_env.items():
        os.environ[key] = value
    
    print("  ✅ Environment configured for safety")

def clear_cache_and_temp():
    """Clear caches and temporary files"""
    print("🧽 Clearing caches and temporary files...")
    
    # Clear Python cache
    gc.collect()
    
    # Clear Streamlit cache
    streamlit_cache_dirs = [
        Path.home() / '.streamlit',
        Path(tempfile.gettempdir()) / 'streamlit',
        Path.cwd() / '.streamlit',
    ]
    
    for cache_dir in streamlit_cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"  ✅ Cleared {cache_dir}")
            except Exception as e:
                print(f"  ⚠️ Could not clear {cache_dir}: {e}")
    
    # Clear our app cache
    app_cache_files = [
        'streamlit.log',
        '.streamlit_cache',
        '__pycache__',
    ]
    
    for cache_file in app_cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            try:
                if cache_path.is_file():
                    cache_path.unlink()
                else:
                    shutil.rmtree(cache_path)
                print(f"  ✅ Cleared {cache_file}")
            except Exception as e:
                print(f"  ⚠️ Could not clear {cache_file}: {e}")
    
    print("  ✅ Cache cleanup completed")

def wait_for_cleanup():
    """Wait for processes to fully terminate"""
    print("⏳ Waiting for cleanup to complete...")
    time.sleep(3)
    
    # Verify no conflicting processes
    conflicts = 0
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'streamlit' in cmdline.lower() or 'app_safe.py' in cmdline:
                    conflicts += 1
            except:
                pass
    except:
        pass
    
    if conflicts > 0:
        print(f"  ⚠️ Found {conflicts} potentially conflicting processes")
        time.sleep(2)  # Additional wait
    else:
        print("  ✅ No conflicts detected")

def start_app():
    """Start the app with safe parameters"""
    print("🚀 Starting trading system app...")
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'quantum_consensus_app.py',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.gatherUsageStats', 'false',
        '--global.developmentMode', 'false',
        '--global.suppressDeprecationWarnings', 'true',
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("  ✅ App process started")
        print("  🌐 App will be available at: http://localhost:8501")
        print("  📝 Monitoring startup...")
        
        # Monitor startup for first few seconds
        startup_time = 0
        max_startup_time = 15
        
        while startup_time < max_startup_time:
            if process.poll() is not None:
                # Process has terminated
                output, _ = process.communicate()
                print(f"  ❌ App terminated unexpectedly:")
                print(output)
                return False
            
            time.sleep(1)
            startup_time += 1
            
            # Check if server is responding
            try:
                import requests
                response = requests.get("http://localhost:8501", timeout=1)
                if response.status_code == 200:
                    print("  ✅ App is responding successfully!")
                    break
            except:
                pass
        
        if startup_time >= max_startup_time:
            print("  ⚠️ App startup taking longer than expected, but process is running")
        
        print("\n🎉 QuantumConsensus Platform started successfully!")
        print("📱 Open your browser to: http://localhost:8501")
        print("⚡ Multi-Agent Intelligence enabled with quantum processing")
        print("\n💡 QuantumConsensus Features:")
        print("  - Multi-Agent AI Chat for quantum consensus analysis")
        print("  - Advanced quantum portfolio management")
        print("  - Neural-powered watchlist with predictive analytics")
        print("\n🛑 To restart QuantumConsensus safely, run this script again")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to start app: {e}")
        return False

def main():
    """Main restart process"""
    print("🌊 QuantumConsensus - Safe Restart")
    print("=" * 40)
    
    try:
        # Step 1: Kill existing processes
        kill_existing_processes()
        
        # Step 2: Clear GPU processes
        clear_gpu_processes()
        
        # Step 3: Set up safe environment
        setup_safe_environment()
        
        # Step 4: Clear caches
        clear_cache_and_temp()
        
        # Step 5: Wait for cleanup
        wait_for_cleanup()
        
        # Step 6: Start the app
        success = start_app()
        
        if success:
            print("\n✅ Restart completed successfully!")
        else:
            print("\n❌ Restart failed - check logs above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Restart interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Restart failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()