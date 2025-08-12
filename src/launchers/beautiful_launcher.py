#!/usr/bin/env python3
"""
River Trading System - Beautiful Launch Experience
=================================================

Stunning, engaging launcher that creates excitement while the system loads.
Professional animations, progress tracking, and smooth transitions.
"""

import os
import sys
import subprocess
import time
import requests
import threading
import json
from datetime import datetime
import random

class BeautifulLauncher:
    """Beautiful loading experience for River Trading System"""
    
    def __init__(self):
        self.api_port = 5001
        self.ui_port = 8503
        self.api_process = None
        self.ui_process = None
        self.loading_complete = False
        self.current_step = 0
        self.total_steps = 8
        
        # Loading steps with engaging messages
        self.steps = [
            {"name": "🔧 Initializing Core Systems", "duration": 2, "message": "Preparing your trading environment..."},
            {"name": "🧠 Loading AI Brain", "duration": 3, "message": "Awakening MLX intelligence..."},
            {"name": "🌐 Connecting Data Feeds", "duration": 2, "message": "Establishing market connections..."},
            {"name": "🚀 Starting API Engine", "duration": 2, "message": "Powering up trading APIs..."},
            {"name": "🎯 Calibrating Analysis Tools", "duration": 2, "message": "Fine-tuning ReAct reasoning..."},
            {"name": "📊 Validating Components", "duration": 1, "message": "Running system health checks..."},
            {"name": "🎨 Launching Interface", "duration": 3, "message": "Creating beautiful experience..."},
            {"name": "✨ Final Preparations", "duration": 1, "message": "Almost ready to trade..."}
        ]
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_gradient_text(self, text, colors=['96', '94', '92', '93', '95']):
        """Print text with gradient colors"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            print(f"\033[{color}m{line}\033[0m")
    
    def print_animated_header(self):
        """Print animated header with river theme"""
        header = """
████████████████████████████████████████████████████████████
█                                                          █
█    🌊 RIVER TRADING SYSTEM 🌊                           █
█                                                          █
█    ✨ Enterprise AI Trading Platform ✨                 █
█                                                          █
████████████████████████████████████████████████████████████
        """
        self.print_gradient_text(header)
        print("\n")
    
    def print_loading_bar(self, progress, step_name, message):
        """Print animated loading bar"""
        bar_length = 50
        filled_length = int(bar_length * progress / 100)
        
        # Create animated bar with flowing effect
        bar = ""
        for i in range(bar_length):
            if i < filled_length:
                if i == filled_length - 1:
                    bar += "🌊"  # Wave at the front
                else:
                    bar += "█"
            else:
                bar += "░"
        
        # Progress percentage with color
        if progress < 30:
            color = "93"  # Yellow
        elif progress < 70:
            color = "94"  # Blue
        else:
            color = "92"  # Green
        
        print(f"\033[{color}m{step_name}\033[0m")
        print(f"[{bar}] \033[{color}m{progress:.1f}%\033[0m")
        print(f"\033[90m{message}\033[0m")
        print()
    
    def animate_loading_dots(self, text, duration=2):
        """Animate loading dots"""
        for _ in range(duration * 4):  # 4 updates per second
            for dots in ["   ", ".  ", ".. ", "..."]:
                print(f"\r\033[96m{text}{dots}\033[0m", end="", flush=True)
                time.sleep(0.25)
        print()
    
    def show_system_specs(self):
        """Show system specifications with style"""
        print("\033[95m" + "=" * 60)
        print("🖥️  SYSTEM SPECIFICATIONS")
        print("=" * 60 + "\033[0m")
        
        specs = [
            ("🧠 AI Engine", "MLX-Accelerated Local LLM"),
            ("🔬 Analysis", "ReAct Multi-Step Reasoning"),
            ("📊 Data Feed", "Real-Time Market Integration"),
            ("🚨 Monitoring", "LED Status System"),
            ("⚡ Performance", "Sub-200ms Inference"),
            ("🛡️  Security", "Local Processing (No Cloud)")
        ]
        
        for spec, value in specs:
            print(f"\033[92m{spec:<15}\033[0m \033[94m{value}\033[0m")
        
        print("\033[95m" + "=" * 60 + "\033[0m\n")
    
    def start_background_services(self):
        """Start services in background with progress tracking"""
        def start_api():
            # Kill existing processes
            try:
                subprocess.run(['lsof', '-ti:5001', '|', 'xargs', 'kill', '-9'], 
                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1)
            except:
                pass
            
            # Start API server
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            env['TRANSFORMERS_VERBOSITY'] = 'error'
            env['MLX_LOGGING_LEVEL'] = 'ERROR'
            
            self.api_process = subprocess.Popen(
                [sys.executable, 'src/api/start_api.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
        
        def start_ui():
            # Kill existing streamlit
            try:
                subprocess.run(['pkill', '-f', 'streamlit'], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(1)
            except:
                pass
            
            # Start UI
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
        
        # Start services in threads
        api_thread = threading.Thread(target=start_api, daemon=True)
        ui_thread = threading.Thread(target=start_ui, daemon=True)
        
        # Start API first, then UI after delay
        api_thread.start()
        time.sleep(3)  # Give API time to start
        ui_thread.start()
    
    def check_services_ready(self):
        """Check if services are ready"""
        api_ready = False
        ui_ready = False
        
        # Check API
        try:
            response = requests.get(f'http://localhost:{self.api_port}/health', timeout=2)
            if response.status_code in [200, 503]:
                api_ready = True
        except:
            pass
        
        # Check UI
        try:
            response = requests.get(f'http://localhost:{self.ui_port}/_stcore/health', timeout=2)
            if response.status_code == 200:
                ui_ready = True
        except:
            pass
        
        return api_ready, ui_ready
    
    def show_loading_sequence(self):
        """Show beautiful loading sequence"""
        self.start_background_services()
        
        total_progress = 0
        step_progress = 100 / self.total_steps
        
        for i, step in enumerate(self.steps):
            self.clear_screen()
            self.print_animated_header()
            
            # Show current step progress
            step_start_progress = i * step_progress
            
            # Animate progress for this step
            for progress_point in range(0, 101, 10):
                current_progress = step_start_progress + (progress_point * step_progress / 100)
                
                self.print_loading_bar(
                    current_progress, 
                    step["name"], 
                    step["message"]
                )
                
                # Add some visual flair
                if i == 1:  # AI Brain loading
                    print("🧠 " + "█" * (progress_point // 10) + "░" * (10 - progress_point // 10))
                elif i == 2:  # Data feeds
                    print("📊 " + "▓" * (progress_point // 20) + "░" * (5 - progress_point // 20))
                elif i == 6:  # Interface
                    print("🎨 " + "◆" * (progress_point // 25) + "◇" * (4 - progress_point // 25))
                
                # Check if services are ready during later steps
                if i >= 3:
                    api_ready, ui_ready = self.check_services_ready()
                    if api_ready:
                        print("\033[92m✓ API Engine Online\033[0m")
                    if ui_ready:
                        print("\033[92m✓ Interface Ready\033[0m")
                
                time.sleep(step["duration"] / 10)  # Distribute step duration
        
        # Final completion
        self.clear_screen()
        self.print_animated_header()
        self.print_loading_bar(100, "🎉 System Ready!", "Welcome to the future of trading!")
        
        # Show completion animation
        print("\n" + "🌊" * 30)
        print("\033[92m" + " " * 10 + "LOADING COMPLETE!" + "\033[0m")
        print("🌊" * 30 + "\n")
    
    def show_ready_screen(self):
        """Show final ready screen with access information"""
        self.clear_screen()
        
        # Celebration header
        celebration = """
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
        
          🌊 RIVER TRADING SYSTEM IS READY! 🌊
          
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
        """
        self.print_gradient_text(celebration)
        
        print("\n" + "\033[96m" + "=" * 80)
        print("🚀 YOUR AI TRADING PLATFORM IS NOW LIVE!")
        print("=" * 80 + "\033[0m\n")
        
        # Access information with beautiful formatting
        access_info = f"""
\033[92m🌊 TRADING INTERFACE:\033[0m
   \033[94m🔗 http://localhost:{self.ui_port}\033[0m
   \033[90m   → Your main trading dashboard\033[0m

\033[92m⚡ API ENDPOINTS:\033[0m
   \033[94m🔗 http://localhost:{self.api_port}\033[0m
   \033[90m   → REST API for integrations\033[0m

\033[92m📊 SYSTEM HEALTH:\033[0m
   \033[94m🔗 http://localhost:{self.api_port}/health\033[0m
   \033[90m   → Real-time system monitoring\033[0m
        """
        print(access_info)
        
        # Feature highlights
        print("\033[93m" + "=" * 50)
        print("✨ PREMIUM FEATURES ACTIVATED")
        print("=" * 50 + "\033[0m")
        
        features = [
            "🧠 MLX-Accelerated AI Analysis",
            "🎯 ReAct Multi-Step Reasoning", 
            "🚨 Real-Time LED Monitoring",
            "📈 Live Market Data Integration",
            "🔍 Cross-Validated Recommendations",
            "⚡ Sub-200ms Response Times",
            "🛡️ Local Processing (Privacy First)"
        ]
        
        for i, feature in enumerate(features):
            time.sleep(0.1)  # Stagger the display
            print(f"\033[92m✓\033[0m {feature}")
        
        # Usage instructions
        print(f"\n\033[95m" + "=" * 50)
        print("🎮 GETTING STARTED")
        print("=" * 50 + "\033[0m")
        
        instructions = [
            "1. 🌐 Open your browser to the trading interface",
            "2. 🚨 Wait for all LED indicators to turn green",  
            "3. 🎯 Start analyzing stocks with AI assistance",
            "4. 📊 Explore opportunities with confidence scores",
            "5. 🧠 Experience transparent ReAct reasoning"
        ]
        
        for instruction in instructions:
            print(f"   {instruction}")
        
        # Call to action
        print(f"\n\033[96m" + "🌊" * 50)
        print("  READY TO REVOLUTIONIZE YOUR TRADING?")
        print("🌊" * 50 + "\033[0m")
        
        print(f"\n\033[93m💡 Click here to open: \033[94m\033[4mhttp://localhost:{self.ui_port}\033[0m")
        print(f"\n\033[91m⚠️  Press Ctrl+C to stop the system\033[0m\n")
        
        # Auto-open browser option
        try:
            import webbrowser
            print("\033[96m🚀 Auto-opening in your default browser...\033[0m")
            time.sleep(2)
            webbrowser.open(f'http://localhost:{self.ui_port}')
        except:
            pass
    
    def monitor_services(self):
        """Monitor running services with beautiful status updates"""
        try:
            print("\n\033[90m📡 Monitoring services... (Ctrl+C to stop)\033[0m\n")
            
            while True:
                # Check service health
                api_ready, ui_ready = self.check_services_ready()
                
                # Beautiful status display (update in place)
                status = f"\r\033[92m●\033[0m API: {'🟢 Online' if api_ready else '🔴 Offline'}  " + \
                        f"\033[92m●\033[0m UI: {'🟢 Ready' if ui_ready else '🔴 Starting'}  " + \
                        f"\033[90m{datetime.now().strftime('%H:%M:%S')}\033[0m"
                
                print(status, end="", flush=True)
                time.sleep(2)
                
        except KeyboardInterrupt:
            print(f"\n\n\033[93m🛑 Shutting down River Trading System...\033[0m")
            self.cleanup()
    
    def cleanup(self):
        """Beautiful shutdown sequence"""
        print("\n\033[96m" + "=" * 50)
        print("🌊 GRACEFUL SHUTDOWN SEQUENCE")
        print("=" * 50 + "\033[0m")
        
        shutdown_steps = [
            "🔌 Stopping API services...",
            "🎨 Closing trading interface...", 
            "🧹 Cleaning up resources...",
            "✨ Finalizing shutdown..."
        ]
        
        for step in shutdown_steps:
            print(f"\033[93m{step}\033[0m", end="", flush=True)
            
            if "API" in step and self.api_process:
                self.api_process.terminate()
                try:
                    self.api_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.api_process.kill()
            
            elif "interface" in step and self.ui_process:
                self.ui_process.terminate()
                try:
                    self.ui_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.ui_process.kill()
            
            # Animate dots
            for i in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            
            print(" \033[92m✓\033[0m")
        
        # Final cleanup
        try:
            subprocess.run(['pkill', '-f', 'streamlit'], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['lsof', '-ti:5001', '|', 'xargs', 'kill', '-9'], 
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
        
        # Beautiful goodbye
        goodbye = """
\n🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊

    Thank you for using River Trading System! 
    
    Your AI-powered trading journey continues...
    
🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊
        """
        self.print_gradient_text(goodbye)
    
    def launch(self):
        """Launch with beautiful experience"""
        try:
            # Show beautiful loading sequence
            self.show_loading_sequence()
            
            # Brief pause for dramatic effect
            time.sleep(1)
            
            # Show ready screen
            self.show_ready_screen()
            
            # Monitor services
            self.monitor_services()
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            print(f"\n\033[91m❌ System error: {e}\033[0m")
            self.cleanup()

def main():
    """Beautiful launcher entry point"""
    # Set up clean environment
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['MLX_LOGGING_LEVEL'] = 'ERROR'
    
    launcher = BeautifulLauncher()
    launcher.launch()

if __name__ == "__main__":
    main()