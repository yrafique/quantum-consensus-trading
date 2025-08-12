"""
Safe AI Model Loader with Memory Management
==========================================

Provides safe loading of AI models with memory monitoring to prevent system crashes.
Includes loading status display and graceful fallback mechanisms.
"""

import os
import psutil
import streamlit as st
import threading
import time
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SafeAILoader:
    """Safe loading of AI models with memory monitoring and status display."""
    
    def __init__(self):
        self.mlx_model = None
        self.model_loaded = False
        self.loading_status = "Not started"
        self.loading_progress = 0.0
        self.error_message = None
        self._lock = threading.Lock()
        
        # Memory thresholds (GB)
        self.MIN_FREE_MEMORY_GB = 2.0  # Minimum free memory required
        self.MODEL_MEMORY_GB = 2.0     # Estimated model memory usage for 3B model
        
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        memory = psutil.virtual_memory()
        
        # Get Apple Silicon unified memory info if available
        total_gb = memory.total / (1024 ** 3)
        available_gb = memory.available / (1024 ** 3)
        used_gb = memory.used / (1024 ** 3)
        percent = memory.percent
        
        return {
            'total_gb': total_gb,
            'available_gb': available_gb,
            'used_gb': used_gb,
            'percent': percent,
            'can_load_model': available_gb >= (self.MIN_FREE_MEMORY_GB + self.MODEL_MEMORY_GB)
        }
    
    def display_loading_status(self, container=None):
        """Display loading status in Streamlit UI."""
        if container is None:
            container = st.container()
            
        with container:
            if self.model_loaded:
                st.success("✅ AI Model loaded successfully")
            elif self.error_message:
                st.error(f"❌ Model loading failed: {self.error_message}")
                st.info("💡 The app will continue with limited AI functionality")
            else:
                # Show loading progress
                st.info(f"🔄 {self.loading_status}")
                if self.loading_progress > 0:
                    st.progress(self.loading_progress)
                
                # Show memory status
                mem_info = self.get_system_memory_info()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("System Memory", f"{mem_info['total_gb']:.1f} GB")
                with col2:
                    st.metric("Available", f"{mem_info['available_gb']:.1f} GB", 
                             f"{100 - mem_info['percent']:.0f}%")
                with col3:
                    status = "✅ Ready" if mem_info['can_load_model'] else "⚠️ Low Memory"
                    st.metric("Model Status", status)
    
    def check_memory_availability(self) -> Tuple[bool, str]:
        """Check if there's enough memory to load the model safely."""
        mem_info = self.get_system_memory_info()
        
        if not mem_info['can_load_model']:
            return False, f"Insufficient memory. Need {self.MIN_FREE_MEMORY_GB + self.MODEL_MEMORY_GB:.1f} GB free, have {mem_info['available_gb']:.1f} GB"
        
        return True, "Memory check passed"
    
    def load_mlx_model_async(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        """Load MLX model asynchronously with progress updates."""
        def _load():
            try:
                with self._lock:
                    self.loading_status = "Checking system memory..."
                    self.loading_progress = 0.1
                
                # Memory check
                can_load, message = self.check_memory_availability()
                if not can_load:
                    with self._lock:
                        self.error_message = message
                        self.loading_status = "Failed: " + message
                    return
                
                with self._lock:
                    self.loading_status = "Initializing MLX framework..."
                    self.loading_progress = 0.2
                
                # Try to import MLX
                try:
                    import mlx.core as mx
                    from mlx_lm import load, generate
                    MLX_AVAILABLE = True
                except ImportError as e:
                    with self._lock:
                        self.error_message = f"MLX not available: {str(e)}"
                        self.loading_status = "Failed: MLX not installed"
                    return
                
                with self._lock:
                    self.loading_status = f"Loading model: {model_name.split('/')[-1]}..."
                    self.loading_progress = 0.3
                
                # Import MLXTradingLLM
                from src.ai.mlx_trading_llm import MLXTradingLLM
                
                # Create model instance
                with self._lock:
                    self.loading_status = "Initializing model (this may take a minute)..."
                    self.loading_progress = 0.5
                
                # Initialize with memory monitoring
                start_mem = psutil.virtual_memory().available
                model = MLXTradingLLM(model_name=model_name)
                end_mem = psutil.virtual_memory().available
                
                mem_used_gb = (start_mem - end_mem) / (1024 ** 3)
                
                with self._lock:
                    self.mlx_model = model
                    self.model_loaded = True
                    self.loading_status = f"Model loaded successfully (used {mem_used_gb:.1f} GB)"
                    self.loading_progress = 1.0
                    
                logger.info(f"MLX model loaded successfully. Memory used: {mem_used_gb:.1f} GB")
                
            except Exception as e:
                logger.error(f"Failed to load MLX model: {str(e)}")
                with self._lock:
                    self.error_message = str(e)
                    self.loading_status = f"Failed: {str(e)}"
                    self.loading_progress = 0.0
        
        # Start loading in background thread
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
    
    def get_model(self) -> Optional[Any]:
        """Get the loaded model if available."""
        with self._lock:
            return self.mlx_model
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        with self._lock:
            return self.model_loaded
    
    def wait_for_model(self, timeout: float = 30.0) -> bool:
        """Wait for model to load with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready() or self.error_message:
                return self.is_ready()
            time.sleep(0.1)
        return False


# Global instance
_safe_loader = None

def get_safe_ai_loader() -> SafeAILoader:
    """Get or create the global safe AI loader instance."""
    global _safe_loader
    if _safe_loader is None:
        _safe_loader = SafeAILoader()
    return _safe_loader


def initialize_ai_safely(model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit") -> SafeAILoader:
    """Initialize AI model safely with memory monitoring."""
    loader = get_safe_ai_loader()
    
    if not loader.is_ready() and not loader.error_message:
        loader.load_mlx_model_async(model_name)
    
    return loader