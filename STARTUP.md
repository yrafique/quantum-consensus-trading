# QuantumConsensus Startup Guide

## Quick Start

To launch the QuantumConsensus trading platform, simply run:

```bash
python start.py
```

or

```bash
./start.py
```

## Safe Startup Features

The platform includes several safety mechanisms to ensure stable operation:

### 1. **Memory Management**
- Checks available system memory before loading AI models
- Gracefully handles low memory situations
- Shows loading progress for AI models
- Falls back to limited functionality if memory is insufficient

### 2. **Process Cleanup**
- Automatically kills any existing Streamlit processes
- Clears GPU memory if applicable
- Removes cache and temporary files
- Ensures clean startup environment

### 3. **AI Model Loading**
- Safe, asynchronous loading of MLX models
- Progress indicators during model initialization
- Memory monitoring during load
- Graceful fallback if models can't be loaded

## Startup Scripts

### `start.py` (Recommended)
Main entry point that uses safe startup mechanism.

### `quantum_start.py`
Safe startup script with full cleanup and environment setup.

### `quantum_consensus_safe.py`
Alternative safe startup with detailed memory checks.

### Direct Launch (Not Recommended)
```bash
streamlit run quantum_consensus_app.py
```
⚠️ This bypasses safety checks and may cause memory issues.

## Troubleshooting

### Low Memory Warning
If you see a memory warning:
1. Close other applications to free memory
2. The app will continue with limited AI functionality
3. Core features (charts, data) will still work

### Port Already in Use
If port 8501 is busy:
1. The startup script will automatically kill existing processes
2. If issues persist, manually run: `lsof -ti:8501 | xargs kill -9`

### AI Models Not Loading
If AI models fail to load:
1. Check available memory (need ~4GB free)
2. The app continues with basic functionality
3. All non-AI features remain available

## System Requirements

- **Memory**: 8GB RAM minimum (16GB recommended)
- **Free Memory**: 4GB for full AI functionality
- **Python**: 3.8 or higher
- **OS**: macOS (Apple Silicon optimized), Linux, Windows

## Features Available Without AI

Even if AI models don't load, you still have:
- Real-time stock data and charts
- Technical indicators
- Portfolio tracking
- Watchlist management
- Basic analysis tools

## Contact & Support

For issues or questions:
- Check the logs in the terminal
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Restart using the safe startup script