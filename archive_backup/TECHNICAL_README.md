# QuantumConsensus Trading Platform - Technical Documentation

## 🌟 Project Overview

QuantumConsensus is a multi-agent AI trading analysis platform that leverages quantum-inspired consensus algorithms to provide comprehensive stock analysis and trading recommendations. The platform combines multiple specialized AI agents to deliver robust investment insights.

## 🏗️ System Architecture

### Core Components

1. **quantum_consensus_app.py** - Main Streamlit application
   - Robinhood-style UI with dark mode support
   - Multi-page interface: Chat, Analysis, Lists (Watchlist), Portfolio
   - Real-time stock data integration with Yahoo Finance
   - Advanced charting with technical indicators (RSI, Moving Averages)
   - Agent-powered analysis with scoring system

2. **quantum_start.py** - Safe application launcher
   - Handles process cleanup and memory management
   - GPU/MLX crash prevention for Apple Silicon
   - Environment variable configuration
   - Automatic restart capabilities

3. **src/** - Core business logic modules
   - `agents/` - Multi-agent trading intelligence system
   - `trading/` - Backtesting, position sizing, signals
   - `data/` - Market data fetching and processing
   - `ai/` - LLM integration and reasoning
   - `utils/` - Data validation, alerts, reporting

## 🤖 Multi-Agent System

### Agent Architecture (src/agents/trading_agents.py)

The platform uses 6 specialized quantum-powered agents:

1. **Quantum Value Investor** - Fundamental analysis with P/E, market cap evaluation
2. **Quantum Technical Momentum** - RSI, MACD, moving average analysis  
3. **Quantum Sentiment Analyzer** - Market sentiment and news analysis
4. **Quantum Risk Manager** - Volatility and risk assessment
5. **Quantum Pattern Hunter** - Chart pattern recognition
6. **Quantum Kelly Criterion** - Optimal position sizing using Kelly formula

### Agent Scoring System
- Each agent provides recommendations with confidence scores (-10 to +10)
- Scores aggregate into consensus recommendations
- Improved scoring logic prevents artificially low scores
- Full utilization of scoring range for better differentiation

## 📊 Key Features

### Real-time Analysis
- **Live price data** - Yahoo Finance integration with force refresh
- **Technical indicators** - RSI, MACD, SMA (20/50/200 day)
- **Volume analysis** - Volume spikes and confirmation signals
- **Market metrics** - Market cap, P/E ratio, Beta, volatility

### Interactive UI Components
- **Watchlist management** - Add/remove stocks with persistence
- **Advanced charting** - Plotly-based charts with overlays
- **Agent chat interface** - Direct interaction with AI agents
- **Portfolio tracking** - Position monitoring and P&L

### Technical Indicators
- **RSI (Relative Strength Index)** - Overbought/oversold levels (30/70)
- **Moving Averages** - 20, 50, 200-day SMAs with trend analysis
- **Volume confirmation** - Volume spike detection (2x+ average)
- **Volatility metrics** - 20-day rolling volatility analysis

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Quick Start
```bash
# Start the platform safely
python quantum_start.py

# Or run directly (development mode)
streamlit run quantum_consensus_app.py --server.port 8501
```

### Configuration
- **quantum_settings.json** - Application settings and preferences
- Environment variables for API keys (if using premium data sources)
- Agent parameters can be tuned in `src/agents/trading_agents.py`

## 📁 Directory Structure

```
quantum_consensus_clean/
├── quantum_consensus_app.py      # Main Streamlit application
├── quantum_start.py              # Safe launcher with cleanup
├── requirements.txt              # Python dependencies
├── quantum_settings.json         # App configuration
├── data/                        # Sample market data
│   ├── AAPL_daily.json
│   ├── MSFT_daily.json
│   └── TSLA_daily.json
└── src/                         # Core modules
    ├── agents/                  # Multi-agent system
    │   └── trading_agents.py    # 6 quantum trading agents
    ├── trading/                 # Trading logic
    │   ├── backtester.py        # Backtesting framework
    │   ├── position_sizer.py    # Kelly Criterion implementation
    │   ├── signals.py           # Technical signal generation
    │   └── config.py            # Trading parameters
    ├── data/                    # Data handling
    │   └── data_fetcher.py      # Yahoo Finance integration
    ├── utils/                   # Utilities
    │   ├── data_ingestion.py    # Data processing
    │   └── alerts.py            # Alert system
    └── ai/                      # AI/LLM integration
        └── llm_reasoner.py      # LLM reasoning engine
```

## 🛠️ Development Guide

### Agent Development
To add new agents, modify `src/agents/trading_agents.py`:
```python
class QuantumNewAgent(BaseAgent):
    def analyze(self, symbol, market_data, context):
        # Analysis logic here
        return {
            'recommendation': 'BUY/HOLD/SELL',
            'confidence': 0.85,  # 0.0 to 1.0
            'score': 7.5,        # -10 to +10
            'reasoning': 'Analysis explanation'
        }
```

### UI Customization
The UI uses Apple-style design principles with dark mode support:
- Colors: Consistent with Apple's design language
- Typography: -apple-system font stack
- Spacing: Grid-based layout with proper padding
- Charts: Plotly with custom styling

### Data Integration
Market data is fetched via `src/data/data_fetcher.py`:
- Primary: Yahoo Finance (free, reliable)
- Fallback: Local cached data
- Real-time updates with configurable refresh intervals

## 🔬 Backtesting Framework

### Features (src/trading/backtester.py)
- **Strategy evaluation** - Compare baseline vs AI-enhanced strategies
- **Kelly Criterion** - Optimal position sizing based on win probability
- **Performance metrics** - Sharpe ratio, max drawdown, CAGR
- **Trade simulation** - Stop-loss and take-profit execution

### Usage Example
```python
from src.trading.backtester import backtest_ticker

# Backtest a strategy
results = backtest_ticker(
    ticker="AAPL",
    strategy="llm",  # or "baseline"
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=100000
)
```

## 🔍 Troubleshooting

### Common Issues

1. **MLX/GPU Crashes on Apple Silicon**
   - Use `quantum_start.py` which disables MLX
   - Set environment variable: `MLX_DISABLE=1`

2. **Data Fetching Errors**
   - Check internet connection
   - Yahoo Finance may have rate limits
   - Use cached data in `data/` folder as fallback

3. **Agent Scoring Issues**
   - Agents now use full -10 to +10 range
   - Low scores may indicate genuine negative sentiment
   - Check `src/agents/trading_agents.py` for scoring logic

4. **UI Display Issues**
   - Dark mode styling issues: Check CSS in app file
   - Button visibility: Ensured with proper color contrast
   - Charts not loading: Verify Plotly installation

### Performance Optimization
- **Memory management** - `quantum_start.py` handles cleanup
- **Process isolation** - Kills dangling Streamlit processes
- **GPU resource management** - Disables unnecessary GPU usage
- **Caching** - Uses Streamlit's caching for data persistence

## 🚀 Future Enhancements

### Planned Features
1. **Advanced Backtesting** - Multi-timeframe and portfolio-level backtesting
2. **Real-time Alerts** - Price and technical indicator alerts
3. **Portfolio Optimization** - Modern Portfolio Theory integration
4. **Options Analysis** - Options chain analysis and strategies
5. **Risk Management** - Position sizing and portfolio risk controls

### Technical Debt
1. **Error Handling** - More robust error handling across modules
2. **Testing** - Comprehensive unit and integration tests
3. **Documentation** - API documentation for all modules
4. **Configuration** - More flexible configuration management

## 📝 Code Quality

### Standards
- **Type hints** - All functions should include type annotations
- **Docstrings** - Google-style docstrings for all classes/functions
- **Error handling** - Graceful degradation with user-friendly messages
- **Logging** - Structured logging for debugging and monitoring

### Key Files for New Developers
1. **Start here**: `quantum_consensus_app.py` - Main application logic
2. **Agent system**: `src/agents/trading_agents.py` - Core AI agents
3. **Data pipeline**: `src/data/data_fetcher.py` - Market data handling
4. **Trading logic**: `src/trading/` - Backtesting and signals

## 🎯 Quick Development Workflow

1. **Start development server**: `python quantum_start.py`
2. **Make changes**: Edit relevant files in `src/` or main app
3. **Test changes**: Restart app automatically cleans processes
4. **Debug**: Check browser console and terminal output
5. **Deploy**: Use same `quantum_start.py` for production

---

**Note**: This platform is designed for educational and research purposes. Always consult with financial professionals before making investment decisions.