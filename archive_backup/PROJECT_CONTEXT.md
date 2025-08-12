# QuantumConsensus Project Context

## 🎯 Current State

**Status**: Clean, functional codebase ready for development  
**Last Updated**: August 2025  
**Version**: QuantumConsensus v1.0 (rebranded from River Trading System)

## 📝 What's Working

✅ **Core Platform**: `quantum_consensus_app.py` - Main Streamlit app with Robinhood-style UI  
✅ **Multi-Agent System**: 6 quantum-powered trading agents with improved scoring (-10 to +10)  
✅ **Real-time Data**: Yahoo Finance integration with force refresh for accurate prices  
✅ **Advanced Charts**: Plotly charts with RSI, moving averages, volume indicators  
✅ **Watchlist**: Add/remove stocks with persistence and expandable analytics  
✅ **Safe Launcher**: `quantum_start.py` handles MLX crashes and process cleanup  
✅ **Kelly Criterion**: Position sizing agent with backtesting framework  

## 🔧 Recent Fixes

- **Agent Scoring**: Fixed low scores (-0.50) by expanding range to full -10 to +10
- **Watchlist Pricing**: Fixed $0.00 prices with force_refresh=True  
- **Button Visibility**: Fixed dark mode color issues with enhanced CSS
- **MLX Crashes**: Created safe mode to prevent segmentation faults on Apple Silicon
- **HTML Rendering**: Replaced raw HTML with native Streamlit components
- **Rebranding**: Complete QuantumConsensus rebrand across entire platform

## 🚨 Known Issues (Fixed)

- ❌ **app_safe.py**: Had critical indentation errors (line 490+) - NOT INCLUDED in clean version
- ❌ **Complex UI**: Original had too many nested components - Simplified in clean version
- ❌ **Process Management**: Dangling processes - Fixed with quantum_start.py cleanup

## 🏗️ Architecture

### Key Files
1. **quantum_consensus_app.py** - Main application (working version from app.py)
2. **quantum_start.py** - Safe launcher with cleanup and MLX prevention  
3. **src/agents/trading_agents.py** - 6 AI agents with quantum branding
4. **src/trading/backtester.py** - Kelly Criterion backtesting framework
5. **src/data/data_fetcher.py** - Yahoo Finance integration

### Agent System
```
QuantumValueInvestor      - Fundamental analysis (P/E, market cap)
QuantumTechnicalMomentum  - RSI, MACD, moving averages  
QuantumSentimentAnalyzer  - Market sentiment analysis
QuantumRiskManager       - Volatility and risk assessment
QuantumPatternHunter     - Chart pattern recognition  
QuantumKellyCriterion    - Optimal position sizing
```

## 🎨 UI Features

- **Robinhood-style Design**: Clean, Apple-inspired interface
- **Dark Mode Support**: Proper color contrast and visibility
- **Multi-page Navigation**: Chat, Analysis, Lists, Portfolio
- **Advanced Analytics**: Expandable sections with technical indicators
- **Real-time Updates**: Live price data with color-coded changes
- **Responsive Charts**: Interactive Plotly charts with overlays

## 🔮 Next Development Steps

### Immediate Priorities
1. **Test Clean Codebase**: Ensure all features work in clean environment
2. **Agent Validation**: Manual testing of all 6 agent outputs  
3. **Backtesting Integration**: Add backtesting buttons to agent recommendations
4. **Error Handling**: Improve robustness across all modules

### Future Enhancements  
1. **Portfolio Optimization**: Modern Portfolio Theory integration
2. **Real-time Alerts**: Price and technical indicator notifications
3. **Options Analysis**: Options chain and strategy analysis
4. **Advanced Backtesting**: Multi-timeframe portfolio backtesting

## 💡 Development Notes

### For Context Restoration
When resuming development, read this file first, then:
1. Review `TECHNICAL_README.md` for architecture details
2. Check `quantum_consensus_app.py` for current UI implementation  
3. Examine `src/agents/trading_agents.py` for agent logic
4. Use `quantum_start.py` for safe app launching

### Performance Optimizations
- Use `quantum_start.py` to prevent MLX crashes on Apple Silicon
- Force refresh data fetching for accurate prices
- Process cleanup prevents memory leaks
- Streamlit caching for data persistence

### Code Quality
- All agents use full -10 to +10 scoring range
- Consistent QuantumConsensus branding throughout
- Apple-style UI design principles
- Robust error handling with graceful degradation

## 🛡️ Safety Features

- **MLX Protection**: Automatic disabling of GPU processes that cause crashes
- **Process Cleanup**: Kills dangling Streamlit and Python processes  
- **Memory Management**: Garbage collection and cache clearing
- **Safe Restart**: Environment variable configuration for stability

---

**Last Known Working State**: Clean codebase with functional Robinhood-style UI, 6 quantum agents, and Kelly Criterion backtesting. Ready for continued development and testing.