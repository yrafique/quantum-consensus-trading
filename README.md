# 🚀 QuantumConsensus — Apple‑Level Trading Intelligence

> **Beautiful. Explainable. Local‑first.**  
> A modern Streamlit platform unifying quant math, MLX‑powered local LLM reasoning, and elegant dashboards for research‑grade trading decisions.

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
  [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#-license)
  
</div>

<div align="center">
  <a href="#-features">Features</a> • 
  <a href="#-screens--explanations">Screens</a> • 
  <a href="#-quickstart">Quickstart</a> • 
  <a href="#-architecture">Architecture</a> •
  <a href="#-technical-documentation">Technical Docs</a> •
  <a href="#-development">Development</a>
</div>

---

## ✨ Features

### 🎯 Core Capabilities

- **🧠 Advanced AI Analysis** — MLX‑optimized Llama 3.1 for sub‑200ms local inference
- **📊 Professional Charting** — 20+ technical indicators with institutional‑grade visualizations  
- **🤖 Multi‑Agent Consensus** — 6 specialized trading agents with quantum voting system
- **💼 Portfolio Management** — Real‑time P&L tracking with Robinhood‑style clarity
- **🔒 Privacy First** — 100% local processing, your data never leaves your machine
- **⚡ Apple Silicon Native** — Optimized for M1/M2/M3 with Metal acceleration

### 🎨 Design Philosophy

- **Streamlit‑native components** for reliability and speed
- **Dark theme optimized** with glass‑morphism effects  
- **Zero configuration** — works out of the box
- **Responsive layouts** that adapt to any screen size

---

## 🖼 Screens & Explanations

### 📈 Advanced Technical Analysis
<p align="center">
  <img src="assets/ui-advanced-aapl-chart.png" alt="AAPL advanced chart with SMA/EMA, Bollinger Bands, RSI, MACD, volume" width="90%">
</p>

**Multi‑pane technical dashboard** featuring:
- Price action with SMA‑20/50/200 and EMA‑21 overlays
- Bollinger Bands for volatility analysis
- RSI/MACD oscillators in synchronized sub‑panels
- Volume profile with anomaly detection
- **Single‑screen situational awareness** optimized for rapid decision making

<details>
<summary><b>📖 How to Add/Customize Indicators</b></summary>

#### Adding New Indicators to Charts

```python
# In quantum_consensus_app.py - Lists page section
# Location: Lines ~1200-1300

# Add checkbox for your indicator
show_custom = st.checkbox("📊 Custom Indicator", value=False, key=f"custom_{symbol}")

# Calculate the indicator
if show_custom:
    chart_df['custom_indicator'] = calculate_custom_indicator(chart_df)
    
    # Add to chart
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df['custom_indicator'],
            mode='lines',
            name='Custom Indicator',
            line=dict(color='#00ff00', width=1.5)
        ),
        row=2, col=1  # Specify subplot position
    )
```

#### Available Technical Indicators
- **Moving Averages**: SMA (20, 50, 200), EMA (21)
- **Oscillators**: RSI (14), MACD (12,26,9), Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume bars, OBV, Volume SMA
- **Custom**: Easy to add any TA-lib indicator

</details>

---

### 🧪 Deep AI Analysis Pipeline
<p align="center">
  <img src="assets/ui-deep-ai-analysis.png" alt="Deep AI analysis panel with analysis modes and debug checklist" width="90%">
</p>

**Transparent AI reasoning** with:
- Analysis modes: `fast` (< 1s), `comprehensive` (full scan), `consensus` (multi‑agent), `real_time` (streaming)
- **Full pipeline visibility**: Data validation → Technical indicators → Agent reasoning → Risk gates → Final recommendation
- **Explainable AI** — see exactly why each decision was made
- Debug mode for complete reasoning traces

---

### 📊 Smart Watchlist Dashboard
<p align="center">
  <img src="assets/ui-watchlist-advanced-chart.png" alt="Watchlist dashboard with candlesticks and moving averages" width="90%">
</p>

**Frictionless monitoring** featuring:
- Quick‑add symbols with autocomplete
- Live quotes with 5‑minute refresh
- Candlestick charts with volume overlays
- **Technical summary cards**: "Above SMA20/50", "RSI Overbought", "MACD Bullish Cross"
- One‑click deep analysis activation

---

### 🤖 Multi‑Agent Trading Intelligence
<p align="center">
  <img src="assets/ui-agent-selector.png" alt="Agent selector with specialized trading strategies" width="90%">
</p>

**6 Specialized Trading Agents**:

| Agent | Strategy | Indicators | Best For |
|-------|----------|------------|----------|
| 📈 **Technical Momentum** | Trend following | RSI, MACD, Moving Averages | Trending markets |
| 🚀 **Short Squeeze Hunter** | Squeeze detection | Short Interest, Borrow Rate, Volume | High volatility plays |
| 🧠 **Quantum Multi‑Factor** | Neural reasoning | 15+ factors, ML scoring | Complex decisions |
| 💎 **Value Investor** | Fundamental analysis | P/E, PEG, Revenue Growth | Long‑term positions |
| 📊 **Kelly Criterion** | Optimal sizing | Win rate, Risk‑reward | Position management |
| 🎯 **Sentiment Analyzer** | Market sentiment | News, Social media | Sentiment shifts |

---

### 💼 Portfolio Excellence
<p align="center">
  <img src="assets/ui-portfolio.png" alt="Portfolio page with total value card and holdings list" width="90%">
</p>

**Institution‑grade portfolio tracking**:
- **Hero metrics** — Total value, Day change, Total gain, Buying power
- **Smart holdings list** — Cost basis, Current value, P&L, Allocation %
- **Performance attribution** — See what's driving returns
- **Risk analytics** — Portfolio beta, Sharpe ratio, Max drawdown

---

## ⚡ Quickstart

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) for MLX acceleration
- **Python 3.9+**
- **8GB RAM** minimum (16GB recommended for AI features)

### Installation

```bash
# Clone the repository
git clone https://github.com/yrafique/quantum-consensus-trading.git
cd quantum-consensus-trading

# Install dependencies
pip3 install -r requirements.txt

# (Optional) Install MLX for Apple Silicon acceleration
pip3 install mlx mlx-lm

# Run the application
python3 quantum_start.py
```

### First Run

1. **Open your browser** to http://localhost:8501
2. **Add symbols** to your watchlist (try AAPL, NVDA, TSLA)
3. **Select an agent** on the AI Trading Agents page
4. **Enable Debug Mode** to see full reasoning traces
5. **Analyze a stock** by entering a symbol and clicking "Analyze"

---

## 🏗 Architecture

### System Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit UI]
        Charts[Plotly Charts]
        RT[Real-time Updates]
    end
    
    subgraph "Intelligence Layer"
        Router[Intelligent Router]
        Agents[6 Trading Agents]
        MLX[MLX LLM Engine]
        RAG[RAG Advisor]
    end
    
    subgraph "Data Layer"
        YF[Yahoo Finance]
        Cache[Smart Cache]
        DB[(SQLite DB)]
    end
    
    UI --> Router
    Router --> Agents
    Agents --> MLX
    Agents --> RAG
    MLX --> YF
    RAG --> Cache
    Cache --> DB
```

### Core Components

| Component | Purpose | Technology | Location |
|-----------|---------|------------|----------|
| **Unified Analyzer** | Orchestrates all analysis | AsyncIO, Circuit Breakers | `src/core/unified_analyzer.py` |
| **Agent Manager** | Routes queries to agents | LangGraph, Type Safety | `src/agents/trading_agents.py` |
| **MLX Engine** | Local LLM inference | Apple MLX, Llama 3.1 | `src/ai/mlx_trading_llm.py` |
| **Data Fetcher** | Real-time & historical data | yfinance, 5-min cache | `src/data/data_fetcher.py` |
| **Risk Manager** | Position sizing & risk gates | Kelly Criterion, VaR | `src/trading/position_sizer.py` |

### File Structure

<details>
<summary><b>📁 Complete Project Structure</b></summary>

```
quantum_consensus_trading/
├── quantum_consensus_app.py      # Main Streamlit application
├── quantum_start.py              # Safe launcher with MLX protection
├── quantum_settings.json         # Application configuration
├── requirements.txt              # Python dependencies
├── README.md                     # This comprehensive guide
├── CLAUDE.md                     # AI assistant context
├── assets/                       # UI screenshots
│   ├── ui-advanced-aapl-chart.png
│   ├── ui-agent-selector.png
│   ├── ui-deep-ai-analysis.png
│   ├── ui-portfolio.png
│   └── ui-watchlist-advanced-chart.png
├── data/                        # Runtime data directory
│   └── market_data.db          # SQLite cache database
├── logs/                        # Application logs
│   └── quantum_consensus.log   # Main log file
└── src/                         # Core source modules
    ├── agents/                  # Multi-Agent Trading System
    │   ├── trading_agents.py    # 6 Quantum trading agents
    │   ├── intelligent_router.py # Agent orchestration
    │   ├── langgraph_integration.py # LangGraph workflow
    │   └── react_trading_agent.py # ReAct pattern
    ├── ai/                      # AI/LLM Integration
    │   ├── mlx_trading_llm.py   # Apple Silicon optimized
    │   ├── llm_reasoner.py      # LLM reasoning coordinator
    │   ├── local_llm.py         # Heuristic fallback
    │   ├── rag_advisor.py       # RAG implementation
    │   └── safe_ai_loader.py    # Memory-safe loading
    ├── trading/                 # Trading Engine
    │   ├── backtester.py        # Strategy backtesting
    │   ├── position_sizer.py    # Kelly Criterion
    │   ├── signals.py           # Signal generation
    │   ├── portfolio.py         # Portfolio management
    │   └── alpaca_trader.py     # Live trading (future)
    ├── data/                    # Data Pipeline
    │   └── data_fetcher.py      # Yahoo Finance + caching
    ├── core/                    # Infrastructure
    │   ├── config.py            # Configuration management
    │   ├── monitoring.py        # Metrics collection
    │   ├── resilience.py        # Circuit breakers
    │   ├── logging_config.py    # Structured logging
    │   ├── unified_analyzer.py  # Master orchestrator
    │   └── exceptions.py        # Exception handling
    ├── api/                     # REST API Layer
    │   ├── main.py              # FastAPI application
    │   ├── dependencies.py      # Dependency injection
    │   ├── middleware.py        # Security middleware
    │   └── routers/             # API endpoints
    └── utils/                   # Utilities
        ├── data_validator.py     # Data validation
        ├── alerts.py            # Alert management
        └── reports.py           # Report generation
```

</details>

---

## 📚 Technical Documentation

### 🤖 Agent System Deep Dive

<details>
<summary><b>Quantum Technical Momentum Agent</b></summary>

**Location**: `src/agents/trading_agents.py:61-246`

```python
class QuantumTechnicalMomentumAgent(TradingAgent):
    """Pure technical analysis focused on momentum and trend following"""
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False):
        # RSI Analysis with weighted scoring
        if rsi < 25:
            signals.append("🟢 RSI extremely oversold - strong reversal signal")
            raw_score += 3.5
        elif rsi < 30:
            signals.append("🟢 RSI oversold - potential bounce")
            raw_score += 2.5
            
        # Moving Average Analysis 
        if price > sma_20 > sma_50 and ma_short_diff > 5:
            signals.append(f"🟢 Strong uptrend - {ma_short_diff:+.1f}% above 20MA")
            raw_score += 3
```

**Scoring Framework**:
- RSI Momentum: -2.5 to +3.5 points
- Moving Average Confluence: -3 to +3 points
- Volume Confirmation: 0 to +2 points
- MACD Crossovers: -2 to +2 points
- Price Action: -1 to +1 points

</details>

<details>
<summary><b>Quantum Short Squeeze Hunter Agent</b></summary>

**Location**: `src/agents/trading_agents.py:247-363`

```python
class QuantumShortSqueezeHunterAgent(TradingAgent):
    """Specialized in detecting short squeeze opportunities"""
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False):
        # Volume spike analysis
        if volume_spike > 3:
            signals.append(f"🔥 Massive volume spike ({volume_spike:.1f}x average)")
            squeeze_score += 3
            
        # Price momentum analysis
        if price_change_1d > 10:
            signals.append(f"🚀 Strong daily move (+{price_change_1d:.1f}%)")
            squeeze_score += 2
```

**Detection Signals**:
- Volume Spike: 1.5x to 3x+ average volume
- Price Momentum: 1-day and 5-day analysis
- RSI Overbought: 60-80 RSI range
- Short Interest: >20% threshold
- Days to Cover: >1.5 days

</details>

<details>
<summary><b>Quantum Kelly Criterion Agent</b></summary>

**Location**: `src/agents/trading_agents.py:590-806`

```python
class QuantumKellyCriterionAgent(TradingAgent):
    """Kelly Criterion optimal position sizing trader"""
    
    def calculate_kelly_fraction(self, win_prob: float, reward_risk: float) -> float:
        """
        Kelly Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to wager
        - b = odds (reward/risk ratio)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        if reward_risk <= 0:
            return 0.0
        q = 1.0 - win_prob
        numerator = win_prob * reward_risk - q
        if numerator <= 0:
            return 0.0
        return numerator / reward_risk
```

**Position Sizing**:
- Kelly Formula: `f = (bp - q) / b`
- Scaling Factor: 0.25x for safety
- Position Limits: 1-10% of portfolio
- Risk Management: Zero allocation for negative EV

</details>

### 🧠 AI/LLM Architecture

<details>
<summary><b>MLX Trading LLM - Apple Silicon Optimization</b></summary>

**Location**: `src/ai/mlx_trading_llm.py`

```python
class MLXTradingLLM:
    """Ultra-fast trading LLM powered by Apple's MLX framework."""
    
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        self.model, self.tokenizer = load(self.model_name)
        
        # Test inference speed
        start_time = datetime.now()
        test_response = self._generate_text("Test", max_tokens=10)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        print(f"   Inference Speed: {inference_time:.0f}ms per response")
```

**Performance Characteristics**:
- **Sub-200ms Response**: Apple Silicon unified memory
- **4-bit Quantization**: Memory-efficient loading
- **Zero API Costs**: Complete local inference
- **Fallback Chain**: MLX → Transformers → Heuristic

</details>

<details>
<summary><b>Intelligent Router System</b></summary>

**Location**: `src/agents/intelligent_router.py`

```python
class IntelligentRouter:
    """Enterprise-grade routing system with ML capabilities"""
    
    def __init__(self):
        self.routing_strategies = {
            "intent_based": self._intent_based_routing,
            "performance_based": self._performance_based_routing,
            "consensus_based": self._consensus_based_routing,
            "adaptive": self._adaptive_routing
        }
```

**Routing Intelligence**:
- Intent Detection for agent selection
- Performance tracking & metrics
- Consensus building algorithms
- Adaptive weight adjustment

</details>

### 📊 Trading Engine

<details>
<summary><b>Signal Generation System</b></summary>

**Location**: `src/trading/signals.py`

```python
def evaluate_signals(ticker: str, df: pd.DataFrame, 
                    short_float: float, days_to_cover: float) -> SignalResult:
    """Evaluate all signal criteria for a given ticker."""
    
    results = {
        "rsi_momentum": _check_rsi_momentum(df),
        "price_crossover": _check_price_crossover(df),
        "short_squeeze": _check_short_squeeze(short_float, days_to_cover),
        "volume_spike": _check_volume_spike(df),
        "bullish_engulfing": _check_bullish_engulfing(df),
    }
    
    # Require both momentum and confirmation signals
    momentum_signals = results["rsi_momentum"] or results["price_crossover"]
    confirmation_signals = (
        results["short_squeeze"] or 
        results["volume_spike"] or 
        results["bullish_engulfing"]
    )
    passes = momentum_signals and confirmation_signals
```

**Signal Criteria**:
1. **RSI Momentum**: >65 threshold with positive slope
2. **Price Crossover**: EMA21 or VWAP breakout
3. **Short Squeeze**: >20% short float, >1.5 days to cover
4. **Volume Spike**: 3x average volume
5. **Bullish Engulfing**: Candlestick pattern

</details>

<details>
<summary><b>Backtesting Framework</b></summary>

**Location**: `src/trading/backtester.py`

```python
def backtest_ticker(
    ticker: str,
    strategy: str = "baseline",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 100_000.0,
) -> Dict[str, object]:
    """Backtest a single ticker using the specified strategy."""
    
    if strategy == "baseline":
        size = capital * 0.10  # Fixed 10% position size
        stop = entry * 0.98    # 2% stop loss
        target = entry * 1.05  # 5% profit target
        
    elif strategy == "llm":
        rec = llm_reasoner.generate_recommendation(ticker, context)
        confidence = rec.get("confidence", 0.9)
        fraction = position_sizer.compute_position_fraction(
            win_prob=confidence,
            reward_to_risk=reward_to_risk,
        )
        size = capital * fraction
```

**Performance Metrics**:
- Total Return & CAGR
- Sharpe Ratio (2% risk-free rate)
- Maximum Drawdown
- Win Rate & Profit Factor
- Risk-adjusted returns

</details>

### 💾 Data Pipeline

<details>
<summary><b>Market Data Fetcher</b></summary>

**Location**: `src/data/data_fetcher.py`

```python
class MarketDataFetcher:
    """Centralized market data fetcher with SQLite caching"""
    
    def fetch_ohlcv(self, symbol: str, period: str = "1mo", 
                    interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        # Three-tier caching strategy
        # 1. Memory cache (fastest)
        if not force_refresh and cache_key in self._cache:
            if datetime.now() - last_fetch < CACHE_DURATION:
                return self._cache[cache_key]
        
        # 2. Database cache (persistent)
        if not force_refresh:
            db_data = self._fetch_from_database(symbol, period, interval)
            if db_data is not None:
                return db_data
        
        # 3. Yahoo Finance API (source of truth)
        fresh_data = self._fetch_from_yahoo(symbol, period, interval)
        self._store_in_database(fresh_data)
        return fresh_data
```

**Caching Architecture**:
- Memory Cache: 5-minute TTL
- SQLite Database: Persistent storage
- Intelligent refresh logic
- Automatic technical indicators

</details>

---

## 🛠 Configuration

### Environment Variables

```bash
# MLX Configuration (Apple Silicon)
export MLX_LAZY_LOAD=1
export MLX_LLM_PATH=~/models/llama31-8b-instruct
export MLX_DISABLE=0  # Set to 1 to disable MLX

# Streamlit Configuration
export STREAMLIT_THEME="dark"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true

# Trading Configuration
export MAX_PORTFOLIO_RISK=0.06      # 6% max risk
export KELLY_SCALING_FACTOR=0.25    # Conservative Kelly
export CACHE_DURATION=300            # 5 minutes
```

### Settings File (`quantum_settings.json`)

```json
{
  "theme": "dark",
  "default_agent": "quantum_multi_factor",
  "cache_duration": 300,
  "debug_mode": false,
  "indicators": ["RSI", "MACD", "BB", "SMA", "EMA"],
  "risk_management": {
    "max_portfolio_risk": 0.06,
    "max_positions": 10,
    "min_trade_size": 100,
    "kelly_scaling": 0.25
  },
  "signal_thresholds": {
    "rsi_momentum": 65,
    "volume_spike": 3.0,
    "short_float": 20.0,
    "days_to_cover": 1.5
  }
}
```

---

## 👩‍💻 Development

### Adding New Trading Agents

<details>
<summary><b>Agent Template & Integration</b></summary>

```python
# Template for new trading agent
class QuantumCustomAgent(TradingAgent):
    def __init__(self):
        super().__init__(
            name="🎯 Quantum Custom Agent",
            description="Your agent description",
            strategy="custom_strategy",
            agent_type="quantum_custom"
        )
    
    def analyze(self, symbol: str, user_query: str = "", 
                debug_mode: bool = False) -> Dict[str, Any]:
        # Fetch market data
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_ohlcv(symbol, period="1mo")
        
        # Implement your analysis logic
        signals = []
        score = 0
        
        # Your custom analysis here
        if your_condition:
            signals.append("📈 Bullish signal detected")
            score += 2
        
        # Return standardized format
        return {
            'recommendation': 'BUY',  # BUY/HOLD/SELL
            'confidence': 0.85,       # 0.0 to 1.0
            'score': score,           # -10 to +10
            'reasoning': 'Detailed explanation',
            'signals': signals,
            'risk_factors': ['List risks'],
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'data_points': len(df)
            }
        }
```

**Integration Steps**:
1. Add agent class to `src/agents/trading_agents.py`
2. Register in `get_all_agents()` function
3. Add to agent selector UI in main app
4. Test with sample symbols

</details>

### Adding Custom Technical Indicators

<details>
<summary><b>Indicator Development Guide</b></summary>

```python
# Step 1: Create indicator calculation function
def calculate_custom_indicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate your custom technical indicator"""
    
    # Example: Custom momentum oscillator
    returns = df['close'].pct_change()
    momentum = returns.rolling(window=period).mean()
    normalized = (momentum - momentum.mean()) / momentum.std()
    
    return normalized

# Step 2: Add to data fetcher (src/data/data_fetcher.py)
def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe"""
    
    # Existing indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    # Add your custom indicator
    df['custom_indicator'] = calculate_custom_indicator(df)
    
    return df

# Step 3: Add UI controls (quantum_consensus_app.py)
# In the Lists page chart section (~line 1200)
show_custom = st.checkbox("📊 Custom Indicator", value=False, 
                          key=f"custom_{symbol}")

if show_custom and 'custom_indicator' in chart_df.columns:
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df['custom_indicator'],
            mode='lines',
            name='Custom Indicator',
            line=dict(color='#00ff00', width=1.5),
            yaxis='y3'  # Use separate y-axis if needed
        ),
        row=3, col=1  # Add to specific subplot
    )
```

</details>

### API Integration

<details>
<summary><b>REST API Endpoints</b></summary>

**Base URL**: `http://localhost:8000/api/v1`

#### Analysis Endpoints

```python
# Multi-agent analysis
POST /analysis/agents
{
    "symbol": "AAPL",
    "agents": ["momentum", "short_squeeze"],
    "mode": "consensus"
}

# Get trading signals
GET /analysis/signals/{symbol}

# Backtest strategy
POST /analysis/backtest
{
    "symbol": "NVDA",
    "strategy": "llm",
    "start_date": "2024-01-01",
    "initial_capital": 100000
}
```

#### Trading Endpoints

```python
# Calculate position size
POST /trading/position_size
{
    "symbol": "TSLA",
    "win_probability": 0.65,
    "reward_risk_ratio": 2.5
}

# Get portfolio analytics
GET /trading/portfolio

# Place order (Alpaca integration)
POST /trading/orders
{
    "symbol": "AAPL",
    "quantity": 100,
    "side": "buy",
    "order_type": "limit",
    "limit_price": 150.00
}
```

</details>

### Testing & Quality

<details>
<summary><b>Testing Framework</b></summary>

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/agents/
pytest tests/trading/
pytest tests/api/

# Run with coverage
pytest --cov=src tests/

# Run performance tests
pytest tests/performance/ --benchmark
```

**Test Structure**:
```
tests/
├── agents/
│   ├── test_momentum_agent.py
│   ├── test_squeeze_hunter.py
│   └── test_kelly_criterion.py
├── trading/
│   ├── test_signals.py
│   ├── test_backtester.py
│   └── test_position_sizer.py
├── api/
│   ├── test_endpoints.py
│   └── test_middleware.py
└── integration/
    ├── test_full_pipeline.py
    └── test_consensus.py
```

</details>

---

## 🚀 Advanced Features

### Quantum Consensus Algorithm

<details>
<summary><b>Mathematical Framework</b></summary>

```python
def quantum_consensus(agent_results: List[AgentResult]) -> ConsensusResult:
    """
    Quantum consensus algorithm using weighted confidence scoring
    
    The algorithm treats each agent recommendation as a quantum state
    that contributes to the final superposition until measurement (decision)
    """
    # Weight by confidence and recent performance
    weighted_scores = []
    total_weight = 0
    
    for result in agent_results:
        # Calculate agent weight based on:
        # 1. Current confidence (0-1)
        # 2. Historical accuracy (0-1)
        # 3. Market regime fitness (0-1)
        weight = (
            result.confidence * 0.4 +
            result.historical_accuracy * 0.4 +
            result.regime_fitness * 0.2
        )
        weighted_scores.append(result.score * weight)
        total_weight += weight
    
    # Calculate consensus score
    consensus_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
    
    # Quantum uncertainty calculation
    variance = sum((score - consensus_score)**2 * weight 
                   for score, weight in zip(scores, weights))
    uncertainty = math.sqrt(variance / total_weight) if total_weight > 0 else 1.0
    
    # Decision mapping
    if consensus_score >= 7:
        action = "STRONG_BUY"
    elif consensus_score >= 4:
        action = "BUY"
    elif consensus_score >= -2:
        action = "HOLD"
    elif consensus_score >= -6:
        action = "SELL"
    else:
        action = "STRONG_SELL"
    
    return {
        "consensus_score": consensus_score,
        "uncertainty": uncertainty,
        "confidence": 1.0 - uncertainty,
        "action": action,
        "agent_contributions": agent_results
    }
```

</details>

### Performance Optimization

<details>
<summary><b>System Performance Tuning</b></summary>

#### Inference Speed Optimization
- **MLX LLM**: <200ms response on Apple Silicon
- **Heuristic Fallback**: <50ms rule-based analysis
- **Data Caching**: 5-minute intelligent cache
- **Parallel Processing**: Async agent execution

#### Memory Management
```python
# In quantum_start.py
def check_memory_available():
    """Ensure sufficient memory for MLX models"""
    import psutil
    
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 4:
        print(f"⚠️ Low memory: {available_gb:.1f}GB available")
        os.environ['MLX_DISABLE'] = '1'
        return False
    return True
```

#### Database Optimization
```sql
-- Create indexes for faster queries
CREATE INDEX idx_symbol_timestamp ON ohlcv_data(symbol, timestamp);
CREATE INDEX idx_indicators ON technical_indicators(symbol, indicator_name);

-- Vacuum database periodically
VACUUM;
ANALYZE;
```

</details>

### Monitoring & Resilience

<details>
<summary><b>Production Monitoring</b></summary>

**Metrics Collection** (`src/core/monitoring.py`):
```python
class MetricsCollector:
    """Thread-safe metrics collection system"""
    
    def collect_metrics(self):
        return {
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'application': {
                'active_agents': len(self.active_agents),
                'cache_hit_rate': self.cache_hits / self.cache_requests,
                'avg_response_time': self.response_times.mean()
            },
            'trading': {
                'signals_generated': self.signal_count,
                'positions_opened': self.position_count,
                'win_rate': self.wins / self.total_trades
            }
        }
```

**Circuit Breakers** (`src/core/resilience.py`):
```python
class CircuitBreaker:
    """Fault protection with exponential backoff"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 60):
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.recovery_timeout = recovery_timeout
```

</details>

---

## 🐛 Troubleshooting

### Common Issues & Solutions

<details>
<summary><b>MLX/GPU Crashes on Apple Silicon</b></summary>

**Problem**: Segmentation fault when loading MLX models

**Solution**:
```bash
# Use the safe launcher
python3 quantum_start.py

# Or disable MLX temporarily
export MLX_DISABLE=1
python3 quantum_consensus_app.py
```

**Prevention**: The safe launcher automatically handles MLX crashes

</details>

<details>
<summary><b>Data Fetching Errors</b></summary>

**Problem**: "No data available" or connection errors

**Solutions**:
1. Check internet connection
2. Clear cache: `rm data/market_data.db`
3. Use force refresh: Add `force_refresh=True` to fetch calls
4. Check Yahoo Finance status

</details>

<details>
<summary><b>Chart Rendering Issues</b></summary>

**Problem**: Blank charts or missing indicators

**Solutions**:
1. Verify data availability for the symbol
2. Check browser console for JavaScript errors
3. Clear Streamlit cache: `streamlit cache clear`
4. Ensure sufficient data points for indicators (min 200 bars)

</details>

---

## 🚀 Roadmap

### Coming Soon (Q1 2025)
- [ ] **WebSocket Integration** — Real-time quote streaming
- [ ] **Advanced Backtesting** — Walk-forward analysis, Monte Carlo
- [ ] **Options Analysis** — Options flow, unusual activity detection
- [ ] **News Sentiment** — Real-time news integration with NLP
- [ ] **Paper Trading** — Virtual portfolio with real-time execution

### Future Vision (Q2-Q4 2025)
- [ ] **Multi-Broker Support** — Alpaca, Interactive Brokers, TD Ameritrade
- [ ] **Cloud Sync** — Cross-device portfolio synchronization
- [ ] **Mobile App** — Native iOS app with SwiftUI
- [ ] **Custom Strategies** — Python scripting for custom strategies
- [ ] **Social Features** — Strategy sharing, leaderboards
- [ ] **Advanced ML** — Deep learning price prediction models
- [ ] **Crypto Integration** — Full cryptocurrency support

---

## 🤝 Contributing

We welcome contributions! Please see the [Issues](https://github.com/yrafique/quantum-consensus-trading/issues) page to get started.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yrafique/quantum-consensus-trading.git
cd quantum-consensus-trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip3 install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black src/
flake8 src/
mypy src/

# Pre-commit hooks
pre-commit install
```

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings required
- **Testing**: Minimum 80% code coverage
- **Formatting**: Black formatter, 88 char line length
- **Linting**: flake8 compliance required

---

## 📖 Documentation

### Available Documentation
- **[Complete Technical Guide](CLAUDE.md)** — AI assistant context with full system details
- **[Startup Guide](STARTUP.md)** — Getting started and configuration
- **This README** — Comprehensive overview with technical deep dives
- **Code Documentation** — Inline docstrings throughout the codebase

### Quick References
- **Agent System** — See expandable sections above for each agent
- **API Endpoints** — Documented in `src/api/routers/` files
- **Configuration** — Examples in this README and `quantum_settings.json`
- **Troubleshooting** — Common issues section above

---

## 🔒 Security & Privacy

### Data Protection
- **100% Local Processing** — All AI inference runs on your machine
- **No External APIs** — Yahoo Finance is the only external data source
- **No Telemetry** — Zero tracking, analytics, or data collection
- **Encrypted Storage** — SQLite database encryption available

### Security Features
- **Input Validation** — All user inputs sanitized
- **Rate Limiting** — API rate limiting and throttling
- **CORS Protection** — Cross-origin request protection
- **SQL Injection Prevention** — Parameterized queries only

---

## 📜 License

This project is licensed under the MIT License

---

## 🙏 Acknowledgments

### Special Thanks
- **MLX Team at Apple** — Revolutionary ML framework for Apple Silicon
- **Streamlit Team** — Beautiful, simple app framework
- **yfinance Contributors** — Reliable market data access
- **LangChain/LangGraph** — Advanced agent orchestration
- **Open Source Community** — Inspiration and collaboration

### Libraries & Frameworks
- **Plotly** — Professional financial charts
- **pandas & NumPy** — Data manipulation
- **scikit-learn** — Machine learning utilities
- **FastAPI** — Modern API framework
- **SQLite** — Embedded database

---

<div align="center">
  
  **Built with ❤️ for traders who think different**
  
  <a href="https://github.com/yrafique/quantum-consensus-trading">⭐ Star on GitHub</a> • 
  <a href="https://github.com/yrafique/quantum-consensus-trading/issues">Report Bug</a> • 
  <a href="https://github.com/yrafique/quantum-consensus-trading/pulls">Contribute</a>
  
  <sub>Local, private, and lightning fast — Your Mac is the trading floor</sub>
  
</div>