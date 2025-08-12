"""
Specialized Trading Agents
=========================

Multiple specialized AI agents for different trading approaches.
Each agent has distinct personality, strategy, and analysis methods.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from ..ai.llm_reasoner import generate_recommendation
from ..ai.rag_advisor import get_trading_advice
from ..trading.signals import evaluate_signals
from ..trading.opportunity_hunter import OpportunityHunter
from ..trading.strategy_validator import StrategyValidator
from ..trading.position_sizer import compute_position_fraction, compute_kelly_fraction
from ..trading.backtester import backtest_ticker
from ..data.data_fetcher import get_data_fetcher

logger = logging.getLogger(__name__)

class AgentType(Enum):
    TECHNICAL_MOMENTUM = "technical_momentum"
    SHORT_SQUEEZE_HUNTER = "short_squeeze_hunter"
    AI_MULTI_FACTOR = "ai_multi_factor"
    SECTOR_ROTATION = "sector_rotation"
    RISK_MANAGER = "risk_manager"
    VALUE_INVESTOR = "value_investor"
    SWING_TRADER = "swing_trader"
    OPTIONS_SPECIALIST = "options_specialist"

class TradingAgent:
    """Base class for trading agents"""
    
    def __init__(self, agent_type: AgentType, name: str, description: str, strategy: str):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.strategy = strategy
        self.data_fetcher = get_data_fetcher()
        self.opportunity_hunter = OpportunityHunter()
        self.validator = StrategyValidator()
        
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        """Analyze a stock using this agent's specific approach"""
        raise NotImplementedError("Subclasses must implement analyze method")
    
    def get_agent_info(self) -> Dict[str, str]:
        """Get agent information for UI display"""
        return {
            'name': self.name,
            'description': self.description,
            'strategy': self.strategy,
            'type': self.agent_type
        }

class TechnicalMomentumAgent(TradingAgent):
    """Pure technical analysis focused on momentum and trend following"""
    
    def __init__(self):
        super().__init__(
            AgentType.TECHNICAL_MOMENTUM,
            "📈 Technical Momentum Analyst",
            "Pure technical analysis with focus on momentum, trends, and chart patterns",
            "RSI • Moving Averages • Volume Analysis • Trend Strength • Pattern Recognition"
        )
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        debug_steps = []
        
        try:
            # Fetch data and calculate indicators
            df = self.data_fetcher.fetch_ohlcv(symbol, period="6mo")
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            df = self.data_fetcher.calculate_technical_indicators(df, symbol)
            latest = df.iloc[-1]
            
            if debug_mode:
                debug_steps.append("📊 **Agent:** Technical Momentum Analyst")
                debug_steps.append(f"📈 **Data Period:** 6 months ({len(df)} trading days)")
            
            # Technical momentum analysis
            rsi = latest.get('rsi', 50)
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            price = latest['close']
            volume = latest['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Enhanced momentum signals with balanced scoring (-10 to +10)
            signals = []
            raw_score = 0
            
            # RSI Analysis (stronger weighting)
            if rsi < 25:
                signals.append("🟢 RSI extremely oversold - strong reversal signal")
                raw_score += 3.5
            elif rsi < 35:
                signals.append("🟢 RSI oversold - potential reversal")
                raw_score += 2.5
            elif rsi > 75:
                signals.append("🔴 RSI extremely overbought - potential correction")
                raw_score -= 2.5
            elif rsi > 65:
                signals.append("🟡 RSI moderately overbought - caution")
                raw_score -= 1.5
            elif rsi > 45 and rsi < 55:
                signals.append(f"🟡 RSI neutral ({rsi:.1f}) - no clear signal")
                raw_score += 0
            else:
                signals.append(f"🟢 RSI in healthy range ({rsi:.1f})")
                raw_score += 1
            
            # Moving Average Analysis (enhanced)
            ma_short_diff = ((price / sma_20) - 1) * 100 if sma_20 > 0 else 0
            ma_long_diff = ((price / sma_50) - 1) * 100 if sma_50 > 0 else 0
            
            if price > sma_20 > sma_50 and ma_short_diff > 5:
                signals.append(f"🟢 Strong uptrend - {ma_short_diff:+.1f}% above 20MA")
                raw_score += 3
            elif price > sma_20 > sma_50:
                signals.append("🟢 Uptrend confirmed - price above both MAs")
                raw_score += 2
            elif price > sma_20 and ma_short_diff > 2:
                signals.append(f"🟡 Short-term bullish - {ma_short_diff:+.1f}% above 20MA")
                raw_score += 1.5
            elif price > sma_20:
                signals.append("🟡 Mild bullish momentum")
                raw_score += 0.5
            elif ma_short_diff < -5:
                signals.append(f"🔴 Strong bearish momentum - {ma_short_diff:.1f}% below 20MA")
                raw_score -= 3
            else:
                signals.append("🔴 Bearish momentum - below moving averages")
                raw_score -= 1.5
            
            # Volume Analysis (enhanced)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 3:
                signals.append(f"🟢 Exceptional volume spike ({volume_ratio:.1f}x avg)")
                raw_score += 2.5
            elif volume_ratio > 1.8:
                signals.append(f"🟢 High volume confirmation ({volume_ratio:.1f}x avg)")
                raw_score += 1.5
            elif volume_ratio > 1.2:
                signals.append(f"🟡 Above average volume ({volume_ratio:.1f}x avg)")
                raw_score += 0.5
            elif volume_ratio < 0.3:
                signals.append("🔴 Very low volume - weak conviction")
                raw_score -= 1.5
            elif volume_ratio < 0.7:
                signals.append("🟡 Below average volume")
                raw_score -= 0.5
            
            # MACD Analysis (enhanced)
            macd_diff = latest.get('macd_diff', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd_diff > 0 and macd_diff > macd_signal:
                signals.append("🟢 MACD bullish crossover with momentum")
                raw_score += 2
            elif macd_diff > 0:
                signals.append("🟡 MACD above signal line")
                raw_score += 1
            elif macd_diff < macd_signal:
                signals.append("🔴 MACD bearish crossover")
                raw_score -= 2
            else:
                signals.append("🟡 MACD mixed signals")
                raw_score -= 0.5
            
            # Price momentum (additional factor)
            price_change_5d = ((price / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
            if price_change_5d > 10:
                signals.append(f"🟢 Strong 5-day momentum (+{price_change_5d:.1f}%)")
                raw_score += 2
            elif price_change_5d > 3:
                signals.append(f"🟡 Positive 5-day momentum (+{price_change_5d:.1f}%)")
                raw_score += 1
            elif price_change_5d < -10:
                signals.append(f"🔴 Weak 5-day performance ({price_change_5d:.1f}%)")
                raw_score -= 2
            elif price_change_5d < -3:
                signals.append(f"🟡 Negative 5-day momentum ({price_change_5d:.1f}%)")
                raw_score -= 1
            
            # Scale to -10 to +10 range and apply logic
            score = max(-10, min(10, raw_score * 1.2))  # Scale factor to utilize full range
            
            # Enhanced recommendation logic
            if score >= 7:
                recommendation = "STRONG BUY"
                confidence = 0.85 + (score - 7) * 0.05
            elif score >= 4:
                recommendation = "BUY"
                confidence = 0.75 + (score - 4) * 0.03
            elif score >= 1:
                recommendation = "WEAK BUY"
                confidence = 0.65 + (score - 1) * 0.03
            elif score >= -1:
                recommendation = "HOLD"
                confidence = 0.55 + abs(score) * 0.05
            elif score >= -4:
                recommendation = "WEAK SELL"
                confidence = 0.65 + abs(score - 1) * 0.03
            elif score >= -7:
                recommendation = "SELL"
                confidence = 0.75 + abs(score - 4) * 0.03
            else:
                recommendation = "STRONG SELL"
                confidence = 0.85 + abs(score - 7) * 0.05
            
            confidence = min(0.95, confidence)  # Cap at 95%
            
            if debug_mode:
                debug_steps.append(f"🔢 **Technical Score:** {score:.1f}/10 (raw: {raw_score:.1f})")
                debug_steps.append(f"🎯 **Recommendation:** {recommendation}")
                debug_steps.append(f"📊 **Confidence:** {confidence:.1%}")
                debug_steps.append(f"📈 **5-day Performance:** {price_change_5d:+.1f}%")
                debug_steps.append(f"📊 **Volume Ratio:** {volume_ratio:.1f}x average")
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'signals': signals,
                'score': round(score, 1),
                'key_metrics': {
                    'RSI': f"{rsi:.1f}",
                    'Price vs 20MA': f"{((price/sma_20-1)*100):+.1f}%" if sma_20 > 0 else "N/A",
                    'Price vs 50MA': f"{((price/sma_50-1)*100):+.1f}%" if sma_50 > 0 else "N/A",
                    'Volume Ratio': f"{volume_ratio:.1f}x"
                },
                'debug_steps': debug_steps if debug_mode else []
            }
            
        except Exception as e:
            logger.error(f"Technical Momentum Agent error: {e}")
            return {"error": str(e), "debug_steps": debug_steps if debug_mode else []}

class ShortSqueezeHunterAgent(TradingAgent):
    """Specialized in detecting short squeeze opportunities"""
    
    def __init__(self):
        super().__init__(
            AgentType.SHORT_SQUEEZE_HUNTER,
            "🚀 Short Squeeze Hunter",
            "Hunts for potential short squeeze opportunities using short interest data",
            "Short Interest • Days to Cover • Float Analysis • Volume Spikes • Momentum"
        )
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        debug_steps = []
        
        try:
            if debug_mode:
                debug_steps.append("🚀 **Agent:** Short Squeeze Hunter")
                debug_steps.append("🔍 **Scanning for:** High short interest + volume spikes")
            
            # Get recent data for volume analysis
            df = self.data_fetcher.fetch_ohlcv(symbol, period="3mo")
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            latest = df.iloc[-1]
            avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_spike = latest['volume'] / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Price momentum
            price_change_5d = ((latest['close'] / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
            price_change_1d = ((latest['close'] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            
            # Simulate short interest analysis (in real system, would use actual data)
            # For demo, we'll use heuristics based on volume and price action
            
            signals = []
            squeeze_score = 0
            
            # Volume spike analysis
            if volume_spike > 3:
                signals.append(f"🔥 Massive volume spike ({volume_spike:.1f}x average)")
                squeeze_score += 3
            elif volume_spike > 2:
                signals.append(f"📈 High volume ({volume_spike:.1f}x average)")
                squeeze_score += 2
            elif volume_spike > 1.5:
                signals.append(f"📊 Elevated volume ({volume_spike:.1f}x average)")
                squeeze_score += 1
            
            # Price momentum analysis
            if price_change_1d > 10:
                signals.append(f"🚀 Strong daily move (+{price_change_1d:.1f}%)")
                squeeze_score += 2
            elif price_change_1d > 5:
                signals.append(f"📈 Solid daily gain (+{price_change_1d:.1f}%)")
                squeeze_score += 1
            
            if price_change_5d > 20:
                signals.append(f"🎯 Major 5-day run (+{price_change_5d:.1f}%)")
                squeeze_score += 2
            elif price_change_5d > 10:
                signals.append(f"📊 Good 5-day momentum (+{price_change_5d:.1f}%)")
                squeeze_score += 1
            
            # Technical indicators for squeeze potential
            df = self.data_fetcher.calculate_technical_indicators(df, symbol)
            latest_tech = df.iloc[-1]
            rsi = latest_tech.get('rsi', 50)
            
            if rsi > 80:
                signals.append("⚠️ Extremely overbought - squeeze may be peaking")
                squeeze_score -= 1
            elif rsi > 70:
                signals.append("🔥 Overbought momentum - active squeeze potential")
                squeeze_score += 1
            elif rsi > 60:
                signals.append("📈 Building momentum - early squeeze phase")
                squeeze_score += 2
            
            # Overall squeeze assessment
            if squeeze_score >= 5:
                recommendation = "HIGH SQUEEZE POTENTIAL"
                confidence = 0.85
            elif squeeze_score >= 3:
                recommendation = "MODERATE SQUEEZE POTENTIAL"
                confidence = 0.65
            elif squeeze_score >= 1:
                recommendation = "LOW SQUEEZE POTENTIAL"
                confidence = 0.45
            else:
                recommendation = "NO SQUEEZE SIGNALS"
                confidence = 0.3
            
            if debug_mode:
                debug_steps.append(f"🎯 **Squeeze Score:** {squeeze_score}/8")
                debug_steps.append(f"🚀 **Assessment:** {recommendation}")
                debug_steps.append(f"📊 **Confidence:** {confidence:.1%}")
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'signals': signals,
                'score': squeeze_score,
                'key_metrics': {
                    'Volume Spike': f"{volume_spike:.1f}x",
                    '1D Change': f"{price_change_1d:+.1f}%",
                    '5D Change': f"{price_change_5d:+.1f}%",
                    'RSI': f"{rsi:.1f}"
                },
                'debug_steps': debug_steps if debug_mode else []
            }
            
        except Exception as e:
            logger.error(f"Short Squeeze Hunter error: {e}")
            return {"error": str(e), "debug_steps": debug_steps if debug_mode else []}

class AIMultiFactorAgent(TradingAgent):
    """Uses existing LLM reasoner and RAG advisor for comprehensive analysis"""
    
    def __init__(self):
        super().__init__(
            AgentType.AI_MULTI_FACTOR,
            "🧠 Quantum Multi-Factor Agent",
            "Multi-dimensional quantum consensus combining all market intelligence vectors",
            "Neural Reasoning • Quantum Knowledge • Technical Analysis • Risk Assessment • Market Context"
        )
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        debug_steps = []
        
        try:
            if debug_mode:
                debug_steps.append("🧠 **Agent:** Quantum Multi-Factor Agent")
                debug_steps.append("⚡ **Using:** Quantum Neural Reasoner + Knowledge Consensus Base")
            
            # Get comprehensive data
            df = self.data_fetcher.fetch_ohlcv(symbol, period="1y")
            stock_info = self.data_fetcher.fetch_stock_info(symbol)
            
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calculate technical indicators
            df = self.data_fetcher.calculate_technical_indicators(df, symbol)
            latest = df.iloc[-1]
            
            # Prepare context for AI analysis
            context = {
                'symbol': symbol,
                'current_price': stock_info['price'],
                'change_percent': stock_info['change_percent'],
                'volume': stock_info['volume'],
                'market_cap': stock_info['market_cap'],
                'pe_ratio': stock_info['pe_ratio'],
                'sector': stock_info.get('sector', 'Unknown'),
                'rsi': latest.get('rsi', 50),
                'sma_20': latest.get('sma_20', 0),
                'sma_50': latest.get('sma_50', 0),
                'macd': latest.get('macd', 0),
                'query': user_query or f"Comprehensive analysis of {symbol}"
            }
            
            if debug_mode:
                debug_steps.append(f"📊 **Market Data:** ${stock_info['price']:.2f} ({stock_info['change_percent']:+.2f}%)")
                debug_steps.append(f"💰 **Market Cap:** ${stock_info['market_cap']:,.0f}")
                debug_steps.append(f"🏢 **Sector:** {stock_info.get('sector', 'Unknown')}")
            
            # Use existing LLM reasoner
            llm_recommendation = generate_recommendation(symbol, df.tail(50), debug_mode=debug_mode)
            
            # Use RAG advisor for additional insights
            rag_query = f"Analyze {symbol} for investment potential considering current market conditions"
            rag_advice = get_trading_advice(rag_query, context)
            
            if debug_mode:
                debug_steps.append("🔬 **LLM Analysis:** Generated recommendation using historical patterns")
                debug_steps.append("📚 **RAG Knowledge:** Retrieved relevant trading insights")
            
            # Combine insights
            confidence = llm_recommendation.get('confidence', 0.5)
            action = llm_recommendation.get('action', 'Hold')
            reasoning = llm_recommendation.get('reasoning', 'No specific reasoning provided')
            
            # Create comprehensive signals list
            signals = [
                f"🧠 AI Recommendation: {action}",
                f"📊 Confidence Level: {confidence:.1%}",
                f"💡 Key Insight: {reasoning[:100]}..." if len(reasoning) > 100 else f"💡 {reasoning}"
            ]
            
            # Add technical context
            rsi = latest.get('rsi', 50)
            if rsi > 70:
                signals.append("⚠️ Technical: Overbought conditions")
            elif rsi < 30:
                signals.append("📈 Technical: Oversold opportunity")
            
            # Format recommendation
            recommendation_map = {
                'Buy': 'BUY',
                'Sell': 'SELL',
                'Hold': 'HOLD',
                'Strong Buy': 'STRONG BUY',
                'Strong Sell': 'STRONG SELL'
            }
            final_recommendation = recommendation_map.get(action, action.upper())
            
            if debug_mode:
                debug_steps.append(f"🎯 **Final Recommendation:** {final_recommendation}")
                debug_steps.append(f"🎪 **Combined Confidence:** {confidence:.1%}")
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'recommendation': final_recommendation,
                'confidence': confidence,
                'signals': signals,
                'llm_reasoning': reasoning,
                'rag_insights': rag_advice if isinstance(rag_advice, str) else str(rag_advice),
                'key_metrics': {
                    'AI Confidence': f"{confidence:.1%}",
                    'P/E Ratio': f"{stock_info['pe_ratio']:.1f}",
                    'RSI': f"{rsi:.1f}",
                    'Sector': stock_info.get('sector', 'Unknown')
                },
                'debug_steps': debug_steps if debug_mode else []
            }
            
        except Exception as e:
            logger.error(f"Quantum Multi-Factor Agent error: {e}")
            return {"error": str(e), "debug_steps": debug_steps if debug_mode else []}

class ValueInvestorAgent(TradingAgent):
    """Long-term value investing approach"""
    
    def __init__(self):
        super().__init__(
            AgentType.VALUE_INVESTOR,
            "💎 Value Investor",
            "Long-term value investing focused on fundamentals and intrinsic value",
            "P/E Analysis • Revenue Growth • Debt Ratios • Dividend Yield • Market Position"
        )
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        debug_steps = []
        
        try:
            if debug_mode:
                debug_steps.append("💎 **Agent:** Value Investor")
                debug_steps.append("🔍 **Focus:** Fundamental analysis & intrinsic value")
            
            stock_info = self.data_fetcher.fetch_stock_info(symbol)
            
            signals = []
            value_score = 0
            
            # P/E Analysis
            pe_ratio = stock_info.get('pe_ratio', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    signals.append(f"💰 Attractive P/E ratio ({pe_ratio:.1f})")
                    value_score += 2
                elif pe_ratio < 25:
                    signals.append(f"📊 Reasonable P/E ratio ({pe_ratio:.1f})")
                    value_score += 1
                else:
                    signals.append(f"⚠️ High P/E ratio ({pe_ratio:.1f})")
                    value_score -= 1
            
            # Market cap analysis (prefer large, stable companies)
            market_cap = stock_info.get('market_cap', 0)
            if market_cap > 100_000_000_000:  # $100B+
                signals.append("🏢 Large-cap stability")
                value_score += 1
            elif market_cap > 10_000_000_000:  # $10B+
                signals.append("📈 Mid-large cap")
                value_score += 0.5
            
            # Dividend analysis
            dividend_yield = stock_info.get('dividend_yield', 0)
            if dividend_yield > 0.03:  # 3%+
                signals.append(f"💰 Good dividend yield ({dividend_yield:.1%})")
                value_score += 1
            elif dividend_yield > 0.01:  # 1%+
                signals.append(f"📊 Modest dividend ({dividend_yield:.1%})")
                value_score += 0.5
            
            # Current valuation vs recent performance
            current_change = stock_info.get('change_percent', 0)
            if current_change < -10:
                signals.append("🛒 Potential buying opportunity on weakness")
                value_score += 1
            elif current_change < -5:
                signals.append("📉 Recent pullback may offer value")
                value_score += 0.5
            
            # Beta analysis (prefer lower volatility)
            beta = stock_info.get('beta', 1.0)
            if beta < 1.2:
                signals.append(f"📊 Lower volatility (β={beta:.2f})")
                value_score += 0.5
            
            # Overall value assessment
            if value_score >= 4:
                recommendation = "STRONG VALUE BUY"
                confidence = 0.8
            elif value_score >= 2:
                recommendation = "VALUE BUY"
                confidence = 0.65
            elif value_score >= 0:
                recommendation = "FAIR VALUE"
                confidence = 0.5
            else:
                recommendation = "OVERVALUED"
                confidence = 0.6
            
            if debug_mode:
                debug_steps.append(f"💰 **Value Score:** {value_score}/6")
                debug_steps.append(f"🎯 **Assessment:** {recommendation}")
                debug_steps.append(f"📊 **Confidence:** {confidence:.1%}")
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'signals': signals,
                'score': value_score,
                'key_metrics': {
                    'P/E Ratio': f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A",
                    'Market Cap': f"${market_cap/1e9:.1f}B" if market_cap > 0 else "N/A",
                    'Dividend Yield': f"{dividend_yield:.2%}" if dividend_yield > 0 else "None",
                    'Beta': f"{beta:.2f}"
                },
                'debug_steps': debug_steps if debug_mode else []
            }
            
        except Exception as e:
            logger.error(f"Value Investor Agent error: {e}")
            return {"error": str(e), "debug_steps": debug_steps if debug_mode else []}

class KellyCriterionAgent(TradingAgent):
    """Kelly Criterion optimal position sizing trader"""
    
    def __init__(self):
        super().__init__(
            AgentType.TECHNICAL_MOMENTUM,  # Will add KELLY_CRITERION to enum
            "📊 Kelly Criterion Trader",
            "Optimal position sizing using Kelly Criterion with backtesting capabilities",
            "Kelly Formula • Win Rate Estimation • Risk-Reward Analysis • Optimal Position Sizing"
        )
        self.agent_type = "kelly_criterion"  # Override for now
    
    def estimate_win_probability(self, df: pd.DataFrame, symbol: str) -> float:
        """Estimate win probability from historical data"""
        if len(df) < 50:
            return 0.5  # Default neutral
        
        # Calculate recent performance
        returns = df['close'].pct_change().dropna()
        positive_returns = (returns > 0).sum()
        total_trades = len(returns)
        
        # Base win rate from recent history
        base_win_rate = positive_returns / total_trades if total_trades > 0 else 0.5
        
        # Adjust based on current technical conditions
        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)
        
        # RSI adjustments
        if rsi < 30:  # Oversold - higher win probability
            base_win_rate += 0.1
        elif rsi > 70:  # Overbought - lower win probability
            base_win_rate -= 0.1
        
        # Moving average trend adjustment
        sma_20 = latest.get('sma_20', 0)
        sma_50 = latest.get('sma_50', 0)
        price = latest['close']
        
        if price > sma_20 > sma_50:  # Strong uptrend
            base_win_rate += 0.05
        elif price < sma_20 < sma_50:  # Strong downtrend
            base_win_rate -= 0.05
        
        # Bound between 0.1 and 0.9
        return max(0.1, min(0.9, base_win_rate))
    
    def estimate_reward_risk_ratio(self, df: pd.DataFrame) -> float:
        """Estimate reward-to-risk ratio from historical volatility"""
        if len(df) < 20:
            return 2.0  # Default 2:1
        
        # Calculate historical volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Estimate average winning vs losing trade sizes
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            reward_risk = avg_win / avg_loss if avg_loss > 0 else 2.0
        else:
            reward_risk = 2.0
        
        # Bound between 1.2 and 5.0
        return max(1.2, min(5.0, reward_risk))
    
    def analyze(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        debug_steps = []
        
        try:
            if debug_mode:
                debug_steps.append("📊 **Agent:** Kelly Criterion Trader")
                debug_steps.append("🎯 **Focus:** Optimal position sizing using Kelly Formula")
            
            # Fetch data and calculate indicators
            df = self.data_fetcher.fetch_ohlcv(symbol, period="1y")
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            df = self.data_fetcher.calculate_technical_indicators(df, symbol)
            stock_info = self.data_fetcher.fetch_stock_info(symbol)
            
            # Estimate Kelly parameters
            win_prob = self.estimate_win_probability(df, symbol)
            reward_risk = self.estimate_reward_risk_ratio(df)
            
            # Calculate Kelly fraction
            kelly_fraction = compute_kelly_fraction(win_prob, reward_risk)
            position_fraction = compute_position_fraction(win_prob, reward_risk)
            
            if debug_mode:
                debug_steps.append(f"🎲 **Win Probability:** {win_prob:.1%}")
                debug_steps.append(f"⚖️ **Reward:Risk Ratio:** {reward_risk:.2f}:1")
                debug_steps.append(f"📊 **Raw Kelly Fraction:** {kelly_fraction:.1%}")
                debug_steps.append(f"🎯 **Scaled Position Size:** {position_fraction:.1%}")
            
            # Generate signals and scoring
            signals = []
            raw_score = 0
            
            # Kelly fraction analysis
            if kelly_fraction > 0.15:  # 15%+
                signals.append(f"🟢 Strong Kelly signal - {kelly_fraction:.1%} allocation")
                raw_score += 4
            elif kelly_fraction > 0.08:  # 8%+
                signals.append(f"🟡 Moderate Kelly signal - {kelly_fraction:.1%} allocation")
                raw_score += 2
            elif kelly_fraction > 0.02:  # 2%+
                signals.append(f"🟡 Weak Kelly signal - {kelly_fraction:.1%} allocation")
                raw_score += 0.5
            else:
                signals.append("🔴 Kelly suggests no position - negative expected value")
                raw_score -= 3
            
            # Win probability assessment
            if win_prob > 0.65:
                signals.append(f"🟢 High win probability ({win_prob:.1%})")
                raw_score += 2
            elif win_prob > 0.55:
                signals.append(f"🟡 Above average win probability ({win_prob:.1%})")
                raw_score += 1
            elif win_prob < 0.4:
                signals.append(f"🔴 Low win probability ({win_prob:.1%})")
                raw_score -= 2
            else:
                signals.append(f"🟡 Neutral win probability ({win_prob:.1%})")
            
            # Risk-reward assessment
            if reward_risk > 3:
                signals.append(f"🟢 Excellent risk-reward ratio ({reward_risk:.1f}:1)")
                raw_score += 2
            elif reward_risk > 2:
                signals.append(f"🟡 Good risk-reward ratio ({reward_risk:.1f}:1)")
                raw_score += 1
            elif reward_risk < 1.5:
                signals.append(f"🔴 Poor risk-reward ratio ({reward_risk:.1f}:1)")
                raw_score -= 2
            else:
                signals.append(f"🟡 Adequate risk-reward ratio ({reward_risk:.1f}:1)")
            
            # Current market conditions
            latest = df.iloc[-1]
            rsi = latest.get('rsi', 50)
            price = latest['close']
            sma_20 = latest.get('sma_20', 0)
            
            if rsi < 30 and price < sma_20 * 0.95:
                signals.append("🟢 Oversold conditions favor entry timing")
                raw_score += 1
            elif rsi > 70:
                signals.append("🔴 Overbought conditions - wait for pullback")
                raw_score -= 1
            
            # Scale score to -10 to +10
            score = max(-10, min(10, raw_score * 1.2))
            
            # Generate recommendation based on Kelly fraction and score
            if kelly_fraction > 0.1 and score > 3:
                recommendation = "STRONG BUY"
                confidence = 0.85 + kelly_fraction
            elif kelly_fraction > 0.05 and score > 1:
                recommendation = "BUY"
                confidence = 0.75 + kelly_fraction * 2
            elif kelly_fraction > 0.02:
                recommendation = "WEAK BUY"
                confidence = 0.65
            elif kelly_fraction > 0:
                recommendation = "HOLD"
                confidence = 0.55
            else:
                recommendation = "SELL"
                confidence = 0.7
            
            confidence = min(0.95, confidence)
            
            # Calculate position size for $100k portfolio
            portfolio_value = 100000
            position_size_dollars = portfolio_value * position_fraction
            shares = int(position_size_dollars / price) if price > 0 else 0
            
            if debug_mode:
                debug_steps.append(f"💰 **Position Size:** ${position_size_dollars:,.0f} ({shares:,} shares)")
                debug_steps.append(f"🎯 **Recommendation:** {recommendation}")
                debug_steps.append(f"📊 **Confidence:** {confidence:.1%}")
                debug_steps.append(f"🔢 **Kelly Score:** {score:.1f}/10")
            
            return {
                'agent': self.name,
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'signals': signals,
                'score': round(score, 1),
                'kelly_fraction': kelly_fraction,
                'position_fraction': position_fraction,
                'win_probability': win_prob,
                'reward_risk_ratio': reward_risk,
                'position_size_dollars': position_size_dollars,
                'shares': shares,
                'key_metrics': {
                    'Kelly Fraction': f"{kelly_fraction:.1%}",
                    'Win Probability': f"{win_prob:.1%}",
                    'Reward:Risk': f"{reward_risk:.1f}:1",
                    'Position Size': f"${position_size_dollars:,.0f}",
                    'Shares': f"{shares:,}"
                },
                'debug_steps': debug_steps if debug_mode else []
            }
            
        except Exception as e:
            logger.error(f"Kelly Criterion Agent error: {e}")
            return {"error": str(e), "debug_steps": debug_steps if debug_mode else []}

class AgentManager:
    """Manages all trading agents and provides unified interface"""
    
    def __init__(self):
        self.agents = {
            AgentType.TECHNICAL_MOMENTUM: TechnicalMomentumAgent(),
            AgentType.SHORT_SQUEEZE_HUNTER: ShortSqueezeHunterAgent(),
            AgentType.AI_MULTI_FACTOR: AIMultiFactorAgent(),
            AgentType.VALUE_INVESTOR: ValueInvestorAgent(),
            "kelly_criterion": KellyCriterionAgent(),  # Custom key for now
        }
    
    def get_all_agents(self) -> Dict[AgentType, Dict[str, str]]:
        """Get information about all available agents"""
        return {agent_type: agent.get_agent_info() for agent_type, agent in self.agents.items()}
    
    def analyze_with_agent(self, agent_type: AgentType, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[str, Any]:
        """Analyze a symbol with a specific agent"""
        if agent_type not in self.agents:
            return {"error": f"Agent type {agent_type} not available"}
        
        return self.agents[agent_type].analyze(symbol, user_query, debug_mode)
    
    def analyze_with_all_agents(self, symbol: str, user_query: str = "", debug_mode: bool = False) -> Dict[AgentType, Dict[str, Any]]:
        """Get analysis from all agents"""
        results = {}
        for agent_type, agent in self.agents.items():
            try:
                results[agent_type] = agent.analyze(symbol, user_query, debug_mode)
            except Exception as e:
                logger.error(f"Error with agent {agent_type}: {e}")
                results[agent_type] = {"error": str(e)}
        
        return results

# Global agent manager instance
_agent_manager = None

def get_agent_manager() -> AgentManager:
    """Get global agent manager instance"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager