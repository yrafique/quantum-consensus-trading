"""
Unified Trading Analyzer
=======================

Comprehensive analysis system that integrates all existing AI components
with advanced debug visualization and clean architecture.
"""

import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

# Import all existing systems
from ..ai.llm_reasoner import generate_recommendation
from ..ai.rag_advisor import get_trading_advice, get_rag_advisor
from ..ai.safe_ai_loader import get_safe_ai_loader, initialize_ai_safely
from ..agents.react_trading_agent import ReActTradingAgent
from ..agents.intelligent_router import IntelligentRouter
from ..trading.opportunity_hunter import OpportunityHunter
from ..trading.signals import evaluate_signals as calc_signals
from ..trading.strategy_validator import StrategyValidator
from ..trading.position_sizer import compute_position_fraction
from ..utils.data_validator import DataValidator
from ..data.data_fetcher import get_data_fetcher
from ..monitoring.connection_monitor import ConnectionMonitor

logger = logging.getLogger(__name__)

@dataclass
class AnalysisStep:
    """Represents a single step in the analysis process"""
    timestamp: datetime
    component: str
    step_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    confidence: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """Complete analysis result with full debug information"""
    symbol: str
    query: str
    final_recommendation: str
    confidence: float
    execution_time: float
    analysis_steps: List[AnalysisStep]
    component_results: Dict[str, Any]
    market_data: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    validation_results: Dict[str, Any]
    system_health: Dict[str, Any]

class AnalysisMode(Enum):
    """Analysis execution modes"""
    FAST = "fast"           # Quick technical analysis
    COMPREHENSIVE = "comprehensive"  # Full multi-agent analysis
    CONSENSUS = "consensus"  # Cross-validation between multiple agents
    REAL_TIME = "real_time"  # Live market analysis

class UnifiedAnalyzer:
    """
    Unified system that orchestrates all trading analysis components
    with comprehensive debug capabilities
    """
    
    def __init__(self):
        # Initialize all components
        self.data_fetcher = get_data_fetcher()
        self.data_validator = DataValidator()
        self.opportunity_hunter = OpportunityHunter()
        self.strategy_validator = StrategyValidator()
        self.position_sizer = None  # Use compute_position_fraction function directly
        self.connection_monitor = ConnectionMonitor()
        
        # AI Components
        self.mlx_llm = None
        self.react_agent = None
        self.intelligent_router = None
        self.rag_advisor = None
        
        # Analysis tracking
        self.analysis_history: List[AnalysisResult] = []
        self.component_performance: Dict[str, List[float]] = {}
        
        self._initialize_ai_components()
    
    def _initialize_ai_components(self):
        """Initialize AI components with error handling"""
        # Initialize MLX safely with memory monitoring
        self.ai_loader = initialize_ai_safely()
        self.mlx_llm = None
        
        # Check if MLX is ready (non-blocking)
        if self.ai_loader.is_ready():
            self.mlx_llm = self.ai_loader.get_model()
            logger.info("MLX LLM loaded from safe loader")
        else:
            logger.info("MLX LLM still loading or unavailable")
        
        try:
            self.react_agent = ReActTradingAgent()
            logger.info("ReAct agent initialized successfully")
        except Exception as e:
            logger.warning(f"ReAct agent initialization failed: {e}")
        
        try:
            self.intelligent_router = IntelligentRouter()
            logger.info("Intelligent router initialized successfully")
        except Exception as e:
            logger.warning(f"Intelligent router initialization failed: {e}")
        
        try:
            self.rag_advisor = get_rag_advisor()
            logger.info("RAG advisor initialized successfully")
        except Exception as e:
            logger.warning(f"RAG advisor initialization failed: {e}")
    
    async def analyze(
        self, 
        symbol: str, 
        query: str = "", 
        mode: AnalysisMode = AnalysisMode.COMPREHENSIVE,
        debug: bool = True
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis using all available components
        """
        start_time = time.time()
        analysis_steps = []
        component_results = {}
        
        # Step 1: System Health Check
        health_step = await self._execute_step(
            "System Health Check",
            "health_check",
            {"symbol": symbol, "query": query},
            self._check_system_health
        )
        analysis_steps.append(health_step)
        
        # Step 2: Data Validation and Fetching
        data_step = await self._execute_step(
            "Data Validation & Fetching",
            "data_fetch",
            {"symbol": symbol, "mode": mode.value},
            self._fetch_and_validate_data,
            symbol
        )
        analysis_steps.append(data_step)
        
        if data_step.errors:
            return self._create_error_result(symbol, query, analysis_steps, "Data fetch failed")
        
        market_data = data_step.output_data
        
        # Step 3: Technical Analysis
        technical_step = await self._execute_step(
            "Technical Analysis",
            "technical",
            {"market_data": market_data},
            self._perform_technical_analysis,
            market_data
        )
        analysis_steps.append(technical_step)
        component_results['technical'] = technical_step.output_data
        
        # Step 4: Opportunity Assessment
        opportunity_step = await self._execute_step(
            "Opportunity Assessment",
            "opportunity",
            {"symbol": symbol, "market_data": market_data},
            self._assess_opportunities,
            symbol, market_data
        )
        analysis_steps.append(opportunity_step)
        component_results['opportunity'] = opportunity_step.output_data
        
        # Step 5: Multi-Agent Analysis (based on mode)
        if mode in [AnalysisMode.COMPREHENSIVE, AnalysisMode.CONSENSUS]:
            ai_step = await self._execute_step(
                "Multi-Agent AI Analysis",
                "ai_analysis",
                {"symbol": symbol, "query": query, "market_data": market_data},
                self._perform_ai_analysis,
                symbol, query, market_data
            )
            analysis_steps.append(ai_step)
            component_results['ai'] = ai_step.output_data
        
        # Step 6: Risk Assessment
        risk_step = await self._execute_step(
            "Risk Assessment",
            "risk",
            {"symbol": symbol, "market_data": market_data, "analysis": component_results},
            self._assess_risk,
            symbol, market_data, component_results
        )
        analysis_steps.append(risk_step)
        
        # Step 7: Consensus Building
        consensus_step = await self._execute_step(
            "Consensus Building",
            "consensus",
            {"components": component_results},
            self._build_consensus,
            component_results
        )
        analysis_steps.append(consensus_step)
        
        # Create final result
        total_time = time.time() - start_time
        
        return AnalysisResult(
            symbol=symbol,
            query=query,
            final_recommendation=consensus_step.output_data.get('recommendation', 'HOLD'),
            confidence=consensus_step.output_data.get('confidence', 0.5),
            execution_time=total_time,
            analysis_steps=analysis_steps,
            component_results=component_results,
            market_data=market_data,
            risk_assessment=risk_step.output_data,
            validation_results=data_step.output_data.get('validation', {}),
            system_health=health_step.output_data
        )
    
    async def _execute_step(
        self, 
        component: str, 
        step_type: str, 
        input_data: Dict[str, Any], 
        func, 
        *args, 
        **kwargs
    ) -> AnalysisStep:
        """Execute a single analysis step with timing and error handling"""
        start_time = time.time()
        timestamp = datetime.now()
        errors = []
        output_data = {}
        
        try:
            if asyncio.iscoroutinefunction(func):
                output_data = await func(*args, **kwargs)
            else:
                output_data = func(*args, **kwargs)
        except Exception as e:
            error_msg = f"{component} failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            output_data = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        # Track component performance
        if component not in self.component_performance:
            self.component_performance[component] = []
        self.component_performance[component].append(execution_time)
        
        return AnalysisStep(
            timestamp=timestamp,
            component=component,
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            confidence=output_data.get('confidence'),
            errors=errors,
            metadata={
                'avg_execution_time': sum(self.component_performance[component]) / len(self.component_performance[component]),
                'component_runs': len(self.component_performance[component])
            }
        )
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check health of all system components"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_health': 'healthy'
        }
        
        # Check data fetcher
        try:
            test_data = self.data_fetcher.fetch_stock_info('AAPL')
            health_data['components']['data_fetcher'] = {
                'status': 'healthy' if test_data.get('price', 0) > 0 else 'degraded',
                'last_update': test_data.get('last_updated', 'unknown')
            }
        except Exception as e:
            health_data['components']['data_fetcher'] = {'status': 'error', 'error': str(e)}
        
        # Check AI components
        health_data['components']['mlx_llm'] = {'status': 'healthy' if self.mlx_llm else 'unavailable'}
        health_data['components']['react_agent'] = {'status': 'healthy' if self.react_agent else 'unavailable'}
        health_data['components']['rag_advisor'] = {'status': 'healthy' if self.rag_advisor else 'unavailable'}
        
        # Check connection monitor
        try:
            monitor_status = self.connection_monitor.get_system_status()
            health_data['components']['connection_monitor'] = {
                'status': 'healthy',
                'details': monitor_status
            }
        except Exception as e:
            health_data['components']['connection_monitor'] = {'status': 'error', 'error': str(e)}
        
        # Determine overall health
        component_statuses = [comp['status'] for comp in health_data['components'].values()]
        if 'error' in component_statuses:
            health_data['overall_health'] = 'degraded'
        elif 'unavailable' in component_statuses:
            health_data['overall_health'] = 'limited'
        
        return health_data
    
    def _fetch_and_validate_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch and validate market data"""
        # Fetch OHLCV data
        df = self.data_fetcher.fetch_ohlcv(symbol, period="1y")
        stock_info = self.data_fetcher.fetch_stock_info(symbol)
        
        # Calculate technical indicators
        df_with_indicators = self.data_fetcher.calculate_technical_indicators(df, symbol)
        
        # Validate data
        validation_results = self.data_validator.validate_stock_data(symbol)
        
        return {
            'symbol': symbol,
            'ohlcv_data': df_with_indicators.to_dict() if not df_with_indicators.empty else {},
            'stock_info': stock_info,
            'data_points': len(df_with_indicators),
            'latest_price': stock_info.get('price', 0),
            'validation': validation_results,
            'indicators_calculated': list(df_with_indicators.columns) if not df_with_indicators.empty else []
        }
    
    def _perform_technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        symbol = market_data['symbol']
        df = pd.DataFrame(market_data['ohlcv_data'])
        
        if df.empty:
            return {'error': 'No market data available'}
        
        # Calculate signals
        signals = calc_signals(df)
        
        # Latest values
        latest = df.iloc[-1] if not df.empty else {}
        
        technical_summary = {
            'signals': signals,
            'current_indicators': {
                'rsi': latest.get('rsi', 50),
                'sma_20': latest.get('sma_20', 0),
                'sma_50': latest.get('sma_50', 0),
                'macd': latest.get('macd', 0),
                'volume_ratio': latest.get('volume', 0) / df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1
            },
            'trend_analysis': self._analyze_trend(df),
            'support_resistance': self._find_support_resistance(df),
            'pattern_analysis': self._analyze_patterns(df)
        }
        
        return technical_summary
    
    def _assess_opportunities(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess opportunities using opportunity hunter"""
        try:
            # Get opportunity score
            opportunities = self.opportunity_hunter.find_opportunities()
            
            # Find our symbol in opportunities
            symbol_opportunity = None
            for opp in opportunities:
                if opp.get('symbol') == symbol:
                    symbol_opportunity = opp
                    break
            
            if symbol_opportunity:
                return {
                    'opportunity_score': symbol_opportunity.get('total_score', 0),
                    'category': symbol_opportunity.get('category', 'unknown'),
                    'factors': symbol_opportunity.get('factors', {}),
                    'ranking': opportunities.index(symbol_opportunity) + 1,
                    'total_opportunities': len(opportunities)
                }
            else:
                return {
                    'opportunity_score': 0,
                    'category': 'not_screened',
                    'factors': {},
                    'ranking': None,
                    'total_opportunities': len(opportunities)
                }
        except Exception as e:
            return {'error': f'Opportunity assessment failed: {str(e)}'}
    
    def _perform_ai_analysis(self, symbol: str, query: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-agent AI analysis"""
        ai_results = {}
        
        # LLM Reasoner
        try:
            df = pd.DataFrame(market_data['ohlcv_data'])
            llm_result = generate_recommendation(symbol, df.tail(50) if not df.empty else df)
            ai_results['llm_reasoner'] = {
                'recommendation': llm_result.get('action', 'Hold'),
                'confidence': llm_result.get('confidence', 0.5),
                'reasoning': llm_result.get('reasoning', ''),
                'entry_price': llm_result.get('entry_price'),
                'stop_loss': llm_result.get('stop_loss'),
                'target_price': llm_result.get('target_price')
            }
        except Exception as e:
            ai_results['llm_reasoner'] = {'error': str(e)}
        
        # MLX LLM (if available)
        if self.mlx_llm:
            try:
                mlx_result = self.mlx_llm.analyze_stock(symbol, market_data)
                ai_results['mlx_llm'] = mlx_result
            except Exception as e:
                ai_results['mlx_llm'] = {'error': str(e)}
        
        # ReAct Agent (if available)
        if self.react_agent:
            try:
                react_result = self.react_agent.analyze(f"Analyze {symbol} for trading potential: {query}")
                ai_results['react_agent'] = react_result
            except Exception as e:
                ai_results['react_agent'] = {'error': str(e)}
        
        # RAG Advisor
        if self.rag_advisor:
            try:
                rag_context = {
                    'symbol': symbol,
                    'current_price': market_data['stock_info']['price'],
                    'market_data': market_data
                }
                rag_result = get_trading_advice(f"Analyze {symbol}: {query}", rag_context)
                ai_results['rag_advisor'] = {
                    'advice': rag_result,
                    'confidence': 0.7  # Default confidence for RAG
                }
            except Exception as e:
                ai_results['rag_advisor'] = {'error': str(e)}
        
        return ai_results
    
    def _assess_risk(self, symbol: str, market_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        stock_info = market_data['stock_info']
        
        # Calculate position size using Kelly Criterion
        win_prob = 0.6  # Default, could be enhanced with historical data
        reward_risk = 2.0  # Default 2:1 ratio
        
        kelly_size = self.position_sizer.calculate_kelly_position(
            win_probability=win_prob,
            reward_to_risk_ratio=reward_risk,
            capital=100000  # Default portfolio size
        )
        
        # Risk metrics
        beta = stock_info.get('beta', 1.0)
        market_cap = stock_info.get('market_cap', 0)
        volume = stock_info.get('volume', 0)
        
        risk_assessment = {
            'position_sizing': {
                'kelly_percentage': kelly_size['percentage'],
                'recommended_shares': kelly_size['shares'],
                'risk_amount': kelly_size['risk_amount']
            },
            'volatility_risk': {
                'beta': beta,
                'risk_level': 'High' if beta > 1.5 else 'Medium' if beta > 0.8 else 'Low'
            },
            'liquidity_risk': {
                'volume': volume,
                'market_cap': market_cap,
                'liquidity_score': min(10, volume / 1000000)  # Simple liquidity score
            },
            'overall_risk_score': self._calculate_overall_risk(beta, market_cap, volume)
        }
        
        return risk_assessment
    
    def _build_consensus(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus from all analysis components"""
        recommendations = []
        confidences = []
        
        # Extract recommendations from each component
        if 'ai' in component_results:
            ai_results = component_results['ai']
            for agent, result in ai_results.items():
                if isinstance(result, dict) and 'error' not in result:
                    rec = result.get('recommendation', result.get('action', 'HOLD'))
                    conf = result.get('confidence', 0.5)
                    recommendations.append(rec.upper())
                    confidences.append(conf)
        
        # Technical analysis contribution
        if 'technical' in component_results:
            tech = component_results['technical']
            if 'signals' in tech and tech['signals']:
                recommendations.append('BUY')
                confidences.append(0.7)
            else:
                recommendations.append('HOLD')
                confidences.append(0.5)
        
        # Opportunity analysis contribution
        if 'opportunity' in component_results:
            opp = component_results['opportunity']
            score = opp.get('opportunity_score', 0)
            if score > 75:
                recommendations.append('BUY')
                confidences.append(0.8)
            elif score > 50:
                recommendations.append('HOLD')
                confidences.append(0.6)
            else:
                recommendations.append('SELL')
                confidences.append(0.7)
        
        # Calculate consensus
        if not recommendations:
            return {'recommendation': 'HOLD', 'confidence': 0.5, 'consensus': 'No data'}
        
        # Count recommendations
        buy_count = recommendations.count('BUY')
        hold_count = recommendations.count('HOLD')
        sell_count = recommendations.count('SELL')
        
        # Determine final recommendation
        if buy_count > hold_count and buy_count > sell_count:
            final_rec = 'BUY'
        elif sell_count > hold_count and sell_count > buy_count:
            final_rec = 'SELL'
        else:
            final_rec = 'HOLD'
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Consensus strength
        total_votes = len(recommendations)
        winning_votes = max(buy_count, hold_count, sell_count)
        consensus_strength = winning_votes / total_votes if total_votes > 0 else 0
        
        return {
            'recommendation': final_rec,
            'confidence': avg_confidence,
            'consensus_strength': consensus_strength,
            'vote_breakdown': {
                'BUY': buy_count,
                'HOLD': hold_count,
                'SELL': sell_count
            },
            'contributing_components': len(recommendations)
        }
    
    # Helper methods
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends"""
        if df.empty or len(df) < 20:
            return {'trend': 'unknown', 'strength': 0}
        
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else 0
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else 0
        current_price = df['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return {'trend': 'uptrend', 'strength': 0.8}
        elif current_price < sma_20 < sma_50:
            return {'trend': 'downtrend', 'strength': 0.8}
        else:
            return {'trend': 'sideways', 'strength': 0.5}
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find support and resistance levels"""
        if df.empty or len(df) < 20:
            return {'support': None, 'resistance': None}
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        return {
            'support': recent_low,
            'resistance': recent_high,
            'current_position': (df['close'].iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        }
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns"""
        if df.empty or len(df) < 10:
            return {'patterns': []}
        
        patterns = []
        
        # Simple pattern detection
        recent_closes = df['close'].tail(5)
        if len(recent_closes) >= 5:
            if recent_closes.iloc[-1] > recent_closes.iloc[-2] > recent_closes.iloc[-3]:
                patterns.append('ascending_trend')
            elif recent_closes.iloc[-1] < recent_closes.iloc[-2] < recent_closes.iloc[-3]:
                patterns.append('descending_trend')
        
        return {'patterns': patterns}
    
    def _calculate_overall_risk(self, beta: float, market_cap: float, volume: int) -> float:
        """Calculate overall risk score (0-10, higher = riskier)"""
        risk_score = 0
        
        # Beta risk
        risk_score += min(5, beta * 2)
        
        # Market cap risk (smaller = riskier)
        if market_cap < 1e9:  # < $1B
            risk_score += 3
        elif market_cap < 10e9:  # < $10B
            risk_score += 1
        
        # Volume risk (lower = riskier)
        if volume < 100000:
            risk_score += 2
        elif volume < 1000000:
            risk_score += 1
        
        return min(10, risk_score)
    
    def _create_error_result(self, symbol: str, query: str, steps: List[AnalysisStep], error: str) -> AnalysisResult:
        """Create error result when analysis fails"""
        return AnalysisResult(
            symbol=symbol,
            query=query,
            final_recommendation='ERROR',
            confidence=0.0,
            execution_time=sum(step.execution_time for step in steps),
            analysis_steps=steps,
            component_results={'error': error},
            market_data={},
            risk_assessment={},
            validation_results={},
            system_health={}
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'components_initialized': {
                'mlx_llm': self.mlx_llm is not None,
                'react_agent': self.react_agent is not None,
                'intelligent_router': self.intelligent_router is not None,
                'rag_advisor': self.rag_advisor is not None
            },
            'analysis_history_count': len(self.analysis_history),
            'component_performance': {
                comp: {
                    'avg_time': sum(times) / len(times) if times else 0,
                    'runs': len(times)
                }
                for comp, times in self.component_performance.items()
            }
        }

# Global instance
_unified_analyzer = None

def get_unified_analyzer() -> UnifiedAnalyzer:
    """Get global unified analyzer instance"""
    global _unified_analyzer
    if _unified_analyzer is None:
        _unified_analyzer = UnifiedAnalyzer()
    return _unified_analyzer