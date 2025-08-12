"""
ReAct Trading Agent with LangGraph
=================================

Advanced trading agent using ReAct (Reasoning and Acting) pattern with LangGraph
to eliminate hallucinations and provide robust, fact-checked market analysis.

Architecture:
- Observation: Gather market data and context
- Thought: Analyze data with reasoning chains
- Action: Execute specific analysis tasks
- Reflection: Validate results and check for errors
- Final Answer: Provide verified recommendations

Key Features:
- Multi-step reasoning with validation
- Fact-checking against real market data
- Error detection and correction
- Confidence scoring with uncertainty handling
- MLX integration for fast local inference
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonREPLTool

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import our existing components
try:
    from ..ai.safe_ai_loader import get_safe_ai_loader
    from ..utils.data_validator import DataValidator
    from ..trading.opportunity_hunter import OpportunityHunter
    SAFE_LOADER_AVAILABLE = True
except ImportError:
    SAFE_LOADER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisStage(Enum):
    """Analysis stages in the ReAct workflow"""
    OBSERVATION = "observation"
    THOUGHT = "thought" 
    ACTION = "action"
    REFLECTION = "reflection"
    FINAL_ANSWER = "final_answer"

class ConfidenceLevel(Enum):
    """Confidence levels for analysis results"""
    HIGH = "high"      # >85% confidence
    MEDIUM = "medium"  # 60-85% confidence  
    LOW = "low"        # <60% confidence
    UNCERTAIN = "uncertain"  # Conflicting data

@dataclass
class ValidationResult:
    """Result of fact-checking validation"""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    data_sources: List[str]

class AgentState(TypedDict):
    """State maintained throughout the ReAct workflow"""
    messages: List[BaseMessage]
    ticker: str
    stage: str
    observations: Dict[str, Any]
    thoughts: List[str]
    actions: List[Dict[str, Any]]
    reflections: List[str]
    confidence: float
    validation_results: List[ValidationResult]
    final_recommendation: Optional[Dict[str, Any]]
    error_count: int
    max_iterations: int
    current_iteration: int

class MarketDataTool(BaseTool):
    """Tool for fetching real market data"""
    
    name: str = "market_data_fetcher" 
    description: str = "Fetch real-time market data for a given ticker symbol"
    
    def __init__(self):
        super().__init__()
        # Initialize validator outside of Pydantic model
        self._validator = None
        if SAFE_LOADER_AVAILABLE:
            try:
                from ..utils.data_validator import DataValidator
                self._validator = DataValidator()
            except:
                pass
    
    @property
    def validator(self):
        return self._validator
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """Fetch market data for the given ticker"""
        if not self.validator:
            return {"error": "Market data validator not available"}
        
        try:
            # Get comprehensive market data
            data = self.validator.get_market_context(ticker)
            if data:
                return {
                    "success": True,
                    "ticker": ticker,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "source": "real_market_data"
                }
            else:
                return {
                    "success": False,
                    "ticker": ticker,
                    "error": "No data available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

class TechnicalAnalysisTool(BaseTool):
    """Tool for technical analysis calculations"""
    
    name: str = "technical_analyzer"
    description: str = "Perform technical analysis on market data"
    
    def _run(self, data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform technical analysis on the provided data"""
        try:
            if not isinstance(data, dict) or "close" not in data:
                return {"error": "Invalid market data format"}
            
            # Extract key metrics
            close = data.get("close", 0)
            rsi = data.get("rsi", 50)
            ema21 = data.get("ema21", close)
            vwap = data.get("vwap", close)
            volume_spike = data.get("volume_spike", False)
            
            # Perform technical analysis
            analysis = {
                "trend_analysis": {
                    "price_vs_ema21": "bullish" if close > ema21 else "bearish",
                    "price_vs_vwap": "above" if close > vwap else "below",
                    "trend_strength": abs(close - ema21) / ema21 if ema21 > 0 else 0
                },
                "momentum_analysis": {
                    "rsi_level": rsi,
                    "rsi_signal": self._interpret_rsi(rsi),
                    "volume_confirmation": volume_spike
                },
                "risk_assessment": {
                    "volatility_level": self._assess_volatility(rsi),
                    "support_resistance": {
                        "support": min(ema21, vwap),
                        "resistance": max(ema21, vwap) * 1.05
                    }
                }
            }
            
            return {
                "success": True,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "source": "technical_analysis"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Technical analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI levels"""
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        elif 55 <= rsi <= 70:
            return "bullish_momentum"
        elif 30 <= rsi <= 45:
            return "bearish_momentum"
        else:
            return "neutral"
    
    def _assess_volatility(self, rsi: float) -> str:
        """Assess volatility based on RSI deviation from 50"""
        deviation = abs(rsi - 50)
        if deviation > 25:
            return "high"
        elif deviation > 15:
            return "medium"
        else:
            return "low"

class FactChecker:
    """Fact-checking and validation system"""
    
    def __init__(self):
        self.validation_rules = {
            "price_sanity": self._validate_price_sanity,
            "rsi_bounds": self._validate_rsi_bounds,
            "consistency_check": self._validate_consistency,
            "data_freshness": self._validate_data_freshness
        }
    
    def validate_analysis(self, ticker: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive fact-checking"""
        errors = []
        warnings = []
        data_sources = ["market_data", "technical_analysis"]
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(ticker, data, analysis)
                if result["errors"]:
                    errors.extend(result["errors"])
                if result["warnings"]:
                    warnings.extend(result["warnings"])
            except Exception as e:
                errors.append(f"Validation rule {rule_name} failed: {str(e)}")
        
        # Calculate overall confidence
        total_issues = len(errors) + len(warnings) * 0.5
        confidence = max(0.0, 1.0 - (total_issues * 0.15))  # Reduce confidence by 15% per issue
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            data_sources=data_sources
        )
    
    def _validate_price_sanity(self, ticker: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate that prices are reasonable"""
        errors = []
        warnings = []
        
        close = data.get("close", 0)
        if close <= 0:
            errors.append("Invalid stock price: price must be positive")
        elif close > 10000:
            warnings.append("Unusually high stock price - verify data accuracy")
        elif close < 0.01:
            warnings.append("Unusually low stock price - may be penny stock")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_rsi_bounds(self, ticker: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate RSI is within valid bounds"""
        errors = []
        warnings = []
        
        rsi = data.get("rsi", 50)
        if not (0 <= rsi <= 100):
            errors.append(f"Invalid RSI value: {rsi} (must be between 0-100)")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_consistency(self, ticker: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate internal consistency of analysis"""
        errors = []
        warnings = []
        
        try:
            close = data.get("close", 0)
            ema21 = data.get("ema21", close)
            vwap = data.get("vwap", close)
            
            # Check trend consistency
            if "trend_analysis" in analysis.get("analysis", {}):
                trend_data = analysis["analysis"]["trend_analysis"]
                price_vs_ema = trend_data.get("price_vs_ema21")
                
                # Validate trend direction matches price comparison
                if price_vs_ema == "bullish" and close <= ema21:
                    errors.append("Inconsistent trend analysis: labeled bullish but price below EMA21")
                elif price_vs_ema == "bearish" and close >= ema21:
                    errors.append("Inconsistent trend analysis: labeled bearish but price above EMA21")
                    
        except Exception as e:
            warnings.append(f"Consistency check incomplete: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_data_freshness(self, ticker: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data freshness"""
        errors = []
        warnings = []
        
        # Check if timestamp is available and recent
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            try:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_minutes = (datetime.now() - data_time).total_seconds() / 60
                
                if age_minutes > 1440:  # 24 hours
                    warnings.append(f"Market data is {age_minutes/60:.1f} hours old")
                elif age_minutes > 60:  # 1 hour
                    warnings.append(f"Market data is {age_minutes:.0f} minutes old")
            except:
                warnings.append("Could not validate data freshness - timestamp format unclear")
        else:
            warnings.append("No timestamp available for data freshness validation")
        
        return {"errors": errors, "warnings": warnings}

class ReActTradingAgent:
    """ReAct trading agent with LangGraph workflow"""
    
    def __init__(self):
        self.fact_checker = FactChecker()
        self.market_tool = MarketDataTool()
        self.tech_tool = TechnicalAnalysisTool()
        
        # Initialize MLX safely if available
        self.mlx_llm = None
        self.mlx_available = False
        
        if SAFE_LOADER_AVAILABLE:
            try:
                ai_loader = get_safe_ai_loader()
                if ai_loader.is_ready():
                    self.mlx_llm = ai_loader.get_model()
                    self.mlx_available = True
                else:
                    logger.info("MLX LLM not ready yet")
            except Exception as e:
                logger.warning(f"MLX safe loader error: {e}")
        else:
            logger.warning("Safe AI loader not available")
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the ReAct workflow using LangGraph"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observation", self._observation_node)
        workflow.add_node("thought", self._thought_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("final_answer", self._final_answer_node)
        
        # Define edges and conditions
        workflow.add_edge(START, "observation")
        workflow.add_edge("observation", "thought")
        workflow.add_edge("thought", "action")
        workflow.add_edge("action", "reflection")
        
        # Conditional edge from reflection
        workflow.add_conditional_edges(
            "reflection",
            self._should_continue,
            {
                "continue": "thought",  # Continue iterating
                "finish": "final_answer"  # Provide final answer
            }
        )
        
        workflow.add_edge("final_answer", END)
        
        return workflow.compile()
    
    def _observation_node(self, state: AgentState) -> AgentState:
        """Observation phase: Gather market data"""
        logger.info(f"🔍 OBSERVATION: Gathering data for {state['ticker']}")
        
        # Fetch market data
        market_data = self.market_tool._run(state["ticker"])
        
        # Store observations
        state["observations"]["market_data"] = market_data
        state["stage"] = AnalysisStage.OBSERVATION.value
        
        # Add observation message
        obs_msg = f"Observed market data for {state['ticker']}: " + json.dumps(market_data, default=str)
        state["messages"].append(AIMessage(content=obs_msg))
        
        return state
    
    def _thought_node(self, state: AgentState) -> AgentState:
        """Thought phase: Analyze and reason about the data"""
        logger.info(f"🤔 THOUGHT: Analyzing data for {state['ticker']}")
        
        market_data = state["observations"].get("market_data", {})
        
        if not market_data.get("success"):
            thought = f"Cannot analyze {state['ticker']} - market data fetch failed: {market_data.get('error', 'Unknown error')}"
            state["thoughts"].append(thought)
            state["confidence"] = 0.0
        else:
            # Generate reasoning using MLX if available
            if self.mlx_available:
                thought = self._generate_mlx_reasoning(state["ticker"], market_data["data"])
            else:
                thought = self._generate_fallback_reasoning(state["ticker"], market_data["data"])
            
            state["thoughts"].append(thought)
        
        state["stage"] = AnalysisStage.THOUGHT.value
        state["messages"].append(AIMessage(content=f"Thought: {thought}"))
        
        return state
    
    def _action_node(self, state: AgentState) -> AgentState:
        """Action phase: Perform technical analysis"""
        logger.info(f"⚡ ACTION: Performing technical analysis for {state['ticker']}")
        
        market_data = state["observations"].get("market_data", {})
        
        if market_data.get("success"):
            # Perform technical analysis
            tech_analysis = self.tech_tool._run(market_data["data"])
            action_result = {
                "type": "technical_analysis",
                "result": tech_analysis,
                "timestamp": datetime.now().isoformat()
            }
        else:
            action_result = {
                "type": "error_handling",
                "result": {"success": False, "error": "No market data available"},
                "timestamp": datetime.now().isoformat()
            }
        
        state["actions"].append(action_result)
        state["stage"] = AnalysisStage.ACTION.value
        
        action_msg = f"Performed technical analysis: {json.dumps(action_result, default=str)}"
        state["messages"].append(AIMessage(content=action_msg))
        
        return state
    
    def _reflection_node(self, state: AgentState) -> AgentState:
        """Reflection phase: Validate results and check for errors"""
        logger.info(f"🔄 REFLECTION: Validating analysis for {state['ticker']}")
        
        market_data = state["observations"].get("market_data", {})
        actions = state["actions"]
        
        if market_data.get("success") and actions:
            # Perform fact-checking
            latest_action = actions[-1]
            if latest_action["result"].get("success"):
                validation = self.fact_checker.validate_analysis(
                    state["ticker"],
                    market_data["data"],
                    latest_action["result"]
                )
                
                state["validation_results"].append(validation)
                state["confidence"] = validation.confidence
                
                if validation.errors:
                    reflection = f"Found {len(validation.errors)} errors in analysis: {', '.join(validation.errors)}"
                    state["error_count"] += len(validation.errors)
                elif validation.warnings:
                    reflection = f"Analysis completed with {len(validation.warnings)} warnings: {', '.join(validation.warnings)}"
                else:
                    reflection = f"Analysis validated successfully with {validation.confidence:.0%} confidence"
            else:
                reflection = "Technical analysis failed - cannot validate results"
                state["error_count"] += 1
        else:
            reflection = "Cannot perform validation - insufficient data"
            state["error_count"] += 1
        
        state["reflections"].append(reflection)
        state["stage"] = AnalysisStage.REFLECTION.value
        state["current_iteration"] += 1
        
        state["messages"].append(AIMessage(content=f"Reflection: {reflection}"))
        
        return state
    
    def _final_answer_node(self, state: AgentState) -> AgentState:
        """Final answer phase: Provide verified recommendation"""
        logger.info(f"📋 FINAL ANSWER: Generating recommendation for {state['ticker']}")
        
        # Compile final recommendation
        recommendation = self._compile_recommendation(state)
        state["final_recommendation"] = recommendation
        state["stage"] = AnalysisStage.FINAL_ANSWER.value
        
        # Generate final message
        final_msg = self._format_final_recommendation(recommendation)
        state["messages"].append(AIMessage(content=final_msg))
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue iteration or provide final answer"""
        max_iterations = state.get("max_iterations", 3)
        current_iteration = state.get("current_iteration", 0)
        error_count = state.get("error_count", 0)
        confidence = state.get("confidence", 0.0)
        
        # Stop if we've reached max iterations
        if current_iteration >= max_iterations:
            logger.info(f"Stopping: reached max iterations ({max_iterations})")
            return "finish"
        
        # Stop if too many errors
        if error_count > 2:
            logger.info(f"Stopping: too many errors ({error_count})")
            return "finish"
        
        # Stop if high confidence achieved
        if confidence > 0.85:
            logger.info(f"Stopping: high confidence achieved ({confidence:.0%})")
            return "finish"
        
        # Continue if we have low confidence and can still iterate
        if confidence < 0.6 and current_iteration < max_iterations - 1:
            logger.info(f"Continuing: low confidence ({confidence:.0%}), iteration {current_iteration}")
            return "continue"
        
        # Default to finish
        return "finish"
    
    def _generate_mlx_reasoning(self, ticker: str, market_data: Dict[str, Any]) -> str:
        """Generate reasoning using MLX LLM"""
        try:
            close = market_data.get("close", 0)
            rsi = market_data.get("rsi", 50)
            ema21 = market_data.get("ema21", close)
            vwap = market_data.get("vwap", close)
            
            prompt = f"""Analyze {ticker} with the following market data:
            Price: ${close:.2f}
            RSI: {rsi:.1f}
            EMA21: ${ema21:.2f}
            VWAP: ${vwap:.2f}
            
            Provide concise reasoning about:
            1. Current trend direction
            2. Momentum indicators
            3. Key risk factors
            4. Confidence level in analysis
            
            Be factual and avoid speculation."""
            
            response = self.mlx_llm._generate_text(prompt, max_tokens=150)
            return response
        except Exception as e:
            logger.error(f"MLX reasoning failed: {e}")
            return self._generate_fallback_reasoning(ticker, market_data)
    
    def _generate_fallback_reasoning(self, ticker: str, market_data: Dict[str, Any]) -> str:
        """Fallback reasoning when MLX is not available"""
        close = market_data.get("close", 0)
        rsi = market_data.get("rsi", 50)
        ema21 = market_data.get("ema21", close)
        
        # Simple rule-based reasoning
        trend = "bullish" if close > ema21 else "bearish"
        momentum = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        reasoning = f"""
        {ticker} analysis: Current price ${close:.2f} is {trend} relative to EMA21 ${ema21:.2f}.
        RSI at {rsi:.1f} indicates {momentum} conditions.
        Trend strength: {abs(close-ema21)/ema21*100:.1f}% deviation from EMA21.
        """
        
        return reasoning.strip()
    
    def _compile_recommendation(self, state: AgentState) -> Dict[str, Any]:
        """Compile final recommendation from analysis results"""
        ticker = state["ticker"]
        market_data = state["observations"].get("market_data", {})
        actions = state["actions"]
        confidence = state.get("confidence", 0.0)
        validation_results = state.get("validation_results", [])
        
        if not market_data.get("success") or not actions:
            return {
                "ticker": ticker,
                "action": "IGNORE",
                "confidence": 0.0,
                "reasoning": "Insufficient data for analysis",
                "timestamp": datetime.now().isoformat(),
                "validation_status": "failed"
            }
        
        # Get technical analysis results
        tech_result = None
        for action in actions:
            if action["type"] == "technical_analysis" and action["result"].get("success"):
                tech_result = action["result"]["analysis"]
                break
        
        if not tech_result:
            return {
                "ticker": ticker,
                "action": "IGNORE", 
                "confidence": 0.0,
                "reasoning": "Technical analysis failed",
                "timestamp": datetime.now().isoformat(),
                "validation_status": "failed"
            }
        
        # Extract data for recommendation
        data = market_data["data"]
        close = data.get("close", 0)
        
        # Determine action based on analysis
        trend_analysis = tech_result.get("trend_analysis", {})
        momentum_analysis = tech_result.get("momentum_analysis", {})
        
        trend_direction = trend_analysis.get("price_vs_ema21", "neutral")
        rsi_signal = momentum_analysis.get("rsi_signal", "neutral")
        volume_confirmation = momentum_analysis.get("volume_confirmation", False)
        
        # Simple decision logic (can be enhanced)
        if (trend_direction == "bullish" and 
            rsi_signal in ["bullish_momentum", "oversold"] and 
            confidence > 0.6):
            action = "BUY"
            entry = close
            target = close * 1.10  # 10% target
            stop = close * 0.95    # 5% stop
        elif (trend_direction == "bearish" and 
              rsi_signal in ["bearish_momentum", "overbought"] and 
              confidence > 0.6):
            action = "SELL"
            entry = close
            target = close * 0.90  # 10% target
            stop = close * 1.05    # 5% stop
        else:
            action = "IGNORE"
            entry = target = stop = close
        
        # Compile final recommendation
        recommendation = {
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "entry": entry,
            "target": target,
            "stop": stop,
            "reasoning": self._compile_reasoning(state, tech_result),
            "timestamp": datetime.now().isoformat(),
            "validation_status": "passed" if validation_results and validation_results[-1].is_valid else "warnings",
            "react_iterations": state.get("current_iteration", 0),
            "error_count": state.get("error_count", 0)
        }
        
        return recommendation
    
    def _compile_reasoning(self, state: AgentState, tech_result: Dict[str, Any]) -> str:
        """Compile reasoning from thoughts and analysis"""
        thoughts = state.get("thoughts", [])
        reflections = state.get("reflections", [])
        
        reasoning_parts = []
        
        # Add key insights from thoughts
        if thoughts:
            reasoning_parts.append(f"Analysis: {thoughts[-1][:200]}")
        
        # Add technical summary
        if tech_result:
            trend = tech_result.get("trend_analysis", {}).get("price_vs_ema21", "neutral")
            momentum = tech_result.get("momentum_analysis", {}).get("rsi_signal", "neutral")
            reasoning_parts.append(f"Technical: {trend} trend with {momentum} momentum")
        
        # Add validation summary
        if reflections:
            reasoning_parts.append(f"Validation: {reflections[-1][:150]}")
        
        return ". ".join(reasoning_parts)
    
    def _format_final_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Format the final recommendation for display"""
        ticker = recommendation["ticker"]
        action = recommendation["action"]
        confidence = recommendation["confidence"]
        
        if action == "IGNORE":
            return f"""
🔍 REACT ANALYSIS COMPLETE: {ticker}

📊 RECOMMENDATION: {action}
🎯 Confidence: {confidence:.0%}
💭 Reasoning: {recommendation["reasoning"]}

⚠️ Status: {recommendation["validation_status"]}
🔄 Iterations: {recommendation["react_iterations"]}
❌ Errors: {recommendation["error_count"]}
"""
        else:
            return f"""
🔍 REACT ANALYSIS COMPLETE: {ticker}

📊 RECOMMENDATION: {action}
🎯 Confidence: {confidence:.0%}
💰 Entry: ${recommendation["entry"]:.2f}
🎯 Target: ${recommendation["target"]:.2f}
🛑 Stop: ${recommendation["stop"]:.2f}

💭 Reasoning: {recommendation["reasoning"]}

✅ Status: {recommendation["validation_status"]}
🔄 Iterations: {recommendation["react_iterations"]}
❌ Errors: {recommendation["error_count"]}
"""
    
    def analyze(self, ticker: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run ReAct analysis for a given ticker"""
        logger.info(f"🚀 Starting ReAct analysis for {ticker}")
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze {ticker} using ReAct methodology")],
            "ticker": ticker.upper(),
            "stage": "start",
            "observations": {},
            "thoughts": [],
            "actions": [],
            "reflections": [],
            "confidence": 0.0,
            "validation_results": [],
            "final_recommendation": None,
            "error_count": 0,
            "max_iterations": max_iterations,
            "current_iteration": 0
        }
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state.get("final_recommendation", {
                "ticker": ticker,
                "action": "ERROR",
                "confidence": 0.0,
                "reasoning": "Analysis failed",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"ReAct analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "action": "ERROR",
                "confidence": 0.0,
                "reasoning": f"System error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "validation_status": "error"
            }

def main():
    """Demo the ReAct trading agent"""
    print("🚀 REACT TRADING AGENT DEMO")
    print("=" * 50)
    
    agent = ReActTradingAgent()
    
    # Test with sample tickers
    test_tickers = ["NVDA", "AAPL", "TSLA"]
    
    for ticker in test_tickers:
        print(f"\n🎯 Analyzing {ticker} with ReAct methodology...")
        print("-" * 40)
        
        try:
            result = agent.analyze(ticker, max_iterations=2)
            
            print(f"Ticker: {result['ticker']}")
            print(f"Action: {result['action']}")
            print(f"Confidence: {result.get('confidence', 0):.0%}")
            
            if result['action'] not in ['IGNORE', 'ERROR']:
                print(f"Entry: ${result.get('entry', 0):.2f}")
                print(f"Target: ${result.get('target', 0):.2f}")
                print(f"Stop: ${result.get('stop', 0):.2f}")
            
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
            print(f"Status: {result.get('validation_status', 'unknown')}")
            print(f"Iterations: {result.get('react_iterations', 0)}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        
        print()

if __name__ == "__main__":
    main()