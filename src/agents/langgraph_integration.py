"""
LangGraph Integration for River Trading System
=============================================

Advanced integration layer that combines LangGraph workflows with
the existing River Trading System, providing enhanced robustness
and hallucination prevention through structured reasoning.

Features:
- Multi-agent workflows for different trading scenarios
- Fact-checking and validation pipelines
- Error recovery and uncertainty handling
- Integration with MLX and existing components
- Conversation memory and context management
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Import our components
try:
    from .react_trading_agent import ReActTradingAgent, AgentState, ValidationResult
    from ..ai.mlx_trading_llm import MLXTradingLLM
    from ..trading.opportunity_hunter import OpportunityHunter
    from ..utils.data_validator import DataValidator
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Components not fully available: {e}")
    COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Different workflow types for various trading scenarios"""
    SINGLE_STOCK_ANALYSIS = "single_stock_analysis"
    OPPORTUNITY_HUNTING = "opportunity_hunting"
    PORTFOLIO_REVIEW = "portfolio_review"
    MARKET_SENTIMENT = "market_sentiment"
    RISK_ASSESSMENT = "risk_assessment"

class UncertaintyLevel(Enum):
    """Levels of uncertainty in analysis"""
    LOW = "low"           # High confidence, consistent data
    MEDIUM = "medium"     # Some conflicting signals
    HIGH = "high"         # Significant uncertainty
    CONFLICTED = "conflicted"  # Contradictory information

@dataclass
class WorkflowResult:
    """Result from a completed workflow"""
    workflow_type: str
    success: bool
    confidence: float
    recommendations: List[Dict[str, Any]]
    uncertainty_level: str
    validation_summary: Dict[str, Any]
    reasoning_chain: List[str]
    error_count: int
    execution_time: float
    timestamp: str

class EnhancedAgentState(AgentState):
    """Extended state for LangGraph workflows"""
    workflow_type: str
    uncertainty_level: str
    cross_validation_results: List[Dict[str, Any]]
    consensus_score: float
    alternative_scenarios: List[Dict[str, Any]]

class LangGraphTradingOrchestrator:
    """Main orchestrator for LangGraph-enhanced trading workflows"""
    
    def __init__(self):
        self.react_agent = None
        self.mlx_llm = None
        self.opportunity_hunter = None
        self.data_validator = None
        
        # Initialize components if available
        if COMPONENTS_AVAILABLE:
            try:
                self.react_agent = ReActTradingAgent()
                self.mlx_llm = MLXTradingLLM() if hasattr(globals(), 'MLXTradingLLM') else None
                self.opportunity_hunter = OpportunityHunter() if hasattr(globals(), 'OpportunityHunter') else None
                self.data_validator = DataValidator() if hasattr(globals(), 'DataValidator') else None
                logger.info("LangGraph orchestrator initialized with full components")
            except Exception as e:
                logger.warning(f"Component initialization partially failed: {e}")
        
        # Build workflows
        self.workflows = self._build_workflows()
    
    def _build_workflows(self) -> Dict[str, StateGraph]:
        """Build different LangGraph workflows for various scenarios"""
        workflows = {}
        
        # Single stock analysis workflow
        workflows[WorkflowType.SINGLE_STOCK_ANALYSIS.value] = self._build_single_stock_workflow()
        
        # Opportunity hunting workflow
        workflows[WorkflowType.OPPORTUNITY_HUNTING.value] = self._build_opportunity_workflow()
        
        # Portfolio review workflow
        workflows[WorkflowType.PORTFOLIO_REVIEW.value] = self._build_portfolio_workflow()
        
        return workflows
    
    def _build_single_stock_workflow(self) -> StateGraph:
        """Build workflow for single stock analysis with cross-validation"""
        
        workflow = StateGraph(EnhancedAgentState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("primary_analysis", self._primary_analysis)
        workflow.add_node("cross_validate", self._cross_validate_analysis)
        workflow.add_node("uncertainty_assessment", self._assess_uncertainty)
        workflow.add_node("consensus_building", self._build_consensus)
        workflow.add_node("final_recommendation", self._generate_final_recommendation)
        
        # Define flow
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "primary_analysis")
        workflow.add_edge("primary_analysis", "cross_validate")
        workflow.add_edge("cross_validate", "uncertainty_assessment")
        workflow.add_edge("uncertainty_assessment", "consensus_building")
        workflow.add_edge("consensus_building", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        return workflow.compile()
    
    def _build_opportunity_workflow(self) -> StateGraph:
        """Build workflow for opportunity hunting with multiple strategies"""
        
        workflow = StateGraph(EnhancedAgentState)
        
        # Add nodes
        workflow.add_node("scan_universe", self._scan_market_universe)
        workflow.add_node("filter_candidates", self._filter_candidates)
        workflow.add_node("detailed_analysis", self._detailed_analysis)
        workflow.add_node("rank_opportunities", self._rank_opportunities)
        workflow.add_node("risk_assessment", self._assess_risks)
        workflow.add_node("final_selection", self._select_final_opportunities)
        
        # Define flow
        workflow.add_edge(START, "scan_universe")
        workflow.add_edge("scan_universe", "filter_candidates")
        workflow.add_edge("filter_candidates", "detailed_analysis")
        workflow.add_edge("detailed_analysis", "rank_opportunities")
        workflow.add_edge("rank_opportunities", "risk_assessment")
        workflow.add_edge("risk_assessment", "final_selection")
        workflow.add_edge("final_selection", END)
        
        return workflow.compile()
    
    def _build_portfolio_workflow(self) -> StateGraph:
        """Build workflow for portfolio review and optimization"""
        
        workflow = StateGraph(EnhancedAgentState)
        
        # Add nodes
        workflow.add_node("portfolio_snapshot", self._take_portfolio_snapshot)
        workflow.add_node("performance_analysis", self._analyze_performance)
        workflow.add_node("risk_analysis", self._analyze_portfolio_risk)
        workflow.add_node("rebalancing_suggestions", self._suggest_rebalancing)
        workflow.add_node("optimization_recommendations", self._optimize_portfolio)
        
        # Define flow
        workflow.add_edge(START, "portfolio_snapshot")
        workflow.add_edge("portfolio_snapshot", "performance_analysis")
        workflow.add_edge("performance_analysis", "risk_analysis")
        workflow.add_edge("risk_analysis", "rebalancing_suggestions")
        workflow.add_edge("rebalancing_suggestions", "optimization_recommendations")
        workflow.add_edge("optimization_recommendations", END)
        
        return workflow.compile()
    
    # Workflow node implementations
    
    def _initialize_analysis(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Initialize analysis with context setup"""
        logger.info(f"🔧 Initializing analysis for {state.get('ticker', 'unknown')}")
        
        state["workflow_type"] = WorkflowType.SINGLE_STOCK_ANALYSIS.value
        state["uncertainty_level"] = UncertaintyLevel.MEDIUM.value
        state["cross_validation_results"] = []
        state["consensus_score"] = 0.0
        state["alternative_scenarios"] = []
        
        # Add initialization message
        init_msg = f"Initialized enhanced analysis workflow for {state.get('ticker', 'target')}"
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(AIMessage(content=init_msg))
        
        return state
    
    def _primary_analysis(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Run primary ReAct analysis"""
        logger.info("🎯 Running primary ReAct analysis")
        
        ticker = state.get("ticker", "")
        if not ticker:
            state["messages"].append(AIMessage(content="Error: No ticker provided for analysis"))
            return state
        
        # Run ReAct analysis if available
        if self.react_agent:
            try:
                react_result = self.react_agent.analyze(ticker, max_iterations=2)
                state["final_recommendation"] = react_result
                
                analysis_msg = f"Primary ReAct analysis complete: {react_result.get('action', 'UNKNOWN')} with {react_result.get('confidence', 0):.0%} confidence"
                state["messages"].append(AIMessage(content=analysis_msg))
                
            except Exception as e:
                error_msg = f"Primary analysis failed: {str(e)}"
                state["messages"].append(AIMessage(content=error_msg))
                logger.error(error_msg)
        else:
            fallback_msg = "ReAct agent not available, using fallback analysis"
            state["messages"].append(AIMessage(content=fallback_msg))
            
            # Fallback analysis
            state["final_recommendation"] = {
                "ticker": ticker,
                "action": "IGNORE",
                "confidence": 0.3,
                "reasoning": "Fallback analysis - ReAct agent unavailable",
                "timestamp": datetime.now().isoformat()
            }
        
        return state
    
    def _cross_validate_analysis(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Cross-validate the primary analysis with alternative methods"""
        logger.info("🔍 Cross-validating analysis")
        
        ticker = state.get("ticker", "")
        primary_result = state.get("final_recommendation", {})
        
        cross_validations = []
        
        # Validation 1: MLX independent analysis
        if self.mlx_llm and self.data_validator:
            try:
                market_data = self.data_validator.get_market_context(ticker)
                if market_data:
                    mlx_analysis = self.mlx_llm.analyze_opportunity(ticker, market_data)
                    cross_validations.append({
                        "method": "mlx_independent",
                        "result": mlx_analysis,
                        "agreement": self._calculate_agreement(primary_result, mlx_analysis)
                    })
            except Exception as e:
                logger.warning(f"MLX cross-validation failed: {e}")
        
        # Validation 2: Simple technical rules
        technical_validation = self._simple_technical_validation(ticker)
        if technical_validation:
            cross_validations.append({
                "method": "technical_rules",
                "result": technical_validation,
                "agreement": self._calculate_agreement(primary_result, technical_validation)
            })
        
        state["cross_validation_results"] = cross_validations
        
        # Calculate average agreement
        if cross_validations:
            avg_agreement = sum(cv["agreement"] for cv in cross_validations) / len(cross_validations)
            validation_msg = f"Cross-validation complete: {len(cross_validations)} methods, {avg_agreement:.0%} average agreement"
        else:
            validation_msg = "Cross-validation unavailable - insufficient methods"
        
        state["messages"].append(AIMessage(content=validation_msg))
        
        return state
    
    def _assess_uncertainty(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Assess uncertainty level based on cross-validation results"""
        logger.info("📊 Assessing uncertainty level")
        
        cross_validations = state.get("cross_validation_results", [])
        primary_confidence = state.get("final_recommendation", {}).get("confidence", 0.0)
        
        if not cross_validations:
            uncertainty_level = UncertaintyLevel.HIGH.value
            uncertainty_msg = "High uncertainty: No cross-validation available"
        else:
            agreements = [cv["agreement"] for cv in cross_validations]
            avg_agreement = sum(agreements) / len(agreements)
            min_agreement = min(agreements)
            
            # Determine uncertainty level
            if avg_agreement > 0.8 and min_agreement > 0.6 and primary_confidence > 0.7:
                uncertainty_level = UncertaintyLevel.LOW.value
            elif avg_agreement > 0.6 and primary_confidence > 0.5:
                uncertainty_level = UncertaintyLevel.MEDIUM.value
            elif min_agreement < 0.3:
                uncertainty_level = UncertaintyLevel.CONFLICTED.value
            else:
                uncertainty_level = UncertaintyLevel.HIGH.value
            
            uncertainty_msg = f"Uncertainty assessment: {uncertainty_level} (avg agreement: {avg_agreement:.0%})"
        
        state["uncertainty_level"] = uncertainty_level
        state["messages"].append(AIMessage(content=uncertainty_msg))
        
        return state
    
    def _build_consensus(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Build consensus from multiple analysis methods"""
        logger.info("🤝 Building consensus recommendation")
        
        primary_result = state.get("final_recommendation", {})
        cross_validations = state.get("cross_validation_results", [])
        uncertainty_level = state.get("uncertainty_level", UncertaintyLevel.HIGH.value)
        
        # Weight the primary analysis and cross-validations
        consensus_score = primary_result.get("confidence", 0.0) * 0.6
        
        if cross_validations:
            # Add cross-validation weight
            avg_agreement = sum(cv["agreement"] for cv in cross_validations) / len(cross_validations)
            consensus_score += avg_agreement * 0.4
        
        # Adjust for uncertainty
        if uncertainty_level == UncertaintyLevel.HIGH.value:
            consensus_score *= 0.7
        elif uncertainty_level == UncertaintyLevel.CONFLICTED.value:
            consensus_score *= 0.5
        
        state["consensus_score"] = consensus_score
        
        # Generate alternative scenarios if uncertainty is high
        if uncertainty_level in [UncertaintyLevel.HIGH.value, UncertaintyLevel.CONFLICTED.value]:
            alternatives = self._generate_alternative_scenarios(state)
            state["alternative_scenarios"] = alternatives
        
        consensus_msg = f"Consensus built: {consensus_score:.0%} consensus score with {uncertainty_level} uncertainty"
        state["messages"].append(AIMessage(content=consensus_msg))
        
        return state
    
    def _generate_final_recommendation(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate final recommendation with uncertainty handling"""
        logger.info("📋 Generating final recommendation")
        
        primary_result = state.get("final_recommendation", {})
        consensus_score = state.get("consensus_score", 0.0)
        uncertainty_level = state.get("uncertainty_level", UncertaintyLevel.HIGH.value)
        alternative_scenarios = state.get("alternative_scenarios", [])
        
        # Modify recommendation based on consensus and uncertainty
        final_recommendation = primary_result.copy()
        final_recommendation["consensus_score"] = consensus_score
        final_recommendation["uncertainty_level"] = uncertainty_level
        final_recommendation["alternative_scenarios"] = alternative_scenarios
        
        # Adjust action based on uncertainty
        if uncertainty_level == UncertaintyLevel.CONFLICTED.value and final_recommendation.get("action") != "IGNORE":
            final_recommendation["action"] = "MONITOR"  # New action for conflicted situations
            final_recommendation["reasoning"] += f" | CONFLICTED: Multiple analysis methods disagree - recommend monitoring"
        elif uncertainty_level == UncertaintyLevel.HIGH.value and consensus_score < 0.6:
            final_recommendation["confidence"] = min(final_recommendation.get("confidence", 0), 0.6)
            final_recommendation["reasoning"] += f" | HIGH UNCERTAINTY: Limited confidence due to data quality"
        
        # Update the state
        state["final_recommendation"] = final_recommendation
        
        # Format final message
        action = final_recommendation.get("action", "UNKNOWN")
        confidence = final_recommendation.get("confidence", 0)
        final_msg = f"Final recommendation: {action} with {confidence:.0%} confidence (consensus: {consensus_score:.0%}, uncertainty: {uncertainty_level})"
        
        state["messages"].append(AIMessage(content=final_msg))
        
        return state
    
    # Opportunity hunting workflow nodes
    
    def _scan_market_universe(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Scan the market universe for potential opportunities"""
        logger.info("🌐 Scanning market universe")
        
        if self.opportunity_hunter:
            try:
                # Get initial screening results
                opportunities = self.opportunity_hunter.hunt_opportunities(max_opportunities=20, debug_mode=True)
                
                if isinstance(opportunities, dict) and "opportunities" in opportunities:
                    candidates = opportunities["opportunities"]
                    stats = opportunities.get("screening_stats", {})
                else:
                    candidates = opportunities if opportunities else []
                    stats = {}
                
                state["observations"] = {
                    "candidates": candidates,
                    "screening_stats": stats,
                    "scan_timestamp": datetime.now().isoformat()
                }
                
                scan_msg = f"Market scan complete: {len(candidates)} candidates identified"
                
            except Exception as e:
                state["observations"] = {"candidates": [], "error": str(e)}
                scan_msg = f"Market scan failed: {str(e)}"
        else:
            state["observations"] = {"candidates": [], "error": "Opportunity hunter not available"}
            scan_msg = "Market scan unavailable - opportunity hunter not loaded"
        
        state["messages"].append(AIMessage(content=scan_msg))
        return state
    
    def _filter_candidates(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Filter candidates based on enhanced criteria"""
        logger.info("🔽 Filtering candidates")
        
        candidates = state.get("observations", {}).get("candidates", [])
        
        # Enhanced filtering criteria
        filtered_candidates = []
        for candidate in candidates:
            rec = candidate.get("recommendation", {})
            confidence = rec.get("confidence", 0)
            composite_score = candidate.get("composite_score", 0)
            
            # Multi-criteria filtering
            if (confidence > 0.75 and 
                composite_score > 0.7 and 
                rec.get("action") in ["BUY", "SELL"]):
                filtered_candidates.append(candidate)
        
        state["observations"]["filtered_candidates"] = filtered_candidates
        
        filter_msg = f"Filtering complete: {len(filtered_candidates)} high-quality candidates remaining"
        state["messages"].append(AIMessage(content=filter_msg))
        
        return state
    
    def _detailed_analysis(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Perform detailed analysis on filtered candidates"""
        logger.info("🔬 Performing detailed analysis")
        
        filtered_candidates = state.get("observations", {}).get("filtered_candidates", [])
        
        detailed_results = []
        for candidate in filtered_candidates[:5]:  # Limit to top 5 for detailed analysis
            ticker = candidate.get("ticker", "")
            if ticker and self.react_agent:
                try:
                    # Run ReAct analysis on each candidate
                    detailed_result = self.react_agent.analyze(ticker, max_iterations=1)
                    detailed_results.append({
                        "ticker": ticker,
                        "original_candidate": candidate,
                        "detailed_analysis": detailed_result
                    })
                except Exception as e:
                    logger.warning(f"Detailed analysis failed for {ticker}: {e}")
        
        state["observations"]["detailed_results"] = detailed_results
        
        analysis_msg = f"Detailed analysis complete: {len(detailed_results)} candidates analyzed"
        state["messages"].append(AIMessage(content=analysis_msg))
        
        return state
    
    def _rank_opportunities(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Rank opportunities based on multiple factors"""
        logger.info("📊 Ranking opportunities")
        
        detailed_results = state.get("observations", {}).get("detailed_results", [])
        
        # Enhanced ranking algorithm
        ranked_opportunities = []
        for result in detailed_results:
            detailed_analysis = result.get("detailed_analysis", {})
            original_candidate = result.get("original_candidate", {})
            
            # Calculate composite ranking score
            confidence = detailed_analysis.get("confidence", 0)
            composite_score = original_candidate.get("composite_score", 0)
            validation_bonus = 0.1 if detailed_analysis.get("validation_status") == "passed" else 0
            
            ranking_score = (confidence * 0.5 + composite_score * 0.4 + validation_bonus)
            
            ranked_opportunities.append({
                **result,
                "ranking_score": ranking_score
            })
        
        # Sort by ranking score
        ranked_opportunities.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        state["observations"]["ranked_opportunities"] = ranked_opportunities
        
        ranking_msg = f"Ranking complete: Top opportunity is {ranked_opportunities[0]['ticker'] if ranked_opportunities else 'None'}"
        state["messages"].append(AIMessage(content=ranking_msg))
        
        return state
    
    def _assess_risks(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Assess risks for ranked opportunities"""
        logger.info("⚠️ Assessing risks")
        
        ranked_opportunities = state.get("observations", {}).get("ranked_opportunities", [])
        
        risk_assessed_opportunities = []
        for opp in ranked_opportunities:
            detailed_analysis = opp.get("detailed_analysis", {})
            
            # Risk assessment factors
            confidence = detailed_analysis.get("confidence", 0)
            error_count = detailed_analysis.get("error_count", 0)
            validation_status = detailed_analysis.get("validation_status", "unknown")
            
            # Calculate risk score (lower is better)
            risk_score = 0.0
            if error_count > 0:
                risk_score += error_count * 0.2
            if validation_status != "passed":
                risk_score += 0.3
            if confidence < 0.7:
                risk_score += (0.7 - confidence)
            
            risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
            
            risk_assessed_opportunities.append({
                **opp,
                "risk_score": risk_score,
                "risk_level": risk_level
            })
        
        state["observations"]["risk_assessed_opportunities"] = risk_assessed_opportunities
        
        risk_msg = f"Risk assessment complete: {len([o for o in risk_assessed_opportunities if o['risk_level'] == 'low'])} low-risk opportunities"
        state["messages"].append(AIMessage(content=risk_msg))
        
        return state
    
    def _select_final_opportunities(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Select final opportunities for recommendation"""
        logger.info("✅ Selecting final opportunities")
        
        risk_assessed_opportunities = state.get("observations", {}).get("risk_assessed_opportunities", [])
        
        # Select top opportunities with acceptable risk
        final_opportunities = []
        for opp in risk_assessed_opportunities:
            if (opp.get("risk_level") in ["low", "medium"] and 
                opp.get("ranking_score", 0) > 0.7 and
                len(final_opportunities) < 3):  # Limit to top 3
                final_opportunities.append(opp)
        
        state["final_recommendation"] = {
            "workflow_type": WorkflowType.OPPORTUNITY_HUNTING.value,
            "opportunities": final_opportunities,
            "total_scanned": len(state.get("observations", {}).get("candidates", [])),
            "final_count": len(final_opportunities),
            "timestamp": datetime.now().isoformat()
        }
        
        selection_msg = f"Final selection: {len(final_opportunities)} opportunities recommended"
        state["messages"].append(AIMessage(content=selection_msg))
        
        return state
    
    # Portfolio workflow nodes (placeholder implementations)
    
    def _take_portfolio_snapshot(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Take a snapshot of current portfolio"""
        # Placeholder implementation
        state["observations"] = {"portfolio_snapshot": "Not implemented"}
        state["messages"].append(AIMessage(content="Portfolio snapshot: Feature not yet implemented"))
        return state
    
    def _analyze_performance(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Analyze portfolio performance"""
        # Placeholder implementation
        state["messages"].append(AIMessage(content="Performance analysis: Feature not yet implemented"))
        return state
    
    def _analyze_portfolio_risk(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Analyze portfolio risk"""
        # Placeholder implementation
        state["messages"].append(AIMessage(content="Risk analysis: Feature not yet implemented"))
        return state
    
    def _suggest_rebalancing(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Suggest portfolio rebalancing"""
        # Placeholder implementation
        state["messages"].append(AIMessage(content="Rebalancing suggestions: Feature not yet implemented"))
        return state
    
    def _optimize_portfolio(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Optimize portfolio allocation"""
        # Placeholder implementation
        state["final_recommendation"] = {"message": "Portfolio optimization not yet implemented"}
        state["messages"].append(AIMessage(content="Portfolio optimization: Feature not yet implemented"))
        return state
    
    # Helper methods
    
    def _calculate_agreement(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calculate agreement between two analysis results"""
        if not result1 or not result2:
            return 0.0
        
        action1 = result1.get("action", "UNKNOWN")
        action2 = result2.get("action", "UNKNOWN")
        conf1 = result1.get("confidence", 0.0)
        conf2 = result2.get("confidence", 0.0)
        
        # Action agreement
        action_agreement = 1.0 if action1 == action2 else 0.0
        
        # Confidence similarity
        conf_similarity = 1.0 - abs(conf1 - conf2)
        
        # Combined agreement
        return (action_agreement * 0.7 + conf_similarity * 0.3)
    
    def _simple_technical_validation(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Simple technical validation using basic rules"""
        if not self.data_validator:
            return None
        
        try:
            data = self.data_validator.get_market_context(ticker)
            if not data:
                return None
            
            close = data.get("close", 0)
            rsi = data.get("rsi", 50)
            ema21 = data.get("ema21", close)
            
            # Simple technical rules
            if close > ema21 and 50 < rsi < 70:
                action = "BUY"
                confidence = 0.6
            elif close < ema21 and 30 < rsi < 50:
                action = "SELL"
                confidence = 0.6
            else:
                action = "IGNORE"
                confidence = 0.4
            
            return {
                "ticker": ticker,
                "action": action,
                "confidence": confidence,
                "reasoning": "Simple technical rules validation",
                "method": "technical_rules"
            }
            
        except Exception as e:
            logger.warning(f"Technical validation failed for {ticker}: {e}")
            return None
    
    def _generate_alternative_scenarios(self, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Generate alternative scenarios for high uncertainty situations"""
        primary_result = state.get("final_recommendation", {})
        
        scenarios = []
        
        # Scenario 1: More conservative approach
        conservative = primary_result.copy()
        conservative["confidence"] = min(conservative.get("confidence", 0), 0.6)
        if conservative.get("action") == "BUY":
            conservative["target"] = conservative.get("entry", 0) * 1.05  # Reduce target
        scenarios.append({
            "name": "conservative",
            "description": "More conservative targets and reduced confidence",
            "recommendation": conservative
        })
        
        # Scenario 2: Wait and monitor
        monitor_scenario = {
            "name": "monitor",
            "description": "Wait for better confirmation signals",
            "recommendation": {
                "ticker": primary_result.get("ticker", ""),
                "action": "MONITOR",
                "confidence": 0.7,
                "reasoning": "High uncertainty suggests waiting for clearer signals"
            }
        }
        scenarios.append(monitor_scenario)
        
        return scenarios
    
    # Public interface methods
    
    def analyze_stock(self, ticker: str, workflow_config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Analyze a single stock using the enhanced LangGraph workflow"""
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting enhanced stock analysis for {ticker}")
        
        # Get the appropriate workflow
        workflow = self.workflows.get(WorkflowType.SINGLE_STOCK_ANALYSIS.value)
        if not workflow:
            return WorkflowResult(
                workflow_type=WorkflowType.SINGLE_STOCK_ANALYSIS.value,
                success=False,
                confidence=0.0,
                recommendations=[],
                uncertainty_level=UncertaintyLevel.HIGH.value,
                validation_summary={},
                reasoning_chain=["Workflow not available"],
                error_count=1,
                execution_time=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=f"Analyze {ticker} using enhanced LangGraph workflow")],
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
                "max_iterations": 3,
                "current_iteration": 0
            }
            
            # Run the workflow
            final_state = workflow.invoke(initial_state)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            recommendation = final_state.get("final_recommendation", {})
            uncertainty_level = final_state.get("uncertainty_level", UncertaintyLevel.MEDIUM.value)
            consensus_score = final_state.get("consensus_score", 0.0)
            
            # Build reasoning chain from messages
            reasoning_chain = [msg.content for msg in final_state.get("messages", []) if isinstance(msg, AIMessage)]
            
            return WorkflowResult(
                workflow_type=WorkflowType.SINGLE_STOCK_ANALYSIS.value,
                success=True,
                confidence=consensus_score,
                recommendations=[recommendation] if recommendation else [],
                uncertainty_level=uncertainty_level,
                validation_summary={
                    "cross_validations": len(final_state.get("cross_validation_results", [])),
                    "consensus_score": consensus_score,
                    "error_count": final_state.get("error_count", 0)
                },
                reasoning_chain=reasoning_chain,
                error_count=final_state.get("error_count", 0),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Enhanced stock analysis failed for {ticker}: {e}")
            
            return WorkflowResult(
                workflow_type=WorkflowType.SINGLE_STOCK_ANALYSIS.value,
                success=False,
                confidence=0.0,
                recommendations=[],
                uncertainty_level=UncertaintyLevel.HIGH.value,
                validation_summary={"error": str(e)},
                reasoning_chain=[f"Workflow failed: {str(e)}"],
                error_count=1,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    def hunt_opportunities(self, config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Hunt for opportunities using the enhanced LangGraph workflow"""
        start_time = datetime.now()
        
        logger.info("🎯 Starting enhanced opportunity hunting")
        
        workflow = self.workflows.get(WorkflowType.OPPORTUNITY_HUNTING.value)
        if not workflow:
            return WorkflowResult(
                workflow_type=WorkflowType.OPPORTUNITY_HUNTING.value,
                success=False,
                confidence=0.0,
                recommendations=[],
                uncertainty_level=UncertaintyLevel.HIGH.value,
                validation_summary={},
                reasoning_chain=["Opportunity hunting workflow not available"],
                error_count=1,
                execution_time=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content="Hunt for market opportunities using enhanced workflow")],
                "ticker": "",
                "stage": "start",
                "observations": {},
                "thoughts": [],
                "actions": [],
                "reflections": [],
                "confidence": 0.0,
                "validation_results": [],
                "final_recommendation": None,
                "error_count": 0,
                "max_iterations": 1,
                "current_iteration": 0
            }
            
            # Run the workflow
            final_state = workflow.invoke(initial_state)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            recommendation = final_state.get("final_recommendation", {})
            opportunities = recommendation.get("opportunities", [])
            
            # Build reasoning chain
            reasoning_chain = [msg.content for msg in final_state.get("messages", []) if isinstance(msg, AIMessage)]
            
            return WorkflowResult(
                workflow_type=WorkflowType.OPPORTUNITY_HUNTING.value,
                success=True,
                confidence=0.8 if opportunities else 0.3,
                recommendations=opportunities,
                uncertainty_level=UncertaintyLevel.MEDIUM.value,
                validation_summary={
                    "total_scanned": recommendation.get("total_scanned", 0),
                    "final_count": recommendation.get("final_count", 0)
                },
                reasoning_chain=reasoning_chain,
                error_count=final_state.get("error_count", 0),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Enhanced opportunity hunting failed: {e}")
            
            return WorkflowResult(
                workflow_type=WorkflowType.OPPORTUNITY_HUNTING.value,
                success=False,
                confidence=0.0,
                recommendations=[],
                uncertainty_level=UncertaintyLevel.HIGH.value,
                validation_summary={"error": str(e)},
                reasoning_chain=[f"Opportunity hunting failed: {str(e)}"],
                error_count=1,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

def main():
    """Demo the LangGraph integration"""
    print("🚀 LANGGRAPH TRADING INTEGRATION DEMO")
    print("=" * 60)
    
    orchestrator = LangGraphTradingOrchestrator()
    
    # Test single stock analysis
    print("\n🎯 Testing Enhanced Stock Analysis...")
    print("-" * 40)
    
    result = orchestrator.analyze_stock("NVDA")
    
    print(f"Workflow: {result.workflow_type}")
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Uncertainty: {result.uncertainty_level}")
    print(f"Recommendations: {len(result.recommendations)}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    if result.recommendations:
        rec = result.recommendations[0]
        print(f"\nRecommendation:")
        print(f"  Action: {rec.get('action', 'N/A')}")
        print(f"  Confidence: {rec.get('confidence', 0):.0%}")
        print(f"  Consensus Score: {rec.get('consensus_score', 0):.0%}")
        print(f"  Uncertainty Level: {rec.get('uncertainty_level', 'N/A')}")
    
    # Test opportunity hunting
    print(f"\n🎯 Testing Enhanced Opportunity Hunting...")
    print("-" * 40)
    
    opp_result = orchestrator.hunt_opportunities()
    
    print(f"Workflow: {opp_result.workflow_type}")
    print(f"Success: {opp_result.success}")
    print(f"Opportunities Found: {len(opp_result.recommendations)}")
    print(f"Execution Time: {opp_result.execution_time:.2f}s")
    
    for i, opp in enumerate(opp_result.recommendations[:3], 1):
        detailed_analysis = opp.get("detailed_analysis", {})
        print(f"  {i}. {detailed_analysis.get('ticker', 'N/A')}: {detailed_analysis.get('action', 'N/A')} ({detailed_analysis.get('confidence', 0):.0%})")

if __name__ == "__main__":
    main()