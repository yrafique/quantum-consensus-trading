"""
RAG-powered Trading Advisor
===========================

Retrieval-Augmented Generation system for providing intelligent trading advice.
Integrates with existing LLM system and adds knowledge base capabilities.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Using fallback RAG implementation.")

# Setup logging
logger = logging.getLogger(__name__)

class TradingKnowledgeBase:
    """Trading-focused knowledge base for RAG"""
    
    def __init__(self):
        self.documents = self._create_trading_knowledge()
        self.chunks = self._split_documents()
        
    def _create_trading_knowledge(self) -> List[Document]:
        """Create a comprehensive trading knowledge base"""
        
        # Technical Analysis Knowledge
        ta_knowledge = """
        Technical Analysis Fundamentals:
        
        Moving Averages:
        - Simple Moving Average (SMA): Average price over N periods
        - Exponential Moving Average (EMA): Gives more weight to recent prices
        - Golden Cross: 50-day MA crosses above 200-day MA (bullish signal)
        - Death Cross: 50-day MA crosses below 200-day MA (bearish signal)
        
        RSI (Relative Strength Index):
        - Oscillator between 0-100
        - Above 70: Potentially overbought (sell signal)
        - Below 30: Potentially oversold (buy signal)
        - Divergence with price can indicate trend reversal
        
        MACD (Moving Average Convergence Divergence):
        - MACD Line: 12-day EMA minus 26-day EMA
        - Signal Line: 9-day EMA of MACD line
        - Histogram: MACD minus Signal line
        - Bullish crossover: MACD crosses above signal line
        - Bearish crossover: MACD crosses below signal line
        
        Bollinger Bands:
        - Middle band: 20-period SMA
        - Upper/Lower bands: 2 standard deviations from middle
        - Price touching upper band: potential resistance
        - Price touching lower band: potential support
        - Band squeeze: low volatility, potential breakout coming
        
        Volume Analysis:
        - High volume confirms price moves
        - Low volume suggests weak conviction
        - Volume spikes often precede major moves
        - On-Balance Volume (OBV) tracks cumulative volume flow
        """
        
        # Risk Management Knowledge
        risk_knowledge = """
        Risk Management Principles:
        
        Position Sizing:
        - Never risk more than 1-2% of account per trade
        - Kelly Criterion: Optimal bet size based on win rate and reward/risk
        - Fixed fractional: Risk fixed percentage of current capital
        
        Stop Losses:
        - Always define exit before entering
        - Technical stops: Based on support/resistance levels
        - Percentage stops: Fixed % below entry
        - Trailing stops: Move with favorable price action
        
        Risk-Reward Ratios:
        - Minimum 1:2 ratio (risk $1 to make $2)
        - Higher ratios allow for lower win rates
        - Account for transaction costs and slippage
        
        Diversification:
        - Don't put all capital in one trade
        - Diversify across sectors and asset classes
        - Correlation matters: avoid highly correlated positions
        
        Portfolio Heat:
        - Total risk across all positions
        - Should not exceed 6-8% of total capital
        - Helps prevent catastrophic losses
        """
        
        # Market Psychology Knowledge
        psychology_knowledge = """
        Market Psychology and Behavioral Finance:
        
        Common Biases:
        - Confirmation Bias: Seeking info that confirms existing beliefs
        - Loss Aversion: Fear of losses stronger than desire for gains
        - Anchoring: Over-relying on first piece of information
        - Herding: Following the crowd without independent analysis
        
        Emotional Trading Mistakes:
        - FOMO (Fear of Missing Out): Chasing rallies
        - Revenge Trading: Trying to quickly recover losses
        - Overconfidence: Taking excessive risk after wins
        - Analysis Paralysis: Over-analyzing instead of acting
        
        Market Sentiment Indicators:
        - VIX: Fear and greed index
        - Put/Call Ratio: Options sentiment
        - Insider Trading: What insiders are doing
        - Margin Debt: Leverage in the system
        
        Contrarian Thinking:
        - When everyone is bullish, consider selling
        - When everyone is bearish, consider buying
        - Extreme sentiment often marks reversals
        - Be greedy when others are fearful, fearful when others are greedy
        """
        
        # Fundamental Analysis Knowledge
        fundamental_knowledge = """
        Fundamental Analysis Basics:
        
        Key Financial Ratios:
        - P/E Ratio: Price to Earnings (valuation metric)
        - P/B Ratio: Price to Book (asset-based valuation)
        - ROE: Return on Equity (profitability)
        - Debt-to-Equity: Financial leverage
        - Current Ratio: Short-term liquidity
        
        Financial Statements:
        - Income Statement: Revenue, expenses, profit
        - Balance Sheet: Assets, liabilities, equity
        - Cash Flow Statement: Operating, investing, financing flows
        - Look for consistent growth and quality earnings
        
        Economic Indicators:
        - GDP Growth: Overall economic health
        - Inflation (CPI): Purchasing power erosion
        - Interest Rates: Cost of capital
        - Employment: Consumer spending power
        - Consumer Confidence: Future spending intentions
        
        Sector Analysis:
        - Cyclical sectors: Benefit from economic growth
        - Defensive sectors: Stable during downturns
        - Growth sectors: High growth potential, higher risk
        - Value sectors: Trading below intrinsic value
        """
        
        # Trading Strategies Knowledge
        strategies_knowledge = """
        Trading Strategies:
        
        Trend Following:
        - Identify established trends
        - Buy in uptrends, sell in downtrends
        - Use moving averages for trend confirmation
        - Works best in trending markets
        
        Mean Reversion:
        - Assumes prices return to average
        - Buy oversold, sell overbought
        - Works best in ranging markets
        - Use RSI, Bollinger Bands for signals
        
        Momentum Trading:
        - Trade in direction of strong moves
        - Look for volume confirmation
        - News catalysts often drive momentum
        - Quick profits, quick exits
        
        Swing Trading:
        - Hold positions for days to weeks
        - Capture price swings within trends
        - Combine technical and fundamental analysis
        - Good work-life balance
        
        Day Trading:
        - Open and close positions same day
        - Requires significant time and attention
        - Focus on high-volume, liquid stocks
        - Small profits, tight risk control
        
        Position Trading:
        - Long-term approach (months to years)
        - Based primarily on fundamentals
        - Less frequent trading, lower costs
        - Requires patience and conviction
        """
        
        # Create Document objects
        documents = [
            Document(page_content=ta_knowledge, metadata={"category": "technical_analysis"}),
            Document(page_content=risk_knowledge, metadata={"category": "risk_management"}),
            Document(page_content=psychology_knowledge, metadata={"category": "psychology"}),
            Document(page_content=fundamental_knowledge, metadata={"category": "fundamental_analysis"}),
            Document(page_content=strategies_knowledge, metadata={"category": "strategies"})
        ]
        
        return documents
    
    def _split_documents(self) -> List[Document]:
        """Split documents into smaller chunks for better retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = []
        for doc in self.documents:
            doc_chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(doc_chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "source": f"{doc.metadata['category']}_chunk_{i}"
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks

class RAGTradingAdvisor:
    """RAG-powered trading advisor"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.knowledge_base = TradingKnowledgeBase()
        self.vectorstore = None
        self.qa_chain = None
        
        if LANGCHAIN_AVAILABLE:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            self.memory = None
        
        if LANGCHAIN_AVAILABLE and self.api_key:
            self._setup_rag_system()
        else:
            logger.warning("RAG system not fully available. Using fallback mode.")
    
    def _setup_rag_system(self):
        """Setup the RAG system with embeddings and vector store"""
        try:
            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                self.knowledge_base.chunks,
                embeddings
            )
            
            # Create LLM
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # Create QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            self.qa_chain = None
    
    def get_advice(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get trading advice using RAG system"""
        
        if not self.qa_chain:
            return self._fallback_advice(query, context)
        
        try:
            # Enhance query with context
            enhanced_query = self._enhance_query(query, context)
            
            # Get response from RAG system
            result = self.qa_chain({"question": enhanced_query})
            
            return {
                "answer": result["answer"],
                "source_documents": [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])],
                "confidence": 0.8,  # High confidence when using RAG
                "method": "rag"
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return self._fallback_advice(query, context)
    
    def _enhance_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Enhance the query with relevant context"""
        if not context:
            return query
        
        enhanced = f"Query: {query}\n\nContext:\n"
        
        if "ticker" in context:
            enhanced += f"- Analyzing stock: {context['ticker']}\n"
        
        if "market_data" in context:
            market_data = context["market_data"]
            enhanced += f"- Current price: ${market_data.get('current_price', 'N/A')}\n"
            enhanced += f"- Volume: {market_data.get('volume', 'N/A')}\n"
            
            if "rsi" in market_data and market_data["rsi"]:
                enhanced += f"- RSI: {market_data['rsi']:.2f}\n"
        
        if "portfolio_context" in context:
            portfolio = context["portfolio_context"]
            enhanced += f"- Portfolio value: ${portfolio.get('total_value', 'N/A')}\n"
            enhanced += f"- Free capital: ${portfolio.get('free_capital', 'N/A')}\n"
            enhanced += f"- Open positions: {portfolio.get('position_count', 0)}\n"
        
        enhanced += f"\nPlease provide specific trading advice based on this information."
        
        return enhanced
    
    def _fallback_advice(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Provide fallback advice when RAG system is not available"""
        
        query_lower = query.lower()
        
        # Pattern matching for common queries
        if any(word in query_lower for word in ["rsi", "relative strength"]):
            advice = """RSI (Relative Strength Index) Analysis:
            
            - RSI above 70: Stock may be overbought (consider selling)
            - RSI below 30: Stock may be oversold (consider buying)
            - RSI between 30-70: Neutral zone
            - Look for divergences between RSI and price for trend reversal signals
            
            Remember: RSI works best in ranging markets, less reliable in strong trends."""
            
        elif any(word in query_lower for word in ["macd", "moving average convergence"]):
            advice = """MACD Analysis:
            
            - MACD line above signal line: Bullish momentum
            - MACD line below signal line: Bearish momentum
            - MACD histogram growing: Momentum increasing
            - MACD histogram shrinking: Momentum decreasing
            - Zero line crossovers indicate trend changes
            
            Best used in conjunction with other indicators for confirmation."""
            
        elif any(word in query_lower for word in ["moving average", "ma", "sma", "ema"]):
            advice = """Moving Average Analysis:
            
            - Price above MA: Uptrend (bullish)
            - Price below MA: Downtrend (bearish)
            - MA slope indicates trend strength
            - Golden Cross (50 MA > 200 MA): Long-term bullish signal
            - Death Cross (50 MA < 200 MA): Long-term bearish signal
            
            Use multiple timeframes for better confirmation."""
            
        elif any(word in query_lower for word in ["risk", "position size", "stop loss"]):
            advice = """Risk Management Guidelines:
            
            - Never risk more than 1-2% of your account per trade
            - Always set stop losses before entering positions
            - Use position sizing to control risk
            - Maintain a risk-reward ratio of at least 1:2
            - Diversify across different stocks and sectors
            
            Risk management is more important than picking winners."""
            
        elif any(word in query_lower for word in ["buy", "sell", "entry", "exit"]):
            advice = """Trade Entry/Exit Guidelines:
            
            - Wait for confirmation signals from multiple indicators
            - Enter on pullbacks in trending markets
            - Set stop losses at logical technical levels
            - Take partial profits at resistance levels
            - Let winners run, cut losers short
            
            Plan your trade and trade your plan."""
            
        else:
            advice = """General Trading Advice:
            
            - Always do your own research before making decisions
            - Use multiple timeframes for analysis
            - Combine technical and fundamental analysis
            - Start with small position sizes while learning
            - Keep a trading journal to track performance
            - Stay disciplined and stick to your trading plan
            
            Remember: No one can predict the market with 100% accuracy."""
        
        return {
            "answer": advice,
            "source_documents": ["built_in_knowledge"],
            "confidence": 0.6,  # Medium confidence for fallback
            "method": "fallback"
        }
    
    def get_relevant_knowledge(self, query: str, k: int = 3) -> List[str]:
        """Get relevant knowledge chunks for a query"""
        if not self.vectorstore:
            return ["RAG system not available"]
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return ["Error retrieving knowledge"]
    
    def add_custom_knowledge(self, content: str, category: str):
        """Add custom knowledge to the knowledge base"""
        try:
            new_doc = Document(
                page_content=content,
                metadata={"category": category, "custom": True}
            )
            
            self.knowledge_base.documents.append(new_doc)
            
            # Re-chunk and update vector store if available
            if self.vectorstore and LANGCHAIN_AVAILABLE:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                chunks = text_splitter.split_text(content)
                new_chunks = []
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            "category": category,
                            "custom": True,
                            "chunk_id": i,
                            "source": f"{category}_custom_chunk_{i}"
                        }
                    )
                    new_chunks.append(chunk_doc)
                
                # Add to existing vector store
                embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                self.vectorstore.add_documents(new_chunks)
                
                logger.info(f"Added custom knowledge: {category}")
                
        except Exception as e:
            logger.error(f"Failed to add custom knowledge: {e}")

# Global RAG advisor instance
_rag_advisor = None

def get_rag_advisor() -> RAGTradingAdvisor:
    """Get global RAG advisor instance"""
    global _rag_advisor
    if _rag_advisor is None:
        _rag_advisor = RAGTradingAdvisor()
    return _rag_advisor

def get_trading_advice(query: str, context: Dict[str, Any] = None) -> str:
    """Simple interface to get trading advice"""
    advisor = get_rag_advisor()
    result = advisor.get_advice(query, context)
    return result["answer"]

# Export main functions
__all__ = ["RAGTradingAdvisor", "get_rag_advisor", "get_trading_advice"]