import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import yfinance as yf
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Import existing AI components instead
    try:
        from src.ai.llm_reasoner import generate_recommendation
        from src.ai.rag_advisor import get_trading_advice
        AI_AVAILABLE = True
    except ImportError:
        AI_AVAILABLE = False
import ta
import os
import time

# Import WebSocket client for real-time data
try:
    from src.websocket.streamlit_websocket_client import StreamlitWebSocketClient, create_websocket_javascript
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    StreamlitWebSocketClient = None

# Page configuration
st.set_page_config(
    page_title="Trading System",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'memory' not in st.session_state:
    if LANGCHAIN_AVAILABLE:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    else:
        st.session_state.memory = None

# Initialize page state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'portfolio'

# Initialize WebSocket client
if WEBSOCKET_AVAILABLE and 'websocket_client' not in st.session_state:
    st.session_state.websocket_client = StreamlitWebSocketClient()

# Robinhood-style header
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    # Navigation buttons in a horizontal layout
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)
    
    with nav_col1:
        if st.button("📊 Portfolio", use_container_width=True, key="nav_portfolio"):
            st.session_state.current_page = 'portfolio'
            st.rerun()
    
    with nav_col2:
        if st.button("📈 Stocks", use_container_width=True, key="nav_stocks"):
            st.session_state.current_page = 'stocks'
            st.rerun()
    
    with nav_col3:
        if st.button("🧠 Deep Analysis", use_container_width=True, key="nav_deep"):
            st.session_state.current_page = 'deep_analysis'
            st.rerun()
    
    with nav_col4:
        if st.button("🪙 Crypto", use_container_width=True, key="nav_crypto"):
            st.session_state.current_page = 'crypto'
            st.rerun()
    
    with nav_col5:
        if st.button("📋 Lists", use_container_width=True, key="nav_lists"):
            st.session_state.current_page = 'lists'
            st.rerun()
    
    with nav_col6:
        if st.button("🤖 AI Chat", use_container_width=True, key="nav_ai"):
            st.session_state.current_page = 'ai_chat'
            st.rerun()

with header_col2:
    st.markdown("<div style='text-align: right; padding: 10px;'>👤 Account</div>", unsafe_allow_html=True)

# Divider line
st.markdown("---")

page = st.session_state.current_page

# Import centralized data fetcher and trading agents
from src.data import get_data_fetcher, fetch_ohlcv_data, fetch_stock_info, calculate_indicators
from src.agents import get_agent_manager, AgentType
from src.core.unified_analyzer import get_unified_analyzer, AnalysisMode

# Get global instances
data_fetcher = get_data_fetcher()
agent_manager = get_agent_manager()
unified_analyzer = get_unified_analyzer()

# Cache functions using centralized fetcher
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch stock data using centralized data fetcher"""
    return data_fetcher.fetch_ohlcv(symbol, period)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_technical_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Calculate technical indicators using centralized data fetcher"""
    return data_fetcher.calculate_technical_indicators(df, symbol)

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """Get current stock information using centralized data fetcher"""
    info = data_fetcher.fetch_stock_info(symbol)
    
    # Ensure all required keys exist with defaults
    required_keys = {
        'price': 0.0,
        'change': 0.0,
        'change_percent': 0.0,
        'volume': 0,
        'market_cap': 0,
        'pe_ratio': 0.0,
        'name': symbol,
        'symbol': symbol
    }
    
    for key, default_value in required_keys.items():
        if key not in info or info[key] is None:
            info[key] = default_value
    
    return info

# Portfolio Page - Robinhood style
if page == "portfolio":
    # Portfolio header with gradient background
    st.markdown("""
    <style>
        .portfolio-header {
            background: linear-gradient(135deg, #00c805 0%, #00d60a 100%);
            padding: 40px;
            border-radius: 16px;
            color: white;
            margin-bottom: 30px;
        }
        .portfolio-value {
            font-size: 56px;
            font-weight: 400;
            margin: 0;
            line-height: 1;
        }
        .portfolio-change {
            font-size: 20px;
            margin-top: 12px;
            opacity: 0.95;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Portfolio value section
    portfolio_value = 125432.78
    daily_change = 2156.43
    daily_change_pct = 1.75
    
    st.markdown(f"""
    <div class="portfolio-header">
        <h1 class="portfolio-value">${portfolio_value:,.2f}</h1>
        <p class="portfolio-change">+${daily_change:,.2f} ({daily_change_pct:+.2f}%) Today</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Buying Power", "$15,234.56")
    with col2:
        st.metric("Day's Gain", f"+${daily_change:,.2f}", f"{daily_change_pct:+.2f}%")
    with col3:
        st.metric("Total Gain", "+$25,432.78", "+25.43%")
    with col4:
        st.metric("Dividends", "$1,234.56")
    
    # Holdings section
    st.markdown("<h2 style='margin-top: 40px;'>Holdings</h2>", unsafe_allow_html=True)
    
    for symbol in st.session_state.watchlist:
        info = get_stock_info(symbol)
        shares = np.random.randint(10, 100)
        value = shares * info['price']
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{symbol}**")
                st.caption(info['name'][:25] + '...' if len(info['name']) > 25 else info['name'])
            
            with col2:
                st.metric("Shares", shares, label_visibility="visible")
            
            with col3:
                color = "green" if info['change_percent'] >= 0 else "red"
                st.markdown(f"<div style='color: {color};'>${info['price']:.2f}<br/><small>{info['change_percent']:+.2f}%</small></div>", unsafe_allow_html=True)
            
            with col4:
                st.metric("Value", f"${value:,.2f}", label_visibility="visible")
            
            with col5:
                if st.button("→", key=f"view_{symbol}"):
                    st.session_state.current_page = 'stocks'
                    st.session_state.selected_stock = symbol
                    st.rerun()
        
        st.divider()

# AI Chat Page
elif page == "ai_chat":
    # Modern header with gradient
    st.markdown("""
    <h1 style="text-align: center; margin-bottom: 8px; font-size: 48px;">
        🤖 AI Trading Agents
    </h1>
    <p style="text-align: center; color: rgba(255, 255, 255, 0.8); font-size: 18px; 
              margin-bottom: 32px; font-weight: 300;">
        Quantum-powered intelligence for superior market analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Import safe AI loader
    from src.ai.safe_ai_loader import get_safe_ai_loader, initialize_ai_safely
    
    # Initialize AI models safely
    ai_loader = initialize_ai_safely()
    
    # Show loading status if model is still loading
    if not ai_loader.is_ready() and not ai_loader.error_message:
        st.info("🚀 Initializing AI models for the first time...")
        loading_container = st.container()
        ai_loader.display_loading_status(loading_container)
        
        # Add a refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Stop here if still loading
        st.stop()
    
    # Show error if model failed to load but continue with limited functionality
    if ai_loader.error_message:
        st.warning("⚠️ AI model could not be loaded. Running with limited functionality.")
        with st.expander("View Details"):
            st.error(ai_loader.error_message)
            mem_info = ai_loader.get_system_memory_info()
            st.write(f"Available Memory: {mem_info['available_gb']:.1f} GB")
            st.write(f"Memory Usage: {mem_info['percent']:.0f}%")
    
    # Create main layout with agents on left and market overview on right
    main_col1, main_col2 = st.columns([3, 1])
    
    with main_col1:
        # Agent selection interface with better header
        st.markdown("""
        <h2 style="text-align: center; margin: 24px 0; font-size: 28px; 
                   background: linear-gradient(90deg, #00c805, #00ff0a); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-weight: 700;">Select Your AI Trading Agent</h2>
        """, unsafe_allow_html=True)
        
        # Get available agents
        available_agents = agent_manager.get_all_agents()
        
        # Initialize selected agent if not set
        if 'selected_agent' not in st.session_state:
            st.session_state.selected_agent = AgentType.AI_MULTI_FACTOR
    
    # Add custom CSS for modern dark theme UI
    st.markdown("""
    <style>
        /* Modern Agent Cards - Dark Theme Optimized */
        .agent-card {
            background: rgba(26, 28, 33, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            min-height: 220px;
        }
        
        .agent-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #00c805, #00ff0a);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .agent-card:hover::before {
            transform: scaleX(1);
        }
        
        .agent-card-selected {
            background: linear-gradient(135deg, rgba(0, 200, 5, 0.1) 0%, rgba(0, 255, 10, 0.05) 100%);
            border: 2px solid #00c805;
            box-shadow: 0 0 20px rgba(0, 200, 5, 0.3), inset 0 0 20px rgba(0, 200, 5, 0.1);
        }
        
        .agent-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 200, 5, 0.5);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .agent-icon {
            font-size: 32px;
            margin-bottom: 12px;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        }
        
        .agent-title {
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: #ffffff;
            letter-spacing: -0.5px;
        }
        
        .agent-description {
            font-size: 13px;
            margin: 0 0 16px 0;
            color: rgba(255, 255, 255, 0.7);
            line-height: 1.5;
            min-height: 40px;
        }
        
        .agent-strategy {
            font-size: 11px;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.6);
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .agent-strategy::before {
            content: '⚡';
            font-size: 14px;
        }
        
        /* Selection Badge */
        .selection-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            background: #00c805;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 4px;
            box-shadow: 0 2px 8px rgba(0, 200, 5, 0.4);
        }
        
        /* Modern Buttons */
        .select-button {
            background: transparent;
            border: 2px solid rgba(0, 200, 5, 0.5);
            color: #00c805;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 8px;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 12px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .select-button:hover {
            background: rgba(0, 200, 5, 0.1);
            border-color: #00c805;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 200, 5, 0.3);
        }
        
        .select-button-selected {
            background: #00c805;
            color: white;
            border-color: #00c805;
        }
        
        .select-button-selected:hover {
            background: #00a004;
            border-color: #00a004;
        }
        
        /* Agent Grid Container */
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }
        
        /* Selected Agent Display */
        .selected-agent-display {
            background: linear-gradient(135deg, rgba(0, 200, 5, 0.1) 0%, transparent 100%);
            border: 1px solid rgba(0, 200, 5, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 24px 0;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .agent-grid {
                grid-template-columns: 1fr;
            }
            
            .agent-card {
                min-height: 180px;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display agent selection cards with modern styling
    cols = st.columns(3)  # 3 columns for better spacing
    
    for i, (agent_type, agent_info) in enumerate(available_agents.items()):
        with cols[i % 3]:
            # Create modern agent card using native Streamlit components
            is_selected = st.session_state.selected_agent == agent_type
            
            # Extract icon from agent name
            icon = agent_info['name'].split()[0]
            name_without_icon = ' '.join(agent_info['name'].split()[1:])
            
            # Create container for the card
            with st.container():
                # Apply custom styling to this specific container
                if is_selected:
                    st.markdown(f"""
                    <style>
                        div[data-testid="stVerticalBlock"] > div:has(> div > div > button[key="select_{agent_type}"]) {{
                            background: linear-gradient(135deg, rgba(0, 200, 5, 0.1) 0%, rgba(0, 255, 10, 0.05) 100%);
                            border: 2px solid #00c805;
                            border-radius: 16px;
                            padding: 20px;
                            box-shadow: 0 0 20px rgba(0, 200, 5, 0.3);
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <style>
                        div[data-testid="stVerticalBlock"] > div:has(> div > div > button[key="select_{agent_type}"]) {{
                            background: rgba(26, 28, 33, 0.95);
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            border-radius: 16px;
                            padding: 20px;
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                
                # Badge for selected agent
                if is_selected:
                    st.markdown("<div style='text-align: right; color: #00c805; font-weight: 600; font-size: 12px; margin-bottom: 8px;'>✓ Active</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                # Icon and title
                st.markdown(f"<div style='font-size: 32px; text-align: center; margin-bottom: 12px;'>{icon}</div>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center; margin: 0 0 8px 0; color: #ffffff;'>{name_without_icon}</h4>", unsafe_allow_html=True)
                
                # Description
                st.markdown(f"<p style='text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 13px; line-height: 1.5; margin-bottom: 16px;'>{agent_info['description']}</p>", unsafe_allow_html=True)
                
                # Strategy tags
                st.markdown(f"<div style='background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 8px 12px; font-size: 11px; color: rgba(255, 255, 255, 0.6); text-align: center;'>⚡ {agent_info['strategy']}</div>", unsafe_allow_html=True)
                
                # Space before button
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Selection button
                button_text = "✓ Selected" if is_selected else "Select Agent"
                button_type = "secondary" if is_selected else "primary"
                
                if st.button(button_text, key=f"select_{agent_type}", use_container_width=True, type=button_type):
                    if not is_selected:
                        st.session_state.selected_agent = agent_type
                        st.rerun()
    
    with main_col2:
        # Market Overview on the right side
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add spacing to align with agents
        st.markdown("""
        <div style='background: rgba(26, 28, 33, 0.95); 
                    border: 1px solid rgba(255, 255, 255, 0.1); 
                    border-radius: 16px; 
                    padding: 20px;
                    margin-top: 20px;'>
            <h4 style='text-align: center; color: #00c805; margin-bottom: 16px;'>Quick Market Overview</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-time WebSocket data integration
        if WEBSOCKET_AVAILABLE:
            ws_client = st.session_state.websocket_client
            
            # Auto-subscribe to watchlist symbols
            for symbol in st.session_state.watchlist[:5]:
                ws_client.subscribe_to_symbol(symbol)
            
            # Display real-time quotes
            for symbol in st.session_state.watchlist[:5]:
                quote_data = ws_client.get_latest_quote(symbol)
                
                if quote_data and ws_client.is_data_fresh(symbol, max_age_seconds=60):
                    # Use real-time WebSocket data
                    price = quote_data.get('price', 0)
                    change_percent = quote_data.get('change_percent', 0)
                    color = "#00c805" if change_percent >= 0 else "#ff4444"
                    timestamp = quote_data.get('timestamp', '')
                    
                    # Real-time indicator
                    live_indicator = "🔴 LIVE" if ws_client.is_data_fresh(symbol, 30) else "🟡 CACHED"
                else:
                    # Fallback to static data
                    info = get_stock_info(symbol)
                    price = info['price']
                    change_percent = info['change_percent']
                    color = "#00c805" if change_percent >= 0 else "#ff4444"
                    live_indicator = "⚪ STATIC"
                
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.02); 
                            border: 1px solid rgba(255, 255, 255, 0.05); 
                            border-radius: 8px; 
                            padding: 12px; 
                            margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-weight: 600; color: #ffffff;'>{symbol}</span>
                            <div style='font-size: 10px; color: rgba(255,255,255,0.5);'>{live_indicator}</div>
                        </div>
                        <div style='text-align: right;'>
                            <div style='color: #ffffff; font-size: 14px;'>${price:.2f}</div>
                            <div style='color: {color}; font-size: 12px;'>{change_percent:+.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Fallback for when WebSocket is not available
            for symbol in st.session_state.watchlist[:5]:  # Show top 5 stocks
                info = get_stock_info(symbol)
                color = "#00c805" if info['change_percent'] >= 0 else "#ff4444"
                
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.02); 
                            border: 1px solid rgba(255, 255, 255, 0.05); 
                            border-radius: 8px; 
                            padding: 12px; 
                            margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-weight: 600; color: #ffffff;'>{symbol}</span>
                        <div style='text-align: right;'>
                            <div style='color: #ffffff; font-size: 14px;'>${info['price']:.2f}</div>
                            <div style='color: {color}; font-size: 12px;'>{info['change_percent']:+.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Display selected agent info with modern styling
    selected_info = available_agents[st.session_state.selected_agent]
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Selected agent display container
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Apply custom styling to the container
            st.markdown("""
            <style>
                .selected-agent-container {
                    background: linear-gradient(135deg, rgba(0, 200, 5, 0.1) 0%, transparent 100%);
                    border: 1px solid rgba(0, 200, 5, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="selected-agent-container">', unsafe_allow_html=True)
            
            # Active agent header
            icon = selected_info['name'].split()[0]
            name = ' '.join(selected_info['name'].split()[1:])
            
            st.markdown(f"<h3 style='color: #00c805; margin: 0 0 8px 0; font-size: 20px;'>{icon} Active Agent: {name}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 14px; line-height: 1.5;'>{selected_info['description']}</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug and settings row
    col_debug, col_mode = st.columns([1, 3])
    with col_debug:
        st.session_state.debug_mode = st.toggle("🔍 Debug Mode", value=st.session_state.debug_mode)
    
    st.markdown("---")
    
    # Analysis interface - full width since market overview is already on the right
    # Agent analysis interface
    st.subheader(f"Analysis with {selected_info['name']}")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Advisor:** {message['content']}")
    
    # User input
    user_input = st.text_input("Enter a stock symbol or ask a question:", key="user_input", placeholder="e.g., AAPL, Tesla analysis, NVDA buy or sell?")
    
    if st.button("Analyze", type="primary"):
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input, 'agent': selected_info['name']})
                
                # Generate analysis using selected agent and unified analyzer
                with st.spinner(f"Analyzing with {selected_info['name']}..."):
                    debug_steps = []
                    if st.session_state.debug_mode:
                        debug_steps.append(f"🤖 **Agent:** {selected_info['name']}")
                        debug_steps.append(f"🔍 **Query:** '{user_input}'")
                    
                    # Extract ticker symbol from input
                    import re
                    ticker = None
                    query_lower = user_input.lower()
                    
                    # Enhanced company mappings including what the user mentioned
                    company_mappings = {
                        'apple': 'AAPL', 'tesla': 'TSLA', 'nvidia': 'NVDA', 'microsoft': 'MSFT',
                        'google': 'GOOGL', 'amazon': 'AMZN', 'meta': 'META', 'netflix': 'NFLX',
                        'figma': None,  # Private company, no ticker
                        'adobe': 'ADBE', 'salesforce': 'CRM', 'shopify': 'SHOP', 'zoom': 'ZM'
                    }
                    
                    # Check for company names first
                    for company, symbol in company_mappings.items():
                        if company in query_lower:
                            ticker = symbol
                            if symbol is None:  # Handle private companies like Figma
                                ai_response = f"""
**{company.title()} Analysis**

❌ **Cannot Analyze:** {company.title()} is a private company and does not have a publicly traded stock ticker.

**Alternative Suggestions:**
• If you're interested in similar companies, try: Adobe (ADBE), Canva (private), or other design software companies
• For public design/creative software stocks: ADBE, ORCL, MSFT
• For SaaS companies: CRM, SHOP, ZM, TEAM

**Available Public Companies:**
• Adobe (ADBE) - Creative software suite
• Microsoft (MSFT) - Office suite and design tools
• Oracle (ORCL) - Enterprise software
"""
                                break
                            break
                    
                    # If no company name found, look for ticker pattern
                    if not ticker and ticker is not None:  # Only if we didn't find a private company
                        ticker_match = re.search(r'\b([A-Z]{1,5})\b', user_input.upper())
                        if ticker_match:
                            ticker = ticker_match.group(1)
                    
                    if ticker and ticker is not None:
                        try:
                            # Use unified analyzer for comprehensive analysis
                            import asyncio
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            unified_result = loop.run_until_complete(
                                unified_analyzer.analyze(
                                    symbol=ticker,
                                    query=user_input,
                                    mode=AnalysisMode.FAST,  # Use fast mode for chat
                                    debug=st.session_state.debug_mode
                                )
                            )
                            loop.close()
                            
                            # Format comprehensive response
                            ai_response = f"""
**🧠 Unified AI Analysis for {ticker}**

**Final Recommendation:** {unified_result.final_recommendation}
**Confidence:** {unified_result.confidence:.1%}
**Analysis Time:** {unified_result.execution_time:.2f}s

**🎯 Key Insights:**"""
                            
                            # Add technical insights
                            if 'technical' in unified_result.component_results:
                                tech_result = unified_result.component_results['technical']
                                if 'current_indicators' in tech_result:
                                    indicators = tech_result['current_indicators']
                                    ai_response += f"""
• **RSI:** {indicators.get('rsi', 0):.1f} {'(Overbought)' if indicators.get('rsi', 0) > 70 else '(Oversold)' if indicators.get('rsi', 0) < 30 else '(Normal)'}
• **Price vs SMA20:** ${unified_result.market_data.get('stock_info', {}).get('price', 0):.2f} vs ${indicators.get('sma_20', 0):.2f}
• **Volume Ratio:** {indicators.get('volume_ratio', 1.0):.1f}x average"""
                            
                            # Add AI agent insights
                            if 'ai' in unified_result.component_results:
                                ai_results = unified_result.component_results['ai']
                                ai_response += "\n\n**🤖 AI Agent Consensus:**"
                                for agent_name, agent_result in ai_results.items():
                                    if isinstance(agent_result, dict) and 'error' not in agent_result:
                                        recommendation = agent_result.get('recommendation', 'N/A')
                                        confidence = agent_result.get('confidence', 0)
                                        ai_response += f"\n• **{agent_name.replace('_', ' ').title()}:** {recommendation} ({confidence:.1%})"
                            
                            # Add risk assessment
                            if unified_result.risk_assessment:
                                risk_score = unified_result.risk_assessment.get('overall_risk_score', 0)
                                kelly_pct = unified_result.risk_assessment.get('position_sizing', {}).get('kelly_percentage', 0)
                                ai_response += f"""

**⚖️ Risk Assessment:**
• **Overall Risk Score:** {risk_score:.1f}/10
• **Suggested Position Size:** {kelly_pct:.1%} of portfolio"""
                            
                            # Add debug information if requested
                            if st.session_state.debug_mode:
                                ai_response += f"""

**🔍 Debug Information:**
• **Components Used:** {len(unified_result.component_results)}
• **Analysis Steps:** {len(unified_result.analysis_steps)}
• **System Health:** {unified_result.system_health.get('overall_health', 'Unknown')}"""
                                
                                for i, step in enumerate(unified_result.analysis_steps):
                                    ai_response += f"\n• **Step {i+1}:** {step.component} ({step.execution_time:.3f}s)"
                        
                        except Exception as e:
                            # Fallback to simple agent analysis
                            result = agent_manager.analyze_with_agent(
                                st.session_state.selected_agent, 
                                ticker, 
                                user_input, 
                                st.session_state.debug_mode
                            )
                            
                            if 'error' in result:
                                ai_response = f"Error analyzing {ticker}: {result['error']}"
                            else:
                                # Format the agent's response
                                signals_text = "\n".join(f"• {signal}" for signal in result.get('signals', []))
                                metrics_text = "\n".join(f"• {key}: {value}" for key, value in result.get('key_metrics', {}).items())
                                
                                ai_response = f"""
**{result.get('agent', 'Agent')} Analysis for {ticker}**

**Recommendation:** {result.get('recommendation', 'N/A')}
**Confidence:** {result.get('confidence', 0):.1%}

**Key Signals:**
{signals_text}

**Key Metrics:**
{metrics_text}
"""
                                
                                # Add debug information if available
                                if st.session_state.debug_mode and result.get('debug_steps'):
                                    debug_info = "\n".join(result['debug_steps'])
                                    ai_response += f"\n\n**🔍 Agent Debug Steps:**\n{debug_info}"
                    
                    else:
                        ai_response = f"""
I couldn't identify a stock ticker in "{user_input}".

**Try these formats:**
• "AAPL" or "Apple"
• "Tesla analysis"
• "NVDA buy or sell?"

**Popular stocks:** AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META
"""
                
                # Add debug information if debug mode is enabled
                if st.session_state.debug_mode and debug_steps:
                    debug_info = "\n\n" + "---\n**🔍 Debug Chain of Thought:**\n\n" + "\n\n".join(debug_steps) + "\n---"
                    ai_response += debug_info
                
                st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
                st.rerun()
    
    # Add WebSocket integration for real-time data updates
    if WEBSOCKET_AVAILABLE:
        # Add auto-refresh functionality
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        with col2:
            # WebSocket connection status
            ws_client = st.session_state.websocket_client
            ws_client.display_connection_status()
        
        with col3:
            # Quick subscription controls
            if st.button("📊 Subscribe to Watchlist", use_container_width=True):
                for symbol in st.session_state.watchlist[:5]:
                    ws_client.subscribe_to_symbol(symbol)
                st.success("Subscribed to watchlist symbols!")
                st.rerun()
        
        # Inject WebSocket JavaScript for browser-level real-time updates
        st.markdown(create_websocket_javascript(), unsafe_allow_html=True)
        
        # Auto-refresh component (updates every 30 seconds)
        st.markdown("""
        <script>
            // Auto-refresh Streamlit app every 30 seconds for real-time data
            setTimeout(function() {
                window.location.reload();
            }, 30000);
        </script>
        """, unsafe_allow_html=True)

# Stocks Page
elif page == "stocks":
    st.title("📊 Stock Analysis Dashboard")
    
    # Watchlist management
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Your Watchlist")
    with col2:
        new_symbol = st.text_input("Add symbol:", key="new_symbol")
        if st.button("Add to Watchlist"):
            if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()} to watchlist")
                st.rerun()
    
    # Display watchlist with technical indicators
    for symbol in st.session_state.watchlist:
        with st.expander(f"{symbol} Analysis", expanded=True):
            # Fetch data
            df = fetch_stock_data(symbol, "3mo")
            if not df.empty:
                df = calculate_technical_indicators(df, symbol)
                
                # Current info
                info = get_stock_info(symbol)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"${info['price']:.2f}", f"{info['change_percent']:.2f}%")
                with col2:
                    st.metric("RSI", f"{df['rsi'].iloc[-1]:.2f}" if 'RSI' in df else "N/A")
                with col3:
                    st.metric("Volume", f"{info['volume']:,.0f}")
                with col4:
                    st.metric("P/E Ratio", f"{info['pe_ratio']:.2f}" if info['pe_ratio'] else "N/A")
                
                # Price chart with indicators
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ))
                
                # Moving averages
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ))
                
                fig.update_layout(
                    title=f"{symbol} - Price and Moving Averages",
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators summary
                st.markdown("**Technical Analysis Summary:**")
                
                # Simple signals
                last_close = df['close'].iloc[-1]
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                rsi = df['rsi'].iloc[-1]
                
                signals = []
                if last_close > sma_20:
                    signals.append("✅ Price above SMA 20")
                else:
                    signals.append("❌ Price below SMA 20")
                
                if last_close > sma_50:
                    signals.append("✅ Price above SMA 50")
                else:
                    signals.append("❌ Price below SMA 50")
                
                if rsi > 70:
                    signals.append("⚠️ RSI indicates overbought")
                elif rsi < 30:
                    signals.append("⚠️ RSI indicates oversold")
                else:
                    signals.append("✅ RSI in normal range")
                
                for signal in signals:
                    st.write(signal)

# Deep Analysis Page - Comprehensive AI Debug Visualization
elif page == "deep_analysis":
    st.title("🧠 Deep AI Analysis & Debug Visualization")
    st.markdown("**Comprehensive multi-agent analysis with full debug chain of thought**")
    
    # Analysis controls
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    with col1:
        analysis_symbol = st.text_input("Stock Symbol:", value="AAPL", key="deep_analysis_symbol")
    
    with col2:
        analysis_mode = st.selectbox("Analysis Mode:", 
            options=[mode.value for mode in AnalysisMode],
            index=1,  # Default to COMPREHENSIVE
            key="analysis_mode"
        )
    
    with col3:
        user_query = st.text_input("Optional Query:", placeholder="e.g., short squeeze potential", key="user_query")
    
    with col4:
        run_analysis = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Initialize session state for analysis results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    if run_analysis and analysis_symbol:
        st.session_state.analysis_result = None  # Clear previous result
        
        with st.spinner(f"Running comprehensive {analysis_mode} analysis on {analysis_symbol.upper()}..."):
            try:
                # Run unified analysis
                import asyncio
                
                # Run the async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    unified_analyzer.analyze(
                        symbol=analysis_symbol.upper(),
                        query=user_query or "",
                        mode=AnalysisMode(analysis_mode),
                        debug=True
                    )
                )
                loop.close()
                
                st.session_state.analysis_result = result
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_result = None
    
    # Display analysis results if available
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Summary header
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Final Recommendation", result.final_recommendation)
        
        with col2:
            st.metric("Confidence", f"{result.confidence:.1%}")
        
        with col3:
            st.metric("Execution Time", f"{result.execution_time:.2f}s")
        
        with col4:
            st.metric("Analysis Steps", len(result.analysis_steps))
        
        with col5:
            st.metric("Components Used", len(result.component_results))
        
        # Main tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Debug Steps", 
            "📈 Market Data", 
            "🤖 AI Components", 
            "⚖️ Risk Assessment", 
            "🏥 System Health"
        ])
        
        with tab1:
            st.markdown("### 🔍 Detailed Analysis Steps")
            st.markdown("*Complete chain of thought showing how each component contributed to the final recommendation*")
            
            for i, step in enumerate(result.analysis_steps):
                with st.expander(f"Step {i+1}: {step.component} ({step.execution_time:.3f}s)", expanded=True):
                    
                    # Step header
                    step_col1, step_col2, step_col3 = st.columns([2, 1, 1])
                    with step_col1:
                        st.markdown(f"**Component:** {step.component}")
                    with step_col2:
                        st.markdown(f"**Type:** {step.step_type}")
                    with step_col3:
                        if step.confidence:
                            st.markdown(f"**Confidence:** {step.confidence:.1%}")
                    
                    # Input data
                    if step.input_data:
                        st.markdown("**📥 Input Data:**")
                        input_display = {}
                        for key, value in step.input_data.items():
                            if isinstance(value, dict) and len(str(value)) > 200:
                                input_display[key] = f"<{type(value).__name__}> ({len(value)} items)"
                            else:
                                input_display[key] = str(value)[:100] + "..." if len(str(value)) > 100 else value
                        st.json(input_display)
                    
                    # Output data
                    if step.output_data:
                        st.markdown("**📤 Output Data:**")
                        # Format output for better readability
                        if isinstance(step.output_data, dict):
                            formatted_output = {}
                            for key, value in step.output_data.items():
                                if key == 'ohlcv_data' and isinstance(value, dict):
                                    formatted_output[key] = f"DataFrame with {len(value.get('close', {}))} price points"
                                elif isinstance(value, dict) and len(str(value)) > 300:
                                    formatted_output[key] = f"<{type(value).__name__}> ({len(value)} items)"
                                else:
                                    formatted_output[key] = value
                            st.json(formatted_output)
                        else:
                            st.json(step.output_data)
                    
                    # Errors
                    if step.errors:
                        st.markdown("**❌ Errors:**")
                        for error in step.errors:
                            st.error(error)
                    
                    # Performance metadata
                    if step.metadata:
                        st.markdown("**⚡ Performance Metadata:**")
                        st.json(step.metadata)
        
        with tab2:
            st.markdown("### 📈 Market Data Analysis")
            
            if result.market_data:
                # Stock info
                stock_info = result.market_data.get('stock_info', {})
                if stock_info:
                    st.markdown("**Current Stock Information:**")
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    
                    with info_col1:
                        st.metric("Current Price", f"${stock_info.get('price', 0):.2f}")
                    with info_col2:
                        st.metric("Volume", f"{stock_info.get('volume', 0):,}")
                    with info_col3:
                        st.metric("Market Cap", f"${stock_info.get('market_cap', 0):,.0f}")
                    with info_col4:
                        st.metric("Data Points", result.market_data.get('data_points', 0))
                
                # Technical indicators
                indicators = result.market_data.get('indicators_calculated', [])
                if indicators:
                    st.markdown("**Technical Indicators Calculated:**")
                    st.write(", ".join(indicators))
                
                # Validation results
                validation = result.validation_results
                if validation:
                    st.markdown("**Data Validation Results:**")
                    st.json(validation)
        
        with tab3:
            st.markdown("### 🤖 AI Components Analysis")
            
            for component_name, component_result in result.component_results.items():
                with st.expander(f"{component_name.title()} Analysis", expanded=True):
                    
                    if component_name == 'technical':
                        # Technical analysis display
                        tech_result = component_result
                        if 'signals' in tech_result:
                            st.markdown("**📊 Technical Signals:**")
                            signals = tech_result['signals']
                            if isinstance(signals, dict):
                                for signal_name, signal_value in signals.items():
                                    st.write(f"• {signal_name}: {signal_value}")
                            else:
                                st.write(f"Signal Status: {signals}")
                        
                        if 'current_indicators' in tech_result:
                            st.markdown("**📈 Current Indicators:**")
                            indicators_col1, indicators_col2 = st.columns(2)
                            indicators = tech_result['current_indicators']
                            mid_point = len(indicators) // 2
                            
                            with indicators_col1:
                                for i, (key, value) in enumerate(list(indicators.items())[:mid_point]):
                                    st.metric(key.upper(), f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                            
                            with indicators_col2:
                                for key, value in list(indicators.items())[mid_point:]:
                                    st.metric(key.upper(), f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                        
                        if 'trend_analysis' in tech_result:
                            st.markdown("**📊 Trend Analysis:**")
                            trend = tech_result['trend_analysis']
                            st.write(f"• Trend Direction: {trend.get('trend', 'Unknown')}")
                            st.write(f"• Trend Strength: {trend.get('strength', 0):.1%}")
                    
                    elif component_name == 'ai':
                        # AI components display
                        ai_results = component_result
                        for agent_name, agent_result in ai_results.items():
                            st.markdown(f"**🤖 {agent_name.replace('_', ' ').title()}:**")
                            if isinstance(agent_result, dict) and 'error' not in agent_result:
                                if 'recommendation' in agent_result:
                                    rec_col1, rec_col2 = st.columns(2)
                                    with rec_col1:
                                        st.write(f"Recommendation: **{agent_result.get('recommendation', 'N/A')}**")
                                    with rec_col2:
                                        st.write(f"Confidence: **{agent_result.get('confidence', 0):.1%}**")
                                
                                if 'reasoning' in agent_result:
                                    st.markdown("*Reasoning:*")
                                    st.write(agent_result['reasoning'][:500] + "..." if len(agent_result['reasoning']) > 500 else agent_result['reasoning'])
                            else:
                                st.error(f"Error: {agent_result.get('error', 'Unknown error')}")
                            st.divider()
                    
                    elif component_name == 'opportunity':
                        # Opportunity analysis display
                        opp_result = component_result
                        if 'error' not in opp_result:
                            opp_col1, opp_col2, opp_col3 = st.columns(3)
                            with opp_col1:
                                st.metric("Opportunity Score", f"{opp_result.get('opportunity_score', 0):.1f}/100")
                            with opp_col2:
                                st.metric("Category", opp_result.get('category', 'Unknown'))
                            with opp_col3:
                                ranking = opp_result.get('ranking')
                                total = opp_result.get('total_opportunities', 0)
                                if ranking:
                                    st.metric("Market Ranking", f"#{ranking} of {total}")
                                else:
                                    st.metric("Market Ranking", "Not ranked")
                            
                            factors = opp_result.get('factors', {})
                            if factors:
                                st.markdown("**Opportunity Factors:**")
                                for factor, value in factors.items():
                                    st.write(f"• {factor}: {value}")
                        else:
                            st.error(f"Error: {opp_result.get('error', 'Unknown error')}")
        
        with tab4:
            st.markdown("### ⚖️ Risk Assessment")
            
            risk_data = result.risk_assessment
            if risk_data:
                # Position sizing
                if 'position_sizing' in risk_data:
                    st.markdown("**💰 Position Sizing (Kelly Criterion):**")
                    pos_sizing = risk_data['position_sizing']
                    
                    sizing_col1, sizing_col2, sizing_col3 = st.columns(3)
                    with sizing_col1:
                        st.metric("Kelly %", f"{pos_sizing.get('kelly_percentage', 0):.1%}")
                    with sizing_col2:
                        st.metric("Recommended Shares", f"{pos_sizing.get('recommended_shares', 0):,.0f}")
                    with sizing_col3:
                        st.metric("Risk Amount", f"${pos_sizing.get('risk_amount', 0):,.2f}")
                
                # Risk metrics
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    if 'volatility_risk' in risk_data:
                        st.markdown("**📊 Volatility Risk:**")
                        vol_risk = risk_data['volatility_risk']
                        st.write(f"• Beta: {vol_risk.get('beta', 1.0):.2f}")
                        st.write(f"• Risk Level: {vol_risk.get('risk_level', 'Unknown')}")
                
                with risk_col2:
                    if 'liquidity_risk' in risk_data:
                        st.markdown("**💧 Liquidity Risk:**")
                        liq_risk = risk_data['liquidity_risk']
                        st.write(f"• Liquidity Score: {liq_risk.get('liquidity_score', 0):.1f}/10")
                        st.write(f"• Daily Volume: {liq_risk.get('volume', 0):,}")
                
                # Overall risk score
                overall_risk = risk_data.get('overall_risk_score', 0)
                st.markdown("**🎯 Overall Risk Score:**")
                risk_color = "red" if overall_risk > 7 else "orange" if overall_risk > 4 else "green"
                st.markdown(f"<h2 style='color: {risk_color};'>{overall_risk:.1f}/10</h2>", unsafe_allow_html=True)
        
        with tab5:
            st.markdown("### 🏥 System Health & Performance")
            
            health_data = result.system_health
            if health_data:
                # Overall health
                overall_health = health_data.get('overall_health', 'unknown')
                health_color = "green" if overall_health == 'healthy' else "orange" if overall_health == 'limited' else "red"
                st.markdown(f"**System Status:** <span style='color: {health_color}; font-weight: bold;'>{overall_health.upper()}</span>", unsafe_allow_html=True)
                
                # Component health
                components = health_data.get('components', {})
                if components:
                    st.markdown("**Component Health Status:**")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    component_items = list(components.items())
                    mid_point = len(component_items) // 2
                    
                    with comp_col1:
                        for comp_name, comp_info in component_items[:mid_point]:
                            status = comp_info.get('status', 'unknown')
                            status_icon = "✅" if status == 'healthy' else "⚠️" if status == 'degraded' else "❌"
                            st.write(f"{status_icon} **{comp_name.replace('_', ' ').title()}**: {status}")
                    
                    with comp_col2:
                        for comp_name, comp_info in component_items[mid_point:]:
                            status = comp_info.get('status', 'unknown')
                            status_icon = "✅" if status == 'healthy' else "⚠️" if status == 'degraded' else "❌"
                            st.write(f"{status_icon} **{comp_name.replace('_', ' ').title()}**: {status}")
                
                # Performance metrics
                st.markdown("**⚡ Performance Analysis:**")
                perf_steps = [step for step in result.analysis_steps if step.execution_time > 0]
                if perf_steps:
                    # Create performance chart
                    step_names = [step.component for step in perf_steps]
                    step_times = [step.execution_time * 1000 for step in perf_steps]  # Convert to ms
                    
                    fig = go.Figure(data=[go.Bar(x=step_names, y=step_times)])
                    fig.update_layout(
                        title="Component Execution Times (ms)",
                        xaxis_title="Component",
                        yaxis_title="Time (milliseconds)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance summary
                    total_time = sum(step_times)
                    avg_time = total_time / len(step_times)
                    slowest_step = max(perf_steps, key=lambda x: x.execution_time)
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.metric("Total Analysis Time", f"{total_time:.0f}ms")
                    with perf_col2:
                        st.metric("Average Step Time", f"{avg_time:.0f}ms")
                    with perf_col3:
                        st.metric("Slowest Component", f"{slowest_step.component}")
        
        # Export functionality
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("💾 Export Analysis", use_container_width=True):
                # Create export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": result.symbol,
                    "query": result.query,
                    "final_recommendation": result.final_recommendation,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time,
                    "analysis_steps": [
                        {
                            "component": step.component,
                            "execution_time": step.execution_time,
                            "confidence": step.confidence,
                            "errors": step.errors
                        }
                        for step in result.analysis_steps
                    ],
                    "component_results": result.component_results,
                    "risk_assessment": result.risk_assessment,
                    "system_health": result.system_health
                }
                
                st.download_button(
                    label="Download Full Analysis",
                    data=pd.Series(export_data).to_json(indent=2),
                    file_name=f"deep_analysis_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🔄 Rerun Analysis", use_container_width=True):
                st.session_state.analysis_result = None
                st.rerun()
    
    else:
        # Show example when no analysis has been run
        st.markdown("---")
        st.markdown("### 🎯 What You'll See")
        st.markdown("""
        **Deep Analysis provides comprehensive AI-driven stock analysis with full transparency:**
        
        **🔍 Debug Steps** - Complete chain of thought showing:
        - System health checks and data validation
        - Technical analysis with 20+ indicators
        - Multi-agent AI analysis (LLM Reasoner, MLX, ReAct, RAG)
        - Opportunity assessment and risk evaluation
        - Consensus building from all components
        
        **📈 Market Data** - Real-time data with validation:
        - Current stock information and volume analysis
        - Technical indicator calculations (RSI, MACD, Bollinger Bands, etc.)
        - Data quality validation and completeness checks
        
        **🤖 AI Components** - Individual agent results:
        - LLM Reasoner with confidence scoring
        - MLX local inference results
        - ReAct agent step-by-step reasoning
        - RAG advisor contextual recommendations
        
        **⚖️ Risk Assessment** - Professional risk management:
        - Kelly Criterion position sizing
        - Volatility and beta analysis
        - Liquidity risk assessment
        - Overall risk scoring (0-10 scale)
        
        **🏥 System Health** - Performance monitoring:
        - Component health status
        - Execution time analysis
        - Performance bottleneck identification
        - System reliability metrics
        """)

# Crypto Page
elif page == "crypto":
    st.markdown("<h1 style='font-weight: 500;'>Crypto</h1>", unsafe_allow_html=True)
    
    # Coming soon message with Robinhood style
    st.markdown("""
    <div style='background-color: #f5f8fa; padding: 40px; border-radius: 16px; text-align: center; margin-top: 40px;'>
        <h2 style='margin: 0; font-weight: 500;'>Cryptocurrency Trading Coming Soon</h2>
        <p style='color: #6e7681; margin-top: 16px;'>We're working on bringing you commission-free crypto trading.</p>
    </div>
    """, unsafe_allow_html=True)

# Lists Page
elif page == "lists":
    st.markdown("<h1 style='font-weight: 500;'>Lists</h1>", unsafe_allow_html=True)
    
    # Watchlist section
    st.markdown("<h3 style='font-weight: 500; margin-top: 32px;'>My Watchlist</h3>", unsafe_allow_html=True)
    
    # Add stock input
    col1, col2 = st.columns([4, 1])
    with col1:
        new_symbol = st.text_input("Add symbol", placeholder="Search for stocks...", label_visibility="collapsed")
    with col2:
        if st.button("Add", type="primary", use_container_width=True):
            if new_symbol and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()}")
                st.rerun()
    
    # Initialize session state for display options
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = 'price_change'  # price_change, total_value, shares
    if 'time_period' not in st.session_state:
        st.session_state.time_period = '1D'  # 1D, 1W, 1M, 1Y
    
    # Add custom CSS for better button visibility
    st.markdown("""
    <style>
        /* Custom styling for buttons and selects */
        .stSelectbox > div > div {
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px 12px;
            font-weight: 500;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #00c805;
            box-shadow: 0 0 0 1px #00c805;
        }
        
        .stButton > button {
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            color: #333;
            font-weight: 500;
            padding: 8px 16px;
        }
        
        .stButton > button:hover {
            background-color: #e8e8e8;
            border-color: #00c805;
        }
        
        .controls-row {
            background-color: #fafafa;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Controls row - Robinhood style with better visibility
    st.markdown('<div class="controls-row">', unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Controls")
    
    col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1])
    
    with col1:
        if st.button("🔄 Refresh Prices", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        # Display mode selection with better styling
        st.markdown("**Display Mode:**")
        display_options = {
            'price_change': '💰 Price Change',
            'total_value': '📊 Total Value', 
            'shares': '🔢 Shares'
        }
        
        # Create radio buttons for better visibility
        display_choice = st.radio(
            "Display Mode",
            options=list(display_options.keys()),
            format_func=lambda x: display_options[x],
            key="display_mode_radio",
            label_visibility="collapsed",
            horizontal=True
        )
        st.session_state.display_mode = display_choice
    
    with col3:
        # Time period selection with better styling
        st.markdown("**Time Period:**")
        time_options = ['1D', '1W', '1M', '1Y']
        time_labels = {
            '1D': '📅 1 Day',
            '1W': '📅 1 Week', 
            '1M': '📅 1 Month',
            '1Y': '📅 1 Year'
        }
        
        # Create radio buttons for time period
        time_choice = st.radio(
            "Time Period",
            options=time_options,
            format_func=lambda x: time_labels[x],
            index=time_options.index(st.session_state.time_period) if st.session_state.time_period in time_options else 0,
            key="time_period_radio",
            label_visibility="collapsed",
            horizontal=True
        )
        st.session_state.time_period = time_choice
    
    with col4:
        # Add a quick stats summary
        st.markdown("**Watchlist:**")
        st.markdown(f"📊 {len(st.session_state.watchlist)} stocks")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Helper function to calculate percentage change for different periods
    def calculate_period_change(symbol, period):
        try:
            # Fetch historical data based on period
            if period == '1D':
                df = data_fetcher.fetch_ohlcv(symbol, period="2d", force_refresh=True)
            elif period == '1W':
                df = data_fetcher.fetch_ohlcv(symbol, period="1mo", force_refresh=True)
            elif period == '1M':
                df = data_fetcher.fetch_ohlcv(symbol, period="3mo", force_refresh=True)
            else:  # 1Y
                df = data_fetcher.fetch_ohlcv(symbol, period="2y", force_refresh=True)
            
            if df.empty or len(df) < 2:
                return 0.0, 0.0
            
            current_price = df['close'].iloc[-1]
            
            if period == '1D':
                previous_price = df['close'].iloc[-2] if len(df) >= 2 else current_price
            elif period == '1W':
                # Get price from 7 days ago
                days_back = min(7, len(df) - 1)
                previous_price = df['close'].iloc[-(days_back + 1)]
            elif period == '1M':
                # Get price from ~30 days ago
                days_back = min(30, len(df) - 1)
                previous_price = df['close'].iloc[-(days_back + 1)]
            else:  # 1Y
                # Get price from ~365 days ago
                days_back = min(365, len(df) - 1)
                previous_price = df['close'].iloc[-(days_back + 1)]
            
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100 if previous_price > 0 else 0.0
            
            return price_change, percent_change
        except Exception as e:
            print(f"Error calculating change for {symbol}: {e}")
            return 0.0, 0.0
    
    # Display watchlist with enhanced Robinhood-style layout
    for symbol in st.session_state.watchlist:
        # Force refresh by bypassing cache
        info = data_fetcher.fetch_stock_info(symbol, force_refresh=True)
        
        # Calculate change based on selected time period
        price_change, percent_change = calculate_period_change(symbol, st.session_state.time_period)
        
        # Determine colors based on change
        if percent_change > 0:
            color = "#00c805"  # Robinhood green
            sign = "+"
        elif percent_change < 0:
            color = "#ff5000"  # Robinhood red
            sign = ""
        else:
            color = "#6e7681"  # Neutral gray
            sign = ""
        
        # Create expandable container for each stock
        with st.expander(f"📈 {symbol} - {info['name']}", expanded=False):
            # Main stock info row
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"""
                <div style='padding: 8px 0;'>
                    <p style='font-weight: 600; margin: 0; font-size: 18px;'>{symbol}</p>
                    <p style='color: #6e7681; font-size: 14px; margin: 0;'>{info['name']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Display based on selected mode
                if st.session_state.display_mode == 'price_change':
                    st.markdown(f"""
                    <div style='text-align: right; padding: 8px 0;'>
                        <p style='font-weight: 600; margin: 0; font-size: 18px;'>${info['price']:.2f}</p>
                        <p style='color: {color}; font-size: 14px; margin: 0; font-weight: 500;'>
                            {sign}${abs(price_change):.2f} ({sign}{abs(percent_change):.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.display_mode == 'total_value':
                    # Simulate portfolio value (in real app, this would be from portfolio data)
                    shares = np.random.randint(1, 100)  # Mock shares owned
                    total_value = shares * info['price']
                    value_change = shares * price_change
                    st.markdown(f"""
                    <div style='text-align: right; padding: 8px 0;'>
                        <p style='font-weight: 600; margin: 0; font-size: 18px;'>${total_value:,.2f}</p>
                        <p style='color: {color}; font-size: 14px; margin: 0; font-weight: 500;'>
                            {sign}${abs(value_change):,.2f} ({sign}{abs(percent_change):.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # shares mode
                    shares = np.random.randint(1, 100)  # Mock shares owned
                    st.markdown(f"""
                    <div style='text-align: right; padding: 8px 0;'>
                        <p style='font-weight: 600; margin: 0; font-size: 18px;'>{shares} shares</p>
                        <p style='color: {color}; font-size: 14px; margin: 0; font-weight: 500;'>
                            {sign}{abs(percent_change):.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='text-align: center; padding: 8px 0;'>
                    <p style='font-weight: 500; margin: 0; font-size: 14px; color: #6e7681;'>Volume</p>
                    <p style='font-weight: 600; margin: 0; font-size: 16px;'>{info['volume']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if st.button("🗑️", key=f"remove_{symbol}", help="Remove from watchlist"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()
            
            # Advanced Chart section with technical indicators
            st.markdown("### 📊 Advanced Price Chart")
            
            # Technical indicator controls in a beautiful layout
            st.markdown("#### 🛠️ Technical Indicators")
            
            # Create columns for technical indicator checkboxes
            tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
            
            with tech_col1:
                show_sma20 = st.checkbox("📈 SMA 20", value=False, key=f"sma20_{symbol}")
                show_sma50 = st.checkbox("📈 SMA 50", value=True, key=f"sma50_{symbol}")
            
            with tech_col2:
                show_sma200 = st.checkbox("📈 SMA 200", value=True, key=f"sma200_{symbol}")
                show_ema21 = st.checkbox("📈 EMA 21", value=False, key=f"ema21_{symbol}")
            
            with tech_col3:
                show_rsi = st.checkbox("⚡ RSI", value=True, key=f"rsi_{symbol}")
                show_macd = st.checkbox("📊 MACD", value=False, key=f"macd_{symbol}")
            
            with tech_col4:
                show_bb = st.checkbox("🎯 Bollinger Bands", value=False, key=f"bb_{symbol}")
                show_volume = st.checkbox("📦 Volume", value=False, key=f"volume_{symbol}")
            
            # Fetch chart data based on time period
            try:
                if st.session_state.time_period == '1D':
                    chart_df = data_fetcher.fetch_ohlcv(symbol, period="5d", interval="1h")  # More data for indicators
                elif st.session_state.time_period == '1W':
                    chart_df = data_fetcher.fetch_ohlcv(symbol, period="1mo", interval="1h")
                elif st.session_state.time_period == '1M':
                    chart_df = data_fetcher.fetch_ohlcv(symbol, period="6mo", interval="1d")
                else:  # 1Y
                    chart_df = data_fetcher.fetch_ohlcv(symbol, period="2y", interval="1d")
                
                if not chart_df.empty:
                    # Calculate technical indicators
                    if len(chart_df) >= 20:
                        chart_df['sma_20'] = chart_df['close'].rolling(window=20).mean()
                    if len(chart_df) >= 50:
                        chart_df['sma_50'] = chart_df['close'].rolling(window=50).mean()
                    if len(chart_df) >= 200:
                        chart_df['sma_200'] = chart_df['close'].rolling(window=200).mean()
                    if len(chart_df) >= 21:
                        chart_df['ema_21'] = chart_df['close'].ewm(span=21).mean()
                    
                    # RSI calculation
                    if len(chart_df) >= 14:
                        delta = chart_df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        chart_df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD calculation
                    if len(chart_df) >= 26:
                        exp1 = chart_df['close'].ewm(span=12).mean()
                        exp2 = chart_df['close'].ewm(span=26).mean()
                        chart_df['macd'] = exp1 - exp2
                        chart_df['macd_signal'] = chart_df['macd'].ewm(span=9).mean()
                        chart_df['macd_histogram'] = chart_df['macd'] - chart_df['macd_signal']
                    
                    # Bollinger Bands
                    if len(chart_df) >= 20:
                        bb_period = 20
                        chart_df['bb_middle'] = chart_df['close'].rolling(window=bb_period).mean()
                        bb_std = chart_df['close'].rolling(window=bb_period).std()
                        chart_df['bb_upper'] = chart_df['bb_middle'] + (bb_std * 2)
                        chart_df['bb_lower'] = chart_df['bb_middle'] - (bb_std * 2)
                    
                    # Determine subplot count based on selected indicators
                    subplot_count = 1  # Main price chart
                    if show_rsi:
                        subplot_count += 1
                    if show_macd:
                        subplot_count += 1
                    if show_volume:
                        subplot_count += 1
                    
                    from plotly.subplots import make_subplots
                    
                    # Create subplot titles
                    subplot_titles = ["Price"]
                    if show_rsi:
                        subplot_titles.append("RSI")
                    if show_macd:
                        subplot_titles.append("MACD")
                    if show_volume:
                        subplot_titles.append("Volume")
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=subplot_count,
                        cols=1,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.05,
                        row_heights=[0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1]
                    )
                    
                    # Main price chart
                    fig.add_trace(
                        go.Scatter(
                            x=chart_df.index,
                            y=chart_df['close'],
                            mode='lines',
                            name='Price',
                            line=dict(color=color, width=2.5)
                        ),
                        row=1, col=1
                    )
                    
                    # Add moving averages
                    if show_sma20 and 'sma_20' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['sma_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='orange', width=1.5, dash='dot')
                            ),
                            row=1, col=1
                        )
                    
                    if show_sma50 and 'sma_50' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['sma_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='blue', width=1.5, dash='dash')
                            ),
                            row=1, col=1
                        )
                    
                    if show_sma200 and 'sma_200' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['sma_200'],
                                mode='lines',
                                name='SMA 200',
                                line=dict(color='red', width=2, dash='dashdot')
                            ),
                            row=1, col=1
                        )
                    
                    if show_ema21 and 'ema_21' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['ema_21'],
                                mode='lines',
                                name='EMA 21',
                                line=dict(color='purple', width=1.5)
                            ),
                            row=1, col=1
                        )
                    
                    # Bollinger Bands
                    if show_bb and 'bb_upper' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['bb_upper'],
                                mode='lines',
                                name='BB Upper',
                                line=dict(color='gray', width=1),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['bb_lower'],
                                mode='lines',
                                name='BB Lower',
                                line=dict(color='gray', width=1),
                                fill='tonexty',
                                fillcolor='rgba(128,128,128,0.1)',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    current_row = 2
                    
                    # RSI subplot
                    if show_rsi and 'rsi' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['rsi'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple', width=2)
                            ),
                            row=current_row, col=1
                        )
                        
                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                    annotation_text="Overbought", annotation_position="bottom right",
                                    row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                    annotation_text="Oversold", annotation_position="top right",
                                    row=current_row, col=1)
                        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                                    row=current_row, col=1)
                        
                        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
                        current_row += 1
                    
                    # MACD subplot
                    if show_macd and 'macd' in chart_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['macd'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue', width=2)
                            ),
                            row=current_row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df.index,
                                y=chart_df['macd_signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='red', width=1.5)
                            ),
                            row=current_row, col=1
                        )
                        fig.add_trace(
                            go.Bar(
                                x=chart_df.index,
                                y=chart_df['macd_histogram'],
                                name='Histogram',
                                marker_color='gray',
                                opacity=0.6
                            ),
                            row=current_row, col=1
                        )
                        current_row += 1
                    
                    # Volume subplot
                    if show_volume:
                        volume_colors = ['red' if chart_df['close'].iloc[i] < chart_df['open'].iloc[i] else 'green' 
                                       for i in range(len(chart_df))]
                        fig.add_trace(
                            go.Bar(
                                x=chart_df.index,
                                y=chart_df['volume'],
                                name='Volume',
                                marker_color=volume_colors,
                                opacity=0.7
                            ),
                            row=current_row, col=1
                        )
                    
                    # Update layout with Robinhood styling
                    fig.update_layout(
                        height=400 + (subplot_count-1)*150,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=11, color="#6e7681")
                    )
                    
                    # Update axes styling
                    for i in range(1, subplot_count + 1):
                        fig.update_xaxes(
                            showgrid=False,
                            showticklabels=True if i == subplot_count else False,
                            color='#6e7681',
                            row=i, col=1
                        )
                        fig.update_yaxes(
                            showgrid=True,
                            gridcolor='rgba(110, 118, 129, 0.2)',
                            showticklabels=True,
                            color='#6e7681',
                            side='right',
                            row=i, col=1
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key metrics row
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        high_price = chart_df['high'].max()
                        st.metric("High", f"${high_price:.2f}")
                    
                    with metric_col2:
                        low_price = chart_df['low'].min()
                        st.metric("Low", f"${low_price:.2f}")
                    
                    with metric_col3:
                        avg_volume = chart_df['volume'].mean()
                        st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    
                    with metric_col4:
                        market_cap = info.get('market_cap', 0)
                        if market_cap > 1e9:
                            market_cap_str = f"${market_cap/1e9:.1f}B"
                        elif market_cap > 1e6:
                            market_cap_str = f"${market_cap/1e6:.1f}M"
                        else:
                            market_cap_str = f"${market_cap:,.0f}"
                        st.metric("Market Cap", market_cap_str)
                
                else:
                    st.warning(f"No chart data available for {symbol}")
                    
            except Exception as e:
                st.error(f"Error loading chart for {symbol}: {str(e)}")
            
            st.markdown("---")

# Historical Charts Page (legacy)
elif page == "historical":
    st.title("📉 Historical Charts")
    
    # Symbol selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol:",
            options=st.session_state.watchlist + ["Other"],
            key="chart_symbol"
        )
        
        if selected_symbol == "Other":
            selected_symbol = st.text_input("Enter symbol:")
    
    with col2:
        period = st.selectbox(
            "Time Period:",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Candlestick", "Line", "OHLC"]
        )
    
    if selected_symbol and selected_symbol != "Other":
        # Fetch data
        df = fetch_stock_data(selected_symbol, period)
        
        if not df.empty:
            df = calculate_technical_indicators(df, selected_symbol)
            
            # Main price chart
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ))
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines',
                    name='Close Price'
                ))
            else:  # OHLC
                fig.add_trace(go.Ohlc(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.1)'
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} - Historical Price Chart",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            
            fig_volume.update_layout(
                title="Trading Volume",
                height=200
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # MACD chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue')
            ))
            
            fig_macd.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='red')
            ))
            
            fig_macd.add_trace(go.Bar(
                x=df.index,
                y=df['MACD_Diff'],
                name='Histogram',
                marker_color='gray'
            ))
            
            fig_macd.update_layout(
                title="MACD Indicator",
                height=300
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # RSI chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple')
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig_rsi.update_layout(
                title="RSI (Relative Strength Index)",
                height=200,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)

# Robinhood-style CSS
st.markdown("""
<style>
    /* Reset and base styles */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Navigation buttons styling - Robinhood style */
    .stButton > button {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 24px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        min-height: 40px !important;
    }
    
    .stButton > button:hover {
        background-color: #e9ecef !important;
        border-color: #adb5bd !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Primary button (selected nav) - Robinhood green */
    .stButton > button[kind="primary"] {
        background-color: #00c805 !important;
        color: #ffffff !important;
        border-color: #00c805 !important;
        box-shadow: 0 2px 8px rgba(0,200,5,0.3) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #00a804 !important;
        border-color: #00a804 !important;
        box-shadow: 0 4px 12px rgba(0,200,5,0.4) !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background-color: #6c757d !important;
        color: #ffffff !important;
        border-color: #6c757d !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #5a6268 !important;
        border-color: #545b62 !important;
    }
    
    /* Small buttons (like arrow buttons) */
    .stButton > button[data-testid*="small"] {
        background-color: #007bff !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        padding: 8px !important;
        min-width: 36px !important;
        min-height: 36px !important;
    }
    
    /* Text and content styling */
    .stApp, .main .block-container {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    /* All text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stText, p, span, div {
        color: #212529 !important;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
    }
    
    div[data-testid="metric-container"] label {
        color: #6c757d !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #212529 !important;
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    
    /* Positive/negative colors */
    div[data-testid="metric-container"] [data-testid="metric-delta"] svg {
        display: none;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #00c805 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-delta"].negative {
        color: #ff5000 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        padding: 12px !important;
        font-size: 16px !important;
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #00c805 !important;
        box-shadow: 0 0 0 2px rgba(0,200,5,0.25) !important;
        outline: none !important;
    }
    
    /* Input labels */
    .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #495057 !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox dropdown */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Dividers */
    hr {
        border: none !important;
        border-top: 1px solid #e3e3e3 !important;
        margin: 24px 0 !important;
    }
    
    /* Charts */
    .js-plotly-plot .plotly {
        background-color: #ffffff !important;
    }
    
    /* Chat messages styling */
    .stMarkdown strong {
        color: #007bff !important;
        font-weight: 600 !important;
    }
    
    /* Code blocks and pre tags */
    .stMarkdown pre, .stMarkdown code {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        border: 1px solid #e9ecef !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    /* Success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 6px !important;
        border: none !important;
    }
    
    .stSuccess {
        background-color: #d1edff !important;
        color: #0c5460 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .stApp, .main .block-container {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stText, p, span, div {
            color: #ffffff !important;
        }
        
        .stButton > button {
            background-color: #2d3748 !important;
            color: #ffffff !important;
            border-color: #4a5568 !important;
        }
        
        div[data-testid="metric-container"] {
            background-color: #2d3748 !important;
            border-color: #4a5568 !important;
        }
        
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            color: #ffffff !important;
        }
        
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            background-color: #2d3748 !important;
            border-color: #4a5568 !important;
            color: #ffffff !important;
        }
        
        hr {
            border-top-color: #4a5568 !important;
        }
        
        .js-plotly-plot .plotly {
            background-color: #1a1a1a !important;
        }
    }
    
    /* Safari fixes */
    @supports (-webkit-appearance: none) {
        .stButton > button {
            -webkit-appearance: none !important;
        }
        
        .stTextInput input, .stSelectbox select {
            -webkit-appearance: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)