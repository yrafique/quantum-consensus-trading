#!/usr/bin/env python3
"""
Streamlit WebSocket Fix
======================

Updates Streamlit to connect to the working WebSocket service.
"""

import streamlit as st

def create_working_websocket_javascript():
    """Generate JavaScript code for working WebSocket connection"""
    return f"""
    <script>
    console.log('🚀 Initializing QuantumConsensus WebSocket...');
    
    let ws = null;
    let clientId = 'streamlit_' + Date.now();
    let subscriptions = new Set(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']);
    let quoteData = {{}};
    
    function connectWebSocket() {{
        // Connect to working service on port 8001
        const wsUrl = 'ws://localhost:8001/quotes/' + clientId;
        ws = new WebSocket(wsUrl);
        
        ws.onopen = function(event) {{
            console.log('✅ WebSocket connected to working service');
            
            // Subscribe to watchlist symbols
            subscriptions.forEach(symbol => {{
                const subscribeMsg = {{
                    type: 'subscribe',
                    data: {{ symbol: symbol }}
                }};
                ws.send(JSON.stringify(subscribeMsg));
                console.log('📊 Subscribed to ' + symbol);
            }});
        }};
        
        ws.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            
            if (message.type === 'quote') {{
                const quote = message.data;
                quoteData[quote.symbol] = quote;
                
                console.log('📈 Quote update:', quote.symbol, '$' + quote.price.toFixed(2));
                
                // Update page elements if they exist
                updateQuoteDisplay(quote);
                
                // Store in localStorage for Streamlit to read
                localStorage.setItem('ws_quotes', JSON.stringify(quoteData));
                localStorage.setItem('ws_connected', 'true');
                localStorage.setItem('ws_last_update', Date.now().toString());
                
            }} else if (message.type === 'status') {{
                console.log('ℹ️ Status:', message.data.status);
                localStorage.setItem('ws_status', message.data.status);
            }}
        }};
        
        ws.onclose = function(event) {{
            console.log('❌ WebSocket disconnected');
            localStorage.setItem('ws_connected', 'false');
            
            // Attempt reconnection after 5 seconds
            setTimeout(connectWebSocket, 5000);
        }};
        
        ws.onerror = function(error) {{
            console.error('❌ WebSocket error:', error);
            localStorage.setItem('ws_connected', 'false');
        }};
    }}
    
    function updateQuoteDisplay(quote) {{
        // Update any existing quote elements on the page
        const elements = document.querySelectorAll('[data-symbol="' + quote.symbol + '"]');
        elements.forEach(element => {{
            if (element.querySelector('.price')) {{
                element.querySelector('.price').textContent = '$' + quote.price.toFixed(2);
            }}
            if (element.querySelector('.change')) {{
                const changeText = (quote.change_percent >= 0 ? '+' : '') + quote.change_percent.toFixed(2) + '%';
                element.querySelector('.change').textContent = changeText;
                element.querySelector('.change').style.color = quote.change_percent >= 0 ? '#00c805' : '#ff4444';
            }}
            if (element.querySelector('.status')) {{
                element.querySelector('.status').textContent = '🔴 LIVE';
                element.querySelector('.status').style.color = '#00c805';
            }}
        }});
    }}
    
    function getConnectionStatus() {{
        const connected = localStorage.getItem('ws_connected') === 'true';
        const lastUpdate = localStorage.getItem('ws_last_update');
        const age = lastUpdate ? (Date.now() - parseInt(lastUpdate)) / 1000 : 999;
        
        return {{
            connected: connected,
            fresh: age < 30,
            lastUpdate: lastUpdate
        }};
    }}
    
    function getQuoteData(symbol) {{
        const quotes = localStorage.getItem('ws_quotes');
        if (quotes) {{
            const data = JSON.parse(quotes);
            return data[symbol] || null;
        }}
        return null;
    }}
    
    // Initialize connection
    connectWebSocket();
    
    // Make functions globally available
    window.quantumWS = {{
        getStatus: getConnectionStatus,
        getQuote: getQuoteData,
        isConnected: () => localStorage.getItem('ws_connected') === 'true'
    }};
    
    console.log('🎉 QuantumConsensus WebSocket initialized');
    </script>
    """

def display_websocket_status():
    """Display WebSocket connection status using JavaScript data"""
    
    # Inject the working WebSocket JavaScript
    st.markdown(create_working_websocket_javascript(), unsafe_allow_html=True)
    
    # Display connection status
    st.markdown("""
    <div id="ws-status">
        <script>
        const status = window.quantumWS ? window.quantumWS.getStatus() : { connected: false };
        const statusText = status.connected ? 
            (status.fresh ? '🔗 Connected to real-time data stream' : '🟡 Connected (data may be stale)') :
            '❌ Real-time data stream disconnected';
        const statusColor = status.connected ? (status.fresh ? '#00c805' : '#ffa500') : '#ff4444';
        
        document.getElementById('ws-status').innerHTML = 
            '<div style="color: ' + statusColor + '; font-weight: 600;">' + statusText + '</div>';
        </script>
    </div>
    """, unsafe_allow_html=True)

def display_live_quote(symbol):
    """Display a live quote with WebSocket data"""
    
    # Get fallback static data (you would replace this with your existing data function)
    fallback_price = 0.00
    fallback_change = 0.00
    
    st.markdown(f"""
    <div data-symbol="{symbol}" style="
        background: rgba(255, 255, 255, 0.02); 
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-radius: 8px; 
        padding: 12px; 
        margin-bottom: 8px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 600; color: #ffffff;">{symbol}</span>
                <div class="status" style="font-size: 10px; color: rgba(255,255,255,0.5);">⚪ STATIC</div>
            </div>
            <div style="text-align: right;">
                <div class="price" style="color: #ffffff; font-size: 14px;">${fallback_price:.2f}</div>
                <div class="change" style="color: #666; font-size: 12px;">{fallback_change:+.2f}%</div>
            </div>
        </div>
    </div>
    
    <script>
    // Update with live data if available
    setTimeout(() => {{
        if (window.quantumWS) {{
            const quote = window.quantumWS.getQuote('{symbol}');
            if (quote) {{
                const element = document.querySelector('[data-symbol="{symbol}"]');
                if (element) {{
                    element.querySelector('.price').textContent = '$' + quote.price.toFixed(2);
                    const changeText = (quote.change_percent >= 0 ? '+' : '') + quote.change_percent.toFixed(2) + '%';
                    element.querySelector('.change').textContent = changeText;
                    element.querySelector('.change').style.color = quote.change_percent >= 0 ? '#00c805' : '#ff4444';
                    element.querySelector('.status').textContent = '🔴 LIVE';
                    element.querySelector('.status').style.color = '#00c805';
                }}
            }}
        }}
    }}, 1000);
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("WebSocket Fix Test")
    
    display_websocket_status()
    
    st.subheader("Live Quotes")
    
    for symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']:
        display_live_quote(symbol)
    
    # Auto-refresh every 10 seconds
    st.markdown("""
    <script>
    setTimeout(() => {
        window.location.reload();
    }, 10000);
    </script>
    """, unsafe_allow_html=True)