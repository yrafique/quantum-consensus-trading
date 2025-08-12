"""
Real-time Quote Streamer
========================

Streams live market data to WebSocket clients using Yahoo Finance.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Set, Any, Optional
from dataclasses import dataclass
import yfinance as yf

from ..core.logging_config import get_logger
from ..data.data_fetcher import MarketDataFetcher
from .websocket_manager import websocket_manager

logger = get_logger(__name__)


@dataclass
class QuoteUpdate:
    """Real-time quote update"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "market_status": self._get_market_status()
        }
    
    def _get_market_status(self) -> str:
        """Determine market status based on time"""
        now = datetime.now()
        # Simple market hours check (Eastern Time approximation)
        hour = now.hour
        
        if 9 <= hour < 16:  # 9 AM to 4 PM EST (approximate)
            return "open"
        elif hour < 9:
            return "pre_market"
        else:
            return "after_hours"


class QuoteStreamer:
    """
    Real-time quote streaming service.
    
    Features:
    - Subscribes to symbols requested by WebSocket clients
    - Fetches live quotes from Yahoo Finance
    - Broadcasts updates to subscribed clients
    - Handles market hours and rate limiting
    """
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.data_fetcher = MarketDataFetcher()
        self.subscribed_symbols: Set[str] = set()
        self.last_quotes: Dict[str, QuoteUpdate] = {}
        self._streaming_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the quote streaming service"""
        if self._streaming_task is not None:
            return
        
        self._running = True
        self._streaming_task = asyncio.create_task(self._streaming_loop())
        logger.info("Quote streamer started")
    
    async def stop(self):
        """Stop the quote streaming service"""
        self._running = False
        
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None
        
        logger.info("Quote streamer stopped")
    
    async def subscribe_symbol(self, symbol: str):
        """
        Subscribe to a symbol for real-time updates.
        
        Args:
            symbol: Stock symbol to subscribe to
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.subscribed_symbols:
            self.subscribed_symbols.add(symbol_upper)
            logger.info(f"Subscribed to real-time quotes for {symbol_upper}")
            
            # Fetch initial quote
            await self._fetch_and_broadcast_quote(symbol_upper)
    
    async def unsubscribe_symbol(self, symbol: str):
        """
        Unsubscribe from a symbol.
        
        Args:
            symbol: Stock symbol to unsubscribe from
        """
        symbol_upper = symbol.upper()
        self.subscribed_symbols.discard(symbol_upper)
        if symbol_upper in self.last_quotes:
            del self.last_quotes[symbol_upper]
        logger.info(f"Unsubscribed from real-time quotes for {symbol_upper}")
    
    async def _streaming_loop(self):
        """Main streaming loop"""
        while self._running:
            try:
                # Update subscribed symbols based on WebSocket connections
                await self._update_subscriptions()
                
                # Fetch and broadcast quotes for all subscribed symbols
                if self.subscribed_symbols:
                    await self._fetch_all_quotes()
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in quote streaming loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_subscriptions(self):
        """Update subscriptions based on WebSocket connections"""
        # Get symbols that clients are subscribed to
        stats = websocket_manager.get_connection_stats()
        client_symbols = set(stats.get("symbol_subscriber_counts", {}).keys())
        
        # Add new symbols
        new_symbols = client_symbols - self.subscribed_symbols
        for symbol in new_symbols:
            await self.subscribe_symbol(symbol)
        
        # Remove symbols with no subscribers
        orphaned_symbols = self.subscribed_symbols - client_symbols
        for symbol in orphaned_symbols:
            await self.unsubscribe_symbol(symbol)
    
    async def _fetch_all_quotes(self):
        """Fetch quotes for all subscribed symbols"""
        if not self.subscribed_symbols:
            return
        
        # Batch fetch quotes (more efficient than individual requests)
        try:
            # Convert to list for yfinance
            symbols_list = list(self.subscribed_symbols)
            
            # Use yfinance for real-time data
            tickers = yf.Tickers(" ".join(symbols_list))
            
            # Fetch current data for each symbol
            for symbol in symbols_list:
                try:
                    ticker = tickers.tickers[symbol]
                    await self._fetch_and_broadcast_quote(symbol, ticker)
                except Exception as e:
                    logger.warning(f"Failed to fetch quote for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to batch fetch quotes: {e}")
            
            # Fallback to individual requests
            for symbol in self.subscribed_symbols:
                await self._fetch_and_broadcast_quote(symbol)
    
    async def _fetch_and_broadcast_quote(self, symbol: str, ticker: Optional[yf.Ticker] = None):
        """
        Fetch quote for a single symbol and broadcast if changed.
        
        Args:
            symbol: Stock symbol
            ticker: Optional pre-created yfinance ticker
        """
        try:
            # Get ticker if not provided
            if ticker is None:
                ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get recent history for price change calculation
            hist = ticker.history(period="2d", interval="1d")
            
            if hist.empty or len(hist) == 0:
                logger.warning(f"No history data for {symbol}")
                return
            
            # Extract current data
            current_price = float(info.get("currentPrice", hist["Close"].iloc[-1]))
            
            # Calculate change
            if len(hist) >= 2:
                previous_close = float(hist["Close"].iloc[-2])
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
            else:
                change = 0.0
                change_percent = 0.0
            
            # Get volume
            volume = int(info.get("volume", hist["Volume"].iloc[-1] if not hist["Volume"].empty else 0))
            
            # Create quote update
            quote_update = QuoteUpdate(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                timestamp=datetime.now()
            )
            
            # Check if quote has changed significantly
            if self._should_broadcast_quote(symbol, quote_update):
                # Update cache
                self.last_quotes[symbol] = quote_update
                
                # Broadcast to WebSocket clients
                await websocket_manager.broadcast_quote(symbol, quote_update.to_dict())
                
                logger.debug(f"Broadcasted quote update for {symbol}: ${current_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
    
    def _should_broadcast_quote(self, symbol: str, new_quote: QuoteUpdate) -> bool:
        """
        Determine if quote should be broadcasted.
        
        Args:
            symbol: Stock symbol
            new_quote: New quote data
            
        Returns:
            bool: True if quote should be broadcasted
        """
        if symbol not in self.last_quotes:
            return True  # First quote for this symbol
        
        last_quote = self.last_quotes[symbol]
        
        # Always broadcast if price changed
        if abs(new_quote.price - last_quote.price) > 0.001:  # Price change > 0.1 cent
            return True
        
        # Broadcast if significant volume change
        if abs(new_quote.volume - last_quote.volume) / max(last_quote.volume, 1) > 0.1:  # 10% volume change
            return True
        
        # Broadcast periodically even if no change (heartbeat)
        time_diff = (new_quote.timestamp - last_quote.timestamp).total_seconds()
        if time_diff > 60:  # Force broadcast every minute
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get streamer status"""
        return {
            "running": self._running,
            "subscribed_symbols": list(self.subscribed_symbols),
            "symbol_count": len(self.subscribed_symbols),
            "last_update": datetime.now().isoformat(),
            "update_interval": self.update_interval
        }


# Global quote streamer instance
quote_streamer = QuoteStreamer()