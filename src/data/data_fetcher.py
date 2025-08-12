"""
Centralized Data Fetcher Module
==============================

Single source of truth for all market data fetching operations.
Handles multiple data sources, caching, and historical data storage.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
import logging
from pathlib import Path
import json
import requests
from functools import lru_cache
import ta

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path(__file__).parent.parent.parent / "data" / "market_data.db"
CACHE_DURATION = timedelta(minutes=5)
MAX_RETRY_ATTEMPTS = 3

class MarketDataFetcher:
    """Centralized market data fetcher with SQLite caching"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._cache = {}
        self._last_fetch = {}
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Historical OHLCV data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT DEFAULT 'yahoo',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            """)
            
            # Stock info/metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    beta REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Technical indicators cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    indicator_name TEXT NOT NULL,
                    value REAL,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, indicator_name, parameters)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON technical_indicators(symbol, timestamp)")
            
            conn.commit()
    
    def fetch_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "1d", 
                    force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch OHLCV data with intelligent caching
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force fetch from API instead of cache
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check memory cache first
        if not force_refresh and cache_key in self._cache:
            last_fetch = self._last_fetch.get(cache_key)
            if last_fetch and datetime.now() - last_fetch < CACHE_DURATION:
                logger.debug(f"Returning cached data for {symbol}")
                return self._cache[cache_key]
        
        # Check database for historical data
        if not force_refresh and interval == "1d":
            db_data = self._load_from_db(symbol, period)
            if not db_data.empty:
                # Check if we need to fetch only recent data
                latest_date = db_data.index.max()
                if latest_date.date() >= (datetime.now().date() - timedelta(days=1)):
                    logger.debug(f"Returning database data for {symbol}")
                    return db_data
        
        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching data for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean and standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Store in database if daily data
            if interval == "1d" and not df.empty:
                self._store_in_db(symbol, df)
            
            # Update caches
            self._cache[cache_key] = df
            self._last_fetch[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # Try to return cached/database data if available
            if cache_key in self._cache:
                return self._cache[cache_key]
            return self._load_from_db(symbol, period)
    
    def fetch_stock_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch detailed stock information
        
        Returns dict with: price, change, change_percent, volume, market_cap, 
                          pe_ratio, dividend_yield, beta, name, sector, industry
        """
        try:
            # Check database first
            if not force_refresh:
                info = self._load_stock_info_from_db(symbol)
                if info and (datetime.now() - info.get('last_updated', datetime.min)) < timedelta(hours=24):
                    return info
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract and standardize data
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'last_updated': datetime.now()
            }
            
            # Store in database
            self._store_stock_info_in_db(stock_info)
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            # Return cached data or defaults
            cached = self._load_stock_info_from_db(symbol)
            if cached:
                logger.debug(f"Returning cached data for {symbol}")
                return cached
            
            logger.warning(f"No cached data available for {symbol}, returning defaults")
            return {
                'symbol': symbol,
                'name': symbol,
                'price': 0.0,
                'change': 0.0,
                'change_percent': 0.0,
                'volume': 0,
                'market_cap': 0,
                'pe_ratio': 0.0,
                'dividend_yield': 0.0,
                'beta': 1.0,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'last_updated': datetime.now()
            }
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculate technical indicators for the given OHLCV data
        
        Adds columns: SMA_20, SMA_50, EMA_12, EMA_26, RSI, MACD, MACD_Signal, 
                     MACD_Diff, BB_Upper, BB_Middle, BB_Lower, ATR, ADX
        """
        if df.empty:
            return df
        
        # Ensure we have lowercase column names
        df.columns = [col.lower() for col in df.columns]
        
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Store indicators in database if symbol provided
        if symbol and not df.empty:
            self._store_indicators_in_db(symbol, df)
        
        return df
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks efficiently"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_ohlcv(symbol, period)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results
    
    def get_realtime_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote (price, bid, ask, volume)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'bid_size': info.get('bidSize', 0),
                'ask_size': info.get('askSize', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {'price': 0, 'bid': 0, 'ask': 0, 'volume': 0}
    
    # Database operations
    def _load_from_db(self, symbol: str, period: str) -> pd.DataFrame:
        """Load OHLCV data from database"""
        try:
            # Convert period to days
            period_days = self._period_to_days(period)
            start_date = datetime.now() - timedelta(days=period_days)
            
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(symbol, start_date), 
                                     parse_dates=['timestamp'], index_col='timestamp')
            return df
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()
    
    def _store_in_db(self, symbol: str, df: pd.DataFrame):
        """Store OHLCV data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data for insertion
                df_to_store = df.reset_index()
                df_to_store['symbol'] = symbol
                df_to_store['timestamp'] = df_to_store['Date'] if 'Date' in df_to_store.columns else df_to_store.index
                
                # Use replace to handle duplicates
                df_to_store[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                    'ohlcv_data', conn, if_exists='replace', index=False
                )
                
        except Exception as e:
            logger.error(f"Error storing in database: {e}")
    
    def _load_stock_info_from_db(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load stock info from database"""
        try:
            query = "SELECT * FROM stock_info WHERE symbol = ?"
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    info = dict(zip(columns, row))
                    if 'last_updated' in info and info['last_updated']:
                        info['last_updated'] = datetime.fromisoformat(info['last_updated'])
                    else:
                        info['last_updated'] = datetime.now()
                    
                    # Ensure all required keys exist with proper defaults
                    defaults = {
                        'price': 0.0,
                        'change': 0.0,
                        'change_percent': 0.0,
                        'volume': 0,
                        'market_cap': 0,
                        'pe_ratio': 0.0,
                        'dividend_yield': 0.0,
                        'beta': 1.0,
                        'sector': 'Unknown',
                        'industry': 'Unknown'
                    }
                    
                    for key, default_value in defaults.items():
                        if key not in info or info[key] is None:
                            info[key] = default_value
                    
                    logger.debug(f"Loaded {symbol} from database with keys: {list(info.keys())}")
                    return info
                    
        except Exception as e:
            logger.error(f"Error loading stock info from database: {e}")
        return None
    
    def _store_stock_info_in_db(self, info: Dict[str, Any]):
        """Store stock info in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_info 
                    (symbol, name, sector, industry, market_cap, pe_ratio, dividend_yield, beta, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    info['symbol'], info['name'], info['sector'], info['industry'],
                    info['market_cap'], info['pe_ratio'], info['dividend_yield'], 
                    info['beta'], datetime.now()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing stock info in database: {e}")
    
    def _store_indicators_in_db(self, symbol: str, df: pd.DataFrame):
        """Store technical indicators in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store each indicator
                indicators = ['rsi', 'macd', 'macd_signal', 'adx', 'atr']
                for indicator in indicators:
                    if indicator in df.columns:
                        for timestamp, value in df[indicator].dropna().items():
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT OR REPLACE INTO technical_indicators
                                (symbol, timestamp, indicator_name, value)
                                VALUES (?, ?, ?, ?)
                            """, (symbol, timestamp, indicator, float(value)))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing indicators in database: {e}")
    
    @staticmethod
    def _period_to_days(period: str) -> int:
        """Convert period string to number of days"""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825,
            '10y': 3650, 'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
            'max': 3650  # Default to 10 years for max
        }
        return period_map.get(period, 30)
    
    def sync_latest_data(self, symbols: List[str] = None, force_refresh: bool = False):
        """
        Sync latest market data for given symbols or all symbols in database
        
        Args:
            symbols: List of symbols to sync. If None, syncs all symbols in database
            force_refresh: Force refresh even if data is recent
        """
        try:
            if symbols is None:
                # Get all unique symbols from database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT symbol FROM ohlcv_data")
                    symbols = [row[0] for row in cursor.fetchall()]
                    
                    # Also get symbols from stock_info table
                    cursor.execute("SELECT DISTINCT symbol FROM stock_info")
                    info_symbols = [row[0] for row in cursor.fetchall()]
                    symbols.extend(info_symbols)
                    symbols = list(set(symbols))  # Remove duplicates
            
            if not symbols:
                logger.info("No symbols to sync")
                return
            
            logger.info(f"Syncing latest data for {len(symbols)} symbols...")
            
            for symbol in symbols:
                try:
                    # Check if we need to update this symbol
                    if not force_refresh and not self._needs_update(symbol):
                        logger.debug(f"Skipping {symbol} - data is recent")
                        continue
                    
                    logger.info(f"Syncing {symbol}...")
                    
                    # Fetch latest OHLCV data (last 5 days to ensure we have latest)
                    df = self.fetch_ohlcv(symbol, period="5d", force_refresh=True)
                    
                    # Fetch latest stock info
                    info = self.fetch_stock_info(symbol, force_refresh=True)
                    
                    logger.debug(f"Updated {symbol} successfully")
                    
                except Exception as e:
                    logger.error(f"Error syncing {symbol}: {e}")
                    continue
            
            logger.info("Data sync completed")
            
        except Exception as e:
            logger.error(f"Error during data sync: {e}")
    
    def sync_historical_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None):
        """
        Sync comprehensive historical data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for historical data (default: 2 years ago)
            end_date: End date for historical data (default: today)
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=730)  # 2 years
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"Syncing historical data for {symbol} from {start_date.date()} to {end_date.date()}")
            
            # Fetch data in chunks to avoid API limits
            current_date = start_date
            chunk_size = timedelta(days=365)  # 1 year chunks
            
            while current_date < end_date:
                chunk_end = min(current_date + chunk_size, end_date)
                
                try:
                    # Calculate period string
                    days_diff = (chunk_end - current_date).days
                    if days_diff <= 5:
                        period = "5d"
                    elif days_diff <= 30:
                        period = "1mo"
                    elif days_diff <= 90:
                        period = "3mo"
                    elif days_diff <= 365:
                        period = "1y"
                    else:
                        period = "2y"
                    
                    # Fetch data
                    df = self.fetch_ohlcv(symbol, period=period, force_refresh=True)
                    
                    if not df.empty:
                        # Filter to the specific date range
                        mask = (df.index >= current_date) & (df.index <= chunk_end)
                        chunk_df = df.loc[mask]
                        
                        if not chunk_df.empty:
                            # Calculate and store technical indicators
                            chunk_df = self.calculate_technical_indicators(chunk_df, symbol)
                            logger.debug(f"Processed {len(chunk_df)} days for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error fetching chunk for {symbol} ({current_date.date()} to {chunk_end.date()}): {e}")
                
                current_date = chunk_end
            
            # Update stock info
            self.fetch_stock_info(symbol, force_refresh=True)
            
            logger.info(f"Historical data sync completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error syncing historical data for {symbol}: {e}")
    
    def bulk_sync_watchlist(self, symbols: List[str]):
        """
        Efficiently sync a watchlist of symbols
        
        Args:
            symbols: List of symbols to sync
        """
        logger.info(f"Bulk syncing watchlist: {symbols}")
        
        # First, sync latest data for all symbols
        self.sync_latest_data(symbols, force_refresh=False)
        
        # Then, ensure we have at least 1 year of history for each
        for symbol in symbols:
            try:
                # Check if we have sufficient historical data
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                        FROM ohlcv_data 
                        WHERE symbol = ?
                    """, (symbol,))
                    result = cursor.fetchone()
                    
                    if result[0] is None:
                        # No data, fetch full history
                        self.sync_historical_data(symbol)
                    else:
                        min_date = datetime.fromisoformat(result[0])
                        max_date = datetime.fromisoformat(result[1])
                        record_count = result[2]
                        
                        # Check if we need more historical data
                        one_year_ago = datetime.now() - timedelta(days=365)
                        if min_date > one_year_ago or record_count < 200:
                            logger.info(f"Fetching more historical data for {symbol}")
                            self.sync_historical_data(symbol, start_date=one_year_ago)
                            
            except Exception as e:
                logger.error(f"Error checking history for {symbol}: {e}")
    
    def _needs_update(self, symbol: str) -> bool:
        """Check if symbol needs data update based on last update time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check latest OHLCV data
                cursor.execute("""
                    SELECT MAX(timestamp) 
                    FROM ohlcv_data 
                    WHERE symbol = ?
                """, (symbol,))
                latest_ohlcv = cursor.fetchone()[0]
                
                # Check latest stock info
                cursor.execute("""
                    SELECT last_updated 
                    FROM stock_info 
                    WHERE symbol = ?
                """, (symbol,))
                result = cursor.fetchone()
                latest_info = result[0] if result else None
                
                now = datetime.now()
                
                # Need update if no data or data is older than 1 day
                if latest_ohlcv is None:
                    return True
                
                latest_ohlcv_dt = datetime.fromisoformat(latest_ohlcv)
                if (now - latest_ohlcv_dt).days >= 1:
                    return True
                
                if latest_info is None:
                    return True
                
                latest_info_dt = datetime.fromisoformat(latest_info)
                if (now - latest_info_dt).hours >= 24:
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking update status for {symbol}: {e}")
            return True  # Update on error to be safe
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get symbol count
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_data")
                symbol_count = cursor.fetchone()[0]
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
                total_records = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv_data")
                date_range = cursor.fetchone()
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Get technical indicators count
                cursor.execute("SELECT COUNT(*) FROM technical_indicators")
                indicator_count = cursor.fetchone()[0]
                
                return {
                    'symbols': symbol_count,
                    'total_records': total_records,
                    'oldest_data': date_range[0],
                    'newest_data': date_range[1],
                    'database_size_bytes': db_size,
                    'database_size_mb': db_size / (1024 * 1024),
                    'technical_indicators': indicator_count,
                    'last_updated': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 365):
        """Remove data older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records to be deleted
                cursor.execute("SELECT COUNT(*) FROM ohlcv_data WHERE timestamp < ?", (cutoff_date,))
                ohlcv_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE timestamp < ?", (cutoff_date,))
                indicator_count = cursor.fetchone()[0]
                
                # Delete old data
                cursor.execute("DELETE FROM ohlcv_data WHERE timestamp < ?", (cutoff_date,))
                cursor.execute("DELETE FROM technical_indicators WHERE timestamp < ?", (cutoff_date,))
                
                conn.commit()
                
                logger.info(f"Cleaned up {ohlcv_count} OHLCV records and {indicator_count} indicator records older than {days} days")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


# Global instance for easy access
_fetcher_instance = None

def get_data_fetcher() -> MarketDataFetcher:
    """Get global data fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = MarketDataFetcher()
    return _fetcher_instance

# Convenience functions for backward compatibility
def fetch_ohlcv_data(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data using global fetcher"""
    return get_data_fetcher().fetch_ohlcv(symbol, period, interval)

def fetch_stock_info(symbol: str) -> Dict[str, Any]:
    """Fetch stock info using global fetcher"""
    return get_data_fetcher().fetch_stock_info(symbol)

def calculate_indicators(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Calculate technical indicators using global fetcher"""
    return get_data_fetcher().calculate_technical_indicators(df, symbol)

def get_realtime_quote(symbol: str) -> Dict[str, float]:
    """Get real-time quote using global fetcher"""
    return get_data_fetcher().get_realtime_quote(symbol)