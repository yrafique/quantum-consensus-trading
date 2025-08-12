"""
Data Module
===========

Centralized data fetching and storage for the trading system.
"""

from .data_fetcher import (
    MarketDataFetcher,
    get_data_fetcher,
    fetch_ohlcv_data,
    fetch_stock_info,
    calculate_indicators,
    get_realtime_quote
)

# Convenience functions for sync operations
def sync_watchlist(symbols):
    """Sync data for a list of symbols"""
    return get_data_fetcher().bulk_sync_watchlist(symbols)

def sync_latest_data(symbols=None, force_refresh=False):
    """Sync latest data for symbols"""
    return get_data_fetcher().sync_latest_data(symbols, force_refresh)

def get_database_stats():
    """Get database statistics"""
    return get_data_fetcher().get_database_stats()

def cleanup_old_data(days=365):
    """Cleanup old data"""
    return get_data_fetcher().cleanup_old_data(days)

__all__ = [
    'MarketDataFetcher',
    'get_data_fetcher',
    'fetch_ohlcv_data',
    'fetch_stock_info',
    'calculate_indicators',
    'get_realtime_quote',
    'sync_watchlist',
    'sync_latest_data',
    'get_database_stats',
    'cleanup_old_data'
]