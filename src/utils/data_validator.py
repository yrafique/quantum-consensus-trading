"""
Data Validation and Real-Time Fetching System
==============================================

This module provides comprehensive data validation, real-time price fetching,
and data freshness monitoring for the trading system.
"""

import json
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

class DataValidator:
    """Comprehensive data validation and real-time fetching system."""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.max_age_hours = 24  # Maximum data age in hours
        self.known_real_prices = {
            # Add current market prices for validation
            "NVDA": 177.0,  # Current approximate price
            "AAPL": 230.0,
            "MSFT": 420.0,
            "TSLA": 250.0,
            "AMZN": 155.0
        }
    
    def validate_data_freshness(self, ticker: str) -> Dict[str, any]:
        """Validate if data is fresh and accurate."""
        file_path = self.data_dir / f"{ticker}_daily.json"
        
        if not file_path.exists():
            return {
                "valid": False,
                "reason": "Data file not found",
                "action": "fetch_new"
            }
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                return {
                    "valid": False,
                    "reason": "Empty data file",
                    "action": "fetch_new"
                }
            
            # Check data age
            latest_date = datetime.strptime(data[-1]["date"], "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.now() - latest_date).total_seconds() / 3600
            
            # Check if data is too old
            if age_hours > self.max_age_hours:
                return {
                    "valid": False,
                    "reason": f"Data is {age_hours:.1f} hours old",
                    "action": "refresh_data",
                    "age_hours": age_hours
                }
            
            # Check if data looks synthetic (future dates)
            if latest_date > datetime.now() + timedelta(days=1):
                return {
                    "valid": False,
                    "reason": "Data contains future dates (synthetic)",
                    "action": "fetch_real_data",
                    "synthetic": True
                }
            
            # Validate price against known ranges
            latest_price = data[-1]["close"]
            if ticker in self.known_real_prices:
                expected_price = self.known_real_prices[ticker]
                price_diff_pct = abs(latest_price - expected_price) / expected_price * 100
                
                if price_diff_pct > 50:  # More than 50% difference
                    return {
                        "valid": False,
                        "reason": f"Price mismatch: showing ${latest_price:.2f}, expected ~${expected_price:.2f}",
                        "action": "fetch_real_data",
                        "price_mismatch": True
                    }
            
            return {
                "valid": True,
                "reason": "Data is fresh and valid",
                "age_hours": age_hours,
                "latest_price": latest_price
            }
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error reading data: {str(e)}",
                "action": "fetch_new"
            }
    
    def get_real_time_price(self, ticker: str) -> Optional[float]:
        """Get real-time price using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            return float(current_price) if current_price else None
        except Exception as e:
            print(f"Error fetching real-time price for {ticker}: {e}")
            return None
    
    def fetch_fresh_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch fresh historical data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                print(f"No data returned for {ticker}")
                return None
            
            # Convert to our format
            df = pd.DataFrame()
            df['date'] = hist.index.strftime('%Y-%m-%d %H:%M:%S')
            df['open'] = hist['Open'].values
            df['high'] = hist['High'].values
            df['low'] = hist['Low'].values
            df['close'] = hist['Close'].values
            df['volume'] = hist['Volume'].values
            
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe."""
        try:
            # RSI calculation
            df['rsi'] = self.calculate_rsi(df['close'], window=14)
            
            # EMA21
            df['ema21'] = df['close'].ewm(span=21).mean()
            
            # VWAP calculation
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Volume moving average
            df['volume_avg50'] = df['volume'].rolling(window=50).mean()
            
            # Bullish engulfing pattern (simplified)
            df['bullish_engulfing'] = (
                (df['close'] > df['open']) &  # Current candle is green
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle was red
                (df['close'] > df['open'].shift(1)) &  # Current close > previous open
                (df['open'] < df['close'].shift(1))  # Current open < previous close
            )
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_fresh_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save fresh data to JSON file."""
        try:
            file_path = self.data_dir / f"{ticker}_daily.json"
            
            # Convert DataFrame to list of dictionaries
            data = df.to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            for record in data:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value)
                    elif isinstance(value, np.bool_):
                        record[key] = bool(value)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Saved fresh data for {ticker}: {len(data)} records")
            return True
            
        except Exception as e:
            print(f"Error saving data for {ticker}: {e}")
            return False
    
    def validate_and_refresh_ticker(self, ticker: str) -> Dict[str, any]:
        """Validate and refresh data for a specific ticker."""
        print(f"🔍 Validating data for {ticker}...")
        
        validation = self.validate_data_freshness(ticker)
        
        if validation["valid"]:
            print(f"✅ {ticker} data is valid")
            return validation
        
        print(f"❌ {ticker}: {validation['reason']}")
        print(f"🔄 Fetching fresh data...")
        
        # Fetch real-time price for immediate validation
        real_time_price = self.get_real_time_price(ticker)
        if real_time_price:
            print(f"📊 Real-time {ticker} price: ${real_time_price:.2f}")
        
        # Fetch fresh historical data
        fresh_df = self.fetch_fresh_data(ticker)
        
        if fresh_df is not None:
            # Save the fresh data
            if self.save_fresh_data(ticker, fresh_df):
                return {
                    "valid": True,
                    "refreshed": True,
                    "reason": "Data refreshed successfully",
                    "latest_price": fresh_df['close'].iloc[-1],
                    "real_time_price": real_time_price,
                    "records": len(fresh_df)
                }
        
        return {
            "valid": False,
            "reason": "Failed to fetch fresh data",
            "real_time_price": real_time_price
        }
    
    def validate_all_watchlist(self, watchlist: List[str]) -> Dict[str, Dict]:
        """Validate and refresh all tickers in watchlist."""
        results = {}
        
        print("🚀 Starting comprehensive data validation...")
        print("=" * 50)
        
        for ticker in watchlist:
            results[ticker] = self.validate_and_refresh_ticker(ticker)
            print()  # Empty line for readability
        
        # Summary
        valid_count = sum(1 for r in results.values() if r["valid"])
        refreshed_count = sum(1 for r in results.values() if r.get("refreshed", False))
        
        print("📊 VALIDATION SUMMARY:")
        print(f"   Valid: {valid_count}/{len(watchlist)}")
        print(f"   Refreshed: {refreshed_count}/{len(watchlist)}")
        
        return results

    def get_market_context(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive market context for LLM analysis."""
        try:
            file_path = self.data_dir / f"{ticker}_daily.json"
            
            if not file_path.exists():
                # Try to fetch fresh data
                self.validate_and_refresh_ticker(ticker)
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if not data:
                return None
                
            # Get the latest data point
            latest = data[-1]
            
            # Calculate derived indicators
            close = latest.get("close", 0)
            ema21 = latest.get("ema21", close)
            vwap = latest.get("vwap", close)
            rsi = latest.get("rsi", 50)
            volume = latest.get("volume", 0)
            volume_avg50 = latest.get("volume_avg50", volume)
            
            # Short interest data (from data ingestion module)
            try:
                from .data_ingestion import get_short_interest
                short_float, days_to_cover = get_short_interest(ticker)
            except ImportError:
                # Fallback values if import fails
                short_float, days_to_cover = 0.1, 1.0
            
            context = {
                "close": close,
                "rsi": rsi,
                "ema21": ema21,
                "vwap": vwap,
                "volume": volume,
                "volume_avg50": volume_avg50,
                "short_float": short_float,
                "days_to_cover": days_to_cover,
                "bullish_engulfing": latest.get("bullish_engulfing", False),
                "volume_spike": volume > volume_avg50 * 1.5 if volume_avg50 > 0 else False,
                "short_squeeze": short_float > 0.15 and days_to_cover > 1.5,
                "price_crossover": close > ema21,
                "rsi_momentum": rsi > 50
            }
            
            return context
            
        except Exception as e:
            print(f"Error getting market context for {ticker}: {e}")
            return None

    def get_comprehensive_data(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive data for opportunity screening."""
        try:
            # First ensure we have fresh data
            validation_result = self.validate_data_freshness(ticker)
            if not validation_result["valid"]:
                refresh_result = self.validate_and_refresh_ticker(ticker)
                if not refresh_result["valid"]:
                    return None
            
            # Get market context
            context = self.get_market_context(ticker)
            if context is None:
                return None
                
            # Add additional screening metrics
            context["ticker"] = ticker
            context["data_age_hours"] = validation_result.get("age_hours", 0)
            context["price_above_ema"] = context["close"] > context["ema21"]
            context["price_above_vwap"] = context["close"] > context["vwap"]
            
            return context
            
        except Exception as e:
            print(f"Error getting comprehensive data for {ticker}: {e}")
            return None


def main():
    """Run data validation as standalone script."""
    validator = DataValidator()
    
    # Default watchlist
    watchlist = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    
    results = validator.validate_all_watchlist(watchlist)
    
    # Print final status
    print("\n🎯 FINAL STATUS:")
    for ticker, result in results.items():
        status = "✅ VALID" if result["valid"] else "❌ INVALID"
        price = result.get("latest_price", "N/A")
        real_price = result.get("real_time_price", "N/A")
        
        print(f"   {ticker}: {status}")
        if price != "N/A":
            print(f"      Latest: ${price:.2f}")
        if real_price != "N/A":
            print(f"      Real-time: ${real_price:.2f}")

if __name__ == "__main__":
    main()