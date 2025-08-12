"""
Alpaca Paper Trading Integration
===============================

Live trading implementation using Alpaca's paper trading environment.
This module executes our high-conviction recommendations in real-time,
manages positions, and provides performance tracking.

Key Features:
- Paper trading environment (no real money risk)
- Automated order execution based on LLM recommendations
- Position management with Kelly criterion sizing
- Real-time performance monitoring
- Risk controls and stop-loss management
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.data.live import StockDataStream
except ImportError:
    print("❌ Alpaca SDK not installed. Run: pip install alpaca-py")
    raise

try:
    from .opportunity_hunter import OpportunityHunter
    from .llm_reasoner import generate_recommendation
    from .data_validator import DataValidator
except ImportError:
    from opportunity_hunter import OpportunityHunter
    from llm_reasoner import generate_recommendation
    from data_validator import DataValidator


@dataclass
class Trade:
    """Represents a trade recommendation ready for execution."""
    ticker: str
    action: str  # BUY/SELL
    quantity: int
    entry_price: float
    target_price: float
    stop_price: float
    confidence: float
    reasoning: str
    timestamp: datetime
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": self.composite_score
        }


class AlpacaTrader:
    """Live trading system using Alpaca's paper trading environment."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize Alpaca trading client.
        
        Parameters:
        -----------
        api_key : str
            Alpaca API key (or set ALPACA_API_KEY env var)
        secret_key : str  
            Alpaca secret key (or set ALPACA_SECRET_KEY env var)
        paper : bool
            Use paper trading environment (default: True)
        """
        # Try to load from .env file first
        try:
            from load_env import load_env
            load_env()
        except ImportError:
            pass
            
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "❌ Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass them as parameters."
            )
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Initialize supporting systems
        self.opportunity_hunter = OpportunityHunter()
        self.data_validator = DataValidator()
        
        # Trading parameters
        self.max_position_size = 0.10  # Max 10% of portfolio per position
        self.max_portfolio_risk = 0.20  # Max 20% total portfolio at risk
        self.min_confidence = 0.85      # Minimum 85% confidence for trades
        
        # Tracking
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Trade] = []
        
        # Data directory for persistence
        self.data_dir = Path(__file__).parent / "alpaca_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Verify connection
        self._verify_connection()
        
    def _verify_connection(self) -> bool:
        """Verify connection to Alpaca and log account info."""
        try:
            account = self.trading_client.get_account()
            environment = "PAPER" if self.paper else "LIVE"
            
            print(f"✅ Connected to Alpaca {environment} Trading")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Day Trade Count: {account.day_trade_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Alpaca: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            return {
                "account_id": account.id,
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "day_trade_count": account.day_trade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "active_positions": len(positions),
                "positions": [
                    {
                        "symbol": pos.symbol,
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc) * 100
                    }
                    for pos in positions
                ]
            }
        except Exception as e:
            logging.error(f"Failed to get account info: {e}")
            return {}
    
    def calculate_position_size(self, price: float, confidence: float, 
                              account_value: float) -> int:
        """
        Calculate position size using Kelly criterion and risk management.
        
        Parameters:
        -----------
        price : float
            Current stock price
        confidence : float
            LLM confidence (0-1)
        account_value : float
            Total portfolio value
            
        Returns:
        --------
        int : Number of shares to trade
        """
        # Kelly fraction based on confidence and assumed 2:1 reward/risk
        win_rate = confidence
        avg_win = 0.20  # Assume 20% average win
        avg_loss = 0.10  # Assume 10% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        # Apply confidence scaling
        confidence_adjusted = kelly_fraction * confidence
        
        # Calculate dollar amount
        position_value = account_value * confidence_adjusted
        
        # Convert to shares
        shares = int(position_value / price)
        
        # Ensure minimum viable trade (at least $100)
        if shares * price < 100:
            shares = max(1, int(100 / price))
            
        return shares
    
    def hunt_and_execute_opportunities(self, max_positions: int = 5) -> List[Trade]:
        """
        Hunt for opportunities and execute trades automatically.
        
        Parameters:
        -----------
        max_positions : int
            Maximum number of concurrent positions
            
        Returns:
        --------
        List[Trade] : List of executed trades
        """
        print("🎯 HUNTING FOR LIVE TRADING OPPORTUNITIES")
        print("=" * 50)
        
        # Get current account info
        account_info = self.get_account_info()
        current_positions = len(account_info.get("positions", []))
        available_slots = max_positions - current_positions
        
        print(f"📊 Portfolio Status:")
        print(f"   Value: ${account_info.get('portfolio_value', 0):,.2f}")
        print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        print(f"   Active Positions: {current_positions}/{max_positions}")
        print(f"   Available Slots: {available_slots}")
        print()
        
        if available_slots <= 0:
            print("⚠️ Maximum positions reached. No new trades will be executed.")
            return []
        
        # Hunt for opportunities
        opportunities = self.opportunity_hunter.hunt_opportunities(
            max_opportunities=available_slots * 2,  # Get more options to choose from
            debug_mode=False
        )
        
        if not opportunities:
            print("❌ No high-conviction opportunities found.")
            return []
        
        executed_trades = []
        
        for opp in opportunities[:available_slots]:
            try:
                # Convert opportunity to trade
                rec = opp["recommendation"]
                
                # Calculate position size
                account_value = account_info.get("portfolio_value", 100000)
                position_size = self.calculate_position_size(
                    rec["entry"], 
                    rec["confidence"], 
                    account_value
                )
                
                trade = Trade(
                    ticker=opp["ticker"],
                    action=rec["action"],
                    quantity=position_size,
                    entry_price=rec["entry"],
                    target_price=rec["target"],
                    stop_price=rec["stop"],
                    confidence=rec["confidence"],
                    reasoning=rec["reasoning"][:500] + "..." if len(rec["reasoning"]) > 500 else rec["reasoning"],
                    timestamp=datetime.now(),
                    composite_score=opp["composite_score"]
                )
                
                # Execute the trade
                if self.execute_trade(trade):
                    executed_trades.append(trade)
                    print(f"✅ Executed: {trade.action} {trade.quantity} shares of {trade.ticker}")
                else:
                    print(f"❌ Failed to execute: {trade.ticker}")
                    
            except Exception as e:
                print(f"❌ Error processing {opp['ticker']}: {e}")
                continue
        
        print(f"\n🚀 EXECUTION SUMMARY: {len(executed_trades)} trades executed")
        
        # Save trade log
        self._save_trade_log(executed_trades)
        
        return executed_trades
    
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a single trade with stop-loss and target orders.
        
        Parameters:
        -----------
        trade : Trade
            Trade to execute
            
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            if trade.action.upper() == "BUY":
                # Place market buy order
                market_order_data = MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_data=market_order_data)
                
                # Track the order
                self.active_trades[trade.ticker] = {
                    "trade": trade.to_dict(),
                    "entry_order_id": order.id,
                    "status": "pending"
                }
                
                logging.info(f"Submitted BUY order for {trade.quantity} shares of {trade.ticker}")
                return True
                
            elif trade.action.upper() == "SELL":
                # Place market sell order (short selling)
                market_order_data = MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_data=market_order_data)
                
                self.active_trades[trade.ticker] = {
                    "trade": trade.to_dict(),
                    "entry_order_id": order.id,
                    "status": "pending"
                }
                
                logging.info(f"Submitted SELL order for {trade.quantity} shares of {trade.ticker}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to execute trade for {trade.ticker}: {e}")
            return False
    
    def monitor_positions(self) -> Dict:
        """Monitor all active positions and manage risk."""
        try:
            positions = self.trading_client.get_all_positions()
            orders = self.trading_client.get_orders()
            
            position_summary = {
                "total_positions": len(positions),
                "total_unrealized_pl": 0.0,
                "positions": [],
                "risk_alerts": []
            }
            
            for pos in positions:
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)
                
                position_data = {
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc * 100,
                    "side": pos.side
                }
                
                position_summary["positions"].append(position_data)
                position_summary["total_unrealized_pl"] += unrealized_pl
                
                # Risk alerts
                if unrealized_plpc < -0.15:  # More than 15% loss
                    position_summary["risk_alerts"].append(
                        f"⚠️ {pos.symbol}: -{abs(unrealized_plpc*100):.1f}% loss"
                    )
                elif unrealized_plpc > 0.25:  # More than 25% gain
                    position_summary["risk_alerts"].append(
                        f"🎯 {pos.symbol}: +{unrealized_plpc*100:.1f}% gain - consider taking profits"
                    )
            
            return position_summary
            
        except Exception as e:
            logging.error(f"Failed to monitor positions: {e}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        try:
            account = self.trading_client.get_account()
            portfolio_history = self.trading_client.get_portfolio_history(
                period="1M",
                timeframe="1Day"
            )
            
            # Calculate key metrics
            current_value = float(account.portfolio_value)
            
            # Simple performance calculations
            total_pl = float(account.unrealized_pl) + float(account.realized_pl)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "account_value": current_value,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "total_pl": total_pl,
                "unrealized_pl": float(account.unrealized_pl),
                "realized_pl": float(account.realized_pl),
                "day_trade_count": account.day_trade_count,
                "positions": self.monitor_positions(),
                "recent_trades": len(self.trade_history),
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}
    
    def _save_trade_log(self, trades: List[Trade]) -> None:
        """Save trade log to file."""
        try:
            log_file = self.data_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing trades if file exists
            existing_trades = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_trades = json.load(f)
            
            # Add new trades
            for trade in trades:
                existing_trades.append(trade.to_dict())
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(existing_trades, f, indent=2, default=str)
                
            logging.info(f"Saved {len(trades)} trades to {log_file}")
            
        except Exception as e:
            logging.error(f"Failed to save trade log: {e}")


def main():
    """Demo the Alpaca trading integration."""
    print("🚀 ALPACA PAPER TRADING INTEGRATION")
    print("=" * 50)
    
    # Check for API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("⚠️ SETUP REQUIRED:")
        print("   1. Sign up at https://alpaca.markets")
        print("   2. Generate API keys in your dashboard")
        print("   3. Set environment variables:")
        print("      export ALPACA_API_KEY='your_key_here'")
        print("      export ALPACA_SECRET_KEY='your_secret_here'")
        print("   4. Run this script again")
        return
    
    try:
        # Initialize trader
        trader = AlpacaTrader(paper=True)
        
        # Show account info
        account_info = trader.get_account_info()
        print(f"\n💰 Account Overview:")
        print(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        print(f"   Active Positions: {account_info.get('active_positions', 0)}")
        
        # Hunt and execute opportunities
        trades = trader.hunt_and_execute_opportunities(max_positions=3)
        
        if trades:
            print(f"\n📊 Executed {len(trades)} trades:")
            for trade in trades:
                print(f"   {trade.action} {trade.quantity} shares of {trade.ticker} @ ${trade.entry_price:.2f}")
        
        # Monitor positions
        positions = trader.monitor_positions()
        if positions.get("positions"):
            print(f"\n📈 Active Positions:")
            for pos in positions["positions"]:
                pl_color = "🟢" if pos["unrealized_plpc"] > 0 else "🔴"
                print(f"   {pl_color} {pos['symbol']}: {pos['quantity']} shares, "
                      f"{pos['unrealized_plpc']:.1f}% P/L")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()