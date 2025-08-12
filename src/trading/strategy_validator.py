"""
Strategy Backtesting and Validation System
==========================================

This module provides comprehensive backtesting and validation of trading strategies
to ensure accuracy and reliability of recommendations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from .signals import generate_signals, evaluate_signals
    from ..ai.llm_reasoner import generate_recommendation
    from .position_sizer import compute_position_fraction
    from ..utils.data_ingestion import fetch_daily_data
except ImportError:
    print("Warning: Could not import some modules. Running in limited mode.")

class StrategyValidator:
    """Comprehensive strategy validation and backtesting system."""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        
    def backtest_signal_accuracy(self, ticker: str, lookback_days: int = 90) -> Dict:
        """Backtest signal accuracy over historical data."""
        print(f"🔍 Backtesting {ticker} signal accuracy over {lookback_days} days...")
        
        try:
            # Load fresh data
            with open(self.data_dir / f"{ticker}_daily.json", 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            if len(df) < lookback_days + 20:
                return {"error": "Insufficient data for backtesting"}
            
            results = {
                "ticker": ticker,
                "total_signals": 0,
                "correct_signals": 0,
                "false_positives": 0,
                "missed_opportunities": 0,
                "average_return": 0.0,
                "win_rate": 0.0,
                "best_return": 0.0,
                "worst_return": 0.0,
                "sharpe_ratio": 0.0,
                "signals_details": []
            }
            
            returns = []
            
            # Walk through historical data
            for i in range(len(df) - lookback_days, len(df) - 5):  # Leave 5 days for validation
                # Get data up to point i
                historical_df = df.iloc[:i+1].copy()
                
                # Generate signal at this point
                try:
                    # Mock short interest data for backtesting
                    short_float = np.random.uniform(0.1, 0.4)
                    days_to_cover = np.random.uniform(0.5, 3.0)
                    
                    signal = evaluate_signals(ticker, historical_df, short_float, days_to_cover)
                    
                    if signal.passes:
                        results["total_signals"] += 1
                        
                        # Look ahead to see if signal was correct (next 5 days)
                        entry_price = historical_df['close'].iloc[-1]
                        future_prices = df['close'].iloc[i+1:i+6]
                        
                        if len(future_prices) >= 3:
                            max_future_price = future_prices.max()
                            min_future_price = future_prices.min()
                            final_price = future_prices.iloc[-1]
                            
                            # Calculate returns
                            max_return = (max_future_price - entry_price) / entry_price * 100
                            final_return = (final_price - entry_price) / entry_price * 100
                            
                            returns.append(final_return)
                            
                            # Determine if signal was correct (positive return within 5 days)
                            if max_return > 2.0:  # At least 2% gain achieved
                                results["correct_signals"] += 1
                            else:
                                results["false_positives"] += 1
                            
                            signal_detail = {
                                "date": historical_df['date'].iloc[-1].strftime('%Y-%m-%d'),
                                "entry_price": entry_price,
                                "max_return": max_return,
                                "final_return": final_return,
                                "rsi": signal.details.get('rsi', 0),
                                "correct": max_return > 2.0
                            }
                            results["signals_details"].append(signal_detail)
                
                except Exception as e:
                    continue
            
            # Calculate summary statistics
            if results["total_signals"] > 0:
                results["win_rate"] = results["correct_signals"] / results["total_signals"] * 100
                
            if returns:
                results["average_return"] = np.mean(returns)
                results["best_return"] = np.max(returns)
                results["worst_return"] = np.min(returns)
                
                # Calculate Sharpe ratio (assuming 3% risk-free rate)
                if np.std(returns) > 0:
                    results["sharpe_ratio"] = (np.mean(returns) - 0.25) / np.std(returns)  # 0.25 = 3%/12 monthly
            
            return results
            
        except Exception as e:
            return {"error": f"Backtesting failed: {str(e)}"}
    
    def validate_llm_recommendations(self, ticker: str, num_tests: int = 20) -> Dict:
        """Validate LLM recommendation accuracy."""
        print(f"🤖 Validating LLM recommendations for {ticker}...")
        
        try:
            # Load data
            with open(self.data_dir / f"{ticker}_daily.json", 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            results = {
                "ticker": ticker,
                "total_recommendations": 0,
                "buy_recommendations": 0,
                "sell_recommendations": 0,
                "successful_trades": 0,
                "average_accuracy": 0.0,
                "confidence_correlation": 0.0,
                "recommendations": []
            }
            
            confidences = []
            accuracies = []
            
            # Test LLM on historical data points
            test_points = np.linspace(50, len(df)-10, min(num_tests, len(df)-60)).astype(int)
            
            for i in test_points:
                try:
                    # Create context from historical data
                    hist_data = df.iloc[:i+1]
                    latest = hist_data.iloc[-1]
                    
                    context = {
                        "close": float(latest['close']),
                        "rsi": float(latest['rsi']) if not pd.isna(latest['rsi']) else 50.0,
                        "ema21": float(latest['ema21']),
                        "vwap": float(latest['vwap']),
                        "volume": float(latest['volume']),
                        "short_float": np.random.uniform(0.1, 0.4),
                        "days_to_cover": np.random.uniform(0.5, 3.0),
                        "bullish_engulfing": bool(latest.get('bullish_engulfing', False)),
                        "volume_spike": latest['volume'] > hist_data['volume'].rolling(50).mean().iloc[-1] * 2,
                        "short_squeeze": True
                    }
                    
                    # Get LLM recommendation
                    rec = generate_recommendation(ticker, context)
                    
                    if rec and rec.get('action') in ['Buy', 'Sell']:
                        results["total_recommendations"] += 1
                        
                        if rec['action'] == 'Buy':
                            results["buy_recommendations"] += 1
                        else:
                            results["sell_recommendations"] += 1
                        
                        # Validate recommendation by looking ahead
                        entry_price = context['close']
                        target_price = rec.get('target', entry_price * 1.05)
                        stop_price = rec.get('stop', entry_price * 0.95)
                        confidence = rec.get('confidence', 0.5)
                        
                        # Look ahead 5-10 days
                        future_prices = df['close'].iloc[i+1:i+11]
                        
                        if len(future_prices) >= 5:
                            if rec['action'] == 'Buy':
                                # Check if target was hit before stop
                                hit_target = any(future_prices >= target_price)
                                hit_stop = any(future_prices <= stop_price)
                                
                                if hit_target and not hit_stop:
                                    success = True
                                elif hit_stop and not hit_target:
                                    success = False
                                else:
                                    # Neither clearly hit, use final price
                                    final_return = (future_prices.iloc[-1] - entry_price) / entry_price
                                    success = final_return > 0.02  # 2% gain
                            else:  # Sell
                                final_return = (entry_price - future_prices.iloc[-1]) / entry_price
                                success = final_return > 0.02
                            
                            if success:
                                results["successful_trades"] += 1
                            
                            # Track confidence vs accuracy
                            confidences.append(confidence)
                            accuracies.append(1.0 if success else 0.0)
                            
                            rec_detail = {
                                "date": hist_data['date'].iloc[-1].strftime('%Y-%m-%d'),
                                "action": rec['action'],
                                "confidence": confidence,
                                "entry": entry_price,
                                "target": target_price,
                                "success": success,
                                "reasoning_length": len(rec.get('reasoning', ''))
                            }
                            results["recommendations"].append(rec_detail)
                
                except Exception as e:
                    continue
            
            # Calculate summary metrics
            if results["total_recommendations"] > 0:
                results["average_accuracy"] = results["successful_trades"] / results["total_recommendations"] * 100
            
            if len(confidences) > 1:
                # Correlation between confidence and accuracy
                results["confidence_correlation"] = np.corrcoef(confidences, accuracies)[0, 1]
            
            return results
            
        except Exception as e:
            return {"error": f"LLM validation failed: {str(e)}"}
    
    def validate_position_sizing(self, ticker: str) -> Dict:
        """Validate position sizing algorithm."""
        print(f"💰 Validating position sizing for {ticker}...")
        
        results = {
            "ticker": ticker,
            "total_tests": 0,
            "kelly_effectiveness": 0.0,
            "risk_adjusted_returns": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "position_sizes": []
        }
        
        try:
            # Test various confidence and risk/reward scenarios
            test_scenarios = [
                (0.6, 1.5), (0.7, 2.0), (0.8, 2.5), (0.9, 3.0),
                (0.5, 1.2), (0.85, 1.8), (0.95, 4.0), (0.75, 2.2)
            ]
            
            portfolio_values = [100000]  # Start with $100k
            
            for confidence, rr_ratio in test_scenarios:
                try:
                    fraction = compute_position_fraction(confidence, rr_ratio)
                    position_size = portfolio_values[-1] * fraction
                    
                    # Simulate trade outcome based on confidence
                    success_prob = confidence
                    if np.random.random() < success_prob:
                        # Winning trade
                        gain = position_size * (rr_ratio * 0.1)  # 10% of risk-reward
                        portfolio_values.append(portfolio_values[-1] + gain)
                    else:
                        # Losing trade
                        loss = position_size * 0.1  # 10% loss
                        portfolio_values.append(portfolio_values[-1] - loss)
                    
                    results["position_sizes"].append({
                        "confidence": confidence,
                        "rr_ratio": rr_ratio,
                        "fraction": fraction,
                        "position_size": position_size,
                        "portfolio_value": portfolio_values[-1]
                    })
                    
                except Exception:
                    continue
            
            # Calculate metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
            
            results["total_tests"] = len(test_scenarios)
            results["final_portfolio_value"] = portfolio_values[-1]
            results["total_return"] = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            
            if len(returns) > 0:
                results["average_return"] = np.mean(returns)
                results["volatility"] = np.std(returns)
                results["sharpe_ratio"] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
                # Calculate max drawdown
                peak = portfolio_values[0]
                max_dd = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak * 100
                    max_dd = max(max_dd, drawdown)
                
                results["max_drawdown"] = max_dd
            
            return results
            
        except Exception as e:
            return {"error": f"Position sizing validation failed: {str(e)}"}
    
    def comprehensive_validation(self, watchlist: List[str]) -> Dict:
        """Run comprehensive validation on all strategies."""
        print("🚀 COMPREHENSIVE STRATEGY VALIDATION")
        print("=" * 50)
        
        overall_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "tickers_tested": len(watchlist),
            "signal_accuracy": {},
            "llm_performance": {},
            "position_sizing": {},
            "summary": {}
        }
        
        # Test each ticker
        for ticker in watchlist:
            print(f"\n📊 Testing {ticker}...")
            
            # Signal accuracy
            signal_results = self.backtest_signal_accuracy(ticker, lookback_days=60)
            overall_results["signal_accuracy"][ticker] = signal_results
            
            # LLM performance
            llm_results = self.validate_llm_recommendations(ticker, num_tests=15)
            overall_results["llm_performance"][ticker] = llm_results
            
            # Position sizing
            sizing_results = self.validate_position_sizing(ticker)
            overall_results["position_sizing"][ticker] = sizing_results
        
        # Calculate overall summary
        self.calculate_summary_metrics(overall_results)
        
        return overall_results
    
    def calculate_summary_metrics(self, results: Dict):
        """Calculate overall summary metrics."""
        signal_accuracies = []
        llm_accuracies = []
        win_rates = []
        
        for ticker in results["signal_accuracy"]:
            signal_data = results["signal_accuracy"][ticker]
            if "win_rate" in signal_data:
                win_rates.append(signal_data["win_rate"])
            
            llm_data = results["llm_performance"][ticker]
            if "average_accuracy" in llm_data:
                llm_accuracies.append(llm_data["average_accuracy"])
        
        results["summary"] = {
            "average_signal_win_rate": np.mean(win_rates) if win_rates else 0,
            "average_llm_accuracy": np.mean(llm_accuracies) if llm_accuracies else 0,
            "total_signals_tested": sum(r.get("total_signals", 0) for r in results["signal_accuracy"].values()),
            "total_recommendations_tested": sum(r.get("total_recommendations", 0) for r in results["llm_performance"].values()),
            "system_reliability_score": (np.mean(win_rates + llm_accuracies)) if (win_rates or llm_accuracies) else 0
        }
    
    def print_validation_report(self, results: Dict):
        """Print a comprehensive validation report."""
        print("\n" + "="*60)
        print("📊 STRATEGY VALIDATION REPORT")
        print("="*60)
        
        summary = results.get("summary", {})
        
        print(f"\n🎯 OVERALL SYSTEM PERFORMANCE:")
        print(f"   • System Reliability Score: {summary.get('system_reliability_score', 0):.1f}%")
        print(f"   • Average Signal Win Rate: {summary.get('average_signal_win_rate', 0):.1f}%")
        print(f"   • Average LLM Accuracy: {summary.get('average_llm_accuracy', 0):.1f}%")
        print(f"   • Total Tests Performed: {summary.get('total_signals_tested', 0) + summary.get('total_recommendations_tested', 0)}")
        
        print(f"\n📈 DETAILED TICKER ANALYSIS:")
        
        for ticker in results["signal_accuracy"]:
            print(f"\n   {ticker}:")
            
            signal_data = results["signal_accuracy"][ticker]
            if "error" not in signal_data:
                print(f"      Signal Win Rate: {signal_data.get('win_rate', 0):.1f}%")
                print(f"      Average Return: {signal_data.get('average_return', 0):.2f}%")
                print(f"      Sharpe Ratio: {signal_data.get('sharpe_ratio', 0):.2f}")
            
            llm_data = results["llm_performance"][ticker]
            if "error" not in llm_data:
                print(f"      LLM Accuracy: {llm_data.get('average_accuracy', 0):.1f}%")
                print(f"      Confidence Correlation: {llm_data.get('confidence_correlation', 0):.2f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        avg_score = summary.get('system_reliability_score', 0)
        
        if avg_score >= 70:
            print("   ✅ System shows strong reliability - Ready for live trading")
        elif avg_score >= 50:
            print("   ⚠️  System shows moderate reliability - Consider parameter tuning")
        else:
            print("   ❌ System needs improvement before live deployment")
        
        print(f"\n📝 Validation completed at: {results.get('validation_timestamp', 'Unknown')}")

def main():
    """Run strategy validation as standalone script."""
    validator = StrategyValidator()
    
    # Default watchlist
    watchlist = ["AAPL", "MSFT", "NVDA"]  # Test with 3 tickers for speed
    
    print("🚀 Starting comprehensive strategy validation...")
    results = validator.comprehensive_validation(watchlist)
    
    # Print detailed report
    validator.print_validation_report(results)
    
    # Save results
    output_file = Path(__file__).parent / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()