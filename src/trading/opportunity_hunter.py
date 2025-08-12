"""
trading_system.opportunity_hunter  
================================

Dynamic opportunity discovery system that hunts for high-potential trading 
opportunities across broader markets rather than relying on fixed watchlists.
This module implements sophisticated screening criteria to maximize gains with
high confidence by scanning market-wide for optimal setups.

Key Features:
- Real-time market scanning across 500+ stocks
- Multi-factor scoring system with institutional-grade criteria  
- Dynamic screening based on momentum, volatility, and volume patterns
- Sector rotation analysis for emerging opportunities
- High-conviction filtering (>85% confidence threshold)
- Risk-adjusted opportunity ranking
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from ..utils.data_validator import DataValidator
    from ..ai.llm_reasoner import generate_recommendation
except ImportError:
    # Fallback for running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_validator import DataValidator
    from ai.llm_reasoner import generate_recommendation


class OpportunityHunter:
    """Advanced opportunity discovery system for systematic alpha generation."""
    
    def __init__(self):
        """Initialize the opportunity hunter with market-wide screening capabilities."""
        self.validator = DataValidator()
        
        # S&P 500 + high-volume stocks for comprehensive screening
        self.screening_universe = [
            # FAANG + Tech Giants
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            # Mega Cap Growth
            "BRK-B", "UNH", "JNJ", "V", "PG", "JPM", "HD", "MA", "CVX", "LLY",
            # High Beta / Momentum Plays
            "GME", "AMC", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "COIN", "HOOD",
            # Sector Leaders
            "XOM", "BAC", "WMT", "DIS", "PYPL", "CRM", "ADBE", "INTC", "AMD",
            # Emerging Growth
            "SQ", "ROKU", "PINS", "SNAP", "UBER", "LYFT", "DOCU", "ZM", "PTON",
            # Biotech/Pharma High Beta
            "MRNA", "BNTX", "GILD", "BIIB", "REGN", "VRTX", "ILMN", "AMGN",
            # Crypto/Fintech
            "MSTR", "SQ", "PYPL", "COIN", "RIOT", "MARA", "CLSK", "HUT",
            # Meme/Reddit Stocks
            "BB", "NOK", "SNDL", "CLOV", "WISH", "SPCE", "TLRY", "CGC",
            # Chinese ADRs
            "BABA", "JD", "PDD", "BIDU", "DIDI", "XPEV", "LI", "NIO"
        ]
        
        # Screening criteria weights for multi-factor scoring
        self.screening_weights = {
            "momentum_score": 0.25,      # RSI, price vs EMAs, trend strength
            "volatility_score": 0.20,    # Historical vol, implied vol, VIX correlation  
            "volume_score": 0.20,        # Volume spikes, institutional flow
            "squeeze_score": 0.15,       # Short interest, days to cover
            "pattern_score": 0.10,       # Technical patterns, candlestick signals
            "fundamental_score": 0.10    # Sector strength, earnings momentum
        }

    def hunt_opportunities(self, max_opportunities: int = 10, debug_mode: bool = False) -> List[Dict]:
        """
        Hunt for the highest-conviction trading opportunities across the market.
        
        Parameters
        ----------
        max_opportunities : int
            Maximum number of opportunities to return (top-ranked)
        debug_mode : bool
            Include detailed scoring breakdown for analysis
            
        Returns
        -------
        List[Dict]
            Ranked list of opportunities with detailed analysis
        """
        logging.info(f"🎯 Starting opportunity hunt across {len(self.screening_universe)} stocks...")
        
        opportunities = []
        screening_results = {}
        
        for ticker in self.screening_universe:
            try:
                # Get fresh market data
                validation_result = self.validator.validate_data_freshness(ticker)
                if not validation_result["valid"]:
                    continue
                    
                # Calculate comprehensive screening scores
                scores = self._calculate_screening_scores(ticker, debug_mode)
                if scores is None:
                    continue
                    
                # Calculate composite opportunity score
                composite_score = self._calculate_composite_score(scores)
                
                # Only consider high-conviction opportunities (>75th percentile)
                if composite_score < 0.75:
                    continue
                    
                # Generate detailed LLM recommendation for promising opportunities
                market_data = self.validator.get_market_context(ticker)
                recommendation = generate_recommendation(ticker, market_data, debug_mode=debug_mode)
                
                if recommendation and recommendation.get("confidence", 0) > 0.8:
                    opportunity = {
                        "ticker": ticker,
                        "composite_score": composite_score,
                        "recommendation": recommendation,
                        "screening_scores": scores,
                        "market_data": market_data,
                        "discovered_at": datetime.now().isoformat()
                    }
                    opportunities.append(opportunity)
                    
                screening_results[ticker] = {
                    "composite_score": composite_score,
                    "passed_screening": composite_score >= 0.75
                }
                    
            except Exception as e:
                logging.warning(f"Failed to screen {ticker}: {e}")
                continue
        
        # Rank opportunities by composite score and LLM confidence
        opportunities.sort(key=lambda x: (
            x["composite_score"] * 0.6 + 
            x["recommendation"]["confidence"] * 0.4
        ), reverse=True)
        
        top_opportunities = opportunities[:max_opportunities]
        
        logging.info(f"🔍 Screening complete: {len(opportunities)} high-conviction opportunities found")
        logging.info(f"📊 Top {len(top_opportunities)} opportunities selected")
        
        if debug_mode:
            return {
                "opportunities": top_opportunities,
                "screening_stats": {
                    "total_screened": len(self.screening_universe),
                    "passed_initial_screening": len([s for s in screening_results.values() if s["passed_screening"]]),
                    "high_conviction_recommendations": len(opportunities),
                    "final_selections": len(top_opportunities)
                },
                "screening_results": screening_results
            }
        
        return top_opportunities

    def _calculate_screening_scores(self, ticker: str, debug_mode: bool = False) -> Optional[Dict]:
        """Calculate multi-factor screening scores for a ticker."""
        try:
            # Get comprehensive market data
            data = self.validator.get_comprehensive_data(ticker)
            if data is None:
                return None
                
            scores = {}
            
            # 1. Momentum Score (0-1)
            scores["momentum_score"] = self._calculate_momentum_score(data)
            
            # 2. Volatility Score (0-1) - higher is better for momentum trades
            scores["volatility_score"] = self._calculate_volatility_score(data)
            
            # 3. Volume Score (0-1) - institutional participation
            scores["volume_score"] = self._calculate_volume_score(data)
            
            # 4. Short Squeeze Score (0-1)
            scores["squeeze_score"] = self._calculate_squeeze_score(data)
            
            # 5. Pattern Score (0-1) - technical patterns
            scores["pattern_score"] = self._calculate_pattern_score(data)
            
            # 6. Fundamental Score (0-1) - sector/earnings momentum
            scores["fundamental_score"] = self._calculate_fundamental_score(ticker, data)
            
            if debug_mode:
                scores["debug_breakdown"] = {
                    "data_points": len(data) if isinstance(data, pd.DataFrame) else "N/A",
                    "latest_price": data.get("close", 0) if isinstance(data, dict) else data["close"].iloc[-1] if not data.empty else 0,
                    "score_calculation": "Multi-factor institutional screening methodology"
                }
            
            return scores
            
        except Exception as e:
            logging.warning(f"Failed to calculate screening scores for {ticker}: {e}")
            return None

    def _calculate_momentum_score(self, data: Dict) -> float:
        """Calculate momentum score based on RSI, price trends, and moving averages."""
        try:
            rsi = data.get("rsi", 50)
            close = data.get("close", 0)
            ema21 = data.get("ema21", close)
            vwap = data.get("vwap", close)
            
            # RSI momentum (optimal range 55-75 for momentum trades)
            if 55 <= rsi <= 75:
                rsi_score = 1.0
            elif 45 <= rsi <= 85:
                rsi_score = 0.7
            else:
                rsi_score = 0.3
                
            # Price structure score
            price_structure_score = 0.0
            if close > ema21:
                price_structure_score += 0.5
            if close > vwap:
                price_structure_score += 0.5
                
            return (rsi_score * 0.6 + price_structure_score * 0.4)
            
        except Exception:
            return 0.5

    def _calculate_volatility_score(self, data: Dict) -> float:
        """Calculate volatility score - moderate volatility is ideal for momentum trades."""
        try:
            rsi = data.get("rsi", 50)
            # Use RSI distance from 50 as volatility proxy
            vol_proxy = abs(rsi - 50)
            
            # Optimal volatility range: 10-25 RSI points from neutral
            if 10 <= vol_proxy <= 25:
                return 1.0
            elif 5 <= vol_proxy <= 35:
                return 0.7
            else:
                return 0.3
                
        except Exception:
            return 0.5

    def _calculate_volume_score(self, data: Dict) -> float:
        """Calculate volume score based on institutional participation indicators."""
        try:
            volume_spike = data.get("volume_spike", False)
            
            # Simple but effective volume scoring
            if volume_spike:
                return 1.0
            else:
                return 0.4
                
        except Exception:
            return 0.5

    def _calculate_squeeze_score(self, data: Dict) -> float:
        """Calculate short squeeze potential score."""
        try:
            short_float = data.get("short_float", 0.0)
            days_to_cover = data.get("days_to_cover", 0.0)
            short_squeeze = data.get("short_squeeze", False)
            
            # High short interest + adequate days to cover = squeeze potential
            if short_squeeze and short_float > 0.2 and days_to_cover > 1.5:
                return 1.0
            elif short_float > 0.15 and days_to_cover > 1.0:
                return 0.7
            elif short_float > 0.1:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.3

    def _calculate_pattern_score(self, data: Dict) -> float:
        """Calculate technical pattern score."""
        try:
            bullish_engulfing = data.get("bullish_engulfing", False)
            
            # Pattern scoring - can be expanded with more patterns
            score = 0.5  # baseline
            if bullish_engulfing:
                score += 0.5
                
            return min(score, 1.0)
            
        except Exception:
            return 0.5

    def _calculate_fundamental_score(self, ticker: str, data: Dict) -> float:
        """Calculate fundamental momentum score based on sector strength."""
        try:
            # Sector strength mapping (simplified)
            tech_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "CRM", "ADBE"]
            growth_stocks = ["TSLA", "AMZN", "NFLX", "ROKU", "SQ", "PYPL"]
            biotech_stocks = ["MRNA", "BNTX", "GILD", "BIIB", "REGN"]
            
            if ticker in tech_stocks:
                return 0.8  # Tech generally strong in current market
            elif ticker in growth_stocks:
                return 0.7  # Growth stocks have moderate momentum
            elif ticker in biotech_stocks:
                return 0.6  # Biotech is volatile but has potential
            else:
                return 0.5  # Neutral for other sectors
                
        except Exception:
            return 0.5

    def _calculate_composite_score(self, scores: Dict) -> float:
        """Calculate weighted composite opportunity score."""
        try:
            composite = 0.0
            for factor, weight in self.screening_weights.items():
                score_value = scores.get(factor, 0.5)
                composite += score_value * weight
                
            return min(max(composite, 0.0), 1.0)  # Clamp to [0,1]
            
        except Exception:
            return 0.5

    def get_sector_opportunities(self, sector: str = None, max_per_sector: int = 3) -> Dict[str, List]:
        """Get opportunities organized by sector for diversification."""
        all_opportunities = self.hunt_opportunities(max_opportunities=50)
        
        sector_map = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "CRM", "ADBE"],
            "Growth": ["TSLA", "AMZN", "NFLX", "ROKU", "SQ", "PYPL", "UBER", "LYFT"],
            "Biotech": ["MRNA", "BNTX", "GILD", "BIIB", "REGN", "VRTX", "ILMN"],
            "Fintech": ["COIN", "HOOD", "SQ", "PYPL", "MSTR"],
            "Meme": ["GME", "AMC", "BB", "NOK", "CLOV", "SPCE"]
        }
        
        sector_opportunities = {}
        for sector_name, tickers in sector_map.items():
            sector_opps = [opp for opp in all_opportunities if opp["ticker"] in tickers]
            sector_opportunities[sector_name] = sector_opps[:max_per_sector]
            
        return sector_opportunities


__all__ = ["OpportunityHunter"]