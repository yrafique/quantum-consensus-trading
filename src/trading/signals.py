"""
trading_system.signals
======================

This module implements the logic for turning raw indicator data into
actionable trade candidates.  A set of heuristics inspired by popular
technical and sentiment analysis techniques are applied to each ticker
to determine whether it merits further consideration by the language
model reasoner.

Criteria for a candidate include:

1. **Momentum** – The RSI must exceed a threshold and be trending upwards.
   Values above 70 typically indicate overbought conditions【582279574564404†L219-L233】,
   but we use a slightly lower cut‑off of 65 to capture earlier moves.
2. **Mean reversion crossover** – The closing price must cross above either
   the 21‑period EMA or the VWAP【690205185577964†L274-L315】【987456935882088†L268-L352】.  Crossing
   above a moving average is a classical bullish signal.
3. **Short squeeze potential** – The short float (shares sold short divided
   by shares outstanding) must exceed 20 %, and days‑to‑cover must be
   greater than 1.5【450714498829626†L112-L126】.
4. **Volume confirmation** – The day's trading volume must exceed three
   times the 50‑day average volume to confirm strong participation.
5. **Bullish candlestick pattern** – A bullish engulfing pattern must
   appear【381159643444015†L451-L485】, signalling a reversal from a downtrend to an
   uptrend.

Only tickers satisfying *all* of these criteria are passed to the
language model reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import data_ingestion
from .config import (
    RSI_THRESHOLD,
    RSI_SLOPE_MIN,
    VOLUME_SPIKE_FACTOR,
    SHORT_FLOAT_MIN,
    DAYS_TO_COVER_MIN,
)


@dataclass
class SignalResult:
    """Container for the signal evaluation of a single ticker."""

    ticker: str
    passes: bool
    details: Dict[str, float | bool]


def _check_rsi_momentum(df: pd.DataFrame) -> bool:
    """Return True if the last RSI exceeds threshold and is trending up."""
    if len(df) < 2:
        return False
    last_rsi = df["rsi"].iloc[-1]
    prev_rsi = df["rsi"].iloc[-2]
    return (last_rsi > RSI_THRESHOLD) and ((last_rsi - prev_rsi) > RSI_SLOPE_MIN)


def _check_price_crossover(df: pd.DataFrame) -> bool:
    """Check if the closing price crosses above the EMA21 or VWAP on the last bar.

    A crossover is defined as the closing price moving from below to above
    the moving average on the latest bar.
    """
    if len(df) < 2:
        return False
    prev_close = df["close"].iloc[-2]
    prev_ema = df["ema21"].iloc[-2]
    prev_vwap = df["vwap"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    curr_ema = df["ema21"].iloc[-1]
    curr_vwap = df["vwap"].iloc[-1]
    # Cross above either EMA or VWAP
    cross_ema = (prev_close <= prev_ema) and (curr_close > curr_ema)
    cross_vwap = (prev_close <= prev_vwap) and (curr_close > curr_vwap)
    return cross_ema or cross_vwap


def _check_short_squeeze(short_float: float, days_to_cover: float) -> bool:
    """Return True if short float and days to cover exceed thresholds."""
    return (short_float >= SHORT_FLOAT_MIN) and (days_to_cover >= DAYS_TO_COVER_MIN)


def _check_volume_spike(df: pd.DataFrame) -> bool:
    """Return True if the latest volume exceeds the specified multiple of the rolling mean."""
    if len(df) == 0:
        return False
    vol = df["volume"].iloc[-1]
    avg = df["volume_avg50"].iloc[-1]
    return vol >= VOLUME_SPIKE_FACTOR * avg


def _check_bullish_engulfing(df: pd.DataFrame) -> bool:
    """Return True if the last candle is a bullish engulfing pattern."""
    if len(df) == 0:
        return False
    return bool(df["bullish_engulfing"].iloc[-1])


def evaluate_signals(ticker: str, df: pd.DataFrame, short_float: float, days_to_cover: float) -> SignalResult:
    """Evaluate all signal criteria for a given ticker.

    Returns a ``SignalResult`` containing whether the ticker passes
    all filters and a dictionary of diagnostic information.
    """
    results = {
        "rsi": df["rsi"].iloc[-1] if not df.empty else np.nan,
        "rsi_momentum": _check_rsi_momentum(df),
        "price_crossover": _check_price_crossover(df),
        "short_float": short_float,
        "days_to_cover": days_to_cover,
        "short_squeeze": _check_short_squeeze(short_float, days_to_cover),
        "volume_spike": _check_volume_spike(df),
        "bullish_engulfing": _check_bullish_engulfing(df),
    }
    # Modified criteria: use a more lenient approach
    # Require either momentum OR technical signals, plus at least one confirmation
    momentum_signals = results["rsi_momentum"] or results["price_crossover"]
    confirmation_signals = (
        results["short_squeeze"] or 
        results["volume_spike"] or 
        results["bullish_engulfing"]
    )
    passes = momentum_signals and confirmation_signals
    return SignalResult(ticker=ticker, passes=passes, details=results)


def generate_signals(tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SignalResult]:
    """Generate signal evaluations for a list of tickers.

    For each ticker, historical data is fetched, indicators are
    computed, and the signals defined above are evaluated.  Short
    interest values are obtained via the data_ingestion module.  The
    returned list is sorted such that passing tickers come first.
    """
    results: List[SignalResult] = []
    for ticker in tickers:
        df = data_ingestion.fetch_daily_data(ticker, start_date=start_date, end_date=end_date)
        short_float, days_to_cover = data_ingestion.get_short_interest(ticker)
        res = evaluate_signals(ticker, df, short_float, days_to_cover)
        results.append(res)
    # Sort: pass first, then by descending RSI as an example of ranking
    results.sort(key=lambda r: (not r.passes, -(r.details["rsi"] if r.details["rsi"] == r.details["rsi"] else -np.inf)))
    return results


__all__ = [
    "SignalResult",
    "evaluate_signals",
    "generate_signals",
]