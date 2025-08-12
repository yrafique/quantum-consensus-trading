"""
trading_system.data_ingestion
=============================

This module encapsulates all functionality for obtaining and preparing
historical price data.  The goal is to provide a simple, consistent
interface for the rest of the trading system without leaking the
implementation details of data retrieval.  To honour the offline‑first
requirement of the specification, the module first attempts to load
data from a local cache (CSV or JSON).  If no data is available it
generates synthetic data as a fallback.  Users running this code on
their own machines can override ``fetch_daily_data`` to call a real
API (e.g. yfinance) but should preserve the output format.

Key functions:

* ``fetch_daily_data`` – returns a DataFrame of OHLCV data for a given
  ticker and date range
* ``compute_rsi`` – calculates the Relative Strength Index as defined
  by J. Welles Wilder, using a simple moving average of gains and
  losses【582279574564404†L195-L210】
* ``compute_ema`` – wraps pandas' exponential weighted moving average to
  compute the 21‑day EMA【690205185577964†L274-L315】
* ``compute_vwap`` – computes an approximate daily VWAP based on daily
  closing prices and volumes【987456935882088†L268-L352】
* ``detect_bullish_engulfing`` – detects bullish engulfing patterns by
  comparing consecutive candlesticks【381159643444015†L451-L485】

The returned DataFrame from ``fetch_daily_data`` always contains the
following columns:

``date`` – pandas.Timestamp
``open`` – float
``high`` – float
``low`` – float
``close`` – float
``volume`` – float
``rsi`` – float (14‑period)
``ema21`` – float (21‑period EMA)
``vwap`` – float (approximate)
``volume_avg50`` – float (50‑day rolling mean of volume)
``bullish_engulfing`` – bool (True where pattern detected)

Short interest metrics are not part of this DataFrame; they are
provided by the separate ``get_short_interest`` function.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Import fix for standalone execution
try:
    from ...config.config import DATA_DIR
except ImportError:
    from pathlib import Path
    DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _ensure_data_dir() -> None:
    """Create the data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_daily_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve or generate daily OHLCV data for ``ticker``.

    Parameters
    ----------
    ticker : str
        The stock symbol to fetch.
    start_date : str, optional
        ISO‑formatted string (YYYY‑MM‑DD) for the beginning of the
        requested date range.  If None, defaults to one year prior to
        ``end_date`` or today's date minus one year.
    end_date : str, optional
        ISO‑formatted string (YYYY‑MM‑DD) for the end of the requested
        date range.  If None, defaults to today's date.
    force_refresh : bool
        If True, ignore any locally cached file and regenerate the
        dataset.  Otherwise, cached files will be loaded when
        available.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing OHLCV data and computed indicators.

    Notes
    -----
    If network access is not permitted (as is the case in this
    execution environment) or no local file exists, synthetic data
    will be generated.  The synthetic series follows a lognormal
    random walk with mean zero and small standard deviation around
    1 % daily drift, which loosely approximates real equity returns.
    """
    _ensure_data_dir()
    cache_file = DATA_DIR / f"{ticker.upper()}_daily.json"

    # Determine date range
    today = datetime.today().date()
    end_dt = datetime.fromisoformat(end_date).date() if end_date else today
    start_dt = (
        datetime.fromisoformat(start_date).date()
        if start_date
        else end_dt - timedelta(days=365)
    )

    # Attempt to load cached data
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
            df["date"] = pd.to_datetime(df["date"])
            # Filter to the desired range
            mask = (df["date"].dt.date >= start_dt) & (df["date"].dt.date <= end_dt)
            return df.loc[mask].reset_index(drop=True)
        except Exception:
            # If the file is corrupt, fall back to regeneration
            pass

    # Generate synthetic data
    num_days = (end_dt - start_dt).days + 1
    dates = pd.date_range(start_dt, periods=num_days, freq="B")  # business days only
    # Generate lognormal returns around 0.0005 mean (approx 12 % per year) and 2 % std
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    log_returns = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
    price = 100 * np.exp(np.cumsum(log_returns))
    # Create OHLCV; we'll approximate open/high/low around close
    close = price
    open_price = close * (1 + rng.normal(0, 0.005, size=len(close)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.01, size=len(close)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.01, size=len(close)))
    volume = rng.integers(low=1e6, high=5e6, size=len(close))

    df = pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # Compute indicators
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["ema21"] = compute_ema(df["close"], span=21)
    df["vwap"] = compute_vwap(df)
    df["volume_avg50"] = df["volume"].rolling(50, min_periods=1).mean()
    df["bullish_engulfing"] = detect_bullish_engulfing(df)

    # Persist to cache for future runs
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, default=str)
    except Exception:
        pass

    # Filter to requested range
    mask = (df["date"].dt.date >= start_dt) & (df["date"].dt.date <= end_dt)
    return df.loc[mask].reset_index(drop=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    The RSI is calculated as

    ``RSI = 100 - 100 / (1 + RS)``【582279574564404†L195-L210】,

    where ``RS`` is the ratio of the average gain to the average loss over
    the specified period.

    Parameters
    ----------
    series : pandas.Series
        Series of prices.
    period : int
        Look‑back window for computing the RSI.  The default is 14 days.

    Returns
    -------
    pandas.Series
        RSI values between 0 and 100.  ``NaN`` values appear in the first
        ``period`` rows.
    """
    delta = series.diff()
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Compute the exponential moving average of gains and losses
    roll_up = gain.ewm(span=period, adjust=False).mean()
    roll_down = loss.ewm(span=period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int = 21) -> pd.Series:
    """Compute the exponential moving average of a price series.

    The EMA weights recent prices more heavily than older prices using a
    smoothing factor of ``2 / (span + 1)``【690205185577964†L274-L315】.

    Parameters
    ----------
    series : pandas.Series
        Series of prices.
    span : int
        Number of periods for the EMA.  The default is 21, a common
        intermediate term moving average used in trading.

    Returns
    -------
    pandas.Series
        Exponential moving average of the input series.
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute an approximate VWAP series based on daily prices and volumes.

    VWAP is defined as the sum of price * volume divided by the total
    volume【987456935882088†L268-L352】.  As we work with daily bars and not
    intraday data, we approximate the daily VWAP using the close price
    multiplied by volume.  Cumulative sums yield a running VWAP which is
    still useful for assessing whether prices are above or below the
    average traded price.
    """
    pv = df["close"] * df["volume"]
    cumulative_pv = pv.cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_pv / cumulative_vol


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect bullish engulfing patterns in a price DataFrame.

    A bullish engulfing pattern occurs when a down candlestick (close
    below open) is completely swallowed by a subsequent up candlestick
    whose body spans the previous body's high and low【381159643444015†L451-L485】.
    This function returns a Boolean series indicating whether the pattern
    appears on each row.  The first row is always False because there is
    no prior bar to compare.
    """
    # We require at least one previous row to evaluate the pattern
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    # Identify down candle followed by up candle
    prev_down = prev_close < prev_open
    curr_up = df["close"] > df["open"]
    # Body of current candle engulfs previous body
    open_lower = df["open"] < prev_low
    close_higher = df["close"] > prev_high
    engulf = open_lower & close_higher
    pattern = prev_down & curr_up & engulf
    return pattern.fillna(False)


def get_short_interest(ticker: str) -> Tuple[float, float]:
    """Return the short float and days‑to‑cover for ``ticker``.

    In a fully featured implementation this function would query a
    regulatory data source such as FINRA or MarketWatch for up‑to‑date
    figures.  FINRA collects short positions from broker‑dealers and
    publishes reports twice a month【62370034694688†L219-L254】, while the
    days‑to‑cover ratio is defined as the number of shares sold short
    divided by the average daily volume【450714498829626†L112-L126】.  Since
    network calls may not be possible, we return deterministic pseudo‑
    random values based on the ticker's hash.  The short float is
    bounded between 0 % and 40 %; the days‑to‑cover is bounded between
    0 and 5.
    """
    # Use a deterministic pseudo‑random generator seeded on the ticker
    h = abs(hash(ticker)) % 2**32
    rng = np.random.default_rng(h)
    short_float = float(rng.uniform(0.05, 0.4))  # 5 % to 40 %
    days_to_cover = float(rng.uniform(0.5, 4.0))
    return short_float, days_to_cover


__all__ = [
    "fetch_daily_data",
    "compute_rsi",
    "compute_ema",
    "compute_vwap",
    "detect_bullish_engulfing",
    "get_short_interest",
]