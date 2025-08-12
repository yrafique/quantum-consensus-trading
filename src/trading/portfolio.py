"""
trading_system.portfolio
=======================

This module defines a simple portfolio class to manage open positions
and track trading history.  All state is stored in two JSON files:

* ``portfolio.json`` – contains current capital and positions
* ``history.json`` – contains a chronological list of executed trades

Both files reside in the ``DATA_DIR`` specified in ``config.py``.

The ``Portfolio`` class provides methods to open and close trades,
update position sizes based on the Kelly criterion and maintain a
running P&L.  It does not connect to any broker; it merely
recommends sizes and logs actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import DATA_DIR, PORTFOLIO_FILE, HISTORY_FILE


def _load_json(path: Path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


@dataclass
class Trade:
    """Record of a single trade recommendation or execution."""

    ticker: str
    action: str  # Buy/Sell/Ignore
    size: float  # capital allocated
    entry: float
    stop: float
    target: float
    confidence: float
    reasoning: str
    timestamp: str

    def to_dict(self):
        return asdict(self)


class Portfolio:
    """Maintain portfolio state and trading history."""

    def __init__(self, initial_capital: float = 100_000.0):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.portfolio_path = PORTFOLIO_FILE
        self.history_path = HISTORY_FILE
        # Load or initialise portfolio
        data = _load_json(self.portfolio_path, default={})
        self.capital = float(data.get("capital", initial_capital))
        self.positions: Dict[str, Dict[str, float]] = data.get("positions", {})
        # Load history list
        self.history: list = _load_json(self.history_path, default=[])

    def save(self):
        data = {
            "capital": self.capital,
            "positions": self.positions,
        }
        _save_json(self.portfolio_path, data)
        _save_json(self.history_path, self.history)

    def open_position(self, trade: Trade) -> None:
        """Record an opening trade and reserve capital."""
        if trade.action == "Ignore":
            return
        ticker = trade.ticker
        # Deduct capital
        allocated = trade.size
        if allocated > self.capital:
            # not enough capital; skip
            return
        self.capital -= allocated
        # Add to positions dict
        self.positions[ticker] = {
            "size": allocated,
            "entry": trade.entry,
            "stop": trade.stop,
            "target": trade.target,
            "confidence": trade.confidence,
            "timestamp": trade.timestamp,
        }
        # Append to history
        self.history.append(trade.to_dict())
        self.save()

    def close_position(self, ticker: str, exit_price: float, timestamp: Optional[str] = None) -> None:
        """Close an open position and update capital based on P&L."""
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        entry = pos["entry"]
        size = pos["size"]
        # Determine profit or loss proportionally to price change
        profit_fraction = (exit_price - entry) / entry
        profit = size * profit_fraction
        self.capital += size + profit  # return principal + profit
        # Log closure
        trade_record = {
            "ticker": ticker,
            "action": "Close",
            "size": size,
            "entry": entry,
            "exit": exit_price,
            "pnl": profit,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        self.history.append(trade_record)
        self.save()

    def get_exposure(self) -> float:
        """Return the total capital currently tied up in open positions."""
        return sum(pos.get("size", 0.0) for pos in self.positions.values())

    def get_free_capital(self) -> float:
        """Return the capital available to allocate to new trades."""
        return self.capital

    def __repr__(self) -> str:
        return f"Portfolio(capital={self.capital:.2f}, positions={list(self.positions.keys())})"


__all__ = ["Trade", "Portfolio"]