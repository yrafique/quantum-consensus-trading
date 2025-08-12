"""
Trading Configuration
====================

Configuration constants for the trading system components.
"""

from pathlib import Path

# Base directories
TRADING_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = TRADING_ROOT / "data"
LOGS_DIR = TRADING_ROOT / "logs"
CONFIG_DIR = TRADING_ROOT / "config"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Portfolio files
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE = DATA_DIR / "history.json"

# Market data files
MARKET_DATA_DIR = DATA_DIR / "market_data"
MARKET_DATA_DIR.mkdir(exist_ok=True)

# Default portfolio settings
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_MAX_POSITION_SIZE = 0.25  # 25% of portfolio per position
DEFAULT_STOP_LOSS_PCT = 0.05      # 5% stop loss
DEFAULT_TARGET_PCT = 0.15         # 15% profit target

# Risk management
MAX_PORTFOLIO_RISK = 0.06         # 6% max total portfolio risk
MAX_POSITIONS = 10                # Maximum concurrent positions
MIN_TRADE_SIZE = 100.0           # Minimum trade size in dollars

# Trading signals thresholds
RSI_THRESHOLD = 65.0             # RSI threshold for momentum signals
RSI_SLOPE_MIN = 0.5              # Minimum RSI slope for trend confirmation
VOLUME_SPIKE_FACTOR = 3.0        # Volume must be 3x average for confirmation
SHORT_FLOAT_MIN = 0.20           # Minimum 20% short float for squeeze potential
DAYS_TO_COVER_MIN = 1.5          # Minimum days to cover for short squeeze
RISK_FREE_RATE = 0.02            # Risk-free rate for Sharpe ratio calculation

# Position sizing parameters (Kelly Criterion)
KELLY_SCALING_FACTOR = 0.25      # Scale down Kelly fraction for safety
MIN_POSITION_FRACTION = 0.01     # Minimum 1% position size
MAX_POSITION_FRACTION = 0.10     # Maximum 10% position size per trade

__all__ = [
    'DATA_DIR', 'LOGS_DIR', 'CONFIG_DIR',
    'PORTFOLIO_FILE', 'HISTORY_FILE', 'MARKET_DATA_DIR',
    'DEFAULT_INITIAL_CAPITAL', 'DEFAULT_MAX_POSITION_SIZE',
    'DEFAULT_STOP_LOSS_PCT', 'DEFAULT_TARGET_PCT',
    'MAX_PORTFOLIO_RISK', 'MAX_POSITIONS', 'MIN_TRADE_SIZE',
    'RSI_THRESHOLD', 'RSI_SLOPE_MIN', 'VOLUME_SPIKE_FACTOR',
    'SHORT_FLOAT_MIN', 'DAYS_TO_COVER_MIN', 'RISK_FREE_RATE',
    'KELLY_SCALING_FACTOR', 'MIN_POSITION_FRACTION', 'MAX_POSITION_FRACTION'
]