"""
trading_system.position_sizer
=============================

Functions to compute position sizes using the Kelly criterion.  The
Kelly formula allocates capital optimally for logarithmic utility
assuming the probability of winning (``W``) and the reward‑to‑risk
ratio (``R``) are known【654862323779160†L24-L37】.  In practice, these
parameters are estimated from historical data or machine learning
models.  To reduce volatility, we multiply the raw Kelly fraction by
a scaling factor and constrain the result between minimum and maximum
fractions defined in ``config.py``.
"""

from __future__ import annotations

from typing import Tuple

from .config import (
    KELLY_SCALING_FACTOR,
    MIN_POSITION_FRACTION,
    MAX_POSITION_FRACTION,
)


def compute_kelly_fraction(win_prob: float, reward_to_risk: float) -> float:
    """Compute the optimal fraction of capital to wager using Kelly.

    The Kelly fraction is ``(W * R - (1 - W)) / R`` where ``W`` is the
    probability of winning and ``R`` is the payoff ratio
    (expected profit divided by expected loss)【654862323779160†L24-L37】.

    Returns zero if the expected value is non‑positive.
    """
    if reward_to_risk <= 0:
        return 0.0
    q = 1.0 - win_prob
    numerator = win_prob * reward_to_risk - q
    if numerator <= 0:
        return 0.0
    return numerator / reward_to_risk


def compute_position_fraction(
    win_prob: float,
    reward_to_risk: float,
    scaling_factor: float = KELLY_SCALING_FACTOR,
    min_fraction: float = MIN_POSITION_FRACTION,
    max_fraction: float = MAX_POSITION_FRACTION,
) -> float:
    """Compute a scaled and clipped Kelly fraction for position sizing.

    Parameters
    ----------
    win_prob : float
        Estimated probability of a profitable trade.
    reward_to_risk : float
        Ratio of expected reward to expected risk.
    scaling_factor : float
        Factor applied to the raw Kelly fraction to temper the bet size.
    min_fraction : float
        Lower bound on the capital fraction per trade.
    max_fraction : float
        Upper bound on the capital fraction per trade.

    Returns
    -------
    float
        A number in ``[min_fraction, max_fraction]`` representing the
        fraction of capital to allocate.
    """
    kelly = compute_kelly_fraction(win_prob, reward_to_risk)
    fraction = kelly * scaling_factor
    # Clip to bounds
    fraction = max(min_fraction, min(max_fraction, fraction))
    return fraction


__all__ = [
    "compute_kelly_fraction",
    "compute_position_fraction",
]