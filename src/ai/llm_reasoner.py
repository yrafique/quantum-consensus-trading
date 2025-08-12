"""
trading_system.llm_reasoner
==========================

Wrapper for selecting and invoking a local language model.  This
module retains the original `generate_recommendation` interface but
dispatches to the underlying ``local_llm`` implementations.  The
default behaviour is to load a transformers model specified by the
``LLM_MODEL`` environment variable; if that fails or is not set, a
deterministic heuristic model is used instead.  This design allows
the rest of the trading system to remain agnostic about which model
is running and how it is invoked.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from . import local_llm


# Lazily initialise the LLM to avoid loading large models at import time.
_LLM_INSTANCE: Optional[local_llm.BaseLLM] = None


def _get_llm() -> local_llm.BaseLLM:
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        try:
            _LLM_INSTANCE = local_llm.get_default_llm()
        except Exception as e:
            logging.error(f"Failed to initialise local LLM: {e}")
            _LLM_INSTANCE = local_llm.HeuristicLLM()
    return _LLM_INSTANCE


def generate_recommendation(ticker: str, context: Dict[str, float | bool], debug_mode: bool = False) -> Optional[Dict[str, object]]:
    """Generate a structured recommendation using the selected local LLM.

    Parameters
    ----------
    ticker : str
        The stock symbol for which to generate a recommendation.
    context : dict
        Dictionary containing indicator values and metadata.
    debug_mode : bool
        If True, include detailed debug information in the response.

    Returns
    -------
    dict or None
        Recommendation dictionary or None if no action is advised.
        If debug_mode=True, includes additional debug_steps, decision_factors,
        regime, and technical_scores fields.
    """
    llm = _get_llm()
    try:
        return llm.recommend(ticker, context, debug_mode=debug_mode)
    except Exception as e:
        logging.error(f"LLM recommendation failed for {ticker}: {e}")
        return None


__all__ = ["generate_recommendation"]