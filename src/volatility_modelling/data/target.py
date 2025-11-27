from __future__ import annotations

import pandas as pd
import numpy as np

# NOTE: This module is DEPRECATED. Use features.py instead.
# The functions here are kept for backward compatibility only.
# The preferred functions are:
#   - features.compute_forward_realized_volatility()
#   - features.compute_backward_realized_volatility()

def realized_volatility_forward_window(
    returns: pd.Series, window: int = 30, ann_factor: int = 252
) -> pd.Series:
    """DEPRECATED: Use features.compute_forward_realized_volatility() instead.
    
    Compute *forward* realized volatility over the next `window` trading days.

    At each date t, RV_{t+window} = sqrt(ann_factor) * std(r_{t+1}, ..., r_{t+window}).

    The result is indexed by the forecast origin date t.
    
    WARNING: This implementation uses a different shift pattern than features.py.
    Both produce the same result, but features.py is clearer and preferred.
    """
    r = returns.sort_index()
    # Calculate rolling std (backward looking by default)
    # At index t+window, it uses returns from t+1 to t+window
    rolling_std = r.rolling(window=window, min_periods=window).std()
    
    # Shift backward by window so that at index t we have the volatility of t+1...t+window
    rv = rolling_std.shift(-window) * np.sqrt(ann_factor)
    
    rv.name = f"RV{window}"
    return rv

def align_target_with_features(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Join features and target on the intersection of dates, dropping any rows with NA."""
    df = features.copy()
    df[target.name] = target
    return df.dropna()
