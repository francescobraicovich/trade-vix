from __future__ import annotations

import pandas as pd
import numpy as np

def realized_volatility_forward_window(
    returns: pd.Series, window: int = 30, ann_factor: int = 252
) -> pd.Series:
    """Compute *forward* realized volatility over the next `window` trading days.


    At each date t, RV_{t+window} = sqrt(ann_factor) * sqrt( (1/window) * sum_{i=1..window} (r_{t+i} - mean_{t+1..t+window})^2 ).

    The result is indexed by the forecast origin date t.

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
